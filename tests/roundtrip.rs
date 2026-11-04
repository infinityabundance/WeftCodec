//! Round-trip exactness across sizes and arbitrary payload bytes.

use weftcodec::dependency::MapResolver;
use weftcodec::frame::Frame;
use weftcodec::hash::hash_domain;
use weftcodec::hash::DOMAIN_CHUNK;
use weftcodec::limits::Limits;
use weftcodec::modes;

mod support;

fn roundtrip(data: &[u8]) {
    let limits = Limits::default();
    let frame = modes::encode_raw(data, None, &limits).expect("encode raw");
    let parsed = Frame::parse(&frame, &limits).expect("parse frame");
    assert_eq!(parsed.header.decoded_len, data.len() as u64);
    assert_eq!(
        parsed.header.content_hash,
        hash_domain(DOMAIN_CHUNK, data),
        "header content hash"
    );
    let decoded = modes::decode_bytes(&frame, &MapResolver::new(), &limits).expect("decode raw");
    assert_eq!(decoded, data);
}

#[test]
fn zero_length() {
    roundtrip(b"");
    let frame = modes::encode_raw(b"", None, &Limits::default()).unwrap();
    assert_eq!(frame.len(), 100, "header + trailer only");
}

#[test]
fn one_byte() {
    roundtrip(b"\x00");
    roundtrip(b"A");
    roundtrip(b"\xff");
}

#[test]
fn four_kib_random() {
    for seed in [1u64, 0xdead_beef, 0x0123_4567_89ab_cdef] {
        roundtrip(&support::pattern(4096, seed));
    }
}

#[test]
fn one_mib_random() {
    roundtrip(&support::pattern(1 << 20, 0x1234_5678_9abc_def0));
}

#[test]
fn four_mib_repetitive() {
    roundtrip(&support::repeat(b"weft-context-", 4 << 20));
}

#[test]
fn invalid_utf8_payload_bytes() {
    let mut data: Vec<u8> = vec![0xff, 0xfe, 0x00, 0xc0, 0x80, 0xed, 0xa0, 0x80];
    data.extend_from_slice(&support::pattern(1024, 42));
    roundtrip(&data);
    // Bytes are never interpreted as text by the payload codec.
    let frame = modes::encode_raw(&data, None, &Limits::default()).unwrap();
    let decoded = modes::decode_bytes(&frame, &MapResolver::new(), &Limits::default()).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn chunked_frames_have_same_content_hash() {
    let limits = Limits::default();
    let data = support::pattern(10_000, 5);
    let unchunked = modes::encode_raw(&data, None, &limits).unwrap();
    let chunked = modes::encode_raw(&data, Some(1024), &limits).unwrap();
    let a = Frame::parse(&unchunked, &limits).unwrap();
    let b = Frame::parse(&chunked, &limits).unwrap();
    assert_eq!(a.header.content_hash, b.header.content_hash);
    assert_eq!(a.header.chunk_count, 0);
    assert_eq!(b.header.chunk_count, 10);
    assert_eq!(
        modes::decode_bytes(&unchunked, &MapResolver::new(), &limits).unwrap(),
        modes::decode_bytes(&chunked, &MapResolver::new(), &limits).unwrap()
    );
}
