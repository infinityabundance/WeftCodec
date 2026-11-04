//! Every payload mode, including chunked forms, dictionaries, references,
//! realistic deltas, and structured WCB values.

use weftcodec::dependency::MapResolver;
use weftcodec::frame::{Mode, DEP_DOMAIN_DICTIONARY};
use weftcodec::hash::hash_domain;
use weftcodec::hash::DOMAIN_CHUNK;
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::wcb::{self, Value};

mod support;

use support::{dep_hash, pattern, repeat, resolver_with};

#[test]
fn raw_and_zstd_chunked() {
    let limits = Limits::default();
    let data = pattern(12_345, 77);
    let empty = MapResolver::new();

    let raw_chunked = modes::encode_raw(&data, Some(1000), &limits).unwrap();
    assert_eq!(
        modes::decode_bytes(&raw_chunked, &empty, &limits).unwrap(),
        data
    );

    let zstd_plain = modes::encode_zstd(&data, 3, None, &limits).unwrap();
    let zstd_chunked = modes::encode_zstd(&data, 3, Some(1000), &limits).unwrap();
    assert_eq!(
        modes::decode_bytes(&zstd_plain, &empty, &limits).unwrap(),
        data
    );
    assert_eq!(
        modes::decode_bytes(&zstd_chunked, &empty, &limits).unwrap(),
        data
    );
    let chunked_header = weftcodec::frame::Frame::parse(&zstd_chunked, &limits).unwrap();
    assert_eq!(chunked_header.header.chunk_count, 13);
}

#[test]
fn zstd_compresses_repetitive_data() {
    let limits = Limits::default();
    let data = repeat(b"weft codec context ", 4096);
    let frame = modes::encode_zstd(&data, 3, None, &limits).unwrap();
    assert!(
        frame.len() < data.len() / 4,
        "zstd frame {} should be much smaller than {}",
        frame.len(),
        data.len()
    );
    assert_eq!(
        modes::decode_bytes(&frame, &MapResolver::new(), &limits).unwrap(),
        data
    );
}

#[test]
fn reference_roundtrip_and_mismatch() {
    let limits = Limits::default();
    let base = repeat(b"reference base", 100);
    let frame = modes::encode_reference(&base, &base, &limits).unwrap();
    let parsed = weftcodec::frame::Frame::parse(&frame, &limits).unwrap();
    assert_eq!(parsed.header.mode, Mode::Reference);
    assert_eq!(parsed.header.body_len, 0);
    assert_eq!(parsed.header.dep_count, 1);
    let decoded = modes::decode_bytes(&frame, &resolver_with(&[base.clone()]), &limits).unwrap();
    assert_eq!(decoded, base);

    // A REFERENCE frame whose base differs is rejected by the encoder.
    let err = modes::encode_reference(b"other", &base, &limits).unwrap_err();
    assert_eq!(err.class(), weftcodec::ErrorClass::Malformed);
}

#[test]
fn delta_realistic_edits() {
    let limits = Limits::default();
    let base = pattern(4096, 0xfeed_face);
    let mut target = Vec::new();
    target.extend_from_slice(&base[..100]);
    target.extend_from_slice(b"INSERTED-REVISION-2");
    target.extend_from_slice(&base[150..500]);
    target.extend_from_slice(b"REPLACED");
    target.extend_from_slice(&base[700..3000]);
    target.extend_from_slice(&base[3200..]);

    let frame = modes::encode_delta(&target, &base, &limits).unwrap();
    let parsed = weftcodec::frame::Frame::parse(&frame, &limits).unwrap();
    assert_eq!(parsed.header.mode, Mode::Delta);
    assert_eq!(parsed.header.dep_count, 1);
    assert!(
        parsed.body.len() < target.len(),
        "delta body {} should be smaller than target {}",
        parsed.body.len(),
        target.len()
    );
    let decoded = modes::decode_bytes(&frame, &resolver_with(&[base.clone()]), &limits).unwrap();
    assert_eq!(decoded, target);
}

#[test]
fn delta_empty_target_and_identical_target() {
    let limits = Limits::default();
    let base = pattern(1000, 9);
    let empty = modes::encode_delta(b"", &base, &limits).unwrap();
    assert_eq!(
        modes::decode_bytes(&empty, &resolver_with(&[base.clone()]), &limits).unwrap(),
        Vec::<u8>::new()
    );
    // Identical target is a single COPY.
    let frame = modes::encode_delta(&base, &base, &limits).unwrap();
    let parsed = weftcodec::frame::Frame::parse(&frame, &limits).unwrap();
    assert_eq!(parsed.body, &[0x00, 0x00, 0xE8, 0x07]);
    assert_eq!(
        modes::decode_bytes(&frame, &resolver_with(&[base.clone()]), &limits).unwrap(),
        base
    );
}

#[test]
fn dictionary_roundtrip() {
    let limits = Limits::default();
    let dict = repeat(b"weft dictionary alpha beta gamma\n", 512);
    let data = repeat(b"weft dictionary alpha beta gamma\n", 200);
    let frame = modes::encode_dictionary(&data, &dict, 3, &limits).unwrap();
    let parsed = weftcodec::frame::Frame::parse(&frame, &limits).unwrap();
    assert_eq!(parsed.header.mode, Mode::Dictionary);
    assert_eq!(parsed.dependencies[0].dep_domain, DEP_DOMAIN_DICTIONARY);
    let decoded = modes::decode_bytes(&frame, &resolver_with(&[dict.clone()]), &limits).unwrap();
    assert_eq!(decoded, data);
    // Wrong dictionary fails dependency verification.
    let wrong = vec![0u8; dict.len()];
    let resolver = MapResolver::new();
    let _ = resolver;
    let mut map = MapResolver::new();
    map.insert(dep_hash(&dict), wrong);
    assert_eq!(
        modes::decode_bytes(&frame, &map, &limits)
            .unwrap_err()
            .class(),
        weftcodec::ErrorClass::DependencyMismatch
    );
}

#[test]
fn structured_values() {
    let limits = Limits::default();
    let values = vec![
        Value::Null,
        Value::Bool(true),
        Value::U64(u64::MAX),
        Value::I64(i64::MIN),
        Value::Timestamp(1_700_000_000_000_000_000),
        Value::Str("héllo 😀".to_owned()),
        Value::Bytes(vec![0, 1, 2, 0xff]),
        Value::Array(vec![Value::U64(1), Value::Null]),
        Value::map([
            ("a", Value::U64(1)),
            ("b", Value::Array(vec![Value::Bool(false)])),
        ]),
    ];
    for value in values {
        let frame = modes::encode_structured(&value, &limits).unwrap();
        let parsed = weftcodec::frame::Frame::parse(&frame, &limits).unwrap();
        assert_eq!(parsed.header.mode, Mode::Structured);
        assert_eq!(parsed.header.domain, weftcodec::frame::DOMAIN_STRUCTURED);
        let body = modes::decode_bytes(&frame, &MapResolver::new(), &limits).unwrap();
        assert_eq!(body, wcb::encode(&value, &limits).unwrap());
        assert_eq!(wcb::decode(&body, &limits).unwrap(), value);
    }
}

#[test]
fn deterministic_encoding() {
    let limits = Limits::default();
    let data = pattern(8192, 123);
    let base = pattern(8192, 123);
    let mut edited = data.clone();
    edited[100] ^= 0xff;

    assert_eq!(
        modes::encode_raw(&data, None, &limits).unwrap(),
        modes::encode_raw(&data, None, &limits).unwrap()
    );
    assert_eq!(
        modes::encode_raw(&data, Some(1024), &limits).unwrap(),
        modes::encode_raw(&data, Some(1024), &limits).unwrap()
    );
    assert_eq!(
        modes::encode_reference(&data, &data, &limits).unwrap(),
        modes::encode_reference(&data, &data, &limits).unwrap()
    );
    assert_eq!(
        modes::encode_delta(&edited, &base, &limits).unwrap(),
        modes::encode_delta(&edited, &base, &limits).unwrap()
    );
    assert_eq!(
        modes::encode_structured(&Value::U64(7), &limits).unwrap(),
        modes::encode_structured(&Value::U64(7), &limits).unwrap()
    );

    let zstd_a = modes::encode_zstd(&data, 3, None, &limits).unwrap();
    let zstd_b = modes::encode_zstd(&data, 3, None, &limits).unwrap();
    assert_eq!(zstd_a, zstd_b, "same zstd library/level is deterministic");
}

#[test]
fn header_hash_is_plaintext_hash() {
    let limits = Limits::default();
    let data = pattern(777, 3);
    let frame = modes::encode_zstd(&data, 5, None, &limits).unwrap();
    let parsed = weftcodec::frame::Frame::parse(&frame, &limits).unwrap();
    assert_eq!(parsed.header.content_hash, hash_domain(DOMAIN_CHUNK, &data));
}
