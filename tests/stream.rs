//! Chunked decode and random-range reads.

use weftcodec::dependency::MapResolver;
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::stream;

mod support;

use support::{fix_trailer, pattern};

#[test]
fn chunked_raw_random_ranges() {
    let limits = Limits::default();
    let data = pattern(10_000, 1234);
    let frame = modes::encode_raw(&data, Some(1000), &limits).unwrap();
    let empty = MapResolver::new();

    assert_eq!(
        stream::read_range(&frame, &empty, &limits, 0, 10_000).unwrap(),
        data
    );
    assert_eq!(
        stream::read_range(&frame, &empty, &limits, 950, 200).unwrap(),
        data[950..1150]
    );
    assert_eq!(
        stream::read_range(&frame, &empty, &limits, 1000, 1000).unwrap(),
        data[1000..2000]
    );
    assert_eq!(
        stream::read_range(&frame, &empty, &limits, 9_999, 100).unwrap(),
        data[9_999..10_000]
    );
    assert!(stream::read_range(&frame, &empty, &limits, 20_000, 10)
        .unwrap()
        .is_empty());
    assert!(stream::read_range(&frame, &empty, &limits, 0, 0)
        .unwrap()
        .is_empty());

    // Sweep deterministic offsets and lengths, including chunk boundaries.
    let mut offset = 0usize;
    while offset < data.len() {
        for len in [1usize, 7, 999, 1000, 1001] {
            let out =
                stream::read_range(&frame, &empty, &limits, offset as u64, len as u64).unwrap();
            let end = (offset + len).min(data.len());
            assert_eq!(out, data[offset..end], "offset {offset} len {len}");
        }
        offset += 997;
    }
}

#[test]
fn chunked_zstd_random_ranges() {
    let limits = Limits::default();
    let data = pattern(6000, 55);
    let frame = modes::encode_zstd(&data, 3, Some(777), &limits).unwrap();
    let empty = MapResolver::new();
    for (offset, len) in [(0u64, 10u64), (700, 200), (1554, 1), (5000, 2000)] {
        let out = stream::read_range(&frame, &empty, &limits, offset, len).unwrap();
        let end = ((offset + len) as usize).min(data.len());
        assert_eq!(out, data[offset as usize..end]);
    }
    assert_eq!(
        stream::decode_chunked(&frame, &empty, &limits).unwrap(),
        data
    );
}

#[test]
fn unchunked_range_read() {
    let limits = Limits::default();
    let data = pattern(4096, 8);
    let frame = modes::encode_raw(&data, None, &limits).unwrap();
    let empty = MapResolver::new();
    assert_eq!(
        stream::read_range(&frame, &empty, &limits, 100, 50).unwrap(),
        data[100..150]
    );
}

#[test]
fn range_read_only_decodes_intersecting_chunks() {
    let limits = Limits::default();
    let data = pattern(5000, 42);
    let frame = modes::encode_raw(&data, Some(1000), &limits).unwrap();
    let parsed = weftcodec::frame::Frame::parse(&frame, &limits).unwrap();
    assert_eq!(parsed.chunks.len(), 5);

    // Corrupt the encoded bytes of chunk 3.
    let mut corrupt = frame.clone();
    let chunk = &parsed.chunks[3];
    let byte_at = 96 + parsed.header.table_len as usize + chunk.encoded_off as usize;
    corrupt[byte_at] ^= 0xff;
    fix_trailer(&mut corrupt);

    let empty = MapResolver::new();
    // Chunks 0..2 are unaffected.
    assert_eq!(
        stream::read_range(&corrupt, &empty, &limits, 0, 1000).unwrap(),
        data[0..1000]
    );
    // Touching chunk 3 fails.
    assert_eq!(
        stream::read_range(&corrupt, &empty, &limits, 3000, 100)
            .unwrap_err()
            .class(),
        weftcodec::ErrorClass::Corrupt
    );
    // Whole-frame decode also fails.
    assert_eq!(
        modes::decode_bytes(&corrupt, &empty, &limits)
            .unwrap_err()
            .class(),
        weftcodec::ErrorClass::Corrupt
    );
}
