//! Malformed-input rejection with exact error classes.

use weftcodec::dependency::{FrameSource, MapResolver};
use weftcodec::error::ErrorClass;
use weftcodec::frame::{
    encode_frame, Dependency, FrameParts, Mode, DEP_DOMAIN_PAYLOAD, DOMAIN_PAYLOAD, FORMAT_MAJOR,
    FORMAT_MINOR,
};
use weftcodec::hash::hash_domain;
use weftcodec::hash::DOMAIN_CHUNK;
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::wcb;

mod support;

use support::{dep_hash, fix_header_and_trailer, fix_trailer, pattern, resolver_with};

fn raw_frame() -> Vec<u8> {
    modes::encode_raw(&pattern(4096, 1), None, &Limits::default()).unwrap()
}

fn decode(frame: &[u8], resolver: &dyn weftcodec::dependency::Resolver) -> weftcodec::ErrorClass {
    modes::decode_bytes(frame, resolver, &Limits::default())
        .expect_err("expected decode failure")
        .class()
}

fn parse(frame: &[u8]) -> weftcodec::ErrorClass {
    weftcodec::frame::Frame::parse(frame, &Limits::default())
        .expect_err("expected parse failure")
        .class()
}

#[test]
fn structural_header_errors() {
    let good = raw_frame();

    let mut bad_magic = good.clone();
    bad_magic[0] ^= 0xff;
    assert_eq!(parse(&bad_magic), ErrorClass::Malformed);

    let mut bad_header_crc = good.clone();
    bad_header_crc[72] ^= 0xff;
    assert_eq!(parse(&bad_header_crc), ErrorClass::Corrupt);

    let mut bad_trailer = good.clone();
    let last = bad_trailer.len() - 1;
    bad_trailer[last] ^= 0xff;
    assert_eq!(parse(&bad_trailer), ErrorClass::Corrupt);

    let mut truncated = good.clone();
    truncated.pop();
    assert_eq!(parse(&truncated), ErrorClass::Malformed);

    let mut major = good.clone();
    major[4] = 2;
    fix_header_and_trailer(&mut major);
    assert_eq!(parse(&major), ErrorClass::UnsupportedVersion);

    let mut minor = good.clone();
    minor[5] = 1;
    fix_header_and_trailer(&mut minor);
    assert_eq!(parse(&minor), ErrorClass::UnsupportedVersion);

    // Modes 6, 7, 8, and 9 are allocated (TENSOR, TOKEN_SEQUENCE,
    // QUANTIZED_TENSOR, FLOAT_TENSOR); 10 is reserved.
    let mut mode = good.clone();
    mode[7] = 10;
    fix_header_and_trailer(&mut mode);
    assert_eq!(parse(&mode), ErrorClass::UnsupportedMode);

    let mut feature = good.clone();
    let features = u16::from_le_bytes([feature[36], feature[37]]) | 0x0004;
    feature[36..38].copy_from_slice(&features.to_le_bytes());
    fix_header_and_trailer(&mut feature);
    assert_eq!(parse(&feature), ErrorClass::UnsupportedFeature);

    let mut flag3 = good.clone();
    let flags = u16::from_le_bytes([flag3[8], flag3[9]]) | 0x0008;
    flag3[8..10].copy_from_slice(&flags.to_le_bytes());
    fix_header_and_trailer(&mut flag3);
    assert_eq!(parse(&flag3), ErrorClass::UnsupportedFeature);

    let mut self_contained = good.clone();
    let flags = u16::from_le_bytes([self_contained[8], self_contained[9]]) | 0x0002;
    self_contained[8..10].copy_from_slice(&flags.to_le_bytes());
    fix_header_and_trailer(&mut self_contained);
    assert_eq!(parse(&self_contained), ErrorClass::UnsupportedFeature);

    let mut reserved = good.clone();
    reserved[38] = 1;
    fix_header_and_trailer(&mut reserved);
    assert_eq!(parse(&reserved), ErrorClass::Malformed);

    let mut table_len = good.clone();
    table_len[76] = 1;
    fix_header_and_trailer(&mut table_len);
    assert_eq!(parse(&table_len), ErrorClass::Malformed);

    let mut body_len = good.clone();
    let value = u64::from_le_bytes(body_len[20..28].try_into().unwrap_or([0; 8])) + 1;
    body_len[20..28].copy_from_slice(&value.to_le_bytes());
    fix_header_and_trailer(&mut body_len);
    assert_eq!(parse(&body_len), ErrorClass::Malformed);

    let mut bad_domain = good.clone();
    bad_domain[6] = 2;
    fix_header_and_trailer(&mut bad_domain);
    assert_eq!(parse(&bad_domain), ErrorClass::Malformed);

    let mut incomplete = vec![0u8; 99];
    incomplete[0..4].copy_from_slice(b"WFT1");
    assert_eq!(parse(&incomplete), ErrorClass::Malformed);
}

#[test]
fn chunk_table_errors() {
    let good = modes::encode_raw(&pattern(3000, 7), Some(1024), &Limits::default()).unwrap();
    let second_off_at = 96 + 48;

    let mut gap = good.clone();
    let off = u64::from_le_bytes(
        gap[second_off_at..second_off_at + 8]
            .try_into()
            .unwrap_or([0; 8]),
    ) + 1;
    gap[second_off_at..second_off_at + 8].copy_from_slice(&off.to_le_bytes());
    fix_trailer(&mut gap);
    assert_eq!(parse(&gap), ErrorClass::Malformed);

    let mut overlap = good.clone();
    overlap[second_off_at..second_off_at + 8].copy_from_slice(&96u64.to_le_bytes());
    fix_trailer(&mut overlap);
    assert_eq!(parse(&overlap), ErrorClass::Malformed);

    let mut zero_len = good.clone();
    zero_len[96 + 8..96 + 12].copy_from_slice(&0u32.to_le_bytes());
    fix_trailer(&mut zero_len);
    assert_eq!(parse(&zero_len), ErrorClass::Malformed);

    let mut sum = good.clone();
    let value = u32::from_le_bytes(sum[96 + 12..96 + 16].try_into().unwrap_or([0; 4])) + 1;
    sum[96 + 12..96 + 16].copy_from_slice(&value.to_le_bytes());
    fix_trailer(&mut sum);
    assert_eq!(parse(&sum), ErrorClass::Malformed);

    // Corrupt one encoded byte: chunk hash mismatch.
    let parsed = weftcodec::frame::Frame::parse(&good, &Limits::default()).unwrap();
    let body_start = 96 + parsed.header.table_len as usize;
    let mut body_corrupt = good.clone();
    body_corrupt[body_start] ^= 0xff;
    fix_trailer(&mut body_corrupt);
    assert_eq!(
        decode(&body_corrupt, &MapResolver::new()),
        ErrorClass::Corrupt
    );
}

#[test]
fn dependency_table_errors() {
    let base = pattern(1000, 3);
    let target = pattern(1200, 4);
    let frame = modes::encode_delta(&target, &base, &Limits::default()).unwrap();
    let dep_at = 96;

    let mut bad_domain = frame.clone();
    bad_domain[dep_at + 40] = 3;
    fix_trailer(&mut bad_domain);
    assert_eq!(parse(&bad_domain), ErrorClass::Malformed);

    let mut no_required = frame.clone();
    no_required[dep_at + 41] = 0;
    fix_trailer(&mut no_required);
    assert_eq!(parse(&no_required), ErrorClass::Malformed);

    let mut unknown_flag = frame.clone();
    unknown_flag[dep_at + 41] = 0x03;
    fix_trailer(&mut unknown_flag);
    assert_eq!(parse(&unknown_flag), ErrorClass::Malformed);

    let mut reserved = frame.clone();
    reserved[dep_at + 42] = 1;
    fix_trailer(&mut reserved);
    assert_eq!(parse(&reserved), ErrorClass::Malformed);

    // Missing dependency.
    assert_eq!(
        decode(&frame, &MapResolver::new()),
        ErrorClass::DependencyMissing
    );

    // Bad declared length: resolver returns the real base.
    let mut bad_len = frame.clone();
    let len = u64::from_le_bytes(
        bad_len[dep_at + 32..dep_at + 40]
            .try_into()
            .unwrap_or([0; 8]),
    ) - 1;
    bad_len[dep_at + 32..dep_at + 40].copy_from_slice(&len.to_le_bytes());
    fix_trailer(&mut bad_len);
    assert_eq!(
        decode(&bad_len, &resolver_with(&[base.clone()])),
        ErrorClass::DependencyMismatch
    );

    // Bad declared hash: the resolver is keyed by the mutated hash but holds
    // the real bytes.
    let mut bad_hash = frame.clone();
    bad_hash[dep_at] ^= 0xff;
    fix_trailer(&mut bad_hash);
    let mut resolver = MapResolver::new();
    let mut wrong = [0u8; 32];
    wrong.copy_from_slice(&bad_hash[dep_at..dep_at + 32]);
    resolver.insert(weftcodec::Hash32::from_bytes(wrong), base.clone());
    assert_eq!(decode(&bad_hash, &resolver), ErrorClass::DependencyMismatch);
}

#[test]
fn delta_op_errors() {
    let base = pattern(1000, 11);
    let dep = Dependency::new(dep_hash(&base), base.len() as u64, DEP_DOMAIN_PAYLOAD, 0);
    let limits = Limits::default();

    let build = |body: &[u8], decoded_len: u64| -> Vec<u8> {
        encode_frame(
            &FrameParts {
                format_major: FORMAT_MAJOR,
                format_minor: FORMAT_MINOR,
                domain: DOMAIN_PAYLOAD,
                mode: Mode::Delta,
                flags: 0,
                level: 0,
                decoded_len,
                content_hash: hash_domain(DOMAIN_CHUNK, &vec![0u8; decoded_len as usize]),
                dependencies: std::slice::from_ref(&dep),
                chunks: &[],
                body,
            },
            &limits,
        )
        .unwrap()
    };
    let resolver = resolver_with(&[base.clone()]);

    // COPY offset beyond the base.
    let body = [0x00, 0xE8, 0x07, 0x01];
    assert_eq!(decode(&build(&body, 1), &resolver), ErrorClass::Malformed);

    // COPY length zero.
    let body = [0x00, 0x00, 0x00];
    assert_eq!(decode(&build(&body, 1), &resolver), ErrorClass::Malformed);

    // INSERT length zero.
    let body = [0x01, 0x00];
    assert_eq!(decode(&build(&body, 1), &resolver), ErrorClass::Malformed);

    // INSERT extends past the body.
    let body = [0x01, 0x05, b'a'];
    assert_eq!(decode(&build(&body, 5), &resolver), ErrorClass::Malformed);

    // Unknown opcode.
    let body = [0x02];
    assert_eq!(decode(&build(&body, 0), &resolver), ErrorClass::Malformed);

    // Output shorter than decoded_len.
    let body = [0x00, 0x00, 0x01];
    assert_eq!(decode(&build(&body, 2), &resolver), ErrorClass::Corrupt);

    // Output longer than decoded_len.
    let body = [0x00, 0x00, 0x05];
    assert_eq!(decode(&build(&body, 2), &resolver), ErrorClass::Corrupt);

    // Non-minimal varint inside the op stream.
    let body = [0x00, 0x80, 0x00, 0x01];
    assert_eq!(
        decode(&build(&body, 1), &resolver),
        ErrorClass::NonCanonical
    );
}

#[test]
fn wcb_canonical_errors() {
    assert_eq!(
        wcb::decode(&[0x03, 0x80, 0x00], &Limits::default())
            .unwrap_err()
            .class(),
        ErrorClass::NonCanonical
    );
    let unsorted = [
        0x08, 0x02, 0x05, 0x01, b'b', 0x03, 0x01, 0x05, 0x01, b'a', 0x03, 0x02,
    ];
    assert_eq!(
        wcb::decode(&unsorted, &Limits::default())
            .unwrap_err()
            .class(),
        ErrorClass::NonCanonical
    );
    let duplicate = [
        0x08, 0x02, 0x05, 0x01, b'a', 0x03, 0x01, 0x05, 0x01, b'a', 0x03, 0x02,
    ];
    assert_eq!(
        wcb::decode(&duplicate, &Limits::default())
            .unwrap_err()
            .class(),
        ErrorClass::NonCanonical
    );
    assert_eq!(
        wcb::decode(&[0x00, 0x00], &Limits::default())
            .unwrap_err()
            .class(),
        ErrorClass::NonCanonical
    );
    assert_eq!(
        wcb::decode(&[0x0a], &Limits::default())
            .unwrap_err()
            .class(),
        ErrorClass::NonCanonical
    );
    assert_eq!(
        wcb::decode(&[0x05, 0x02, b'a'], &Limits::default())
            .unwrap_err()
            .class(),
        ErrorClass::Malformed
    );
    let mut deep = Vec::new();
    for _ in 0..65 {
        deep.push(0x07);
        deep.push(0x01);
    }
    deep.push(0x00);
    assert_eq!(
        wcb::decode(&deep, &Limits::default()).unwrap_err().class(),
        ErrorClass::LimitExceeded
    );
}

#[test]
fn structured_frame_non_canonical_body() {
    let limits = Limits::default();
    let frame =
        modes::encode_structured(&wcb::Value::map([("a", wcb::Value::U64(1))]), &limits).unwrap();
    let mut corrupt = frame.clone();
    let body_start = 96;
    corrupt[body_start] = 0x0a;
    fix_trailer(&mut corrupt);
    assert_eq!(
        decode(&corrupt, &MapResolver::new()),
        ErrorClass::NonCanonical
    );
}

#[test]
fn limit_enforcement() {
    let frame = raw_frame();
    let limits = Limits::default().with_max_decoded_bytes(100);
    assert_eq!(
        weftcodec::frame::Frame::parse(&frame, &limits)
            .unwrap_err()
            .class(),
        ErrorClass::LimitExceeded
    );
    let limits = Limits::default().with_max_frame_bytes(50);
    assert_eq!(
        weftcodec::frame::Frame::parse(&frame, &limits)
            .unwrap_err()
            .class(),
        ErrorClass::LimitExceeded
    );
    let chunked = modes::encode_raw(&pattern(5000, 1), Some(1024), &Limits::default()).unwrap();
    let limits = Limits::default().with_max_chunks(2);
    assert_eq!(
        weftcodec::frame::Frame::parse(&chunked, &limits)
            .unwrap_err()
            .class(),
        ErrorClass::LimitExceeded
    );
}

/// Custom source returning an unrelated frame to test content-hash binding.
struct WrongSource(Vec<u8>);

impl FrameSource for WrongSource {
    fn get_frame(&self, _content_hash: &weftcodec::Hash32) -> weftcodec::Result<Option<Vec<u8>>> {
        Ok(Some(self.0.clone()))
    }
}

#[test]
fn frame_source_content_hash_binding() {
    let base = b"base-bytes";
    let delta = modes::encode_delta(b"base-bytes-plus", base, &Limits::default()).unwrap();
    let other = modes::encode_raw(b"other", None, &Limits::default()).unwrap();
    let err =
        weftcodec::dependency::decode_frame_dag(&WrongSource(other), &delta, &Limits::default())
            .expect_err("wrong frame must be rejected");
    assert_eq!(err.class(), ErrorClass::DependencyMismatch);
}
