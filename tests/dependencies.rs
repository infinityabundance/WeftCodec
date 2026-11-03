//! Dependency DAG decoding, cycle detection, and WFA1 archives.

use weftcodec::archive;
use weftcodec::dependency::{decode_frame_dag, ArchiveResolver, MapFrameSource, MapResolver};
use weftcodec::error::ErrorClass;
use weftcodec::frame::{
    encode_frame, Dependency, FrameParts, Mode, DEP_DOMAIN_PAYLOAD, DOMAIN_PAYLOAD, FORMAT_MAJOR,
    FORMAT_MINOR,
};
use weftcodec::hash::{hash_domain, Hash32, DOMAIN_CHUNK};
use weftcodec::limits::Limits;
use weftcodec::modes;

mod support;

use support::pattern;

fn reference_parts(content_hash: Hash32, dep: Hash32, decoded_len: u64) -> Vec<u8> {
    encode_frame(
        &FrameParts {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            domain: DOMAIN_PAYLOAD,
            mode: Mode::Reference,
            flags: 0,
            level: 0,
            decoded_len,
            content_hash,
            dependencies: &[Dependency::new(dep, decoded_len, DEP_DOMAIN_PAYLOAD, 0)],
            chunks: &[],
            body: &[],
        },
        &Limits::default(),
    )
    .unwrap()
}

#[test]
fn frame_dag_success() {
    let limits = Limits::default();
    let base = pattern(2048, 10);
    let target = {
        let mut target = base.clone();
        target[100] ^= 0xff;
        target.extend_from_slice(b"tail");
        target
    };
    let base_frame = modes::encode_raw(&base, None, &limits).unwrap();
    let delta_frame = modes::encode_delta(&target, &base, &limits).unwrap();
    let mut source = MapFrameSource::new();
    source.insert_frame(base_frame).unwrap();
    source.insert_frame(delta_frame.clone()).unwrap();
    assert_eq!(
        decode_frame_dag(&source, &delta_frame, &limits).unwrap(),
        target
    );
}

#[test]
fn frame_dag_cycle() {
    let limits = Limits::default();
    let a_hash = Hash32::ZERO;
    let b_hash = Hash32::from_bytes([2u8; 32]);
    let a = reference_parts(a_hash, b_hash, 1);
    let b = reference_parts(b_hash, a_hash, 1);
    let mut source = MapFrameSource::new();
    source.insert_frame(a.clone()).unwrap();
    source.insert_frame(b.clone()).unwrap();
    let err = decode_frame_dag(&source, &a, &limits).unwrap_err();
    assert_eq!(err.class(), ErrorClass::DependencyCycle);
}

#[test]
fn frame_dag_depth_limit() {
    let limits = Limits::default();
    let mut previous = b"a".to_vec();
    let mut source = MapFrameSource::new();
    let mut root = modes::encode_raw(&previous, None, &limits).unwrap();
    source.insert_frame(root.clone()).unwrap();
    for index in 0..10u8 {
        let mut next = previous.clone();
        next.push(b'a' + index);
        let frame = modes::encode_delta(&next, &previous, &limits).unwrap();
        source.insert_frame(frame.clone()).unwrap();
        root = frame;
        previous = next;
    }
    let err = decode_frame_dag(&source, &root, &limits).unwrap_err();
    assert_eq!(err.class(), ErrorClass::LimitExceeded);
}

#[test]
fn frame_dag_fetch_limit_and_missing() {
    let limits = Limits::default();
    let base = pattern(1000, 2);
    let target = pattern(1200, 3);
    let base_frame = modes::encode_raw(&base, None, &limits).unwrap();
    let delta = modes::encode_delta(&target, &base, &limits).unwrap();
    let mut source = MapFrameSource::new();
    source.insert_frame(base_frame).unwrap();
    source.insert_frame(delta.clone()).unwrap();

    let tiny = Limits::default().with_max_total_fetched_bytes(10);
    assert_eq!(
        decode_frame_dag(&source, &delta, &tiny)
            .unwrap_err()
            .class(),
        ErrorClass::LimitExceeded
    );

    let empty = MapFrameSource::new();
    assert_eq!(
        decode_frame_dag(&empty, &delta, &limits)
            .unwrap_err()
            .class(),
        ErrorClass::DependencyMissing
    );
}

fn pack_archive(frames: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"WFA1");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    for frame in frames {
        out.extend_from_slice(&(frame.len() as u64).to_le_bytes());
        out.extend_from_slice(frame);
    }
    let crc = crc32c::crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

#[test]
fn archive_roundtrip_and_topological_order() {
    let limits = Limits::default();
    let base = pattern(4096, 5);
    let target = {
        let mut target = base.clone();
        target[10] ^= 1;
        target.extend_from_slice(b"archive-tail");
        target
    };
    let base_frame = modes::encode_raw(&base, None, &limits).unwrap();
    let delta = modes::encode_delta(&target, &base, &limits).unwrap();

    // Input order is deliberately reversed; creation must sort topologically.
    let bytes = archive::create_archive(&[delta.clone(), base_frame.clone()], &limits).unwrap();
    let parsed = archive::parse_archive(&bytes, &limits).unwrap();
    assert_eq!(parsed.entries.len(), 2);
    assert_eq!(
        parsed.entries[0].frame.header.content_hash,
        hash_domain(DOMAIN_CHUNK, &base)
    );
    assert_eq!(
        parsed.entries[1].frame.header.content_hash,
        hash_domain(DOMAIN_CHUNK, &target)
    );

    let decoded = archive::decode_archive(&bytes, &limits).unwrap();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].bytes, base);
    assert_eq!(decoded[1].bytes, target);

    // ArchiveResolver can supply the base.
    let resolver = ArchiveResolver::from_archive(&bytes, &limits).unwrap();
    let plain = modes::decode_bytes(&delta, &resolver, &limits).unwrap();
    assert_eq!(plain, target);
}

#[test]
fn archive_missing_dependency_rejected() {
    let limits = Limits::default();
    let base = pattern(100, 1);
    let delta = modes::encode_delta(&pattern(120, 2), &base, &limits).unwrap();
    let err = archive::create_archive(&[delta], &limits).unwrap_err();
    assert_eq!(err.class(), ErrorClass::DependencyMissing);
}

#[test]
fn archive_non_topological_rejected() {
    let limits = Limits::default();
    let base = pattern(1000, 1);
    let target = pattern(1200, 2);
    let base_frame = modes::encode_raw(&base, None, &limits).unwrap();
    let delta = modes::encode_delta(&target, &base, &limits).unwrap();
    // Delta before its base: structurally fine, order is not.
    let bytes = pack_archive(&[&delta, &base_frame]);
    let err = archive::parse_archive(&bytes, &limits).unwrap_err();
    assert_eq!(err.class(), ErrorClass::Malformed);
}

#[test]
fn archive_truncation_and_crc() {
    let limits = Limits::default();
    let frame = modes::encode_raw(b"archive payload", None, &limits).unwrap();
    let bytes = archive::create_archive(&[frame], &limits).unwrap();

    let mut bad_crc = bytes.clone();
    let last = bad_crc.len() - 1;
    bad_crc[last] ^= 0xff;
    assert_eq!(
        archive::parse_archive(&bad_crc, &limits)
            .unwrap_err()
            .class(),
        ErrorClass::Corrupt
    );

    let mut truncated = bytes.clone();
    truncated.pop();
    let err = archive::parse_archive(&truncated, &limits).unwrap_err();
    assert!(matches!(
        err.class(),
        ErrorClass::Corrupt | ErrorClass::Malformed
    ));

    // Truncate inside the frame payload: the CRC no longer matches.
    let short = &bytes[..bytes.len() - 10];
    let err = archive::parse_archive(short, &limits).unwrap_err();
    assert!(matches!(
        err.class(),
        ErrorClass::Corrupt | ErrorClass::Malformed
    ));
}

#[test]
fn archive_duplicate_identical_frame_accepted() {
    let limits = Limits::default();
    let frame = modes::encode_raw(b"duplicate", None, &limits).unwrap();
    let bytes = archive::create_archive(&[frame.clone(), frame], &limits).unwrap();
    let decoded = archive::decode_archive(&bytes, &limits).unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].bytes, b"duplicate");
}

#[test]
fn archive_duplicate_conflicting_frame_rejected() {
    let limits = Limits::default();
    let frame = modes::encode_raw(b"same content", None, &limits).unwrap();
    let mut altered = frame.clone();
    // Change the informational level field: content hash stays the same but
    // the bytes differ.
    altered[10] = 7;
    support::fix_header_and_trailer(&mut altered);
    let err = archive::create_archive(&[frame, altered], &limits).unwrap_err();
    assert_eq!(err.class(), ErrorClass::Malformed);
}

#[test]
fn archive_dependency_graph_cycle_rejected() {
    let a_hash = Hash32::ZERO;
    let b_hash = Hash32::from_bytes([2u8; 32]);
    let a = reference_parts(a_hash, b_hash, 1);
    let b = reference_parts(b_hash, a_hash, 1);
    let err = archive::create_archive(&[a, b], &Limits::default()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::DependencyCycle);
}

#[test]
fn cycle_broken_by_resolver_is_still_rejected() {
    // A reference cycle cannot be resolved by a plain resolver either.
    let a_hash = Hash32::ZERO;
    let b_hash = Hash32::from_bytes([3u8; 32]);
    let a = reference_parts(a_hash, b_hash, 1);
    let _ = a;
    let mut map = MapResolver::new();
    map.insert(b_hash, vec![0u8; 1]);
    // The resolver returns bytes but their hash is not b_hash.
    assert_eq!(
        modes::decode_bytes(&a, &map, &Limits::default())
            .unwrap_err()
            .class(),
        ErrorClass::DependencyMismatch
    );
}
