//! Mode 20 `ENGRAM_SHARD` round-trip, region and conformance replay
//! (`spec/WEFT-ENGRAM.md`).

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use weftcodec::dependency::MapResolver;
use weftcodec::engram;
use weftcodec::frame::Frame;
use weftcodec::Limits;

fn v1_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1")
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    assert!(hex.len() % 2 == 0, "odd hex length");
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).unwrap())
        .collect()
}

fn load(path: &Path) -> Json {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
fn engram_round_trip_is_exact_and_deterministic() {
    let limits = Limits::default();
    let mut state = 0x1234u64;
    let table: Vec<u8> = (0..8 * 3 * 2)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xff) as u8
        })
        .collect();
    let input = engram::EngramInput::inline(
        weftcodec::TensorDtype::Bf16,
        vec![8, 3],
        vec![0, 3, 8],
        engram::ENGRAM_HASH_BLAKE3,
        8,
    );
    let first = engram::encode_engram(&table, &input, &limits).unwrap();
    let second = engram::encode_engram(&table, &input, &limits).unwrap();
    assert_eq!(first, second, "encoding must be deterministic");
    let frame = Frame::parse(&first, &limits).unwrap();
    assert_eq!(frame.header.mode, weftcodec::Mode::EngramShard);
    let decoded = engram::decode_engram_without_dependencies(&frame, &limits).unwrap();
    assert_eq!(decoded, table);
    let descriptor = engram::parse_engram(&frame, &limits).unwrap();
    assert_eq!(descriptor.meta.groups.len(), 2);
    for group in &descriptor.meta.groups {
        assert!(matches!(
            group.storage,
            engram::EngramStorage::Inline { .. }
        ));
    }
    let root = descriptor.meta.descriptor_root(&limits).unwrap();
    assert_eq!(
        root,
        engram::parse_engram(&frame, &limits)
            .unwrap()
            .meta
            .descriptor_root(&limits)
            .unwrap()
    );
}

#[test]
fn untouched_groups_are_not_decoded() {
    let limits = Limits::default();
    let mut state = 0x55u64;
    let table: Vec<u8> = (0..64 * 4 * 2)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xff) as u8
        })
        .collect();
    let edges: Vec<u64> = (0..=16).map(|index| index * 4).collect();
    let input = engram::EngramInput::inline(
        weftcodec::TensorDtype::Bf16,
        vec![64, 4],
        edges,
        engram::ENGRAM_HASH_BLAKE3,
        64,
    );
    let frame_bytes = engram::encode_engram(&table, &input, &limits).unwrap();
    let frame = Frame::parse(&frame_bytes, &limits).unwrap();
    let descriptor = engram::parse_engram(&frame, &limits).unwrap();
    let resolver = MapResolver::new();
    let (bytes, touched) =
        engram::read_engram_region(&frame, &descriptor, 17, 19, &resolver, &limits).unwrap();
    assert_eq!(touched, vec![4]);
    let row = 4 * 2;
    assert_eq!(bytes, table[17 * row..19 * row]);
    let (groups, touched) =
        engram::read_engram_groups(&frame, &descriptor, &[3, 11], &resolver, &limits).unwrap();
    assert_eq!(touched, vec![3, 11]);
    assert_eq!(groups.len(), 2 * 4 * row);
}

#[test]
fn conformance_engram_vectors_replay() {
    let dir = v1_dir().join("engram");
    let limits = Limits::default();
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("read engram vectors")
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    assert!(files.len() >= 20, "expected at least 20 engram vectors");
    let mut valid = 0;
    let mut rejects = 0;
    for path in files {
        let vector = load(&path);
        let frame_bytes = hex_bytes(vector["frame_hex"].as_str().unwrap());
        if let Some(expected) = vector.get("expect_error").and_then(Json::as_str) {
            let mut resolver = MapResolver::new();
            for frame_hex in vector["dependency_frames"]
                .as_array()
                .cloned()
                .unwrap_or_default()
            {
                let bytes = hex_bytes(frame_hex.as_str().unwrap());
                let parsed = Frame::parse(&bytes, &limits).unwrap();
                resolver.insert(parsed.header.content_hash, bytes);
            }
            let result = Frame::parse(&frame_bytes, &limits)
                .and_then(|frame| engram::decode_frame(&frame, &resolver, &limits));
            let err = result.expect_err("reject vector decoded successfully");
            assert_eq!(
                err.class().as_str(),
                expected,
                "{}: expected {expected}, got {}",
                path.display(),
                err
            );
            rejects += 1;
            continue;
        }
        let frame = Frame::parse(&frame_bytes, &limits).unwrap();
        let descriptor = engram::parse_engram(&frame, &limits).unwrap();
        let root = descriptor.meta.descriptor_root(&limits).unwrap();
        assert_eq!(
            root.to_hex(),
            vector["descriptor_root"].as_str().unwrap(),
            "{}: descriptor root",
            path.display()
        );
        let decoded = engram::decode_engram_without_dependencies(&frame, &limits).unwrap();
        let table = hex_bytes(vector["table_hex"].as_str().unwrap());
        assert_eq!(decoded, table, "{}: table bytes", path.display());
        if let Some(region) = vector.get("region") {
            let start = region["row_start"].as_u64().unwrap();
            let end = region["row_end"].as_u64().unwrap();
            let resolver = MapResolver::new();
            let (bytes, touched) =
                engram::read_engram_region(&frame, &descriptor, start, end, &resolver, &limits)
                    .unwrap();
            assert_eq!(
                bytes,
                hex_bytes(region["bytes_hex"].as_str().unwrap()),
                "{}: region bytes",
                path.display()
            );
            let expected: Vec<u64> = region["groups_touched"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect();
            assert_eq!(
                touched
                    .iter()
                    .map(|value| u64::from(*value))
                    .collect::<Vec<_>>(),
                expected,
                "{}: touched groups",
                path.display()
            );
        }
        valid += 1;
    }
    assert_eq!((valid, rejects), (6, 14));
}

#[test]
fn reference_and_supersession_vectors_are_reachable() {
    // A referenced group decodes through a MapResolver holding the earlier
    // frame; the same mechanism drives `update_engram_groups` in the store.
    let limits = Limits::default();
    let table: Vec<u8> = (0..8u64 * 2 * 2).map(|index| (index * 7) as u8).collect();
    let first_input = engram::EngramInput::inline(
        weftcodec::TensorDtype::Bf16,
        vec![8, 2],
        vec![0, 4, 8],
        engram::ENGRAM_HASH_BLAKE3,
        8,
    );
    let first = engram::encode_engram(&table, &first_input, &limits).unwrap();
    let first_frame = Frame::parse(&first, &limits).unwrap();
    let first_meta = engram::engram_metadata(&first_frame, &limits).unwrap();
    let mut resolver = MapResolver::new();
    resolver.insert(first_frame.header.content_hash, first.clone());

    let mut updated = table.clone();
    updated[0] ^= 0xff;
    let mut input = engram::EngramInput::inline(
        weftcodec::TensorDtype::Bf16,
        vec![8, 2],
        vec![0, 4, 8],
        engram::ENGRAM_HASH_BLAKE3,
        8,
    );
    input.references.insert(
        1,
        engram::EngramReference {
            group_root: first_meta.groups[1].group_root,
            dependency: 0,
        },
    );
    input.supersedes = vec![Some(first_meta.groups[0].group_root), None];
    let second = engram::encode_engram_with_dependencies(
        &updated,
        &input,
        &[first_frame.header.content_hash],
        &limits,
    )
    .unwrap();
    let second_frame = Frame::parse(&second, &limits).unwrap();
    let decoded = engram::decode_engram(&second_frame, &resolver, &limits).unwrap();
    assert_eq!(decoded, updated);
    let descriptor = engram::parse_engram(&second_frame, &limits).unwrap();
    assert!(matches!(
        descriptor.meta.groups[1].storage,
        engram::EngramStorage::Reference { .. }
    ));
    assert!(descriptor.meta.payload_len < first_meta.payload_len);
}
