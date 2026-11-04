//! Mode 14 `SEMANTIC_DELTA`, semantic roots, and candidate selection.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;

use weftcodec::dependency::{decode_frame_dag, MapFrameSource, MapResolver};
use weftcodec::frame::Frame;
use weftcodec::hash::{hash_domain, DOMAIN_CHUNK};
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::semantic::{
    select_semantic_base, semantic_matches, semantic_record, semantic_root, SemanticPolicy,
    SemanticRootRecord,
};
use weftcodec::semantic_delta::{
    decode_with_base_frame, encode_semantic_delta, semantic_delta_metadata,
};
use weftcodec::tensor::{LogicalRecord, TensorDtype};
use weftcodec::weight_delta::WeightDtype;
use weftcodec::ErrorClass;

fn default_limits() -> Limits {
    Limits::default()
}

fn v1_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1")
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    files.sort();
    files
}

fn load(path: &Path) -> Json {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hex_field(vector: &Json, key: &str) -> Vec<u8> {
    let text = vector
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("vector missing {key}"));
    hex::decode(text).unwrap_or_else(|e| panic!("bad hex in {key}: {e}"))
}

fn shape_of(vector: &Json) -> Vec<u64> {
    vector["shape"]
        .as_array()
        .expect("shape array")
        .iter()
        .map(|value| value.as_u64().expect("shape extent"))
        .collect()
}

fn policy_of(value: &Json) -> SemanticPolicy {
    let scheme = value["scheme"].as_str().expect("policy scheme");
    match scheme {
        "none" => SemanticPolicy::None,
        "mantissa_truncate" => SemanticPolicy::MantissaTruncate {
            bits: value["bits"].as_u64().expect("bits") as u32,
        },
        "block_quantize" => SemanticPolicy::BlockQuantize {
            block_size: value["block_size"].as_u64().expect("block_size"),
            levels: value["levels"].as_u64().expect("levels"),
        },
        other => panic!("unknown policy {other}"),
    }
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn bf16_bytes(words: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

#[test]
fn semantic_root_vectors_replay() {
    let dir = v1_dir().join("semantic-root");
    let files = json_files(&dir);
    assert_eq!(files.len(), 9, "expected 9 semantic-root vectors");
    for path in files {
        let vector = load(&path);
        let name = vector["name"].as_str().unwrap_or("?");
        if let Some(class) = vector.get("expect_error").and_then(Json::as_str) {
            let record = hex_field(&vector, "record_hex");
            let error = SemanticRootRecord::from_canonical_bytes(&record, &default_limits())
                .expect_err(name);
            assert_eq!(error.class().as_str(), class, "{name}: {error}");
            continue;
        }
        let dtype = WeightDtype::from_name(vector["dtype"].as_str().unwrap()).unwrap();
        let shape = shape_of(&vector);
        let bytes = hex_field(&vector, "values_hex");
        let policy = policy_of(&vector["policy"]);
        let record = semantic_record(dtype, &shape, &bytes, policy, &default_limits()).expect(name);
        assert_eq!(
            record.quantized_values_hash.to_hex(),
            vector["quantized_values_hash"].as_str().unwrap(),
            "{name} values hash"
        );
        let canonical = record.canonical_bytes(&default_limits()).unwrap();
        assert_eq!(
            hex::encode(&canonical),
            vector["canonical_hex"].as_str().unwrap(),
            "{name} canonical bytes"
        );
        assert_eq!(
            record.root(&default_limits()).unwrap().to_hex(),
            vector["semantic_root"].as_str().unwrap(),
            "{name} root"
        );
        let decoded =
            SemanticRootRecord::from_canonical_bytes(&canonical, &default_limits()).unwrap();
        assert_eq!(decoded, record, "{name} record round trip");
    }
}

#[test]
fn semantic_delta_vectors_replay() {
    let dir = v1_dir().join("semantic-delta");
    let files = json_files(&dir);
    assert_eq!(files.len(), 9, "expected 9 SEMANTIC_DELTA vectors");
    for path in files {
        let vector = load(&path);
        let name = vector["name"].as_str().unwrap_or("?");
        let delta_bytes = hex_field(&vector, "delta_frame_hex");
        let parsed = Frame::parse(&delta_bytes, &default_limits()).expect("parse delta frame");
        if let Some(class) = vector.get("expect_error").and_then(Json::as_str) {
            let use_source = vector["use_frame_source"].as_bool().unwrap_or(false);
            let error = if use_source {
                let source = MapFrameSource::new();
                decode_frame_dag(&source, &delta_bytes, &default_limits()).expect_err(name)
            } else if vector["base_frame_hex"].is_string() {
                let base = hex_field(&vector, "base_frame_hex");
                decode_with_base_frame(&parsed, &base, &MapResolver::new(), &default_limits())
                    .expect_err(name)
            } else {
                weftcodec::semantic_delta::decode_frame(
                    &parsed,
                    &MapResolver::new(),
                    &default_limits(),
                )
                .expect_err(name)
            };
            assert_eq!(error.class().as_str(), class, "{name}: {error}");
            continue;
        }
        let base = hex_field(&vector, "base_frame_hex");
        let target = hex_field(&vector, "target_hex");
        let decoded =
            decode_with_base_frame(&parsed, &base, &MapResolver::new(), &default_limits())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(decoded, target, "{name} reconstruction");
        assert_eq!(
            hash_domain(DOMAIN_CHUNK, &decoded).to_hex(),
            vector["target_hash"].as_str().unwrap(),
            "{name} target hash"
        );
        let meta = semantic_delta_metadata(&parsed, &default_limits()).unwrap();
        assert_eq!(meta.dtype.as_str(), vector["dtype"].as_str().unwrap());
        assert_eq!(meta.shape, shape_of(&vector));
        assert_eq!(
            meta.element_count,
            vector["element_count"].as_u64().unwrap()
        );
        assert_eq!(meta.word_bits, vector["word_bits"].as_u64().unwrap());
        assert_eq!(meta.base.to_hex(), vector["base"].as_str().unwrap());
        assert_eq!(
            meta.base_dtype.as_str(),
            vector["base_dtype"].as_str().unwrap()
        );
        assert_eq!(meta.base_shape, shape_of(&vector));
        assert_eq!(
            meta.base_semantic_root.to_hex(),
            vector["base_semantic_root"].as_str().unwrap()
        );
        assert_eq!(policy_of(&vector["policy"]), meta.policy);
        assert_eq!(meta.plane_codec, vector["plane_codec"].as_str().unwrap());
        assert_eq!(
            meta.delta_root.to_hex(),
            vector["delta_root"].as_str().unwrap()
        );
        assert_eq!(meta.payload_len, vector["payload_len"].as_u64().unwrap());
        let reencoded = encode_semantic_delta(
            &target,
            meta.dtype,
            &meta.shape,
            &base,
            meta.policy,
            &default_limits(),
        )
        .unwrap();
        assert_eq!(reencoded, delta_bytes, "{name} byte-identical re-encode");
    }
}

#[test]
fn near_duplicate_pair_selects_the_right_base_and_stays_exact() {
    let limits = default_limits();
    let element_count = 500_000usize; // 1 MB of bf16, 0.1% = 500 elements
    let shape = vec![element_count as u64];
    let mut state = 0x5EED_1234_ABCD_9876u64;
    let mut base_bytes = Vec::with_capacity(element_count * 2);
    for _ in 0..element_count {
        // Keep finite exponents so the low-bit changes stay inside the mantissa.
        let word = 0x3C00u16 | ((xorshift(&mut state) & 0x03FF) as u16);
        base_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut target_bytes = base_bytes.clone();
    let mut changed = 0usize;
    for index in (0..element_count).step_by(1000) {
        target_bytes[index * 2] ^= 0x05; // low three mantissa bits
        changed += 1;
    }
    assert_eq!(changed, 500);

    let policy = SemanticPolicy::MantissaTruncate { bits: 3 };
    let target_root =
        semantic_root(WeightDtype::Bf16, &shape, &target_bytes, policy.clone()).unwrap();
    let base_root = semantic_root(WeightDtype::Bf16, &shape, &base_bytes, policy.clone()).unwrap();
    assert_eq!(
        target_root, base_root,
        "0.1% low-bit changes share the root"
    );

    let base_frame = modes::encode_tensor(&base_bytes, TensorDtype::Bf16, &shape, &limits).unwrap();
    // A different semantic root under the same policy.
    let other_bytes: Vec<u8> = (0..element_count)
        .flat_map(|index| {
            let word = 0x4000u16 ^ ((index as u16) << 4) ^ (index as u16 & 0x0F);
            word.to_le_bytes()
        })
        .collect();
    let other_frame =
        modes::encode_tensor(&other_bytes, TensorDtype::Bf16, &shape, &limits).unwrap();

    let candidates: [&[u8]; 2] = [&base_frame, &other_frame];
    let selected = select_semantic_base(
        &target_bytes,
        WeightDtype::Bf16,
        &shape,
        &policy,
        &candidates,
        &limits,
    )
    .unwrap();
    assert_eq!(selected, Some(0), "the near-duplicate must be chosen");

    // A candidate with a different semantic root is never offered.
    let only_other: [&[u8]; 1] = [&other_frame];
    let none = select_semantic_base(
        &target_bytes,
        WeightDtype::Bf16,
        &shape,
        &policy,
        &only_other,
        &limits,
    )
    .unwrap();
    assert_eq!(none, None);
    let other_record = semantic_record(
        WeightDtype::Bf16,
        &shape,
        &other_bytes,
        policy.clone(),
        &limits,
    )
    .unwrap();
    let target_record = semantic_record(
        WeightDtype::Bf16,
        &shape,
        &target_bytes,
        policy.clone(),
        &limits,
    )
    .unwrap();
    assert!(!semantic_matches(&target_record, &other_record));

    // The delta reconstructs every bit.
    let delta = encode_semantic_delta(
        &target_bytes,
        WeightDtype::Bf16,
        &shape,
        &base_frame,
        policy,
        &limits,
    )
    .unwrap();
    let parsed = Frame::parse(&delta, &limits).unwrap();
    let decoded =
        decode_with_base_frame(&parsed, &base_frame, &MapResolver::new(), &default_limits())
            .unwrap();
    assert_eq!(decoded, target_bytes);
    let original_logical =
        LogicalRecord::from_row_major(TensorDtype::Bf16, &shape, &target_bytes, &limits)
            .unwrap()
            .root_with_limits(&limits)
            .unwrap();
    let decoded_logical =
        LogicalRecord::from_row_major(TensorDtype::Bf16, &shape, &decoded, &limits)
            .unwrap()
            .root_with_limits(&limits)
            .unwrap();
    assert_eq!(original_logical, decoded_logical);

    // Independent encoding for comparison: zstd over the target element bytes.
    let zstd = zstd::bulk::compress(&target_bytes, 3).unwrap().len();
    let factor = zstd as f64 / delta.len() as f64;
    assert!(
        delta.len() < zstd,
        "semantic delta {} bytes must beat zstd {} bytes",
        delta.len(),
        zstd
    );
    println!(
        "near-duplicate: changed={changed}/{element_count} delta={} bytes zstd3={} bytes factor={factor:.2}x",
        delta.len(),
        zstd
    );
}

#[test]
fn block_quantize_keeps_small_changes_in_one_group() {
    let words: [u16; 32] = std::array::from_fn(|index| 0x3C00 | ((index as u16) * 7 & 0x03FF));
    let base = bf16_bytes(&words);
    let mut target = base.clone();
    // Change only the lowest bit so the signed block quantizer is unaffected.
    target[2] ^= 0x01;
    let policy = SemanticPolicy::BlockQuantize {
        block_size: 16,
        levels: 64,
    };
    let a = semantic_root(WeightDtype::Bf16, &[32], &base, policy.clone()).unwrap();
    let b = semantic_root(WeightDtype::Bf16, &[32], &target, policy).unwrap();
    assert_eq!(a, b);
}

#[test]
fn different_shapes_are_never_matched() {
    let limits = default_limits();
    let bytes = bf16_bytes(&[0x3F80, 0x4000, 0x0000, 0x0001]);
    let policy = SemanticPolicy::None;
    let a = semantic_record(WeightDtype::Bf16, &[2, 2], &bytes, policy.clone(), &limits).unwrap();
    let b = semantic_record(WeightDtype::Bf16, &[4], &bytes, policy, &limits).unwrap();
    assert!(!semantic_matches(&a, &b));
}

#[test]
fn descriptor_rejects_have_typed_classes() {
    let limits = default_limits();
    let bf16 = bf16_bytes(&[0x3F80, 0x4000]);
    let base = modes::encode_tensor(&bf16, TensorDtype::Bf16, &[2], &limits).unwrap();
    let delta = encode_semantic_delta(
        &bf16,
        WeightDtype::Bf16,
        &[2],
        &base,
        SemanticPolicy::None,
        &limits,
    )
    .unwrap();
    let parsed = Frame::parse(&delta, &limits).unwrap();
    let meta = semantic_delta_metadata(&parsed, &limits).unwrap();
    assert_eq!(meta.policy, SemanticPolicy::None);

    // Unknown policy scheme is Malformed at descriptor parse time.
    let mut bad = meta.clone();
    bad.policy = SemanticPolicy::MantissaTruncate { bits: 100 };
    let descriptor_len = u32::from_le_bytes([
        parsed.body[0],
        parsed.body[1],
        parsed.body[2],
        parsed.body[3],
    ]) as usize;
    let payload = &parsed.body[4 + descriptor_len..];
    let frame = weftcodec::semantic_delta::encode_descriptor_frame(
        &bad,
        payload,
        parsed.header.decoded_len,
        parsed.header.content_hash,
        6,
        &limits,
    )
    .unwrap();
    let parsed_bad = Frame::parse(&frame, &limits).unwrap();
    let error = semantic_delta_metadata(&parsed_bad, &limits).expect_err("bad policy must fail");
    assert_eq!(error.class(), ErrorClass::Malformed);
    let _ = &bf16;
}
