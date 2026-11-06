//! Mode 16 `SPARSE_DELTA` exactness, sparse densities, run-stream rejects, and
//! committed-vector replay.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use weftcodec::dependency::{decode_frame_dag, MapFrameSource, MapResolver};
use weftcodec::frame::Frame;
use weftcodec::hash::{hash_domain, Hash32, DOMAIN_CHUNK};
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::sparse_delta::{
    decode_with_base_frame, encode_descriptor_frame, encode_sparse_delta, sparse_delta_metadata,
    SparseDeltaMeta, SPARSE_DELTA_CODEC, SPARSE_DELTA_OP,
};
use weftcodec::tensor;
use weftcodec::weight_delta::{encode_weight_delta, WeightDtype};
use weftcodec::{ErrorClass, TensorDtype};

fn limits() -> Limits {
    Limits::default()
}

fn base_frame(dtype: TensorDtype, shape: &[u64], bytes: &[u8]) -> Vec<u8> {
    modes::encode_tensor(bytes, dtype, shape, &limits()).expect("base frame")
}

fn decode(delta: &[u8], base: Option<&[u8]>) -> weftcodec::Result<Vec<u8>> {
    let parsed = Frame::parse(delta, &limits())?;
    let resolver = MapResolver::new();
    match base {
        Some(bytes) => decode_with_base_frame(&parsed, bytes, &resolver, &limits()),
        None => weftcodec::sparse_delta::decode_frame(&parsed, &resolver, &limits()),
    }
}

fn payload_of(frame_bytes: &[u8]) -> Vec<u8> {
    let parsed = Frame::parse(frame_bytes, &limits()).expect("parse");
    let body = parsed.body;
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    body[4 + meta_len..].to_vec()
}

fn run_count(frame_bytes: &[u8]) -> u32 {
    let payload = payload_of(frame_bytes);
    u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]])
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn words(bytes: &[u8], width: usize) -> Vec<u32> {
    bytes
        .chunks_exact(width)
        .map(|chunk| match width {
            1 => u32::from(chunk[0]),
            2 => u32::from(u16::from_le_bytes([chunk[0], chunk[1]])),
            _ => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
        })
        .collect()
}

fn put_words(out: &mut Vec<u8>, values: &[u32], width: usize) {
    for value in values {
        match width {
            1 => out.push(*value as u8),
            2 => out.extend_from_slice(&(*value as u16).to_le_bytes()),
            _ => out.extend_from_slice(&value.to_le_bytes()),
        }
    }
}

// ---------------------------------------------------------------------------
// Round trips
// ---------------------------------------------------------------------------

#[test]
fn bf16_roundtrip_preserves_every_bit_pattern() {
    let base_words: [u16; 8] = [
        0x3F80, // 1.0
        0xC000, // -2.0
        0x0000, // +0.0
        0x8000, // -0.0
        0x7FC0, // quiet NaN
        0x7F81, // signalling NaN payload
        0x0001, // denormal
        0x7F80, // +infinity
    ];
    let mut target_words = base_words;
    target_words[1] = 0x4000;
    target_words[4] = 0x7FC1;
    target_words[6] = 0x8002;
    let mut base_bytes = Vec::new();
    let mut target_bytes = Vec::new();
    for word in base_words {
        base_bytes.extend_from_slice(&word.to_le_bytes());
    }
    for word in target_words {
        target_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let base = base_frame(TensorDtype::Bf16, &[2, 4], &base_bytes);
    let delta = encode_sparse_delta(&target_bytes, WeightDtype::Bf16, &[2, 4], &base, &limits())
        .expect("encode");
    assert_eq!(decode(&delta, Some(&base)).expect("decode"), target_bytes);

    let parsed = Frame::parse(&delta, &limits()).expect("parse");
    let meta = sparse_delta_metadata(&parsed, &limits()).expect("meta");
    assert_eq!(meta.dtype, WeightDtype::Bf16);
    assert_eq!(meta.storage_bytes, 2);
    assert_eq!(meta.op, SPARSE_DELTA_OP);
    assert_eq!(meta.sparse_codec, SPARSE_DELTA_CODEC);
    assert_eq!(
        meta.base,
        Frame::parse(&base, &limits()).unwrap().header.content_hash
    );
    assert_eq!(run_count(&delta), 3);

    // Deterministic re-encoding.
    let again =
        encode_sparse_delta(&target_bytes, WeightDtype::Bf16, &[2, 4], &base, &limits()).unwrap();
    assert_eq!(again, delta);
}

#[test]
fn fp16_fp32_and_q8_0_roundtrip() {
    // fp16: one changed word.
    let mut fp16_base = Vec::new();
    for index in 0..5u16 {
        fp16_base.extend_from_slice(&(index * 3).to_le_bytes());
    }
    let mut fp16_target = fp16_base.clone();
    fp16_target[4..6].copy_from_slice(&0x3C01u16.to_le_bytes());
    let frame = base_frame(TensorDtype::F16, &[5], &fp16_base);
    let delta =
        encode_sparse_delta(&fp16_target, WeightDtype::Fp16, &[5], &frame, &limits()).unwrap();
    assert_eq!(decode(&delta, Some(&frame)).unwrap(), fp16_target);
    assert_eq!(run_count(&delta), 1);

    // fp32: three changed words.
    let mut fp32_base = Vec::new();
    for index in 0..8u32 {
        fp32_base.extend_from_slice(&(0x3F00_0000u32 + index * 7).to_le_bytes());
    }
    let mut fp32_target = fp32_base.clone();
    for index in [0usize, 3, 7] {
        let at = index * 4;
        fp32_target[at..at + 4].copy_from_slice(&(0x7FC0_0001u32 + index as u32).to_le_bytes());
    }
    let frame = base_frame(TensorDtype::F32, &[2, 4], &fp32_base);
    let delta =
        encode_sparse_delta(&fp32_target, WeightDtype::Fp32, &[2, 4], &frame, &limits()).unwrap();
    assert_eq!(decode(&delta, Some(&frame)).unwrap(), fp32_target);
    assert_eq!(run_count(&delta), 3);

    // q8_0: wrapping deltas in both directions.
    let q8_base: Vec<u8> = (0..64u16)
        .map(|index| ((index * 37 + 11) % 251) as u8)
        .collect();
    let mut q8_target = q8_base.clone();
    for (index, byte) in q8_target.iter_mut().enumerate() {
        if index % 11 == 0 {
            *byte = byte.wrapping_add(7);
        }
    }
    let frame = base_frame(TensorDtype::I8, &[8, 8], &q8_base);
    let delta =
        encode_sparse_delta(&q8_target, WeightDtype::Q8_0, &[8, 8], &frame, &limits()).unwrap();
    assert_eq!(decode(&delta, Some(&frame)).unwrap(), q8_target);
    assert_eq!(run_count(&delta), 6);
}

#[test]
fn q4_0_roundtrip_keeps_nibbles_in_range() {
    let q4_base: Vec<u8> = (0..48u16).map(|index| (index % 16) as u8).collect();
    let mut q4_target = q4_base.clone();
    for index in [2usize, 9, 47] {
        q4_target[index] = (q4_target[index] + 1) % 16;
    }
    let frame = base_frame(TensorDtype::U8, &[6, 8], &q4_base);
    let delta =
        encode_sparse_delta(&q4_target, WeightDtype::Q4_0, &[6, 8], &frame, &limits()).unwrap();
    let parsed = Frame::parse(&delta, &limits()).unwrap();
    let meta = sparse_delta_metadata(&parsed, &limits()).unwrap();
    assert_eq!(meta.storage_bytes, 1);
    assert_eq!(run_count(&delta), 3);
    assert_eq!(decode(&delta, Some(&frame)).unwrap(), q4_target);
}

#[test]
fn all_zero_delta_is_a_run_count_of_zero() {
    let mut bytes = Vec::new();
    for index in 0..6u16 {
        bytes.extend_from_slice(&(index * 5).to_le_bytes());
    }
    let base = base_frame(TensorDtype::Bf16, &[2, 3], &bytes);
    let delta = encode_sparse_delta(&bytes, WeightDtype::Bf16, &[2, 3], &base, &limits()).unwrap();
    assert_eq!(run_count(&delta), 0);
    // The payload is exactly the u32le run_count: no records, no trailing zeros.
    assert_eq!(payload_of(&delta).len(), 4);
    assert_eq!(decode(&delta, Some(&base)).unwrap(), bytes);
    let parsed = Frame::parse(&delta, &limits()).unwrap();
    let meta = sparse_delta_metadata(&parsed, &limits()).unwrap();
    assert_eq!(
        meta.delta_root,
        hash_domain(tensor::DOMAIN_TENSOR_VALUES, &vec![0u8; 12])
    );
}

// ---------------------------------------------------------------------------
// Sparse densities
// ---------------------------------------------------------------------------

struct DensityCase {
    base: Vec<u8>,
    target: Vec<u8>,
    changed: u32,
}

fn density_case(density_ppm: u64) -> DensityCase {
    let element_count = 1_048_576usize; // 2 MiB of bf16
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut base_words = Vec::with_capacity(element_count);
    for _ in 0..element_count {
        base_words.push((xorshift(&mut state) & 0xFFFF) as u32);
    }
    let mut target_words = base_words.clone();
    let mut changed = 0u32;
    for word in target_words.iter_mut() {
        let draw = xorshift(&mut state) % 1_000_000;
        if draw < density_ppm {
            // Always a non-zero delta of the same word width.
            let delta = ((xorshift(&mut state) as u32) & 0xFFFF) | 1;
            *word = (word.wrapping_add(delta)) & 0xFFFF;
            changed += 1;
        }
    }
    let mut base = Vec::new();
    put_words(&mut base, &base_words, 2);
    let mut target = Vec::new();
    put_words(&mut target, &target_words, 2);
    DensityCase {
        base,
        target,
        changed,
    }
}

#[test]
fn sparse_densities_roundtrip_exactly() {
    let element_count = 1_048_576usize;
    let shape = vec![element_count as u64];
    for density_ppm in [0u64, 10, 10_000, 100_000, 500_000, 1_000_000] {
        let case = density_case(density_ppm);
        let base = base_frame(TensorDtype::Bf16, &shape, &case.base);
        let delta = encode_sparse_delta(&case.target, WeightDtype::Bf16, &shape, &base, &limits())
            .unwrap_or_else(|err| panic!("density {density_ppm} ppm encode: {err}"));
        let observed = run_count(&delta);
        assert_eq!(
            u64::from(observed),
            u64::from(case.changed),
            "density {density_ppm} ppm run count"
        );
        let restored = decode(&delta, Some(&base)).expect("decode");
        assert_eq!(restored, case.target, "density {density_ppm} ppm bytes");
        assert_eq!(
            words(&restored, 2),
            words(&case.target, 2),
            "density {density_ppm} ppm words"
        );
        // One u32le count plus one record per changed word, each at least
        // one varint byte and one element word.
        let payload = payload_of(&delta).len() as u64;
        assert!(payload >= 4 + 3 * u64::from(case.changed));
    }
}

#[test]
fn mode16_is_smaller_than_mode11_on_sparse_and_larger_on_dense() {
    let element_count = 1_048_576usize;
    let shape = vec![element_count as u64];
    let case = density_case(0);
    let base = base_frame(TensorDtype::Bf16, &shape, &case.base);

    // Sparse: 19 changed words with a guaranteed non-zero delta each.
    let base_words = words(&case.base, 2);
    let mut target_words = base_words.clone();
    for index in [3usize, 1000, 65_535, 100_000, 500_000, 1_048_575] {
        let delta = ((index as u32) & 0xFFFE) | 1;
        target_words[index] = target_words[index].wrapping_add(delta) & 0xFFFF;
    }
    for index in (7..32).step_by(2) {
        target_words[index] = target_words[index].wrapping_add(3) & 0xFFFF;
    }
    let mut sparse_target = Vec::new();
    put_words(&mut sparse_target, &target_words, 2);
    let sparse_delta =
        encode_sparse_delta(&sparse_target, WeightDtype::Bf16, &shape, &base, &limits()).unwrap();
    let weight_delta =
        encode_weight_delta(&sparse_target, WeightDtype::Bf16, &shape, &base, &limits()).unwrap();
    assert_eq!(run_count(&sparse_delta), 19);
    assert!(
        sparse_delta.len() < weight_delta.len() / 100,
        "mode 16 {} bytes vs mode 11 {} bytes",
        sparse_delta.len(),
        weight_delta.len()
    );
    assert_eq!(decode(&sparse_delta, Some(&base)).unwrap(), sparse_target);

    // Dense: every word changed. Mode 16 must lose.
    let mut dense_target_words = base_words.clone();
    for (index, word) in dense_target_words.iter_mut().enumerate() {
        *word = (word.wrapping_add(((index as u32) % 0xFFFF) + 1)) & 0xFFFF;
    }
    let mut dense_target = Vec::new();
    put_words(&mut dense_target, &dense_target_words, 2);
    let sparse_dense =
        encode_sparse_delta(&dense_target, WeightDtype::Bf16, &shape, &base, &limits()).unwrap();
    let weight_dense =
        encode_weight_delta(&dense_target, WeightDtype::Bf16, &shape, &base, &limits()).unwrap();
    assert!(sparse_dense.len() > weight_dense.len());
    assert_eq!(decode(&sparse_dense, Some(&base)).unwrap(), dense_target);
}

// ---------------------------------------------------------------------------
// Rejects
// ---------------------------------------------------------------------------

#[test]
fn base_dtype_and_shape_mismatches_fail_closed() {
    let base_bytes = vec![0u8; 16];
    let target_bytes = vec![1u8; 16];
    let bf16_base = base_frame(TensorDtype::Bf16, &[2, 4], &base_bytes);
    let delta = encode_sparse_delta(
        &target_bytes,
        WeightDtype::Bf16,
        &[2, 4],
        &bf16_base,
        &limits(),
    )
    .unwrap();

    let fp32_base = base_frame(TensorDtype::F32, &[2, 4], &vec![0u8; 32]);
    assert_eq!(
        decode(&delta, Some(&fp32_base)).unwrap_err().class(),
        ErrorClass::DependencyMismatch
    );

    let wrong_shape_base = base_frame(TensorDtype::Bf16, &[4, 2], &base_bytes);
    assert_eq!(
        decode(&delta, Some(&wrong_shape_base)).unwrap_err().class(),
        ErrorClass::DependencyMismatch
    );
}

#[test]
fn missing_base_fails_closed() {
    let base = base_frame(TensorDtype::Bf16, &[2, 2], &[0u8; 8]);
    let delta =
        encode_sparse_delta(&[1u8; 8], WeightDtype::Bf16, &[2, 2], &base, &limits()).unwrap();
    assert_eq!(
        decode(&delta, None).unwrap_err().class(),
        ErrorClass::DependencyMissing
    );
}

#[test]
fn dependency_cycle_fails_closed() {
    let target = vec![0u8; 12];
    let content_hash = hash_domain(DOMAIN_CHUNK, &target);
    let meta = SparseDeltaMeta {
        dtype: WeightDtype::Bf16,
        shape: vec![2, 3],
        element_count: 6,
        storage_bytes: 2,
        op: SPARSE_DELTA_OP.to_owned(),
        sparse_codec: SPARSE_DELTA_CODEC.to_owned(),
        // The base equals the frame's own content hash.
        base: content_hash,
        delta_root: hash_domain(tensor::DOMAIN_TENSOR_VALUES, &target),
    };
    let payload = 0u32.to_le_bytes().to_vec();
    let frame =
        encode_descriptor_frame(&meta, &payload, 12, content_hash, 6, &limits()).expect("frame");
    let parsed = Frame::parse(&frame, &limits()).unwrap();

    let mut source = MapFrameSource::new();
    source
        .insert_frame_with_limits(frame.clone(), &limits())
        .unwrap();
    assert_eq!(
        decode_frame_dag(&source, &frame, &limits())
            .unwrap_err()
            .class(),
        ErrorClass::DependencyCycle
    );
    assert_eq!(
        weftcodec::sparse_delta::decode_frame(&parsed, &MapResolver::new(), &limits())
            .unwrap_err()
            .class(),
        ErrorClass::DependencyMissing
    );
}

#[test]
fn descriptor_and_run_stream_rejects_have_typed_classes() {
    let base_bytes = vec![0u8; 12];
    let mut target_bytes = base_bytes.clone();
    target_bytes[0..2].copy_from_slice(&1u16.to_le_bytes());
    target_bytes[10..12].copy_from_slice(&5u16.to_le_bytes());
    let base = base_frame(TensorDtype::Bf16, &[2, 3], &base_bytes);
    let valid =
        encode_sparse_delta(&target_bytes, WeightDtype::Bf16, &[2, 3], &base, &limits()).unwrap();
    let parsed = Frame::parse(&valid, &limits()).unwrap();
    let meta = sparse_delta_metadata(&parsed, &limits()).unwrap();
    let payload = payload_of(&valid);
    let content_hash = parsed.header.content_hash;

    let frame_of = |meta: &SparseDeltaMeta, payload: &[u8]| {
        encode_descriptor_frame(meta, payload, 12, content_hash, 6, &limits()).unwrap()
    };

    let mut unknown_op = meta.clone();
    unknown_op.op = "xor".to_owned();
    assert_eq!(
        decode(&frame_of(&unknown_op, &payload), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Malformed
    );

    let mut unknown_codec = meta.clone();
    unknown_codec.sparse_codec = "rle-v2".to_owned();
    assert_eq!(
        decode(&frame_of(&unknown_codec, &payload), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Malformed
    );

    let mut storage = meta.clone();
    storage.storage_bytes = 1;
    assert_eq!(
        decode(&frame_of(&storage, &payload), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Malformed
    );

    let mut root = meta.clone();
    let mut flipped = root.delta_root.into_bytes();
    flipped[0] ^= 1;
    root.delta_root = Hash32::from_bytes(flipped);
    assert_eq!(
        decode(&frame_of(&root, &payload), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Corrupt
    );

    // run_count 3 > element_count 6? no: 7 > 6.
    let mut over_count = 7u32.to_le_bytes().to_vec();
    over_count.push(0);
    assert_eq!(
        decode(&frame_of(&meta, &over_count), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Malformed
    );

    // A run that skips all six words before emitting.
    let mut past_end = 1u32.to_le_bytes().to_vec();
    past_end.push(6); // varint(6)
    past_end.extend_from_slice(&[1, 0]);
    assert_eq!(
        decode(&frame_of(&meta, &past_end), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Malformed
    );

    // A truncated record: one of two word bytes.
    let mut truncated = 1u32.to_le_bytes().to_vec();
    truncated.push(0); // varint(0)
    truncated.push(1);
    assert_eq!(
        decode(&frame_of(&meta, &truncated), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Malformed
    );

    // Trailing bytes after the declared records.
    let mut trailing = payload.clone();
    trailing.push(0);
    assert_eq!(
        decode(&frame_of(&meta, &trailing), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Corrupt
    );

    // A record whose stored word is zero.
    let mut zero_word = 1u32.to_le_bytes().to_vec();
    zero_word.push(0);
    zero_word.extend_from_slice(&[0, 0]);
    assert_eq!(
        decode(&frame_of(&meta, &zero_word), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Corrupt
    );

    // A payload shorter than the run_count field.
    assert_eq!(
        decode(&frame_of(&meta, &[0u8, 0]), Some(&base))
            .unwrap_err()
            .class(),
        ErrorClass::Malformed
    );

    // The run bound is enforced.
    let tight = limits().with_max_sparse_runs(1);
    let parsed_valid = Frame::parse(&valid, &limits()).unwrap();
    assert_eq!(
        decode_with_base_frame(&parsed_valid, &base, &MapResolver::new(), &tight)
            .unwrap_err()
            .class(),
        ErrorClass::LimitExceeded
    );
}

// ---------------------------------------------------------------------------
// Committed vectors and CLI
// ---------------------------------------------------------------------------

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_weftcodec")
}

fn decode_hex(text: &str) -> Vec<u8> {
    hex::decode(text).expect("hex")
}

#[test]
fn committed_vectors_replay() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1/sparse-delta");
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("vector dir")
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 18, "expected 18 SPARSE_DELTA vectors");
    let limits = limits();
    for path in paths {
        let text = fs::read_to_string(&path).expect("read vector");
        let vector: serde_json::Value = serde_json::from_str(&text).expect("JSON vector");
        let delta_bytes = decode_hex(vector["delta_frame_hex"].as_str().unwrap());
        let parsed = Frame::parse(&delta_bytes, &limits).expect("parse delta frame");
        if let Some(expected) = vector["expect_error"].as_str() {
            let class = if vector["use_frame_source"].as_bool().unwrap_or(false) {
                let mut source = MapFrameSource::new();
                source
                    .insert_frame_with_limits(delta_bytes.clone(), &limits)
                    .unwrap();
                decode_frame_dag(&source, &delta_bytes, &limits)
                    .unwrap_err()
                    .class()
            } else if let Some(base_hex) = vector["base_frame_hex"].as_str() {
                let base_bytes = decode_hex(base_hex);
                let limits = match vector.get("limits") {
                    Some(override_) => {
                        let mut limits = limits.clone();
                        if let Some(value) =
                            override_.get("max_sparse_runs").and_then(|v| v.as_u64())
                        {
                            limits = limits.with_max_sparse_runs(value);
                        }
                        limits
                    }
                    None => limits.clone(),
                };
                decode_with_base_frame(&parsed, &base_bytes, &MapResolver::new(), &limits)
                    .unwrap_err()
                    .class()
            } else {
                decode(&delta_bytes, None).unwrap_err().class()
            };
            assert_eq!(class.as_str(), expected, "{}", path.display());
            continue;
        }
        let base_bytes = decode_hex(vector["base_frame_hex"].as_str().unwrap());
        let target = decode_hex(vector["target_hex"].as_str().unwrap());
        let decoded = decode(&delta_bytes, Some(&base_bytes)).expect("decode valid vector");
        assert_eq!(decoded, target, "{}", path.display());
        let meta = sparse_delta_metadata(&parsed, &limits).expect("metadata");
        assert_eq!(meta.dtype.as_str(), vector["dtype"].as_str().unwrap());
        assert_eq!(
            meta.storage_bytes,
            vector["storage_bytes"].as_u64().unwrap()
        );
        assert_eq!(
            meta.delta_root.to_hex(),
            vector["delta_root"].as_str().unwrap()
        );
        assert_eq!(meta.base.to_hex(), vector["base"].as_str().unwrap());
        let reencoded = encode_sparse_delta(&target, meta.dtype, &meta.shape, &base_bytes, &limits)
            .expect("re-encode");
        assert_eq!(reencoded, delta_bytes, "{}", path.display());
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(binary()).args(args).output().expect("run")
}

fn assert_ok(output: &Output) {
    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_encode_decode_inspect_roundtrip() {
    let dir = std::env::temp_dir().join(format!("weftcodec-sparse-delta-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");

    let base_bytes = vec![0u8; 12];
    let mut target_bytes = base_bytes.clone();
    target_bytes[4..6].copy_from_slice(&7u16.to_le_bytes());
    let base = base_frame(TensorDtype::Bf16, &[2, 3], &base_bytes);
    let base_path = dir.join("base.wft1");
    let target_path = dir.join("target.bin");
    let delta_path = dir.join("delta.wft1");
    let out_path = dir.join("out.bin");
    fs::write(&base_path, &base).unwrap();
    fs::write(&target_path, &target_bytes).unwrap();

    assert_ok(&run(&[
        "encode",
        "--input",
        target_path.to_str().unwrap(),
        "--output",
        delta_path.to_str().unwrap(),
        "--mode",
        "sparse-delta",
        "--base",
        base_path.to_str().unwrap(),
        "--dtype",
        "bf16",
        "--shape",
        "2,3",
    ]));

    let output = run(&["inspect", delta_path.to_str().unwrap(), "--json"]);
    assert_ok(&output);
    let info: serde_json::Value = serde_json::from_slice(&output.stdout).expect("inspect JSON");
    assert_eq!(info["mode"], "sparse_delta");
    assert_eq!(info["mode_value"], 16);
    assert_eq!(info["dtype"], "bf16");
    assert_eq!(info["storage_bytes"], 2);
    assert_eq!(info["op"], "mod-sub");
    assert_eq!(info["sparse_codec"], "zero-run-v1");

    assert_ok(&run(&[
        "decode",
        "--input",
        delta_path.to_str().unwrap(),
        "--output",
        out_path.to_str().unwrap(),
        "--base",
        base_path.to_str().unwrap(),
    ]));
    assert_eq!(fs::read(&out_path).unwrap(), target_bytes);
    let _ = fs::remove_dir_all(&dir);
}
