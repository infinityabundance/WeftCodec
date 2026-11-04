//! Mode 11 `WEIGHT_DELTA` exactness, base resolution, and 4 MiB evidence.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use weftcodec::dependency::{decode_frame_dag, MapFrameSource, MapResolver};
use weftcodec::frame::Frame;
use weftcodec::hash::{hash_domain, Hash32, DOMAIN_CHUNK};
use weftcodec::huffman;
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::tensor::{self, LogicalRecord};
use weftcodec::weight_delta::{
    decode_with_base_frame, encode_descriptor_frame, encode_weight_delta, weight_delta_metadata,
    WeightDeltaMeta, WeightDtype, WEIGHT_DELTA_OP, WEIGHT_DELTA_PLANE_CODEC,
};
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
        None => weftcodec::weight_delta::decode_frame(&parsed, &resolver, &limits()),
    }
}

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
    let delta = encode_weight_delta(&target_bytes, WeightDtype::Bf16, &[2, 4], &base, &limits())
        .expect("encode");
    assert_eq!(decode(&delta, Some(&base)).expect("decode"), target_bytes);

    let parsed = Frame::parse(&delta, &limits()).expect("parse");
    let meta = weight_delta_metadata(&parsed, &limits()).expect("meta");
    assert_eq!(meta.dtype, WeightDtype::Bf16);
    assert_eq!(meta.word_bits, 16);
    assert_eq!(meta.planes, 2);
    assert_eq!(meta.op, WEIGHT_DELTA_OP);
    assert_eq!(meta.plane_codec, WEIGHT_DELTA_PLANE_CODEC);
    assert_eq!(
        meta.base,
        Frame::parse(&base, &limits()).unwrap().header.content_hash
    );
    assert_eq!(
        meta.base,
        Frame::parse(&base, &limits()).unwrap().header.content_hash
    );

    // Deterministic re-encoding.
    let again =
        encode_weight_delta(&target_bytes, WeightDtype::Bf16, &[2, 4], &base, &limits()).unwrap();
    assert_eq!(again, delta);
}

#[test]
fn fp32_all_zero_delta_reconstructs_exactly() {
    let words: [u32; 6] = [
        0x7FC0_0001,
        0x8000_0000,
        0x0000_0001,
        0x7F80_0000,
        0xFF80_0000,
        0x0000_0000,
    ];
    let mut bytes = Vec::new();
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let base = base_frame(TensorDtype::F32, &[3, 2], &bytes);
    let delta = encode_weight_delta(&bytes, WeightDtype::Fp32, &[3, 2], &base, &limits())
        .expect("encode zero delta");
    let parsed = Frame::parse(&delta, &limits()).unwrap();
    let meta = weight_delta_metadata(&parsed, &limits()).unwrap();
    // Four all-zero planes, each a single-symbol Huffman table and no bits.
    assert_eq!(meta.payload_len, 4 * huffman::TABLE_BYTES as u64);
    assert_eq!(
        meta.delta_root,
        hash_domain(tensor::DOMAIN_TENSOR_VALUES, &vec![0u8; bytes.len()])
    );
    assert_eq!(decode(&delta, Some(&base)).unwrap(), bytes);
}

#[test]
fn q8_0_and_q4_0_roundtrip() {
    let q8_base: Vec<u8> = (0..64u16)
        .map(|index| ((index * 37 + 11) % 251) as u8)
        .collect();
    let mut q8_target = q8_base.clone();
    for (index, byte) in q8_target.iter_mut().enumerate() {
        if index % 5 == 0 {
            *byte = byte.wrapping_add(7);
        }
    }
    let q8_frame = base_frame(TensorDtype::I8, &[8, 8], &q8_base);
    let q8_delta =
        encode_weight_delta(&q8_target, WeightDtype::Q8_0, &[8, 8], &q8_frame, &limits()).unwrap();
    assert_eq!(decode(&q8_delta, Some(&q8_frame)).unwrap(), q8_target);

    let q4_base: Vec<u8> = (0..48u16).map(|index| (index % 16) as u8).collect();
    let mut q4_target = q4_base.clone();
    for (index, byte) in q4_target.iter_mut().enumerate() {
        *byte = (*byte + (index % 7) as u8) % 16;
    }
    let q4_frame = base_frame(TensorDtype::U8, &[6, 8], &q4_base);
    let q4_delta =
        encode_weight_delta(&q4_target, WeightDtype::Q4_0, &[6, 8], &q4_frame, &limits()).unwrap();
    let parsed = Frame::parse(&q4_delta, &limits()).unwrap();
    let meta = weight_delta_metadata(&parsed, &limits()).unwrap();
    assert_eq!(meta.word_bits, 4);
    assert_eq!(meta.planes, 1);
    assert_eq!(decode(&q4_delta, Some(&q4_frame)).unwrap(), q4_target);
}

#[test]
fn base_dtype_and_shape_mismatches_fail_closed() {
    let base_bytes = vec![0u8; 16];
    let target_bytes = vec![1u8; 16];
    let bf16_base = base_frame(TensorDtype::Bf16, &[2, 4], &base_bytes);
    let delta = encode_weight_delta(
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
        encode_weight_delta(&[1u8; 8], WeightDtype::Bf16, &[2, 2], &base, &limits()).unwrap();
    assert_eq!(
        decode(&delta, None).unwrap_err().class(),
        ErrorClass::DependencyMissing
    );
}

#[test]
fn dependency_cycle_fails_closed() {
    let shape = vec![2u64, 3];
    let element_count = 6u64;
    let target = vec![0u8; 12];
    let content_hash = hash_domain(DOMAIN_CHUNK, &target);
    let mut payload = Vec::new();
    for _ in 0..2 {
        payload.extend_from_slice(&huffman::encode(&[0u8; 6]));
    }
    let meta = WeightDeltaMeta {
        dtype: WeightDtype::Bf16,
        shape: shape.clone(),
        element_count,
        word_bits: 16,
        // The base equals the frame's own content hash.
        base: content_hash,
        base_dtype: WeightDtype::Bf16,
        base_shape: shape,
        op: WEIGHT_DELTA_OP.to_owned(),
        planes: 2,
        plane_codec: WEIGHT_DELTA_PLANE_CODEC.to_owned(),
        delta_root: hash_domain(tensor::DOMAIN_TENSOR_VALUES, &[0u8; 12]),
        payload_len: payload.len() as u64,
    };
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
    // The resolver-only path reports the same cycle through the same check.
    assert_eq!(
        weftcodec::weight_delta::decode_frame(&parsed, &MapResolver::new(), &limits())
            .unwrap_err()
            .class(),
        ErrorClass::DependencyMissing
    );
}

#[test]
fn descriptor_rejects_have_typed_classes() {
    let base_bytes = vec![0u8; 12];
    let target_bytes = vec![0x11u8; 12];
    let base = base_frame(TensorDtype::Bf16, &[2, 3], &base_bytes);
    let valid =
        encode_weight_delta(&target_bytes, WeightDtype::Bf16, &[2, 3], &base, &limits()).unwrap();
    let parsed = Frame::parse(&valid, &limits()).unwrap();
    let meta = weight_delta_metadata(&parsed, &limits()).unwrap();
    let body = &parsed.body;
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let payload = body[4 + meta_len..].to_vec();
    let content_hash = parsed.header.content_hash;

    let mut unknown_op = meta.clone();
    unknown_op.op = "xor".to_owned();
    let frame =
        encode_descriptor_frame(&unknown_op, &payload, 12, content_hash, 6, &limits()).unwrap();
    assert_eq!(
        decode(&frame, Some(&base)).unwrap_err().class(),
        ErrorClass::Malformed
    );

    let mut word_bits = meta.clone();
    word_bits.word_bits = 8;
    let frame =
        encode_descriptor_frame(&word_bits, &payload, 12, content_hash, 6, &limits()).unwrap();
    assert_eq!(
        decode(&frame, Some(&base)).unwrap_err().class(),
        ErrorClass::Malformed
    );

    let mut delta_root = meta.clone();
    let mut flipped = delta_root.delta_root.into_bytes();
    flipped[0] ^= 1;
    delta_root.delta_root = Hash32::from_bytes(flipped);
    let frame =
        encode_descriptor_frame(&delta_root, &payload, 12, content_hash, 6, &limits()).unwrap();
    assert_eq!(
        decode(&frame, Some(&base)).unwrap_err().class(),
        ErrorClass::Corrupt
    );

    let mut payload_len = meta.clone();
    payload_len.payload_len += 1;
    let frame =
        encode_descriptor_frame(&payload_len, &payload, 12, content_hash, 6, &limits()).unwrap();
    assert_eq!(
        decode(&frame, Some(&base)).unwrap_err().class(),
        ErrorClass::Malformed
    );
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn four_mib_bf16_one_percent_delta_is_small_and_exact() {
    let limits = limits();
    let element_count: usize = 2 * 1024 * 1024; // 4 MiB of bf16
    let shape = vec![element_count as u64];
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut base_bytes = Vec::with_capacity(element_count * 2);
    for _ in 0..element_count {
        let word = (xorshift(&mut state) & 0xFFFF) as u16;
        base_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut target_bytes = base_bytes.clone();
    let mut changed = 0usize;
    for index in (0..element_count).step_by(100) {
        target_bytes[index * 2] ^= 0x01;
        target_bytes[index * 2 + 1] ^= 0x20;
        changed += 1;
    }
    assert_eq!(changed, element_count.div_ceil(100));

    let base = modes::encode_tensor(&base_bytes, TensorDtype::Bf16, &shape, &limits).unwrap();
    let delta =
        encode_weight_delta(&target_bytes, WeightDtype::Bf16, &shape, &base, &limits).unwrap();
    let decoded = decode_with_base_frame(
        &Frame::parse(&delta, &limits).unwrap(),
        &base,
        &MapResolver::new(),
        &limits,
    )
    .unwrap();
    assert_eq!(decoded, target_bytes, "reconstruction must be byte-exact");

    let original_root =
        LogicalRecord::from_row_major(TensorDtype::Bf16, &shape, &target_bytes, &limits)
            .unwrap()
            .root_with_limits(&limits)
            .unwrap();
    let decoded_root = LogicalRecord::from_row_major(TensorDtype::Bf16, &shape, &decoded, &limits)
        .unwrap()
        .root_with_limits(&limits)
        .unwrap();
    assert_eq!(decoded_root, original_root);

    let zstd_bytes = zstd::bulk::compress(&target_bytes, 3).expect("zstd").len();
    eprintln!(
        "4 MiB bf16 1% delta: delta_frame={} independent_zstd_l3={} base_frame={} changed_words={}",
        delta.len(),
        zstd_bytes,
        base.len(),
        changed
    );
    assert!(
        delta.len() * 4 < zstd_bytes,
        "delta frame {} must be dramatically smaller than independent zstd {}",
        delta.len(),
        zstd_bytes
    );
}

// ---------------------------------------------------------------------------
// CLI round-trip
// ---------------------------------------------------------------------------

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_weftcodec")
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "weftcodec-weight-delta-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn decode_hex(text: &str) -> Vec<u8> {
    hex::decode(text).expect("hex")
}

#[test]
fn committed_vectors_replay() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1/weight-delta");
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("vector dir")
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 11, "expected 11 WEIGHT_DELTA vectors");
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
                decode(&delta_bytes, Some(&base_bytes)).unwrap_err().class()
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
        let meta = weight_delta_metadata(&parsed, &limits).expect("metadata");
        assert_eq!(meta.dtype.as_str(), vector["dtype"].as_str().unwrap());
        assert_eq!(meta.word_bits, vector["word_bits"].as_u64().unwrap());
        assert_eq!(meta.planes, vector["planes"].as_u64().unwrap());
        assert_eq!(
            meta.delta_root.to_hex(),
            vector["delta_root"].as_str().unwrap()
        );
        assert_eq!(meta.base.to_hex(), vector["base"].as_str().unwrap());
        let reencoded = encode_weight_delta(&target, meta.dtype, &meta.shape, &base_bytes, &limits)
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

fn inspect_json(path: &Path) -> serde_json::Value {
    let output = run(&["inspect", path.to_str().unwrap(), "--json"]);
    assert_ok(&output);
    serde_json::from_slice(&output.stdout).expect("inspect JSON")
}

#[test]
fn cli_encode_decode_inspect_roundtrip() {
    let dir = temp_dir("cli");
    let base_bytes = vec![0u8; 12];
    let target_bytes = vec![0x22u8; 12];
    let base_frame = base_frame(TensorDtype::Bf16, &[2, 3], &base_bytes);
    let base_path = dir.join("base.wft1");
    let target_path = dir.join("target.bin");
    let delta_path = dir.join("delta.wft1");
    let out_path = dir.join("out.bin");
    fs::write(&base_path, &base_frame).unwrap();
    fs::write(&target_path, &target_bytes).unwrap();

    let output = run(&[
        "encode",
        "--input",
        target_path.to_str().unwrap(),
        "--output",
        delta_path.to_str().unwrap(),
        "--mode",
        "weight-delta",
        "--base",
        base_path.to_str().unwrap(),
        "--dtype",
        "bf16",
        "--shape",
        "2,3",
    ]);
    assert_ok(&output);

    let info = inspect_json(&delta_path);
    assert_eq!(info["mode"], "weight_delta");
    assert_eq!(info["mode_value"], 11);
    assert_eq!(info["dtype"], "bf16");
    assert_eq!(info["word_bits"], 16);
    assert_eq!(info["planes"], 2);
    assert_eq!(info["op"], "mod-sub");
    assert_eq!(info["plane_codec"], "huffman-canonical-v1");

    let output = run(&[
        "decode",
        "--input",
        delta_path.to_str().unwrap(),
        "--output",
        out_path.to_str().unwrap(),
        "--base",
        base_path.to_str().unwrap(),
    ]);
    assert_ok(&output);
    assert_eq!(fs::read(&out_path).unwrap(), target_bytes);
}

#[test]
fn cli_requires_a_base_frame_for_encoding() {
    let dir = temp_dir("cli-no-base");
    let target_path = dir.join("target.bin");
    let delta_path = dir.join("delta.wft1");
    fs::write(&target_path, vec![0u8; 12]).unwrap();
    let output = run(&[
        "encode",
        "--input",
        target_path.to_str().unwrap(),
        "--output",
        delta_path.to_str().unwrap(),
        "--mode",
        "weight-delta",
        "--dtype",
        "bf16",
        "--shape",
        "2,3",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires --base"));
}
