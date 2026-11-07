//! Generation of the committed v1 conformance vectors.
//!
//! Vectors are append-only once committed. `conformance/v1/codec/*.json`
//! contains complete frames (hex), plaintext, dependency bytes, and the
//! expected decode outcome. `structured/*.json` freezes canonical WCB bytes and
//! non-canonical rejects. `selectors/*.json` freezes canonical selector bytes
//! and ids.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::{json, Value as Json};

use crate::activation_stream;
use crate::capability::{self, CapabilityEnvelope};
use crate::context::{self, Budget, CausalClosure, ContextRoot, DEFAULT_RESOLVER};
use crate::error::{Error, ErrorClass, Result};
use crate::frame::{
    encode_frame, Dependency, Frame, FrameParts, Mode, DEP_DOMAIN_DICTIONARY, DEP_DOMAIN_PAYLOAD,
    DOMAIN_PAYLOAD, DOMAIN_STRUCTURED, FORMAT_MAJOR, FORMAT_MINOR,
};
use crate::hash::{
    hash_domain, Hash32, DOMAIN_CHUNK, DOMAIN_FLOAT_TENSOR_PAYLOAD,
    DOMAIN_QUANTIZED_TENSOR_PAYLOAD, DOMAIN_SELECTOR, DOMAIN_TENSOR_PAYLOAD,
    DOMAIN_TENSOR_SHARDED_PAYLOAD, DOMAIN_TOKENS_PAYLOAD,
};
use crate::kv_cache;
use crate::limits::Limits;
use crate::modes;
use crate::selector;
use crate::semantic::{SemanticPolicy, SemanticRootRecord, SEMANTIC_SCHEMA};
use crate::semantic_delta::{self, SemanticDeltaMeta, SEMANTIC_DELTA_SCHEMA};
use crate::sparse_delta::{self, SparseDeltaMeta};
use crate::tensor::{self, LayoutDescriptor, LogicalRecord, TensorLayout};
use crate::tensor_sharded;
use crate::varint;
use crate::wcb::{self, Value};
use crate::weight_delta::{self, WeightDeltaMeta, WeightDtype};

/// Generate every v1 vector into `dir` (creating `codec/`, `structured/`,
/// `selectors/`, `context/`, `causal/`, and `tensor/`).
pub fn generate_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let codec_dir = dir.join("codec");
    let structured_dir = dir.join("structured");
    let selectors_dir = dir.join("selectors");
    let context_dir = dir.join("context");
    let causal_dir = dir.join("causal");
    let tensor_dir = dir.join("tensor");
    let sharded_dir = dir.join("sharded");
    let weight_delta_dir = dir.join("weight-delta");
    let semantic_root_dir = dir.join("semantic-root");
    let semantic_delta_dir = dir.join("semantic-delta");
    let kv_cache_dir = dir.join("kv-cache");
    let activation_stream_dir = dir.join("activation-stream");
    let capability_dir = dir.join("capability");
    let tensor_tiled_dir = dir.join("tensor-tiled");
    let sparse_delta_dir = dir.join("sparse-delta");
    for sub in [
        &codec_dir,
        &structured_dir,
        &selectors_dir,
        &context_dir,
        &causal_dir,
        &tensor_dir,
        &sharded_dir,
        &weight_delta_dir,
        &semantic_root_dir,
        &semantic_delta_dir,
        &kv_cache_dir,
        &activation_stream_dir,
        &capability_dir,
        &tensor_tiled_dir,
        &sparse_delta_dir,
    ] {
        fs::create_dir_all(sub)
            .map_err(|e| Error::io(format!("cannot create {}: {e}", sub.display())))?;
    }

    write_codec_vectors(&codec_dir, limits)?;
    write_structured_vectors(&structured_dir, limits)?;
    write_selector_vectors(&selectors_dir, limits)?;
    write_context_vectors(&context_dir, limits)?;
    write_causal_vectors(&causal_dir, limits)?;
    write_tensor_vectors(&tensor_dir, limits)?;
    write_sharded_vectors(&sharded_dir, limits)?;
    write_weight_delta_vectors(&weight_delta_dir, limits)?;
    write_semantic_root_vectors(&semantic_root_dir, limits)?;
    write_semantic_delta_vectors(&semantic_delta_dir, limits)?;
    write_kv_cache_vectors(&kv_cache_dir, limits)?;
    write_activation_stream_vectors(&activation_stream_dir, limits)?;
    write_capability_vectors(&capability_dir, limits)?;
    write_tiled_vectors(&tensor_tiled_dir, limits)?;
    write_sparse_delta_vectors(&sparse_delta_dir, limits)?;
    Ok(())
}

/// Generate only the authenticated capability-envelope vectors into
/// `<dir>/capability`.
pub fn generate_capability_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let capability_dir = dir.join("capability");
    fs::create_dir_all(&capability_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", capability_dir.display())))?;
    write_capability_vectors(&capability_dir, limits)
}

/// Generate only the additive WEFT-TENSOR vectors into `<dir>/tensor`.
pub fn generate_tensor_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let tensor_dir = dir.join("tensor");
    fs::create_dir_all(&tensor_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", tensor_dir.display())))?;
    write_tensor_vectors(&tensor_dir, limits)
}

/// Generate only the TENSOR_SHARDED (mode 10) vectors into `<dir>/sharded`.
pub fn generate_sharded_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let sharded_dir = dir.join("sharded");
    fs::create_dir_all(&sharded_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", sharded_dir.display())))?;
    write_sharded_vectors(&sharded_dir, limits)
}

/// Generate only the WEIGHT_DELTA (mode 11) vectors into
/// `<dir>/weight-delta`.
pub fn generate_weight_delta_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let weight_delta_dir = dir.join("weight-delta");
    fs::create_dir_all(&weight_delta_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", weight_delta_dir.display())))?;
    write_weight_delta_vectors(&weight_delta_dir, limits)
}

/// Generate only the SPARSE_DELTA (mode 16) vectors into `<dir>/sparse-delta`.
pub fn generate_sparse_delta_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let sparse_delta_dir = dir.join("sparse-delta");
    fs::create_dir_all(&sparse_delta_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", sparse_delta_dir.display())))?;
    write_sparse_delta_vectors(&sparse_delta_dir, limits)
}

/// Generate only the SEMANTIC_DELTA (mode 14) and semantic-root vectors into
/// `<dir>/semantic-delta` and `<dir>/semantic-root`.
pub fn generate_semantic_delta_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let root_dir = dir.join("semantic-root");
    let delta_dir = dir.join("semantic-delta");
    for sub in [&root_dir, &delta_dir] {
        fs::create_dir_all(sub)
            .map_err(|e| Error::io(format!("cannot create {}: {e}", sub.display())))?;
    }
    write_semantic_root_vectors(&root_dir, limits)?;
    write_semantic_delta_vectors(&delta_dir, limits)
}

/// Generate only the KV_CACHE (mode 12) vectors into `<dir>/kv-cache`.
pub fn generate_kv_cache_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let kv_cache_dir = dir.join("kv-cache");
    fs::create_dir_all(&kv_cache_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", kv_cache_dir.display())))?;
    write_kv_cache_vectors(&kv_cache_dir, limits)
}

/// Generate only the ACTIVATION_STREAM (mode 13) vectors into
/// `<dir>/activation-stream`.
pub fn generate_activation_stream_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let activation_dir = dir.join("activation-stream");
    fs::create_dir_all(&activation_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", activation_dir.display())))?;
    write_activation_stream_vectors(&activation_dir, limits)
}

fn write_json(path: &Path, value: &Json) -> Result<()> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| Error::io(format!("cannot serialize vector: {e}")))?;
    fs::write(path, format!("{text}\n"))
        .map_err(|e| Error::io(format!("cannot write {}: {e}", path.display())))
}

fn pattern(len: usize, mut state: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state & 0xff) as u8);
    }
    out
}

fn repeat(text: &[u8], len: usize) -> Vec<u8> {
    text.iter().copied().cycle().take(len).collect()
}

fn hexs(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

fn deps_json(deps: &[(Hash32, Vec<u8>, u8)]) -> Json {
    Json::Array(
        deps.iter()
            .map(|(hash, bytes, domain)| {
                json!({
                    "hash": hash.to_hex(),
                    "decoded_len": bytes.len(),
                    "dep_domain": domain,
                    "bytes_hex": hexs(bytes),
                })
            })
            .collect(),
    )
}

fn push_ok(
    vectors: &mut Vec<Json>,
    name: &str,
    frame: &[u8],
    plaintext: &[u8],
    deps: &[(Hash32, Vec<u8>, u8)],
) -> Result<()> {
    let parsed = Frame::parse(frame, &Limits::default())?;
    vectors.push(json!({
        "name": name,
        "mode": parsed.header.mode.as_str(),
        "mode_value": parsed.header.mode.as_u8(),
        "format_major": parsed.header.format_major,
        "format_minor": parsed.header.format_minor,
        "domain": parsed.header.domain,
        "flags": parsed.header.flags,
        "features": parsed.header.features,
        "decoded_len": parsed.header.decoded_len,
        "body_len": parsed.header.body_len,
        "frame_len": frame.len(),
        "chunk_count": parsed.header.chunk_count,
        "content_hash": parsed.header.content_hash.to_hex(),
        "plaintext_hex": hexs(plaintext),
        "plaintext_hash": hash_domain(DOMAIN_CHUNK, plaintext).to_hex(),
        "frame_hex": hexs(frame),
        "dependencies": deps_json(deps),
        "expect": { "ok": true },
    }));
    Ok(())
}

fn push_error(
    vectors: &mut Vec<Json>,
    name: &str,
    mode: &str,
    frame: &[u8],
    deps: &[(Hash32, Vec<u8>, u8)],
    class: ErrorClass,
    limits_override: Option<Json>,
) {
    let mut value = json!({
        "name": name,
        "mode": mode,
        "frame_hex": hexs(frame),
        "dependencies": deps_json(deps),
        "expect": { "error": class.as_str() },
    });
    if let Some(limits) = limits_override {
        value["limits"] = limits;
    }
    vectors.push(value);
}

fn push_error_superseded(
    vectors: &mut Vec<Json>,
    name: &str,
    mode: &str,
    frame: &[u8],
    deps: &[(Hash32, Vec<u8>, u8)],
    class: ErrorClass,
    limits_override: Option<Json>,
    superseded_by: &str,
    note: &str,
) {
    let mut value = json!({
        "name": name,
        "mode": mode,
        "frame_hex": hexs(frame),
        "dependencies": deps_json(deps),
        "expect": { "error": class.as_str() },
        "superseded_by": superseded_by,
        "note": note,
    });
    if let Some(limits) = limits_override {
        value["limits"] = limits;
    }
    vectors.push(value);
}

fn refix_trailer(frame: &mut [u8]) {
    let at = frame.len() - 4;
    let crc = crc32c::crc32c(&frame[..at]);
    frame[at..].copy_from_slice(&crc.to_le_bytes());
}

fn refix_header_and_trailer(frame: &mut [u8]) {
    let mut crc_input = Vec::with_capacity(92);
    crc_input.extend_from_slice(&frame[0..72]);
    crc_input.extend_from_slice(&frame[76..96]);
    let crc = crc32c::crc32c(&crc_input);
    frame[72..76].copy_from_slice(&crc.to_le_bytes());
    refix_trailer(frame);
}

fn ai_body(meta: &Value, payload: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(meta, limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("AI metadata does not fit u32"))?;
    let mut body = Vec::with_capacity(4 + meta_bytes.len() + payload.len());
    body.extend_from_slice(&meta_len.to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(payload);
    Ok(body)
}

fn ai_frame(mode: Mode, meta: &Value, payload: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let body = ai_body(meta, payload, limits)?;
    build_frame(
        mode,
        DOMAIN_PAYLOAD,
        0,
        payload.len() as u64,
        hash_domain(DOMAIN_CHUNK, payload),
        &[],
        &body,
        limits,
    )
}

fn tensor_meta_value(
    dtype: &str,
    shape: &[u64],
    count: u64,
    payload_bytes: u64,
    payload_hash: &[u8],
    layout: &str,
    endianness: &str,
) -> Value {
    Value::map([
        ("schema", Value::Str("weft.tensor.v1".to_owned())),
        ("dtype", Value::Str(dtype.to_owned())),
        (
            "shape",
            Value::Array(shape.iter().map(|d| Value::U64(*d)).collect()),
        ),
        ("layout", Value::Str(layout.to_owned())),
        ("endianness", Value::Str(endianness.to_owned())),
        ("element_count", Value::U64(count)),
        ("payload_bytes", Value::U64(payload_bytes)),
        ("payload_hash", Value::Bytes(payload_hash.to_vec())),
    ])
}

fn quantized_meta_value(
    dtype: &str,
    shape: &[u64],
    count: u64,
    payload_bytes: u64,
    payload_hash: &[u8],
    layout: &str,
    quantization: &str,
    scale_bits: &[u8],
    zero_point: i64,
) -> Value {
    Value::map([
        ("schema", Value::Str("weft.quantized-tensor.v1".to_owned())),
        ("dtype", Value::Str(dtype.to_owned())),
        (
            "shape",
            Value::Array(shape.iter().map(|d| Value::U64(*d)).collect()),
        ),
        ("layout", Value::Str(layout.to_owned())),
        ("quantization", Value::Str(quantization.to_owned())),
        ("scale_bits_le", Value::Bytes(scale_bits.to_vec())),
        ("zero_point", Value::I64(zero_point)),
        ("element_count", Value::U64(count)),
        ("payload_bytes", Value::U64(payload_bytes)),
        ("payload_hash", Value::Bytes(payload_hash.to_vec())),
    ])
}

fn token_meta_value(
    vocab_schema: &str,
    encoding: &str,
    count: u64,
    payload_bytes: u64,
    payload_hash: &[u8],
) -> Value {
    Value::map([
        ("schema", Value::Str("weft.tokens.v1".to_owned())),
        ("vocab_schema", Value::Str(vocab_schema.to_owned())),
        ("encoding", Value::Str(encoding.to_owned())),
        ("token_count", Value::U64(count)),
        ("payload_bytes", Value::U64(payload_bytes)),
        ("payload_hash", Value::Bytes(payload_hash.to_vec())),
    ])
}

fn float_tensor_meta_value(
    dtype: &str,
    shape: &[u64],
    count: u64,
    sign_bits: u64,
    exponent_bits: u64,
    mantissa_bits: u64,
    exponent_plane_len: u64,
    mantissa_plane_len: u64,
    payload_hash: &[u8],
) -> Value {
    Value::map([
        ("schema", Value::Str("weft.ai.float_tensor.v1".to_owned())),
        ("dtype", Value::Str(dtype.to_owned())),
        (
            "shape",
            Value::Array(shape.iter().map(|d| Value::U64(*d)).collect()),
        ),
        ("element_count", Value::U64(count)),
        ("sign_bits", Value::U64(sign_bits)),
        ("exponent_bits", Value::U64(exponent_bits)),
        ("mantissa_bits", Value::U64(mantissa_bits)),
        (
            "exponent_coder",
            Value::Str(modes::FLOAT_TENSOR_EXPONENT_CODER.to_owned()),
        ),
        ("exponent_plane_len", Value::U64(exponent_plane_len)),
        ("mantissa_plane_len", Value::U64(mantissa_plane_len)),
        ("payload_hash", Value::Bytes(payload_hash.to_vec())),
    ])
}

/// Build a FLOAT_TENSOR frame whose decoded length is the reconstructed element
/// length, independent of the stored plane length.
fn float_ai_frame(
    meta: &Value,
    stored: &[u8],
    decoded_len: u64,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let body = ai_body(meta, stored, limits)?;
    let plaintext = vec![0u8; decoded_len as usize];
    build_frame(
        Mode::FloatTensor,
        DOMAIN_PAYLOAD,
        0,
        decoded_len,
        hash_domain(DOMAIN_CHUNK, &plaintext),
        &[],
        &body,
        limits,
    )
}

fn build_frame(
    mode: Mode,
    domain: u8,
    flags: u16,
    decoded_len: u64,
    content_hash: Hash32,
    deps: &[Dependency],
    body: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    encode_frame(
        &FrameParts {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            domain,
            mode,
            flags,
            level: 0,
            decoded_len,
            content_hash,
            dependencies: deps,
            chunks: &[],
            body,
        },
        limits,
    )
}

fn write_codec_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    let mut vectors: Vec<Json> = Vec::new();
    let mut index = 0usize;
    let mut name = |base: &str| -> String {
        index += 1;
        format!("{index:03}_{base}")
    };

    // RAW: empty, one byte, 4 KiB, 1 MiB, repetitive.
    let frame = modes::encode_raw(b"", None, limits)?;
    push_ok(&mut vectors, &name("raw_empty"), &frame, b"", &[])?;
    let frame = modes::encode_raw(b"A", None, limits)?;
    push_ok(&mut vectors, &name("raw_one_byte"), &frame, b"A", &[])?;
    let data_4k = pattern(4096, 0x1234_5678_9abc_def0);
    let frame = modes::encode_raw(&data_4k, None, limits)?;
    push_ok(&mut vectors, &name("raw_4k"), &frame, &data_4k, &[])?;
    let data_1m = pattern(1 << 20, 0x0fed_cba9_8765_4321);
    let frame = modes::encode_raw(&data_1m, None, limits)?;
    push_ok(&mut vectors, &name("raw_1m"), &frame, &data_1m, &[])?;
    let data_rep = repeat(b"weft", 64 << 10);
    let frame = modes::encode_raw(&data_rep, None, limits)?;
    push_ok(
        &mut vectors,
        &name("raw_64k_repetitive"),
        &frame,
        &data_rep,
        &[],
    )?;

    // Chunked RAW and ZSTD.
    let data_chunked = pattern(5000, 0x9999_1111_2222_3333);
    let frame = modes::encode_raw(&data_chunked, Some(1024), limits)?;
    push_ok(
        &mut vectors,
        &name("raw_chunked_1024"),
        &frame,
        &data_chunked,
        &[],
    )?;
    let frame = modes::encode_zstd(&data_chunked, 3, Some(1024), limits)?;
    push_ok(
        &mut vectors,
        &name("zstd_chunked_1024"),
        &frame,
        &data_chunked,
        &[],
    )?;

    // ZSTD 4 KiB.
    let frame = modes::encode_zstd(&data_4k, 3, None, limits)?;
    push_ok(&mut vectors, &name("zstd_4k"), &frame, &data_4k, &[])?;

    // REFERENCE.
    let base = repeat(b"reference base payload:", 200);
    let frame = modes::encode_reference(&base, &base, limits)?;
    let dep = (
        hash_domain(DOMAIN_CHUNK, &base),
        base.clone(),
        DEP_DOMAIN_PAYLOAD,
    );
    push_ok(
        &mut vectors,
        &name("reference_equal"),
        &frame,
        &base,
        &[dep],
    )?;

    // DELTA with insert/delete/replace.
    let base = data_4k.clone();
    let mut target = Vec::new();
    target.extend_from_slice(&base[..100]);
    target.extend_from_slice(b"INSERTED-REVISION-2");
    target.extend_from_slice(&base[150..500]);
    target.extend_from_slice(b"REPLACED-BLOCK");
    target.extend_from_slice(&base[700..3000]);
    target.extend_from_slice(&base[3200..]);
    let frame = modes::encode_delta(&target, &base, limits)?;
    let dep = (
        hash_domain(DOMAIN_CHUNK, &base),
        base.clone(),
        DEP_DOMAIN_PAYLOAD,
    );
    push_ok(&mut vectors, &name("delta_edits"), &frame, &target, &[dep])?;

    // DICTIONARY.
    let dict = repeat(b"weft codec dictionary line alpha beta gamma\n", 1024);
    let data = repeat(b"weft codec dictionary line alpha beta gamma\n", 300);
    let frame = modes::encode_dictionary(&data, &dict, 3, limits)?;
    let dep = (
        hash_domain(DOMAIN_CHUNK, &dict),
        dict.clone(),
        DEP_DOMAIN_DICTIONARY,
    );
    push_ok(&mut vectors, &name("dictionary_raw"), &frame, &data, &[dep])?;

    // STRUCTURED.
    let value = Value::map([
        ("name", Value::Str("vector".to_owned())),
        ("count", Value::U64(42)),
        ("signed", Value::I64(-7)),
        ("flags", Value::Array(vec![Value::Bool(true), Value::Null])),
    ]);
    let frame = modes::encode_structured(&value, limits)?;
    let structured_body = wcb::encode(&value, limits)?;
    push_ok(
        &mut vectors,
        &name("structured_map"),
        &frame,
        &structured_body,
        &[],
    )?;

    // --- Error vectors -----------------------------------------------------
    let good = modes::encode_raw(&data_4k, None, limits)?;

    let mut bad_magic = good.clone();
    bad_magic[0] ^= 0xff;
    push_error(
        &mut vectors,
        &name("bad_magic"),
        "raw",
        &bad_magic,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let mut bad_header_crc = good.clone();
    bad_header_crc[72] ^= 0xff;
    push_error(
        &mut vectors,
        &name("bad_header_crc"),
        "raw",
        &bad_header_crc,
        &[],
        ErrorClass::Corrupt,
        None,
    );

    let mut bad_trailer = good.clone();
    let last = bad_trailer.len() - 1;
    bad_trailer[last] ^= 0xff;
    push_error(
        &mut vectors,
        &name("bad_trailer_crc"),
        "raw",
        &bad_trailer,
        &[],
        ErrorClass::Corrupt,
        None,
    );

    let mut truncated = good.clone();
    truncated.pop();
    push_error(
        &mut vectors,
        &name("truncated"),
        "raw",
        &truncated,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let mut unknown_mode = good.clone();
    unknown_mode[7] = 14;
    refix_header_and_trailer(&mut unknown_mode);
    push_error_superseded(
        &mut vectors,
        &name("unknown_mode_reserved14"),
        "raw",
        &unknown_mode,
        &[],
        ErrorClass::UnsupportedMode,
        None,
        "unknown_mode_reserved15",
        "Historical: this vector asserted that mode value 14 was reserved. WEFT-AI-CODEC allocates 14 to SEMANTIC_DELTA, so the authoritative unknown-mode vector is 017_unknown_mode_reserved15. The frozen bytes and original expectation are retained unchanged.",
    );

    let mut unknown_mode = good.clone();
    unknown_mode[7] = 15;
    refix_header_and_trailer(&mut unknown_mode);
    push_error_superseded(
        &mut vectors,
        "017_unknown_mode_reserved15",
        "raw",
        &unknown_mode,
        &[],
        ErrorClass::UnsupportedMode,
        None,
        "017_unknown_mode_reserved16",
        "Historical: this vector asserted that mode value 15 was reserved. WEFT-AI-CODEC allocates 15 to TENSOR_TILED, so the authoritative unknown-mode vector is 017_unknown_mode_reserved16. The frozen bytes and original expectation are retained unchanged.",
    );

    let mut unknown_mode = good.clone();
    unknown_mode[7] = 16;
    refix_header_and_trailer(&mut unknown_mode);
    push_error_superseded(
        &mut vectors,
        "017_unknown_mode_reserved16",
        "raw",
        &unknown_mode,
        &[],
        ErrorClass::UnsupportedMode,
        None,
        "017_unknown_mode_reserved17",
        "Historical: this vector asserted that mode value 16 was reserved. WEFT-AI-CODEC allocates 16 to SPARSE_DELTA, so the authoritative unknown-mode vector is 017_unknown_mode_reserved17. The frozen bytes and original expectation are retained unchanged.",
    );

    let mut unknown_mode = good.clone();
    unknown_mode[7] = 17;
    refix_header_and_trailer(&mut unknown_mode);
    push_error_superseded(
        &mut vectors,
        "017_unknown_mode_reserved17",
        "raw",
        &unknown_mode,
        &[],
        ErrorClass::UnsupportedMode,
        None,
        "017_unknown_mode_reserved18",
        "Historical: this vector asserted that mode value 17 was reserved. WEFT-AI-CODEC allocates 17 to FLOAT_TILED, so the authoritative unknown-mode vector is 017_unknown_mode_reserved18. The frozen bytes and original expectation are retained unchanged.",
    );

    let mut unknown_mode = good.clone();
    unknown_mode[7] = 18;
    refix_header_and_trailer(&mut unknown_mode);
    push_error(
        &mut vectors,
        "017_unknown_mode_reserved18",
        "raw",
        &unknown_mode,
        &[],
        ErrorClass::UnsupportedMode,
        None,
    );

    let mut unknown_feature = good.clone();
    let features = u16::from_le_bytes([unknown_feature[36], unknown_feature[37]]) | 0x0004;
    unknown_feature[36..38].copy_from_slice(&features.to_le_bytes());
    refix_header_and_trailer(&mut unknown_feature);
    push_error(
        &mut vectors,
        &name("unknown_feature"),
        "raw",
        &unknown_feature,
        &[],
        ErrorClass::UnsupportedFeature,
        None,
    );

    let mut self_contained = good.clone();
    let flags = u16::from_le_bytes([self_contained[8], self_contained[9]]) | 0x0002;
    self_contained[8..10].copy_from_slice(&flags.to_le_bytes());
    refix_header_and_trailer(&mut self_contained);
    push_error(
        &mut vectors,
        &name("self_contained_reserved"),
        "raw",
        &self_contained,
        &[],
        ErrorClass::UnsupportedFeature,
        None,
    );

    let mut bad_version = good.clone();
    bad_version[4] = 2;
    refix_header_and_trailer(&mut bad_version);
    push_error(
        &mut vectors,
        &name("unsupported_version"),
        "raw",
        &bad_version,
        &[],
        ErrorClass::UnsupportedVersion,
        None,
    );

    let mut table_len_bad = good.clone();
    let table_len = u64::from_le_bytes(table_len_bad[76..84].try_into().unwrap_or([0; 8])) + 1;
    table_len_bad[76..84].copy_from_slice(&table_len.to_le_bytes());
    refix_header_and_trailer(&mut table_len_bad);
    push_error(
        &mut vectors,
        &name("bad_table_len"),
        "raw",
        &table_len_bad,
        &[],
        ErrorClass::Malformed,
        None,
    );

    // Chunk gap: second chunk offset moved by one.
    let chunked = modes::encode_raw(&pattern(3000, 7), Some(1024), limits)?;
    let mut chunk_gap = chunked.clone();
    let second_off_at = 96 + 48;
    let off = u64::from_le_bytes(
        chunk_gap[second_off_at..second_off_at + 8]
            .try_into()
            .unwrap_or([0; 8]),
    ) + 1;
    chunk_gap[second_off_at..second_off_at + 8].copy_from_slice(&off.to_le_bytes());
    refix_trailer(&mut chunk_gap);
    push_error(
        &mut vectors,
        &name("chunk_gap"),
        "raw",
        &chunk_gap,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let mut chunk_overlap = chunked.clone();
    chunk_overlap[second_off_at..second_off_at + 8].copy_from_slice(&96u64.to_le_bytes());
    refix_trailer(&mut chunk_overlap);
    push_error(
        &mut vectors,
        &name("chunk_overlap"),
        "raw",
        &chunk_overlap,
        &[],
        ErrorClass::Malformed,
        None,
    );

    // Bad dependency length on a DELTA frame.
    let delta = modes::encode_delta(&target, &base, limits)?;
    let mut bad_dep_len = delta.clone();
    let new_dep_len = (base.len() as u64) - 1;
    bad_dep_len[128..136].copy_from_slice(&new_dep_len.to_le_bytes());
    refix_trailer(&mut bad_dep_len);
    let dep = (
        hash_domain(DOMAIN_CHUNK, &base),
        base.clone(),
        DEP_DOMAIN_PAYLOAD,
    );
    push_error(
        &mut vectors,
        &name("bad_dependency_length"),
        "delta",
        &bad_dep_len,
        &[dep.clone()],
        ErrorClass::DependencyMismatch,
        None,
    );

    // Bad dependency hash: dependency table points at a hash whose bytes do
    // not match. The vector supplies the (wrong) hash with the real base bytes.
    let mut bad_dep_hash = delta.clone();
    bad_dep_hash[96] ^= 0xff;
    refix_trailer(&mut bad_dep_hash);
    let mut wrong_hash_bytes = [0u8; 32];
    wrong_hash_bytes.copy_from_slice(&bad_dep_hash[96..128]);
    let wrong_hash = Hash32::from_bytes(wrong_hash_bytes);
    push_error(
        &mut vectors,
        &name("bad_dependency_hash"),
        "delta",
        &bad_dep_hash,
        &[(wrong_hash, base.clone(), DEP_DOMAIN_PAYLOAD)],
        ErrorClass::DependencyMismatch,
        None,
    );

    // Delta COPY offset out of range.
    let dep_entry = Dependency::new(
        hash_domain(DOMAIN_CHUNK, &base),
        base.len() as u64,
        DEP_DOMAIN_PAYLOAD,
        0,
    );
    let mut body = vec![0x00];
    varint::encode(1 << 40, &mut body);
    varint::encode(1, &mut body);
    let frame = build_frame(
        Mode::Delta,
        DOMAIN_PAYLOAD,
        0,
        1,
        hash_domain(DOMAIN_CHUNK, b"x"),
        &[dep_entry.clone()],
        &body,
        limits,
    )?;
    push_error(
        &mut vectors,
        &name("delta_offset_out_of_range"),
        "delta",
        &frame,
        &[(
            hash_domain(DOMAIN_CHUNK, &base),
            base.clone(),
            DEP_DOMAIN_PAYLOAD,
        )],
        ErrorClass::Malformed,
        None,
    );

    // Delta emits fewer bytes than decoded_len.
    let mut body = vec![0x00];
    varint::encode(0, &mut body);
    varint::encode(1, &mut body);
    let frame = build_frame(
        Mode::Delta,
        DOMAIN_PAYLOAD,
        0,
        2,
        hash_domain(DOMAIN_CHUNK, b"xy"),
        &[dep_entry],
        &body,
        limits,
    )?;
    push_error(
        &mut vectors,
        &name("delta_length_mismatch"),
        "delta",
        &frame,
        &[(
            hash_domain(DOMAIN_CHUNK, &base),
            base.clone(),
            DEP_DOMAIN_PAYLOAD,
        )],
        ErrorClass::Corrupt,
        None,
    );

    // Structured body that is not canonical WCB.
    let bad_wcb = vec![0x0au8];
    let frame = build_frame(
        Mode::Structured,
        DOMAIN_STRUCTURED,
        0,
        1,
        hash_domain(DOMAIN_CHUNK, &bad_wcb),
        &[],
        &bad_wcb,
        limits,
    )?;
    push_error(
        &mut vectors,
        &name("structured_non_canonical"),
        "structured",
        &frame,
        &[],
        ErrorClass::NonCanonical,
        None,
    );

    // Limit exceeded: frame decodes 4096 bytes but the harness limit is 1024.
    push_error(
        &mut vectors,
        &name("limits_exceeded"),
        "raw",
        &good,
        &[],
        ErrorClass::LimitExceeded,
        Some(json!({ "max_decoded_bytes": 1024 })),
    );

    // --- AI-mode error vectors --------------------------------------------
    let tensor_payload = vec![1u8, 2, 3, 4, 5, 6];
    let tensor_hash = hash_domain(DOMAIN_TENSOR_PAYLOAD, &tensor_payload);

    // Shape product != element_count.
    let meta = tensor_meta_value(
        "u8",
        &[2, 3],
        5,
        6,
        tensor_hash.as_bytes(),
        "row_major",
        "little",
    );
    let frame = ai_frame(Mode::Tensor, &meta, &tensor_payload, limits)?;
    push_error(
        &mut vectors,
        "120_tensor_shape_mismatch",
        "tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    // payload_bytes disagrees with element_count * element_size.
    let meta = tensor_meta_value(
        "u8",
        &[2, 3],
        6,
        7,
        tensor_hash.as_bytes(),
        "row_major",
        "little",
    );
    let frame = ai_frame(Mode::Tensor, &meta, &tensor_payload, limits)?;
    push_error(
        &mut vectors,
        "121_tensor_payload_bytes_mismatch",
        "tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    // Declared payload hash does not bind the payload.
    let meta = tensor_meta_value("u8", &[2, 3], 6, 6, &[0u8; 32], "row_major", "little");
    let frame = ai_frame(Mode::Tensor, &meta, &tensor_payload, limits)?;
    push_error(
        &mut vectors,
        "122_tensor_payload_hash_mismatch",
        "tensor",
        &frame,
        &[],
        ErrorClass::Corrupt,
        None,
    );

    // Specified-but-not-implemented layout and dtype.
    let meta = tensor_meta_value(
        "u8",
        &[6],
        6,
        6,
        tensor_hash.as_bytes(),
        "column_major",
        "little",
    );
    let frame = ai_frame(Mode::Tensor, &meta, &tensor_payload, limits)?;
    push_error(
        &mut vectors,
        "123_tensor_layout_not_implemented",
        "tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );
    let meta = tensor_meta_value(
        "q4",
        &[1],
        1,
        1,
        tensor_hash.as_bytes(),
        "row_major",
        "little",
    );
    let frame = ai_frame(Mode::Tensor, &meta, &tensor_payload[..1], limits)?;
    push_error(
        &mut vectors,
        "124_tensor_unknown_dtype",
        "tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    // Checked shape arithmetic and resource limits.
    let empty_hash = hash_domain(DOMAIN_TENSOR_PAYLOAD, b"");
    let meta = tensor_meta_value(
        "u8",
        &[u64::MAX, 2],
        0,
        0,
        empty_hash.as_bytes(),
        "row_major",
        "little",
    );
    let frame = ai_frame(Mode::Tensor, &meta, b"", limits)?;
    push_error(
        &mut vectors,
        "125_tensor_shape_overflow",
        "tensor",
        &frame,
        &[],
        ErrorClass::LimitExceeded,
        None,
    );
    let meta = tensor_meta_value(
        "u8",
        &[2, 3],
        6,
        6,
        tensor_hash.as_bytes(),
        "row_major",
        "little",
    );
    let frame = ai_frame(Mode::Tensor, &meta, &tensor_payload, limits)?;
    push_error(
        &mut vectors,
        "126_tensor_element_limit",
        "tensor",
        &frame,
        &[],
        ErrorClass::LimitExceeded,
        Some(json!({ "max_tensor_elements": 5 })),
    );
    let one = vec![0u8; 1];
    let one_hash = hash_domain(DOMAIN_TENSOR_PAYLOAD, &one);
    let meta = tensor_meta_value(
        "u8",
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        1,
        1,
        one_hash.as_bytes(),
        "row_major",
        "little",
    );
    let frame = ai_frame(Mode::Tensor, &meta, &one, limits)?;
    push_error(
        &mut vectors,
        "127_tensor_rank_limit",
        "tensor",
        &frame,
        &[],
        ErrorClass::LimitExceeded,
        Some(json!({ "max_tensor_rank": 8 })),
    );

    // Token count, encoding, canonicality, and hash errors.
    let token_ids = vec![1u64, 2, 3];
    let token_payload = modes::encode_token_payload(&token_ids, limits)?;
    let token_hash = hash_domain(DOMAIN_TOKENS_PAYLOAD, &token_payload);
    let meta = token_meta_value(
        "weft.test.vocab.v1",
        modes::TOKEN_ENCODING,
        2,
        token_payload.len() as u64,
        token_hash.as_bytes(),
    );
    let frame = ai_frame(Mode::TokenSequence, &meta, &token_payload, limits)?;
    push_error(
        &mut vectors,
        "130_tokens_count_mismatch",
        "tokens",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );
    let meta = token_meta_value(
        "weft.test.vocab.v1",
        "u32-fixed-little",
        3,
        token_payload.len() as u64,
        token_hash.as_bytes(),
    );
    let frame = ai_frame(Mode::TokenSequence, &meta, &token_payload, limits)?;
    push_error(
        &mut vectors,
        "131_tokens_encoding_not_implemented",
        "tokens",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );
    let non_minimal = vec![0x81u8, 0x00];
    let nm_hash = hash_domain(DOMAIN_TOKENS_PAYLOAD, &non_minimal);
    let meta = token_meta_value(
        "weft.test.vocab.v1",
        modes::TOKEN_ENCODING,
        1,
        2,
        nm_hash.as_bytes(),
    );
    let frame = ai_frame(Mode::TokenSequence, &meta, &non_minimal, limits)?;
    push_error(
        &mut vectors,
        "132_tokens_non_minimal_varint",
        "tokens",
        &frame,
        &[],
        ErrorClass::NonCanonical,
        None,
    );
    let trailing = vec![0x01u8, 0x02];
    let tr_hash = hash_domain(DOMAIN_TOKENS_PAYLOAD, &trailing);
    let meta = token_meta_value(
        "weft.test.vocab.v1",
        modes::TOKEN_ENCODING,
        1,
        2,
        tr_hash.as_bytes(),
    );
    let frame = ai_frame(Mode::TokenSequence, &meta, &trailing, limits)?;
    push_error(
        &mut vectors,
        "133_tokens_trailing_bytes",
        "tokens",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );
    let meta = token_meta_value(
        "weft.test.vocab.v1",
        modes::TOKEN_ENCODING,
        3,
        token_payload.len() as u64,
        &[0u8; 32],
    );
    let frame = ai_frame(Mode::TokenSequence, &meta, &token_payload, limits)?;
    push_error(
        &mut vectors,
        "134_tokens_payload_hash_mismatch",
        "tokens",
        &frame,
        &[],
        ErrorClass::Corrupt,
        None,
    );
    let meta = token_meta_value(
        "weft.test.vocab.v1",
        modes::TOKEN_ENCODING,
        3,
        token_payload.len() as u64,
        token_hash.as_bytes(),
    );
    let frame = ai_frame(Mode::TokenSequence, &meta, &token_payload, limits)?;
    push_error(
        &mut vectors,
        "135_tokens_count_limit",
        "tokens",
        &frame,
        &[],
        ErrorClass::LimitExceeded,
        Some(json!({ "max_tokens": 2 })),
    );

    // --- Newly allocated AI modes (WEFT-AI-CODEC) -------------------------
    let tensor_f32 = pattern(24, 0x1357_9bdf_2468_ace0);
    let frame = modes::encode_tensor(&tensor_f32, modes::TensorDtype::F32, &[2, 3], limits)?;
    push_ok(&mut vectors, "100_tensor_f32_2x3", &frame, &tensor_f32, &[])?;
    let tensor_scalar = vec![0x00u8, 0x3cu8]; // f16 1.0, little-endian
    let frame = modes::encode_tensor(&tensor_scalar, modes::TensorDtype::F16, &[], limits)?;
    push_ok(
        &mut vectors,
        "101_tensor_f16_scalar",
        &frame,
        &tensor_scalar,
        &[],
    )?;
    let frame = modes::encode_tensor(b"", modes::TensorDtype::U8, &[0, 4], limits)?;
    push_ok(&mut vectors, "102_tensor_u8_zero_element", &frame, b"", &[])?;
    let tokens = vec![0u64, 1, 0, u64::MAX, u64::MAX - 1, 42, 42, 41];
    let frame = modes::encode_token_sequence(&tokens, "weft.test.vocab.v1", limits)?;
    let payload = modes::encode_token_payload(&tokens, limits)?;
    push_ok(
        &mut vectors,
        "103_tokens_delta_boundaries",
        &frame,
        &payload,
        &[],
    )?;
    let frame = modes::encode_token_sequence(&[], "weft.test.vocab.v1", limits)?;
    push_ok(&mut vectors, "104_tokens_empty", &frame, b"", &[])?;

    // --- QUANTIZED_TENSOR (mode 8) -----------------------------------------
    let quant_payload = vec![0x00u8, 0x7f, 0x80, 0xff, 0x01, 0xfe];
    let frame = modes::encode_quantized_tensor(
        &quant_payload,
        modes::TensorDtype::I8,
        &[2, 3],
        0.25,
        -3,
        limits,
    )?;
    push_ok(
        &mut vectors,
        "105_quantized_i8_2x3",
        &frame,
        &quant_payload,
        &[],
    )?;
    let frame =
        modes::encode_quantized_tensor(b"", modes::TensorDtype::U8, &[0, 4], 1.0, 128, limits)?;
    push_ok(
        &mut vectors,
        "106_quantized_u8_zero_element",
        &frame,
        b"",
        &[],
    )?;
    let scalar = vec![0x80u8];
    let frame =
        modes::encode_quantized_tensor(&scalar, modes::TensorDtype::I8, &[], 0.1, 0, limits)?;
    push_ok(
        &mut vectors,
        "107_quantized_i8_scalar",
        &frame,
        &scalar,
        &[],
    )?;

    // --- FLOAT_TENSOR (mode 9) --------------------------------------------
    // bf16 rank 2: +0, -0, smallest denormal, and a NaN with a non-zero
    // payload (all-zero exponent plane, all-ones exponent for the NaN).
    let bf16_payload: Vec<u8> = [0x0000u16, 0x8000, 0x0001, 0x7fc1]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let frame =
        modes::encode_float_tensor(&bf16_payload, modes::FloatDtype::Bf16, &[2, 2], limits)?;
    push_ok(
        &mut vectors,
        "108_float_bf16_2x2",
        &frame,
        &bf16_payload,
        &[],
    )?;
    // fp16 rank 3: +inf, -inf, a denormal, and a NaN with a non-zero payload.
    let fp16_payload: Vec<u8> = [0x7c00u16, 0xfc00, 0x0001, 0x7e01]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let frame =
        modes::encode_float_tensor(&fp16_payload, modes::FloatDtype::F16, &[2, 1, 2], limits)?;
    push_ok(
        &mut vectors,
        "109_float_fp16_3d",
        &frame,
        &fp16_payload,
        &[],
    )?;
    // fp32 rank 4: signed zeros, +inf, and a NaN with a non-zero payload.
    let fp32_payload: Vec<u8> = [0x0000_0000u32, 0x8000_0000, 0x7f80_0000, 0x7f80_0001]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let frame =
        modes::encode_float_tensor(&fp32_payload, modes::FloatDtype::F32, &[2, 1, 1, 2], limits)?;
    push_ok(
        &mut vectors,
        "110_float_fp32_4d",
        &frame,
        &fp32_payload,
        &[],
    )?;

    let q_hash = hash_domain(DOMAIN_QUANTIZED_TENSOR_PAYLOAD, &quant_payload);
    let scale_bits = 0.25f64.to_bits().to_le_bytes();
    let q_layout = "row_major";
    let q_quant = modes::QUANTIZED_TENSOR_QUANTIZATION;

    let meta = quantized_meta_value(
        "f32",
        &[2, 3],
        6,
        6,
        q_hash.as_bytes(),
        q_layout,
        q_quant,
        &scale_bits,
        -3,
    );
    let frame = ai_frame(Mode::QuantizedTensor, &meta, &quant_payload, limits)?;
    push_error(
        &mut vectors,
        &name("quantized_dtype_f32"),
        "quantized_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let nan_bits = f64::NAN.to_bits().to_le_bytes();
    let meta = quantized_meta_value(
        "i8",
        &[2, 3],
        6,
        6,
        q_hash.as_bytes(),
        q_layout,
        q_quant,
        &nan_bits,
        -3,
    );
    let frame = ai_frame(Mode::QuantizedTensor, &meta, &quant_payload, limits)?;
    push_error(
        &mut vectors,
        &name("quantized_scale_nan"),
        "quantized_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = quantized_meta_value(
        "i8",
        &[2, 3],
        6,
        6,
        q_hash.as_bytes(),
        q_layout,
        q_quant,
        &scale_bits,
        200,
    );
    let frame = ai_frame(Mode::QuantizedTensor, &meta, &quant_payload, limits)?;
    push_error(
        &mut vectors,
        &name("quantized_zero_point_out_of_range"),
        "quantized_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = quantized_meta_value(
        "i8",
        &[2, 3],
        6,
        7,
        q_hash.as_bytes(),
        q_layout,
        q_quant,
        &scale_bits,
        -3,
    );
    let frame = ai_frame(Mode::QuantizedTensor, &meta, &quant_payload, limits)?;
    push_error(
        &mut vectors,
        &name("quantized_payload_bytes_mismatch"),
        "quantized_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = quantized_meta_value(
        "i8",
        &[2, 3],
        6,
        6,
        &[0u8; 32],
        q_layout,
        q_quant,
        &scale_bits,
        -3,
    );
    let frame = ai_frame(Mode::QuantizedTensor, &meta, &quant_payload, limits)?;
    push_error(
        &mut vectors,
        &name("quantized_payload_hash_mismatch"),
        "quantized_tensor",
        &frame,
        &[],
        ErrorClass::Corrupt,
        None,
    );

    let meta = quantized_meta_value(
        "i8",
        &[2, 3],
        6,
        6,
        q_hash.as_bytes(),
        q_layout,
        q_quant,
        &scale_bits[..4],
        -3,
    );
    let frame = ai_frame(Mode::QuantizedTensor, &meta, &quant_payload, limits)?;
    push_error(
        &mut vectors,
        &name("quantized_scale_bytes_len"),
        "quantized_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = quantized_meta_value(
        "i8",
        &[2, 3],
        6,
        6,
        q_hash.as_bytes(),
        "column_major",
        q_quant,
        &scale_bits,
        -3,
    );
    let frame = ai_frame(Mode::QuantizedTensor, &meta, &quant_payload, limits)?;
    push_error(
        &mut vectors,
        &name("quantized_layout_not_implemented"),
        "quantized_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let mut wrong_zp = quantized_meta_value(
        "i8",
        &[2, 3],
        6,
        6,
        q_hash.as_bytes(),
        q_layout,
        q_quant,
        &scale_bits,
        -3,
    );
    if let Value::Map(map) = &mut wrong_zp {
        map.insert("zero_point".to_owned(), Value::U64(0));
    }
    let frame = ai_frame(Mode::QuantizedTensor, &wrong_zp, &quant_payload, limits)?;
    push_error(
        &mut vectors,
        &name("quantized_zero_point_not_i64"),
        "quantized_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    // --- FLOAT_TENSOR rejects (mode 9) ------------------------------------
    let fp32_two: Vec<u8> = [0x3f80_0000u32, 0x7f80_0001]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let (exp_plane, mant_plane) =
        modes::split_float_tensor_payload(&fp32_two, modes::FloatDtype::F32, 2)?;
    let mut float_stored = exp_plane.clone();
    float_stored.extend_from_slice(&mant_plane);
    let float_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &float_stored);
    let exp_len = exp_plane.len() as u64;
    let mant_len = mant_plane.len() as u64;

    let meta = float_tensor_meta_value(
        "fp64",
        &[2],
        2,
        1,
        8,
        23,
        exp_len,
        mant_len,
        float_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &float_stored, 8, limits)?;
    push_error(
        &mut vectors,
        "140_float_tensor_unknown_dtype",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = float_tensor_meta_value(
        "fp32",
        &[2],
        3,
        1,
        8,
        23,
        exp_len,
        mant_len,
        float_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &float_stored, 8, limits)?;
    push_error(
        &mut vectors,
        "141_float_tensor_element_count_mismatch",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = float_tensor_meta_value(
        "fp32",
        &[],
        1,
        1,
        8,
        23,
        exp_len,
        mant_len,
        float_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &float_stored, 8, limits)?;
    push_error(
        &mut vectors,
        "142_float_tensor_rank_zero",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = float_tensor_meta_value(
        "fp32",
        &[2, 0],
        0,
        1,
        8,
        23,
        exp_len,
        mant_len,
        float_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &float_stored, 8, limits)?;
    push_error(
        &mut vectors,
        "143_float_tensor_zero_dimension",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = float_tensor_meta_value(
        "fp16",
        &[2],
        2,
        1,
        8,
        7,
        exp_len,
        mant_len,
        float_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &float_stored, 8, limits)?;
    push_error(
        &mut vectors,
        "144_float_tensor_field_widths_mismatch",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = float_tensor_meta_value("fp8-e4m3", &[2], 2, 1, 4, 3, u64::MAX, 1, &[0u8; 32]);
    let frame = float_ai_frame(&meta, b"", 2, limits)?;
    push_error(
        &mut vectors,
        "145_float_tensor_plane_length_overflow",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::LimitExceeded,
        None,
    );

    let mut garbage = vec![0xffu8; 3];
    garbage.extend_from_slice(&mant_plane);
    let garbage_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &garbage);
    let meta = float_tensor_meta_value(
        "fp32",
        &[2],
        2,
        1,
        8,
        23,
        3,
        mant_len,
        garbage_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &garbage, 8, limits)?;
    push_error(
        &mut vectors,
        "152_float_tensor_exponent_table_truncated",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Corrupt,
        None,
    );

    let mut meta = float_tensor_meta_value(
        "fp32",
        &[2],
        2,
        1,
        8,
        23,
        exp_len,
        mant_len,
        float_hash.as_bytes(),
    );
    if let Value::Map(map) = &mut meta {
        map.insert("exponent_coder".to_owned(), Value::Str("zstd".to_owned()));
    }
    let frame = float_ai_frame(&meta, &float_stored, 8, limits)?;
    push_error(
        &mut vectors,
        "146_float_tensor_unknown_exponent_coder",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Malformed,
        None,
    );

    let meta = float_tensor_meta_value(
        "fp32",
        &[2],
        2,
        1,
        8,
        23,
        exp_len,
        mant_len,
        float_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &float_stored, 7, limits)?;
    push_error(
        &mut vectors,
        "147_float_tensor_decoded_len_mismatch",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Corrupt,
        None,
    );

    // Huffman table and bitstream rejects. Count is 1 for the table-only
    // vectors; fp8-e4m3 has a one-byte mantissa plane.
    let one_mantissa = [0u8];
    let over = {
        let mut table = vec![1u8; 256];
        table.extend_from_slice(&one_mantissa);
        table
    };
    let over_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &over);
    let meta = float_tensor_meta_value("fp8-e4m3", &[1], 1, 1, 4, 3, 256, 1, over_hash.as_bytes());
    let frame = float_ai_frame(&meta, &over, 1, limits)?;
    push_error(
        &mut vectors,
        "148_float_tensor_exponent_over_subscribed",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::NonCanonical,
        None,
    );

    let mut incomplete = vec![0u8; 256];
    incomplete[0] = 2;
    incomplete[1] = 2;
    incomplete.extend_from_slice(&one_mantissa);
    let incomplete_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &incomplete);
    let meta = float_tensor_meta_value(
        "fp8-e4m3",
        &[1],
        1,
        1,
        4,
        3,
        256,
        1,
        incomplete_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &incomplete, 1, limits)?;
    push_error(
        &mut vectors,
        "149_float_tensor_exponent_incomplete",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::NonCanonical,
        None,
    );

    let mut too_long = vec![0u8; 256];
    too_long[0] = 25;
    too_long.extend_from_slice(&one_mantissa);
    let too_long_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &too_long);
    let meta = float_tensor_meta_value(
        "fp8-e4m3",
        &[1],
        1,
        1,
        4,
        3,
        256,
        1,
        too_long_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &too_long, 1, limits)?;
    push_error(
        &mut vectors,
        "150_float_tensor_exponent_code_length_too_long",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::NonCanonical,
        None,
    );

    let mut padding = vec![0u8; 256];
    padding[0] = 1;
    padding[1] = 1;
    padding.push(0b0100_0000);
    padding.extend_from_slice(&one_mantissa);
    let padding_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &padding);
    let meta = float_tensor_meta_value(
        "fp8-e4m3",
        &[1],
        1,
        1,
        4,
        3,
        257,
        1,
        padding_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &padding, 1, limits)?;
    push_error(
        &mut vectors,
        "151_float_tensor_exponent_nonzero_padding",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Corrupt,
        None,
    );

    let mut truncated = vec![0u8; 256];
    truncated[0] = 1;
    truncated[1] = 1;
    truncated.extend_from_slice(&one_mantissa);
    let truncated_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &truncated);
    let meta = float_tensor_meta_value(
        "fp8-e4m3",
        &[2],
        2,
        1,
        4,
        3,
        256,
        1,
        truncated_hash.as_bytes(),
    );
    let frame = float_ai_frame(&meta, &truncated, 2, limits)?;
    push_error(
        &mut vectors,
        "153_float_tensor_exponent_stream_truncated",
        "float_tensor",
        &frame,
        &[],
        ErrorClass::Corrupt,
        None,
    );

    for vector in &vectors {
        let file_name = format!("{}.json", vector["name"].as_str().unwrap_or("vector"));
        write_json(&dir.join(file_name), vector)?;
    }
    Ok(())
}

fn write_structured_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    let values: Vec<(&str, Value)> = vec![
        ("null", Value::Null),
        ("true", Value::Bool(true)),
        ("false", Value::Bool(false)),
        ("u64_zero", Value::U64(0)),
        ("u64_max", Value::U64(u64::MAX)),
        ("i64_min", Value::I64(i64::MIN)),
        ("i64_neg_one", Value::I64(-1)),
        ("timestamp", Value::Timestamp(1_700_000_000_123_456_789)),
        ("text", Value::Str("h\u{e9}llo \u{1f600}".to_owned())),
        ("bytes", Value::Bytes(vec![0, 1, 2, 0xff])),
        (
            "array",
            Value::Array(vec![Value::Null, Value::U64(1), Value::Bool(false)]),
        ),
        (
            "map",
            Value::map([
                ("a", Value::U64(1)),
                ("b", Value::Str("two".to_owned())),
                ("c", Value::Null),
            ]),
        ),
    ];
    let mut index = 0usize;
    for (name, value) in values {
        index += 1;
        let canonical = wcb::encode(&value, limits)?;
        let vector = json!({
            "name": name,
            "description": format!("canonical WCB value: {value:?}"),
            "canonical_hex": hexs(&canonical),
            "expect": "ok",
        });
        write_json(&dir.join(format!("{index:03}_{name}.json")), &vector)?;
    }

    let rejects: Vec<(&str, Vec<u8>, ErrorClass)> = vec![
        (
            "non_minimal_varint",
            vec![0x03, 0x80, 0x00],
            ErrorClass::NonCanonical,
        ),
        (
            "unsorted_map_keys",
            vec![
                0x08, 0x02, 0x05, 0x01, b'b', 0x03, 0x01, 0x05, 0x01, b'a', 0x03, 0x02,
            ],
            ErrorClass::NonCanonical,
        ),
        (
            "duplicate_map_keys",
            vec![
                0x08, 0x02, 0x05, 0x01, b'a', 0x03, 0x01, 0x05, 0x01, b'a', 0x03, 0x02,
            ],
            ErrorClass::NonCanonical,
        ),
        (
            "non_text_map_key",
            vec![0x08, 0x01, 0x03, 0x01, 0x03, 0x02],
            ErrorClass::NonCanonical,
        ),
        ("trailing_bytes", vec![0x00, 0x00], ErrorClass::NonCanonical),
        ("unknown_tag", vec![0x0a], ErrorClass::NonCanonical),
        (
            "truncated_text",
            vec![0x05, 0x05, b'a'],
            ErrorClass::Malformed,
        ),
    ];
    for (name, bytes, class) in rejects {
        index += 1;
        let vector = json!({
            "name": name,
            "hex": hexs(&bytes),
            "expect_error": class.as_str(),
        });
        write_json(&dir.join(format!("{index:03}_{name}.json")), &vector)?;
    }

    // Depth limit rejection uses an explicit limit.
    let mut deep = Vec::new();
    for _ in 0..65 {
        deep.push(0x07);
        deep.push(0x01);
    }
    deep.push(0x00);
    index += 1;
    let vector = json!({
        "name": "depth_limit",
        "hex": hexs(&deep),
        "limits": { "max_structured_depth": 64 },
        "expect_error": ErrorClass::LimitExceeded.as_str(),
    });
    write_json(&dir.join(format!("{index:03}_depth_limit.json")), &vector)?;
    Ok(())
}

fn write_selector_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    let sources = [
        "select true",
        "select false",
        "select not false",
        "select not not true",
        "select has(attrs.project)",
        "select tag(\"demo\")",
        "select relation(derived_from, obj:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)",
        "select attrs.project == \"demo\"",
        "select attrs.size != 10",
        "select attrs.size < -1",
        "select attrs.size <= 10 and attrs.size >= 0",
        "select attrs.a == 1 and (attrs.b == 2 and attrs.c == 3)",
        "select attrs.a == 1 or attrs.b == 2 or attrs.c == 3",
        "select not has(attrs.x)",
        "select attrs.x == 1 order by attrs.seq desc limit 10",
        "select attrs.x == 1 order by attrs.seq asc, attrs.y offset 5",
        "select attrs.x == 1 offset 0",
        "select attrs.x == 1 and attrs.y == 2",
        "select attrs.y == 2 and attrs.x == 1",
        "select attrs.e == \"\\u00e9\\uD83D\\uDE00\\n\\t\\\"\\\\\\/\"",
        "select attrs.p == null",
        "select attrs.b == true and attrs.n == false",
    ];
    for (index, source) in sources.iter().enumerate() {
        let selector = selector::parse_with(source, selector::DEFAULT_NAMESPACE, limits)?;
        let canonical = selector.canonical_bytes_with_limits(limits)?;
        let id = hash_domain(DOMAIN_SELECTOR, &canonical);
        let vector = json!({
            "name": format!("{}_{}", index + 1, slug(source)),
            "source": source,
            "namespace": selector::DEFAULT_NAMESPACE,
            "canonical_hex": hexs(&canonical),
            "selector_id": id.to_hex(),
        });
        write_json(
            &dir.join(format!("{:03}_{}.json", index + 1, slug(source))),
            &vector,
        )?;
    }
    Ok(())
}

fn slug(source: &str) -> String {
    let mut out = String::new();
    for ch in source.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
        if out.len() >= 40 {
            break;
        }
    }
    out.trim_matches('_').to_owned()
}

fn hash_hex(byte: u8) -> String {
    format!("{:02x}", byte).repeat(32)
}

fn context_map(root: &ContextRoot) -> BTreeMap<String, Value> {
    match root.to_wcb_value() {
        Value::Map(map) => map,
        _ => unreachable!("context root encodes as a map"),
    }
}

fn write_context_error(
    dir: &Path,
    name: &str,
    record: Value,
    expected: &str,
    limits: &Limits,
) -> Result<()> {
    let bytes = wcb::encode(&record, limits)?;
    let error = ContextRoot::from_canonical_bytes(&bytes, limits)
        .expect_err("rejection vector unexpectedly decoded");
    if error.class().as_str() != expected {
        return Err(Error::io(format!(
            "{name}: expected class {expected}, generated {} ({error})",
            error.class()
        )));
    }
    write_json(
        &dir.join(format!("{name}.json")),
        &json!({
            "name": name,
            "record_hex": hexs(&bytes),
            "expect_error": expected,
        }),
    )
}

fn write_context_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    struct Case {
        name: &'static str,
        source: &'static str,
        namespace: &'static str,
        snapshot: u8,
        resolver: &'static str,
        causal: &'static [u8],
        capability: Option<u8>,
        budget: Option<&'static str>,
    }

    let cases = [
        Case {
            name: "001_root_true",
            source: "select true",
            namespace: "default",
            snapshot: 0x11,
            resolver: DEFAULT_RESOLVER,
            causal: &[],
            capability: None,
            budget: None,
        },
        Case {
            name: "002_root_filter_budget",
            source: "select attrs.project == \"demo\" limit 10",
            namespace: "default",
            snapshot: 0x22,
            resolver: DEFAULT_RESOLVER,
            causal: &[],
            capability: None,
            budget: Some("max_results=10,max_bytes=1048576"),
        },
        Case {
            name: "003_root_causal_capability",
            source: "select attrs.project == \"demo\" and tag(\"reviewed\")",
            namespace: "proj",
            snapshot: 0x33,
            resolver: DEFAULT_RESOLVER,
            causal: &[0xBB, 0xAA],
            capability: Some(0x44),
            budget: Some("max_results=8,max_depth=2"),
        },
        Case {
            name: "004_root_descriptor_escaping",
            source: "select attrs.e == \"\\u00e9\\uD83D\\uDE00\\n\\t\\\"\\\\\\/\"",
            namespace: "agents.prod-1",
            snapshot: 0x55,
            resolver: DEFAULT_RESOLVER,
            causal: &[0x11],
            capability: None,
            budget: Some("max_depth=3"),
        },
        Case {
            name: "005_root_custom_resolver",
            source: "select relation(derived_from, obj:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)",
            namespace: "default",
            snapshot: 0x66,
            resolver: "weft-resolver-experimental-1",
            causal: &[0xAA, 0xBB, 0xCC],
            capability: None,
            budget: None,
        },
        Case {
            name: "006_root_empty_causal",
            source: "select false",
            namespace: "_internal",
            snapshot: 0x77,
            resolver: DEFAULT_RESOLVER,
            causal: &[],
            capability: None,
            budget: Some("max_work=1"),
        },
    ];

    for case in cases {
        let causal: Vec<Hash32> = case
            .causal
            .iter()
            .map(|byte| Hash32::from_hex(&hash_hex(*byte)).unwrap())
            .collect();
        let capability_hash = case
            .capability
            .map(|byte| Hash32::from_hex(&hash_hex(byte)).unwrap());
        let budget = match case.budget {
            Some(spec) => Some(Budget::parse_spec(spec)?),
            None => None,
        };
        let root = context::build_context_root(
            case.source,
            case.namespace,
            &hash_hex(case.snapshot),
            case.resolver,
            &causal,
            capability_hash,
            budget,
        )?;
        let record = root.canonical_bytes_with_limits(limits)?;
        let decoded = ContextRoot::from_canonical_bytes(&record, limits)?;
        if decoded != root {
            return Err(Error::io(format!(
                "{}: context root did not round-trip",
                case.name
            )));
        }
        let id = root.id()?;
        let uri = root.authoritative_uri()?;
        let descriptor = root.descriptor_uri(case.source)?;
        let parsed = context::parse_context_uri(&descriptor, limits)?;
        if parsed.canonical_uri(limits)? != uri {
            return Err(Error::io(format!(
                "{}: descriptor did not canonicalize to the authoritative URI",
                case.name
            )));
        }
        let vector = json!({
            "name": case.name,
            "schema": context::CONTEXT_ROOT_SCHEMA,
            "namespace": root.namespace,
            "selector_source": case.source,
            "selector_canonical_hex": hexs(&root.selector_bytes),
            "selector_id": root.selector_id.to_hex(),
            "snapshot": root.snapshot.to_hex(),
            "resolver": root.resolver,
            "causal": root.causal.iter().map(|h| h.to_hex()).collect::<Vec<_>>(),
            "capability_hash": root.capability_hash.map(|h| h.to_hex()),
            "budget": root.budget.as_ref().map(|b| b.to_json()),
            "ordering": root.ordering,
            "canonical_hex": hexs(&record),
            "context_root_id": id.to_hex(),
            "uri": uri,
            "descriptor": descriptor,
        });
        write_json(&dir.join(format!("{}.json", case.name)), &vector)?;
    }

    // Rejection vectors. Each one is a valid WCB map with exactly one broken
    // context-root invariant.
    let base = context::build_context_root(
        "select true",
        "default",
        &hash_hex(0x11),
        DEFAULT_RESOLVER,
        &[],
        None,
        None,
    )?;
    let a = Hash32::from_hex(&hash_hex(0xAA)).unwrap();
    let b = Hash32::from_hex(&hash_hex(0xBB)).unwrap();

    let mut map = context_map(&base);
    map.insert(
        "causal".to_owned(),
        Value::Array(vec![bytes_value(b), bytes_value(a)]),
    );
    write_context_error(
        dir,
        "101_unsorted_causal",
        Value::Map(map),
        "NonCanonical",
        limits,
    )?;

    let mut map = context_map(&base);
    map.insert(
        "causal".to_owned(),
        Value::Array(vec![bytes_value(a), bytes_value(a)]),
    );
    write_context_error(
        dir,
        "102_duplicate_causal",
        Value::Map(map),
        "NonCanonical",
        limits,
    )?;

    let mut map = context_map(&base);
    map.insert("snapshot".to_owned(), bytes_value(Hash32::ZERO));
    write_context_error(
        dir,
        "103_zero_snapshot",
        Value::Map(map),
        "NonCanonical",
        limits,
    )?;

    let mut map = context_map(&base);
    map.insert("selector_id".to_owned(), bytes_value(b));
    write_context_error(
        dir,
        "104_selector_id_mismatch",
        Value::Map(map),
        "NonCanonical",
        limits,
    )?;

    let mut map = context_map(&base);
    map.insert("namespace".to_owned(), Value::Str("other".to_owned()));
    write_context_error(
        dir,
        "105_namespace_mismatch",
        Value::Map(map),
        "NonCanonical",
        limits,
    )?;

    let mut map = context_map(&base);
    map.insert(
        "namespace".to_owned(),
        Value::Str("bad namespace".to_owned()),
    );
    write_context_error(
        dir,
        "106_bad_namespace",
        Value::Map(map),
        "NonCanonical",
        limits,
    )?;

    let mut map = context_map(&base);
    map.insert(
        "schema".to_owned(),
        Value::Str("weft.context-root.v2".to_owned()),
    );
    write_context_error(
        dir,
        "107_unknown_schema",
        Value::Map(map),
        "UnknownSchema",
        limits,
    )?;

    let mut map = context_map(&base);
    map.insert(
        "budget".to_owned(),
        Value::map([("max_results", Value::U64(0))]),
    );
    write_context_error(
        dir,
        "108_bad_budget_zero",
        Value::Map(map),
        "NonCanonical",
        limits,
    )?;

    let mut map = context_map(&base);
    map.insert(
        "ordering".to_owned(),
        Value::Str("object_id_asc".to_owned()),
    );
    write_context_error(
        dir,
        "109_unknown_ordering",
        Value::Map(map),
        "UnsupportedVersion",
        limits,
    )?;

    let mut map = context_map(&base);
    map.insert("extra".to_owned(), Value::Bool(true));
    write_context_error(
        dir,
        "110_unknown_key",
        Value::Map(map),
        "NonCanonical",
        limits,
    )?;

    Ok(())
}

fn bytes_value(hash: Hash32) -> Value {
    Value::Bytes(hash.as_bytes().to_vec())
}

fn write_causal_error(
    dir: &Path,
    name: &str,
    record: Value,
    expected: &str,
    limits: &Limits,
) -> Result<()> {
    let bytes = wcb::encode(&record, limits)?;
    let error = CausalClosure::from_canonical_bytes(&bytes, limits)
        .expect_err("causal rejection vector unexpectedly decoded");
    if error.class().as_str() != expected {
        return Err(Error::io(format!(
            "{name}: expected class {expected}, generated {} ({error})",
            error.class()
        )));
    }
    write_json(
        &dir.join(format!("{name}.json")),
        &json!({
            "name": name,
            "record_hex": hexs(&bytes),
            "expect_error": expected,
        }),
    )
}

fn write_causal_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    let a = Hash32::from_hex(&hash_hex(0xAA)).unwrap();
    let b = Hash32::from_hex(&hash_hex(0xBB)).unwrap();
    let c = Hash32::from_hex(&hash_hex(0xCC)).unwrap();

    let cases: [(&str, Vec<Hash32>); 4] = [
        ("001_single_parent", vec![a]),
        ("002_two_parents_unsorted_input", vec![b, a]),
        ("003_three_parents", vec![c, a, b]),
        ("004_empty_closure", vec![]),
    ];
    for (name, parents) in cases {
        let closure = CausalClosure::from_parents(&parents)?;
        let record = closure.canonical_bytes_with_limits(limits)?;
        let decoded = CausalClosure::from_canonical_bytes(&record, limits)?;
        if decoded != closure {
            return Err(Error::io(format!(
                "{name}: causal closure did not round-trip"
            )));
        }
        let vector = json!({
            "name": name,
            "schema": context::CAUSAL_SCHEMA,
            "parents_input": parents.iter().map(|h| h.to_hex()).collect::<Vec<_>>(),
            "parents": closure.parents.iter().map(|h| h.to_hex()).collect::<Vec<_>>(),
            "closure": closure.closure.iter().map(|h| h.to_hex()).collect::<Vec<_>>(),
            "canonical_hex": hexs(&record),
            "closure_hash": closure.id()?.to_hex(),
        });
        write_json(&dir.join(format!("{name}.json")), &vector)?;
    }

    let unsorted = Value::map([
        ("schema", Value::Str(context::CAUSAL_SCHEMA.to_owned())),
        (
            "parents",
            Value::Array(vec![bytes_value(b), bytes_value(a)]),
        ),
        (
            "closure",
            Value::Array(vec![bytes_value(b), bytes_value(a)]),
        ),
    ]);
    write_causal_error(dir, "101_unsorted", unsorted, "NonCanonical", limits)?;

    let duplicate = Value::map([
        ("schema", Value::Str(context::CAUSAL_SCHEMA.to_owned())),
        (
            "parents",
            Value::Array(vec![bytes_value(a), bytes_value(a)]),
        ),
        (
            "closure",
            Value::Array(vec![bytes_value(a), bytes_value(a)]),
        ),
    ]);
    write_causal_error(dir, "102_duplicate", duplicate, "NonCanonical", limits)?;

    let missing = Value::map([
        ("schema", Value::Str(context::CAUSAL_SCHEMA.to_owned())),
        ("parents", Value::Array(vec![bytes_value(a)])),
        ("closure", Value::Array(vec![bytes_value(b)])),
    ]);
    write_causal_error(
        dir,
        "103_closure_missing_parent",
        missing,
        "NonCanonical",
        limits,
    )?;

    let zero = Value::map([
        ("schema", Value::Str(context::CAUSAL_SCHEMA.to_owned())),
        ("parents", Value::Array(vec![bytes_value(Hash32::ZERO)])),
        ("closure", Value::Array(vec![bytes_value(Hash32::ZERO)])),
    ]);
    write_causal_error(dir, "104_zero_entry", zero, "NonCanonical", limits)?;

    let schema = Value::map([
        ("schema", Value::Str("weft.causal.v2".to_owned())),
        ("parents", Value::Array(vec![])),
        ("closure", Value::Array(vec![])),
    ]);
    write_causal_error(dir, "105_unknown_schema", schema, "UnknownSchema", limits)?;

    let extra = Value::map([
        ("schema", Value::Str(context::CAUSAL_SCHEMA.to_owned())),
        ("parents", Value::Array(vec![])),
        ("closure", Value::Array(vec![])),
        ("extra", Value::Bool(true)),
    ]);
    write_causal_error(dir, "106_unknown_key", extra, "NonCanonical", limits)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// WEFT-TENSOR vectors
// ---------------------------------------------------------------------------

fn write_tensor_logical_error(
    dir: &Path,
    name: &str,
    description: &str,
    record: Value,
    expected: &str,
    limits: &Limits,
) -> Result<()> {
    let bytes = wcb::encode(&record, limits)?;
    let error = LogicalRecord::from_canonical_bytes(&bytes, limits)
        .expect_err("rejection vector unexpectedly decoded");
    if error.class().as_str() != expected {
        return Err(Error::io(format!(
            "{name}: expected class {expected}, generated {} ({error})",
            error.class()
        )));
    }
    write_json(
        &dir.join(format!("{name}.json")),
        &json!({
            "name": name,
            "kind": "logical",
            "description": description,
            "record_hex": hexs(&bytes),
            "expect_error": expected,
        }),
    )
}

fn write_tensor_layout_error(
    dir: &Path,
    name: &str,
    description: &str,
    record: Value,
    expected: &str,
    limits: &Limits,
) -> Result<()> {
    let bytes = wcb::encode(&record, limits)?;
    let error = LayoutDescriptor::from_canonical_bytes(&bytes, limits)
        .expect_err("rejection vector unexpectedly decoded");
    if error.class().as_str() != expected {
        return Err(Error::io(format!(
            "{name}: expected class {expected}, generated {} ({error})",
            error.class()
        )));
    }
    write_json(
        &dir.join(format!("{name}.json")),
        &json!({
            "name": name,
            "kind": "layout",
            "description": description,
            "record_hex": hexs(&bytes),
            "expect_error": expected,
        }),
    )
}

fn shape_value(shape: &[u64]) -> Value {
    Value::Array(shape.iter().map(|dim| Value::U64(*dim)).collect())
}

fn logical_map(dtype: &str, shape: &[u64], element_count: u64, values_hash: &[u8]) -> Value {
    Value::map([
        ("schema", Value::Str(tensor::LOGICAL_SCHEMA.to_owned())),
        ("dtype", Value::Str(dtype.to_owned())),
        ("shape", shape_value(shape)),
        ("element_count", Value::U64(element_count)),
        ("values_hash", Value::Bytes(values_hash.to_vec())),
    ])
}

fn layout_map_value(dtype: &str, shape: &[u64], layout: &str, extras: &[(&str, Value)]) -> Value {
    let mut pairs: Vec<(String, Value)> = vec![
        (
            "schema".to_owned(),
            Value::Str(tensor::LAYOUT_SCHEMA.to_owned()),
        ),
        ("dtype".to_owned(), Value::Str(dtype.to_owned())),
        ("shape".to_owned(), shape_value(shape)),
        ("layout".to_owned(), Value::Str(layout.to_owned())),
    ];
    for (key, value) in extras {
        pairs.push(((*key).to_owned(), value.clone()));
    }
    Value::map(pairs)
}

fn write_tensor_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    // Valid logical records and their layout-independent roots.
    let valid: [(&str, modes::TensorDtype, &[u64], u64); 3] = [
        (
            "001_logical_bf16_2x3x4",
            modes::TensorDtype::Bf16,
            &[2, 3, 4],
            0x1001,
        ),
        ("002_logical_f32_3", modes::TensorDtype::F32, &[3], 0x1002),
        (
            "003_logical_i8_2x2",
            modes::TensorDtype::I8,
            &[2, 2],
            0x1003,
        ),
    ];
    for (name, dtype, shape, seed) in valid {
        let count = tensor::validate_shape(shape, limits)?;
        let len = usize::try_from(tensor::payload_len(count, dtype)?)
            .map_err(|_| Error::limit_exceeded("vector payload does not fit usize"))?;
        let values = pattern(len, seed);
        let record = LogicalRecord::from_row_major(dtype, shape, &values, limits)?;
        let canonical = record.canonical_bytes_with_limits(limits)?;
        let root = record.root_with_limits(limits)?;
        let vector = json!({
            "name": name,
            "kind": "logical",
            "dtype": dtype.as_str(),
            "shape": shape,
            "element_count": count,
            "values_hex": hexs(&values),
            "values_hash": record.values_hash.to_hex(),
            "canonical_hex": hexs(&canonical),
            "logical_root": root.to_hex(),
        });
        write_json(&dir.join(format!("{name}.json")), &vector)?;
    }

    // Logical rejects.
    let good_hash = tensor::values_hash(b"weft-tensor-logical-vector");
    write_tensor_logical_error(
        dir,
        "101_logical_rank_zero",
        "rank 0 is not a v1 logical tensor",
        logical_map("bf16", &[], 1, good_hash.as_bytes()),
        "NonCanonical",
        limits,
    )?;
    write_tensor_logical_error(
        dir,
        "102_logical_zero_dimension",
        "a zero dimension is rejected",
        logical_map("bf16", &[2, 0, 3], 0, good_hash.as_bytes()),
        "NonCanonical",
        limits,
    )?;
    write_tensor_logical_error(
        dir,
        "103_logical_product_overflow",
        "the shape product overflows u64",
        logical_map(
            "bf16",
            &[4_294_967_296, 4_294_967_296, 4_294_967_296],
            1,
            good_hash.as_bytes(),
        ),
        "LimitExceeded",
        limits,
    )?;
    write_tensor_logical_error(
        dir,
        "104_logical_element_count_mismatch",
        "element_count must equal the shape product",
        logical_map("bf16", &[2, 3, 4], 23, good_hash.as_bytes()),
        "NonCanonical",
        limits,
    )?;
    write_tensor_logical_error(
        dir,
        "105_logical_unknown_schema",
        "schema must be weft.tensor.logical.v1",
        Value::map([
            ("schema", Value::Str("weft.tensor.logical.v2".to_owned())),
            ("dtype", Value::Str("bf16".to_owned())),
            ("shape", shape_value(&[2, 3, 4])),
            ("element_count", Value::U64(24)),
            ("values_hash", Value::Bytes(good_hash.as_bytes().to_vec())),
        ]),
        "UnknownSchema",
        limits,
    )?;
    write_tensor_logical_error(
        dir,
        "106_logical_unknown_dtype",
        "unknown dtype is UnsupportedFeature in the identity layer",
        logical_map("f8", &[2, 3, 4], 24, good_hash.as_bytes()),
        "UnsupportedFeature",
        limits,
    )?;
    write_tensor_logical_error(
        dir,
        "107_logical_short_values_hash",
        "values_hash must be 32 bytes",
        logical_map("bf16", &[2, 3, 4], 24, &good_hash.as_bytes()[..31]),
        "NonCanonical",
        limits,
    )?;
    write_tensor_logical_error(
        dir,
        "108_logical_unknown_key",
        "unknown keys are rejected",
        Value::map([
            ("schema", Value::Str(tensor::LOGICAL_SCHEMA.to_owned())),
            ("dtype", Value::Str("bf16".to_owned())),
            ("shape", shape_value(&[2, 3, 4])),
            ("element_count", Value::U64(24)),
            ("values_hash", Value::Bytes(good_hash.as_bytes().to_vec())),
            ("layout", Value::Str("row-major".to_owned())),
        ]),
        "NonCanonical",
        limits,
    )?;
    let nine = [1u64; 9];
    write_tensor_logical_error(
        dir,
        "109_logical_rank_exceeds_limit",
        "rank 9 exceeds max_tensor_rank 8",
        logical_map("i8", &nine, 1, good_hash.as_bytes()),
        "LimitExceeded",
        limits,
    )?;
    write_tensor_logical_error(
        dir,
        "110_logical_count_exceeds_limit",
        "100000 x 100000 exceeds max_tensor_elements",
        logical_map("i8", &[100_000, 100_000], 1, good_hash.as_bytes()),
        "LimitExceeded",
        limits,
    )?;

    // Valid layout descriptors.
    let layouts = [
        (
            "201_layout_row_major",
            LayoutDescriptor::row_major(modes::TensorDtype::Bf16, &[2, 3, 4]),
        ),
        (
            "202_layout_column_major",
            LayoutDescriptor::column_major(modes::TensorDtype::Bf16, &[2, 3, 4]),
        ),
        (
            "203_layout_blocked",
            LayoutDescriptor::blocked(modes::TensorDtype::Bf16, &[4, 3, 2], &[2, 2, 2]),
        ),
        (
            "204_layout_sharded",
            LayoutDescriptor::sharded(modes::TensorDtype::Bf16, &[4, 3, 2], 0, 4, 2),
        ),
    ];
    for (name, descriptor) in layouts {
        let canonical = descriptor.canonical_bytes_with_limits(limits)?;
        let mut vector = json!({
            "name": name,
            "kind": "layout",
            "dtype": descriptor.dtype.as_str(),
            "shape": descriptor.shape,
            "layout": descriptor.layout.name(),
            "canonical_hex": hexs(&canonical),
        });
        if let TensorLayout::Blocked { block_shape } = &descriptor.layout {
            vector["block_shape"] = json!(block_shape);
        }
        if let TensorLayout::Sharded {
            axis,
            shards,
            index,
        } = &descriptor.layout
        {
            vector["axis"] = json!(axis);
            vector["shards"] = json!(shards);
            vector["index"] = json!(index);
        }
        write_json(&dir.join(format!("{name}.json")), &vector)?;
    }

    // Layout rejects.
    write_tensor_layout_error(
        dir,
        "301_layout_unknown_layout",
        "unknown layout names are UnsupportedFeature",
        layout_map_value("bf16", &[4, 3, 2], "tiled", &[]),
        "UnsupportedFeature",
        limits,
    )?;
    write_tensor_layout_error(
        dir,
        "302_sharded_not_divisible",
        "shape[axis] must be divisible by shards",
        layout_map_value(
            "bf16",
            &[2, 3, 4],
            "sharded",
            &[
                ("axis", Value::U64(0)),
                ("shards", Value::U64(3)),
                ("index", Value::U64(0)),
            ],
        ),
        "NonCanonical",
        limits,
    )?;
    write_tensor_layout_error(
        dir,
        "303_shard_index_out_of_range",
        "index must be below shards",
        layout_map_value(
            "bf16",
            &[4, 3, 2],
            "sharded",
            &[
                ("axis", Value::U64(0)),
                ("shards", Value::U64(4)),
                ("index", Value::U64(4)),
            ],
        ),
        "NonCanonical",
        limits,
    )?;
    write_tensor_layout_error(
        dir,
        "304_sharded_zero_shards",
        "shards must be at least 1",
        layout_map_value(
            "bf16",
            &[4, 3, 2],
            "sharded",
            &[
                ("axis", Value::U64(0)),
                ("shards", Value::U64(0)),
                ("index", Value::U64(0)),
            ],
        ),
        "NonCanonical",
        limits,
    )?;
    write_tensor_layout_error(
        dir,
        "305_blocked_rank_mismatch",
        "block_shape rank must equal shape rank",
        layout_map_value(
            "bf16",
            &[4, 3, 2],
            "blocked",
            &[("block_shape", shape_value(&[2, 2]))],
        ),
        "NonCanonical",
        limits,
    )?;
    write_tensor_layout_error(
        dir,
        "306_blocked_zero_block",
        "block_shape dimensions must be >= 1",
        layout_map_value(
            "bf16",
            &[4, 3, 2],
            "blocked",
            &[("block_shape", shape_value(&[2, 2, 0]))],
        ),
        "NonCanonical",
        limits,
    )?;
    write_tensor_layout_error(
        dir,
        "307_column_major_rank_zero",
        "column-major cannot express a rank-0 shape",
        layout_map_value("bf16", &[], "column-major", &[]),
        "NonCanonical",
        limits,
    )?;
    write_tensor_layout_error(
        dir,
        "308_layout_extra_field",
        "row-major must not carry a shard field",
        layout_map_value("bf16", &[4, 3, 2], "row-major", &[("axis", Value::U64(0))]),
        "NonCanonical",
        limits,
    )?;
    write_tensor_layout_error(
        dir,
        "309_shard_axis_out_of_range",
        "axis must be below rank",
        layout_map_value(
            "bf16",
            &[4, 3, 2],
            "sharded",
            &[
                ("axis", Value::U64(3)),
                ("shards", Value::U64(1)),
                ("index", Value::U64(0)),
            ],
        ),
        "NonCanonical",
        limits,
    )?;
    write_tensor_layout_error(
        dir,
        "310_layout_shape_overflow",
        "layout shape product overflows u64",
        layout_map_value(
            "bf16",
            &[4_294_967_296, 4_294_967_296, 4_294_967_296],
            "row-major",
            &[],
        ),
        "LimitExceeded",
        limits,
    )?;

    // Shard reassembly vectors, including byte-identical containers.
    let shard_cases: [(&str, modes::TensorDtype, &[u64], u32, u32, u64); 2] = [
        (
            "401_shards_bf16_4x3x2_axis0",
            modes::TensorDtype::Bf16,
            &[4, 3, 2],
            0,
            4,
            0x2001,
        ),
        (
            "402_shards_f32_4x3x2_axis1",
            modes::TensorDtype::F32,
            &[4, 3, 2],
            1,
            3,
            0x2002,
        ),
    ];
    for (name, dtype, shape, axis, shard_count, seed) in shard_cases {
        let count = tensor::validate_shape(shape, limits)?;
        let len = usize::try_from(tensor::payload_len(count, dtype)?)
            .map_err(|_| Error::limit_exceeded("vector payload does not fit usize"))?;
        let values = pattern(len, seed);
        let shards = tensor::split_shards(dtype, shape, axis, shard_count, &values, limits)?;
        let mut shard_hex = Vec::with_capacity(shards.len());
        let mut descriptor_hex = Vec::with_capacity(shards.len());
        let mut container_hex = Vec::with_capacity(shards.len());
        let mut decoded = Vec::with_capacity(shards.len());
        for (index, payload) in shards.iter().enumerate() {
            let descriptor =
                LayoutDescriptor::sharded(dtype, shape, axis, shard_count, index as u32);
            let container = tensor::encode_shard(&descriptor, payload, limits)?;
            let parsed = tensor::decode_shard(&container, limits)?;
            shard_hex.push(hexs(payload));
            descriptor_hex.push(hexs(&descriptor.canonical_bytes_with_limits(limits)?));
            container_hex.push(hexs(&container));
            decoded.push(parsed);
        }
        let root = tensor::shard_root(&decoded, limits)?;
        let expected = tensor::logical_root(dtype, shape, &values, limits)?;
        if root != expected {
            return Err(Error::io(format!(
                "{name}: shard root differs from logical root"
            )));
        }
        let vector = json!({
            "name": name,
            "kind": "shards",
            "dtype": dtype.as_str(),
            "shape": shape,
            "axis": axis,
            "shards": shard_count,
            "element_count": count,
            "values_hex": hexs(&values),
            "values_hash": tensor::values_hash(&values).to_hex(),
            "logical_root": root.to_hex(),
            "shard_hex": shard_hex,
            "shard_descriptor_hex": descriptor_hex,
            "shard_container_hex": container_hex,
        });
        write_json(&dir.join(format!("{name}.json")), &vector)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// TENSOR_SHARDED (mode 10) vectors
// ---------------------------------------------------------------------------

fn sharded_frame_with_value(
    meta_value: Value,
    planes: &[u8],
    stored: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(&meta_value, limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("TENSOR_SHARDED metadata does not fit u32"))?;
    let mut body = Vec::with_capacity(4 + meta_bytes.len() + stored.len());
    body.extend_from_slice(&meta_len.to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(stored);
    build_frame(
        Mode::TensorSharded,
        DOMAIN_PAYLOAD,
        0,
        planes.len() as u64,
        hash_domain(DOMAIN_CHUNK, planes),
        &[],
        &body,
        limits,
    )
}

fn sharded_frame(
    meta: &tensor_sharded::TensorShardedMeta,
    planes: &[u8],
    stored: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    let payload_hash = hash_domain(DOMAIN_TENSOR_SHARDED_PAYLOAD, stored);
    sharded_frame_with_value(
        tensor_sharded::descriptor_value(meta, &payload_hash),
        planes,
        stored,
        limits,
    )
}

fn sharded_words_hex(words: &[i8]) -> String {
    let bytes: Vec<u8> = words.iter().map(|word| *word as u8).collect();
    hexs(&bytes)
}

fn sharded_pattern_q8(count: usize) -> Vec<i8> {
    (0..count)
        .map(|i| (((i * 7 + 3) % 251) as i32 - 125) as i8)
        .collect()
}

fn sharded_pattern_q4(count: usize) -> Vec<i8> {
    (0..count).map(|i| (i % 16) as i8 - 8).collect()
}

fn write_sharded_valid(
    dir: &Path,
    name: &str,
    input: &tensor_sharded::ShardedTensorInput,
    limits: &Limits,
) -> Result<()> {
    let frames = tensor_sharded::encode_sharded_tensor(input, limits)?;
    let root = tensor_sharded::logical_root(
        input.dtype,
        &input.shape,
        &input.words,
        &input.scales,
        &input.zero_points,
        limits,
    )?;
    let mut frame_json = Vec::with_capacity(frames.len());
    for frame in frames.iter() {
        let parsed = Frame::parse(frame, limits)?;
        let shard = tensor_sharded::decode_shard_frame(&parsed, limits)?;
        frame_json.push(json!({
            "index": shard.meta.shard_index,
            "frame_hex": hexs(frame),
        }));
    }
    // Deliberately shuffled: the descriptor index is authoritative.
    frame_json.reverse();
    let vector = json!({
        "name": name,
        "schema": tensor_sharded::TENSOR_SHARDED_SCHEMA,
        "dtype": input.dtype.as_str(),
        "shape": input.shape,
        "element_count": input.words.len(),
        "axis": input.shard_axis,
        "shards": input.shard_count,
        "block_size": input.block_size,
        "scales": input.scales,
        "zero_points": input.zero_points,
        "words_hex": sharded_words_hex(&input.words),
        "layout_root": root.to_hex(),
        "frames": frame_json,
        "expect": { "ok": true },
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_sharded_reject(
    dir: &Path,
    name: &str,
    frames: &[Vec<u8>],
    expected: ErrorClass,
    note: &str,
) -> Result<()> {
    let mut frame_json = Vec::with_capacity(frames.len());
    for frame in frames {
        frame_json.push(json!({
            "frame_hex": hexs(frame),
        }));
    }
    let vector = json!({
        "name": name,
        "expect_error": expected.as_str(),
        "note": note,
        "frames": frame_json,
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_sharded_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    let q8 = tensor_sharded::ShardedTensorInput {
        dtype: tensor_sharded::ShardedDtype::Q8_0,
        shape: vec![4, 8],
        shard_axis: 0,
        shard_count: 2,
        block_size: 16,
        scales: vec![0.5f32.to_bits(), 0.25f32.to_bits()],
        zero_points: vec![0, 3],
        words: sharded_pattern_q8(32),
    };
    write_sharded_valid(dir, "001_q8_0_rank2_2shards", &q8, limits)?;

    let q4 = tensor_sharded::ShardedTensorInput {
        dtype: tensor_sharded::ShardedDtype::Q4_0,
        shape: vec![2, 4, 8],
        shard_axis: 1,
        shard_count: 4,
        block_size: 16,
        scales: vec![
            0.5f32.to_bits(),
            0.25f32.to_bits(),
            0.125f32.to_bits(),
            0.0625f32.to_bits(),
        ],
        zero_points: vec![0, 1, -1, 2],
        words: sharded_pattern_q4(64),
    };
    write_sharded_valid(dir, "002_q4_0_rank3_4shards", &q4, limits)?;

    let single = tensor_sharded::ShardedTensorInput {
        dtype: tensor_sharded::ShardedDtype::Q8_0,
        shape: vec![16],
        shard_axis: 0,
        shard_count: 1,
        block_size: 16,
        scales: vec![1.0f32.to_bits()],
        zero_points: vec![0],
        words: sharded_pattern_q8(16),
    };
    write_sharded_valid(dir, "003_q8_0_rank1_1shard", &single, limits)?;

    // Rejection frames are derived from the first q8_0 shard.
    let frames = tensor_sharded::encode_sharded_tensor(&q8, limits)?;
    let parsed = Frame::parse(&frames[0], limits)?;
    let shard = tensor_sharded::decode_shard_frame(&parsed, limits)?;
    let planes = shard.planes.clone();
    let stored = crate::huffman::encode(&planes);
    let mut base = shard.meta.clone();
    base.payload_len = stored.len() as u64;

    let unknown_dtype_frame = {
        let payload_hash = hash_domain(DOMAIN_TENSOR_SHARDED_PAYLOAD, &stored);
        let value = tensor_sharded::descriptor_value(&base, &payload_hash);
        let overridden = match value {
            Value::Map(mut map) => {
                map.insert("dtype".to_owned(), Value::Str("q5_1".to_owned()));
                Value::Map(map)
            }
            other => other,
        };
        sharded_frame_with_value(overridden, &planes, &stored, limits)?
    };
    write_sharded_reject(
        dir,
        "101_unknown_dtype",
        &[unknown_dtype_frame],
        ErrorClass::Malformed,
        "dtype q5_1 has no v1 wire value",
    )?;

    let mut mismatched_planes = planes.clone();
    let last = mismatched_planes.len() - 1;
    mismatched_planes[last] ^= 0x01;
    let mismatched_stored = crate::huffman::encode(&mismatched_planes);
    let mut mismatched_meta = base.clone();
    mismatched_meta.payload_len = mismatched_stored.len() as u64;
    let scale_mismatch = sharded_frame(
        &mismatched_meta,
        &mismatched_planes,
        &mismatched_stored,
        limits,
    )?;
    write_sharded_reject(
        dir,
        "102_quant_scale_mismatch",
        &[scale_mismatch],
        ErrorClass::Corrupt,
        "scale plane bit differs from the descriptor scales array",
    )?;

    let mut element_meta = base.clone();
    element_meta.element_count = 31;
    element_meta.payload_len = stored.len() as u64;
    let element_frame = sharded_frame(&element_meta, &planes, &stored, limits)?;
    write_sharded_reject(
        dir,
        "103_element_count_mismatch",
        &[element_frame],
        ErrorClass::Malformed,
        "element_count is not the shape product",
    )?;

    let mut block_meta = base.clone();
    block_meta.block_size = 24;
    block_meta.payload_len = stored.len() as u64;
    let block_frame = sharded_frame(&block_meta, &planes, &stored, limits)?;
    write_sharded_reject(
        dir,
        "104_block_size_invalid",
        &[block_frame],
        ErrorClass::Malformed,
        "block_size 24 is not in {16,32,64,128,256}",
    )?;

    let mut axis_meta = base.clone();
    axis_meta.shard_axis = 1;
    axis_meta.shard_count = 3;
    axis_meta.payload_len = stored.len() as u64;
    let axis_frame = sharded_frame(&axis_meta, &planes, &stored, limits)?;
    write_sharded_reject(
        dir,
        "105_shape_not_divisible",
        &[axis_frame],
        ErrorClass::Malformed,
        "shape[1] = 8 is not divisible by shard_count 3",
    )?;

    let mut index_meta = base.clone();
    index_meta.shard_index = 2;
    index_meta.payload_len = stored.len() as u64;
    let index_frame = sharded_frame(&index_meta, &planes, &stored, limits)?;
    write_sharded_reject(
        dir,
        "106_index_out_of_range",
        &[index_frame],
        ErrorClass::Malformed,
        "shard_index 2 is not below shard_count 2",
    )?;

    let second_parsed = Frame::parse(&frames[1], limits)?;
    let second = tensor_sharded::decode_shard_frame(&second_parsed, limits)?;
    let second_stored = crate::huffman::encode(&second.planes);
    let mut duplicate_meta = second.meta.clone();
    duplicate_meta.shard_index = 0;
    duplicate_meta.payload_len = second_stored.len() as u64;
    let duplicate_frame = sharded_frame(&duplicate_meta, &second.planes, &second_stored, limits)?;
    write_sharded_reject(
        dir,
        "107_duplicate_shard",
        &[frames[0].clone(), duplicate_frame],
        ErrorClass::NonCanonical,
        "two shards declare index 0",
    )?;

    let mut truncated = stored.clone();
    truncated.truncate(truncated.len() - 1);
    let mut huffman_meta = base.clone();
    huffman_meta.payload_len = truncated.len() as u64;
    let huffman_frame = sharded_frame(&huffman_meta, &planes, &truncated, limits)?;
    write_sharded_reject(
        dir,
        "108_malformed_huffman",
        &[huffman_frame],
        ErrorClass::Corrupt,
        "truncated huffman-canonical-v1 bitstream",
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Mode 11 WEIGHT_DELTA
// ---------------------------------------------------------------------------

fn weight_delta_bf16_case(limits: &Limits) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let base_words: [u16; 6] = [0x3F80, 0xC000, 0x0000, 0x7FC0, 0x8000, 0x0001];
    let mut target_words = base_words;
    // Exactly one word changes; both byte planes carry a non-zero symbol.
    target_words[2] = 0x3F80;
    let mut base_bytes = Vec::with_capacity(12);
    let mut target_bytes = Vec::with_capacity(12);
    for word in base_words {
        base_bytes.extend_from_slice(&word.to_le_bytes());
    }
    for word in target_words {
        target_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let base_frame = modes::encode_tensor(&base_bytes, modes::TensorDtype::Bf16, &[2, 3], limits)?;
    Ok((target_bytes, base_bytes, base_frame))
}

fn weight_delta_fp32_zero_case(limits: &Limits) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let words: [u32; 4] = [0x7FC0_0001, 0x8000_0000, 0x0000_0001, 0x7F80_0000];
    let mut bytes = Vec::with_capacity(16);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let base_frame = modes::encode_tensor(&bytes, modes::TensorDtype::F32, &[1, 2, 2], limits)?;
    Ok((bytes.clone(), bytes, base_frame))
}

fn weight_delta_q8_case(limits: &Limits) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let base: [u8; 8] = [0x80, 0xFF, 0x00, 0x01, 0x7F, 0x2A, 0xF9, 0x64];
    let mut target = base;
    // Mixed signed changes, applied to the raw word representation.
    target[0] = base[0].wrapping_add(1);
    target[1] = base[1].wrapping_sub(1);
    target[3] = base[3].wrapping_add(2);
    target[4] = base[4].wrapping_sub(1);
    target[6] = base[6].wrapping_add(1);
    target[7] = base[7].wrapping_sub(2);
    let base_frame = modes::encode_tensor(&base, modes::TensorDtype::I8, &[2, 4], limits)?;
    Ok((target.to_vec(), base.to_vec(), base_frame))
}

fn write_weight_delta_valid(
    dir: &Path,
    name: &str,
    dtype: WeightDtype,
    shape: &[u64],
    base_frame: &[u8],
    target: &[u8],
    limits: &Limits,
) -> Result<()> {
    let delta_frame = weight_delta::encode_weight_delta(target, dtype, shape, base_frame, limits)?;
    let parsed = Frame::parse(&delta_frame, limits)?;
    let meta = weight_delta::weight_delta_metadata(&parsed, limits)?;
    let decoded = weight_delta::decode_with_base_frame(
        &parsed,
        base_frame,
        &crate::dependency::MapResolver::new(),
        limits,
    )?;
    if decoded != target {
        return Err(Error::corrupt("WEIGHT_DELTA vector does not reconstruct"));
    }
    let vector = json!({
        "name": name,
        "schema": weight_delta::WEIGHT_DELTA_SCHEMA,
        "dtype": meta.dtype.as_str(),
        "shape": meta.shape,
        "element_count": meta.element_count,
        "word_bits": meta.word_bits,
        "base": meta.base.to_hex(),
        "base_dtype": meta.base_dtype.as_str(),
        "base_shape": meta.base_shape,
        "op": meta.op,
        "planes": meta.planes,
        "plane_codec": meta.plane_codec,
        "delta_root": meta.delta_root.to_hex(),
        "payload_len": meta.payload_len,
        "target_hash": blake3_hex(target),
        "target_hex": hexs(target),
        "base_frame_hex": hexs(base_frame),
        "delta_frame_hex": hexs(&delta_frame),
        "expect": { "ok": true },
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_weight_delta_reject(
    dir: &Path,
    name: &str,
    delta_frame: &[u8],
    base_frame: Option<&[u8]>,
    expected: ErrorClass,
    note: &str,
    use_frame_source: bool,
) -> Result<()> {
    let vector = json!({
        "name": name,
        "expect_error": expected.as_str(),
        "note": note,
        "base_frame_hex": base_frame.map(hexs),
        "delta_frame_hex": hexs(delta_frame),
        "use_frame_source": use_frame_source,
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn weight_delta_frame(
    meta: &WeightDeltaMeta,
    payload: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    base_mode: u32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    weight_delta::encode_descriptor_frame(
        meta,
        payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )
}

fn write_weight_delta_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    // --- valid ------------------------------------------------------------
    let (bf16_target, _bf16_base, bf16_base_frame) = weight_delta_bf16_case(limits)?;
    write_weight_delta_valid(
        dir,
        "001_bf16_small_changes",
        WeightDtype::Bf16,
        &[2, 3],
        &bf16_base_frame,
        &bf16_target,
        limits,
    )?;

    let (fp32_bytes, _fp32_base, fp32_base_frame) = weight_delta_fp32_zero_case(limits)?;
    write_weight_delta_valid(
        dir,
        "002_fp32_zero_delta",
        WeightDtype::Fp32,
        &[1, 2, 2],
        &fp32_base_frame,
        &fp32_bytes,
        limits,
    )?;

    let (q8_target, _q8_base, q8_base_frame) = weight_delta_q8_case(limits)?;
    write_weight_delta_valid(
        dir,
        "003_q8_0_mixed_changes",
        WeightDtype::Q8_0,
        &[2, 4],
        &q8_base_frame,
        &q8_target,
        limits,
    )?;

    // --- rejects derived from the bf16 case -------------------------------
    let delta_frame = weight_delta::encode_weight_delta(
        &bf16_target,
        WeightDtype::Bf16,
        &[2, 3],
        &bf16_base_frame,
        limits,
    )?;
    let parsed = Frame::parse(&delta_frame, limits)?;
    let meta = weight_delta::weight_delta_metadata(&parsed, limits)?;
    let payload = {
        let body = &parsed.body;
        let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
        body[4 + meta_len..].to_vec()
    };
    let content_hash = parsed.header.content_hash;
    let decoded_len = parsed.header.decoded_len;
    let base_mode = 6u32;

    let mut unknown_op = meta.clone();
    unknown_op.op = "xor".to_owned();
    let frame = weight_delta_frame(
        &unknown_op,
        &payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )?;
    write_weight_delta_reject(
        dir,
        "101_base_dtype_mismatch",
        &{
            let fp32_base =
                modes::encode_tensor(&vec![0u8; 24], modes::TensorDtype::F32, &[2, 3], limits)?;
            let fp32_parsed = Frame::parse(&fp32_base, limits)?;
            let mut mismatch = meta.clone();
            mismatch.base = fp32_parsed.header.content_hash;
            weight_delta_frame(&mismatch, &payload, decoded_len, content_hash, 6, limits)?
        },
        Some(&modes::encode_tensor(
            &vec![0u8; 24],
            modes::TensorDtype::F32,
            &[2, 3],
            limits,
        )?),
        ErrorClass::DependencyMismatch,
        "base frame dtype f32 does not match the declared bf16 dtype",
        false,
    )?;
    write_weight_delta_reject(
        dir,
        "102_base_shape_mismatch",
        &{
            let bf16_base =
                modes::encode_tensor(&vec![0u8; 12], modes::TensorDtype::Bf16, &[3, 2], limits)?;
            let bf16_parsed = Frame::parse(&bf16_base, limits)?;
            let mut mismatch = meta.clone();
            mismatch.base = bf16_parsed.header.content_hash;
            weight_delta_frame(&mismatch, &payload, decoded_len, content_hash, 6, limits)?
        },
        Some(&modes::encode_tensor(
            &vec![0u8; 12],
            modes::TensorDtype::Bf16,
            &[3, 2],
            limits,
        )?),
        ErrorClass::DependencyMismatch,
        "base frame shape [3,2] does not match the declared shape [2,3]",
        false,
    )?;
    write_weight_delta_reject(
        dir,
        "103_unknown_op",
        &frame,
        Some(&bf16_base_frame),
        ErrorClass::Malformed,
        "op xor has no v1 implementation",
        false,
    )?;

    let mut word_bits = meta.clone();
    word_bits.word_bits = 8;
    let frame = weight_delta_frame(
        &word_bits,
        &payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )?;
    write_weight_delta_reject(
        dir,
        "104_word_bits_inconsistent",
        &frame,
        Some(&bf16_base_frame),
        ErrorClass::Malformed,
        "word_bits 8 disagrees with bf16",
        false,
    )?;

    let mut delta_root = meta.clone();
    let mut flipped = delta_root.delta_root.into_bytes();
    flipped[0] ^= 0x01;
    delta_root.delta_root = Hash32::from_bytes(flipped);
    let frame = weight_delta_frame(
        &delta_root,
        &payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )?;
    write_weight_delta_reject(
        dir,
        "105_delta_root_mismatch",
        &frame,
        Some(&bf16_base_frame),
        ErrorClass::Corrupt,
        "declared delta_root does not match the reconstructed delta words",
        false,
    )?;

    let mut payload_len = meta.clone();
    payload_len.payload_len = payload.len() as u64 + 1;
    let frame = weight_delta_frame(
        &payload_len,
        &payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )?;
    write_weight_delta_reject(
        dir,
        "106_payload_length_mismatch",
        &frame,
        Some(&bf16_base_frame),
        ErrorClass::Malformed,
        "declared payload_len is one byte larger than the stored payload",
        false,
    )?;

    write_weight_delta_reject(
        dir,
        "107_missing_base",
        &delta_frame,
        None,
        ErrorClass::DependencyMissing,
        "no resolver entry and no --base frame was supplied",
        false,
    )?;

    // A self-referential base: the descriptor's base equals the frame's own
    // content hash, so the DAG resolver re-enters the root frame.
    let zero_target = vec![0u8; 12];
    let cycle_hash = hash_domain(DOMAIN_CHUNK, &zero_target);
    let zero_payload = {
        let planes: [Vec<u8>; 2] = [vec![0u8; 6], vec![0u8; 6]];
        let mut out = Vec::new();
        for plane in &planes {
            out.extend_from_slice(&crate::huffman::encode(plane));
        }
        out
    };
    let mut cycle_meta = meta.clone();
    cycle_meta.base = cycle_hash;
    cycle_meta.delta_root = hash_domain(tensor::DOMAIN_TENSOR_VALUES, &vec![0u8; 12]);
    cycle_meta.payload_len = zero_payload.len() as u64;
    let cycle_frame = weight_delta_frame(
        &cycle_meta,
        &zero_payload,
        zero_target.len() as u64,
        cycle_hash,
        base_mode,
        limits,
    )?;
    write_weight_delta_reject(
        dir,
        "108_dependency_cycle",
        &cycle_frame,
        None,
        ErrorClass::DependencyCycle,
        "the descriptor base equals the frame's own content hash",
        true,
    )?;

    Ok(())
}

fn blake3_hex(bytes: &[u8]) -> String {
    hash_domain(DOMAIN_CHUNK, bytes).to_hex()
}

// ---------------------------------------------------------------------------
// Mode 16 SPARSE_DELTA
// ---------------------------------------------------------------------------

/// Build a `zero-run-v1` payload from explicit records.
fn sparse_payload(run_count: u32, records: &[(u64, [u8; 4])], storage: usize) -> Vec<u8> {
    let mut out = run_count.to_le_bytes().to_vec();
    for (zero_run, word) in records {
        varint::encode(*zero_run, &mut out);
        out.extend_from_slice(&word[..storage]);
    }
    out
}

fn sparse_delta_frame(
    meta: &SparseDeltaMeta,
    payload: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    base_mode: u32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    sparse_delta::encode_descriptor_frame(
        meta,
        payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )
}

fn sparse_parts(frame_bytes: &[u8], limits: &Limits) -> Result<(SparseDeltaMeta, Vec<u8>)> {
    let parsed = Frame::parse(frame_bytes, limits)?;
    let meta = sparse_delta::sparse_delta_metadata(&parsed, limits)?;
    let body = parsed.body;
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    Ok((meta, body[4 + meta_len..].to_vec()))
}

fn write_sparse_delta_valid(
    dir: &Path,
    name: &str,
    dtype: WeightDtype,
    shape: &[u64],
    base_frame: &[u8],
    target: &[u8],
    limits: &Limits,
) -> Result<()> {
    let frame = sparse_delta::encode_sparse_delta(target, dtype, shape, base_frame, limits)?;
    let (meta, payload) = sparse_parts(&frame, limits)?;
    let parsed = Frame::parse(&frame, limits)?;
    let decoded = sparse_delta::decode_with_base_frame(
        &parsed,
        base_frame,
        &crate::dependency::MapResolver::new(),
        limits,
    )?;
    if decoded != target {
        return Err(Error::corrupt("SPARSE_DELTA vector does not reconstruct"));
    }
    if payload.len() < 4 {
        return Err(Error::corrupt("SPARSE_DELTA payload has no run_count"));
    }
    let run_count = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let vector = json!({
        "name": name,
        "schema": sparse_delta::SPARSE_DELTA_SCHEMA,
        "dtype": meta.dtype.as_str(),
        "shape": meta.shape,
        "element_count": meta.element_count,
        "storage_bytes": meta.storage_bytes,
        "op": meta.op,
        "sparse_codec": meta.sparse_codec,
        "base": meta.base.to_hex(),
        "delta_root": meta.delta_root.to_hex(),
        "run_count": run_count,
        "payload_len": payload.len(),
        "target_hash": blake3_hex(target),
        "target_hex": hexs(target),
        "base_frame_hex": hexs(base_frame),
        "delta_frame_hex": hexs(&frame),
        "expect": { "ok": true },
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_sparse_delta_reject(
    dir: &Path,
    name: &str,
    delta_frame: &[u8],
    base_frame: Option<&[u8]>,
    expected: ErrorClass,
    note: &str,
    use_frame_source: bool,
    limits_override: Option<Json>,
) -> Result<()> {
    let mut vector = json!({
        "name": name,
        "expect_error": expected.as_str(),
        "note": note,
        "base_frame_hex": base_frame.map(hexs),
        "delta_frame_hex": hexs(delta_frame),
        "use_frame_source": use_frame_source,
    });
    if let Some(limits) = limits_override {
        vector["limits"] = limits;
    }
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_sparse_delta_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    // --- valid ------------------------------------------------------------
    // 001: an all-zero delta. run_count = 0 reconstructs the base exactly.
    let bf16_words: [u16; 6] = [0x3F80, 0xC000, 0x0000, 0x7FC0, 0x8000, 0x0001];
    let mut bf16_bytes = Vec::with_capacity(12);
    for word in bf16_words {
        bf16_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let bf16_base = modes::encode_tensor(&bf16_bytes, modes::TensorDtype::Bf16, &[2, 3], limits)?;
    write_sparse_delta_valid(
        dir,
        "001_bf16_all_zero_delta",
        WeightDtype::Bf16,
        &[2, 3],
        &bf16_base,
        &bf16_bytes,
        limits,
    )?;

    // 002: exactly one non-zero fp32 word.
    let fp32_words: [u32; 4] = [0x7FC0_0001, 0x8000_0000, 0x0000_0001, 0x7F80_0000];
    let mut fp32_base_bytes = Vec::with_capacity(16);
    for word in fp32_words {
        fp32_base_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut fp32_target_bytes = fp32_base_bytes.clone();
    fp32_target_bytes[8..12].copy_from_slice(&0x3F80_0000u32.to_le_bytes());
    let fp32_base =
        modes::encode_tensor(&fp32_base_bytes, modes::TensorDtype::F32, &[2, 2], limits)?;
    write_sparse_delta_valid(
        dir,
        "002_fp32_one_nonzero_word",
        WeightDtype::Fp32,
        &[2, 2],
        &fp32_base,
        &fp32_target_bytes,
        limits,
    )?;

    // 003: several runs, with the last record emitting the final element.
    let q8_base_bytes: [u8; 8] = [0x80, 0xFF, 0x00, 0x01, 0x7F, 0x2A, 0xF9, 0x64];
    let mut q8_target_bytes = q8_base_bytes;
    q8_target_bytes[0] = 0x81; // index 0
    q8_target_bytes[3] = 0x03; // index 3
    q8_target_bytes[7] = 0x62; // index 7 = element_count - 1
    let q8_base = modes::encode_tensor(&q8_base_bytes, modes::TensorDtype::I8, &[2, 4], limits)?;
    write_sparse_delta_valid(
        dir,
        "003_q8_0_several_runs_to_the_end",
        WeightDtype::Q8_0,
        &[2, 4],
        &q8_base,
        &q8_target_bytes,
        limits,
    )?;

    // 004: a fully dense fp16 delta (run_count == element_count).
    let fp16_words: [u16; 5] = [0x3C00, 0xBC00, 0x0000, 0x8000, 0x7E00];
    let mut fp16_base_bytes = Vec::with_capacity(10);
    for word in fp16_words {
        fp16_base_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut fp16_target_bytes = fp16_base_bytes.clone();
    for (index, word) in fp16_words.iter().enumerate() {
        let next = word.wrapping_add(1 + index as u16);
        fp16_target_bytes[index * 2..index * 2 + 2].copy_from_slice(&next.to_le_bytes());
    }
    let fp16_base = modes::encode_tensor(&fp16_base_bytes, modes::TensorDtype::F16, &[5], limits)?;
    write_sparse_delta_valid(
        dir,
        "004_fp16_dense_every_word",
        WeightDtype::Fp16,
        &[5],
        &fp16_base,
        &fp16_target_bytes,
        limits,
    )?;

    // --- rejects derived from the bf16 case -------------------------------
    let mut bf16_target = bf16_bytes.clone();
    bf16_target[4..6].copy_from_slice(&0x3F80u16.to_le_bytes());
    bf16_target[10..12].copy_from_slice(&0x7F81u16.to_le_bytes());
    let delta_frame = sparse_delta::encode_sparse_delta(
        &bf16_target,
        WeightDtype::Bf16,
        &[2, 3],
        &bf16_base,
        limits,
    )?;
    let (meta, payload) = sparse_parts(&delta_frame, limits)?;
    let parsed = Frame::parse(&delta_frame, limits)?;
    let content_hash = parsed.header.content_hash;
    let decoded_len = parsed.header.decoded_len;
    let base_mode = 6u32;

    let mut unknown_op = meta.clone();
    unknown_op.op = "xor".to_owned();
    write_sparse_delta_reject(
        dir,
        "101_unknown_op",
        &sparse_delta_frame(
            &unknown_op,
            &payload,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&bf16_base),
        ErrorClass::Malformed,
        "op xor has no v1 implementation",
        false,
        None,
    )?;

    let mut unknown_codec = meta.clone();
    unknown_codec.sparse_codec = "rle-v2".to_owned();
    write_sparse_delta_reject(
        dir,
        "102_unknown_sparse_codec",
        &sparse_delta_frame(
            &unknown_codec,
            &payload,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&bf16_base),
        ErrorClass::Malformed,
        "sparse_codec rle-v2 has no v1 implementation",
        false,
        None,
    )?;

    let mut storage_bytes = meta.clone();
    storage_bytes.storage_bytes = 1;
    write_sparse_delta_reject(
        dir,
        "103_storage_bytes_inconsistent",
        &sparse_delta_frame(
            &storage_bytes,
            &payload,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&bf16_base),
        ErrorClass::Malformed,
        "storage_bytes 1 disagrees with bf16 (2 bytes)",
        false,
        None,
    )?;

    let mut delta_root = meta.clone();
    let mut flipped = delta_root.delta_root.into_bytes();
    flipped[0] ^= 0x01;
    delta_root.delta_root = Hash32::from_bytes(flipped);
    write_sparse_delta_reject(
        dir,
        "104_delta_root_mismatch",
        &sparse_delta_frame(
            &delta_root,
            &payload,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&bf16_base),
        ErrorClass::Corrupt,
        "declared delta_root does not match the decoded run stream",
        false,
        None,
    )?;

    let fp32_base_frame =
        modes::encode_tensor(&vec![0u8; 24], modes::TensorDtype::F32, &[2, 3], limits)?;
    let fp32_parsed = Frame::parse(&fp32_base_frame, limits)?;
    let mut dtype_mismatch = meta.clone();
    dtype_mismatch.base = fp32_parsed.header.content_hash;
    write_sparse_delta_reject(
        dir,
        "105_base_dtype_mismatch",
        &sparse_delta_frame(
            &dtype_mismatch,
            &payload,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&fp32_base_frame),
        ErrorClass::DependencyMismatch,
        "base frame dtype f32 does not match the declared bf16 dtype",
        false,
        None,
    )?;

    let wrong_shape_base =
        modes::encode_tensor(&vec![0u8; 12], modes::TensorDtype::Bf16, &[3, 2], limits)?;
    let wrong_shape_parsed = Frame::parse(&wrong_shape_base, limits)?;
    let mut shape_mismatch = meta.clone();
    shape_mismatch.base = wrong_shape_parsed.header.content_hash;
    write_sparse_delta_reject(
        dir,
        "106_base_shape_mismatch",
        &sparse_delta_frame(
            &shape_mismatch,
            &payload,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&wrong_shape_base),
        ErrorClass::DependencyMismatch,
        "base frame shape [3,2] does not match the declared shape [2,3]",
        false,
        None,
    )?;

    write_sparse_delta_reject(
        dir,
        "107_missing_base",
        &delta_frame,
        None,
        ErrorClass::DependencyMissing,
        "no resolver entry and no --base frame was supplied",
        false,
        None,
    )?;

    // run_count larger than element_count: no record stream can be consistent.
    let over_count = sparse_payload(7, &[], 2);
    write_sparse_delta_reject(
        dir,
        "108_run_count_exceeds_element_count",
        &sparse_delta_frame(
            &meta,
            &over_count,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&bf16_base),
        ErrorClass::Malformed,
        "run_count 7 exceeds element_count 6",
        false,
        None,
    )?;

    // A run that would place its word past element_count.
    let past_end = sparse_payload(1, &[(6, [0x01, 0x00, 0x00, 0x00])], 2);
    write_sparse_delta_reject(
        dir,
        "109_run_past_element_count",
        &sparse_delta_frame(
            &meta,
            &past_end,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&bf16_base),
        ErrorClass::Malformed,
        "the first run skips 6 of 6 words, so its non-zero word has no slot",
        false,
        None,
    )?;

    // A record whose stored word is truncated (1 of 2 bytes).
    let truncated = {
        let mut out = 1u32.to_le_bytes().to_vec();
        varint::encode(0, &mut out);
        out.push(0x01);
        out
    };
    write_sparse_delta_reject(
        dir,
        "110_run_record_truncated",
        &sparse_delta_frame(
            &meta,
            &truncated,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&bf16_base),
        ErrorClass::Malformed,
        "a bf16 word needs 2 bytes; only 1 remains in the payload",
        false,
        None,
    )?;

    // Trailing bytes after the declared records.
    let mut trailing = payload.clone();
    trailing.push(0x00);
    write_sparse_delta_reject(
        dir,
        "111_trailing_bytes",
        &sparse_delta_frame(
            &meta,
            &trailing,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&bf16_base),
        ErrorClass::Corrupt,
        "one byte follows the declared run records",
        false,
        None,
    )?;

    // A record that stores an all-zero word (zeros are implicit).
    let zero_word = sparse_payload(1, &[(0, [0x00, 0x00, 0x00, 0x00])], 2);
    write_sparse_delta_reject(
        dir,
        "112_zero_nonzero_word",
        &sparse_delta_frame(
            &meta,
            &zero_word,
            decoded_len,
            content_hash,
            base_mode,
            limits,
        )?,
        Some(&bf16_base),
        ErrorClass::Corrupt,
        "a run record stores a zero word, which the canonical encoding never emits",
        false,
        None,
    )?;

    // The run bound is a real limit, not an advisory.
    write_sparse_delta_reject(
        dir,
        "113_run_count_limit",
        &delta_frame,
        Some(&bf16_base),
        ErrorClass::LimitExceeded,
        "max_sparse_runs lowered below the frame's run count",
        false,
        Some(json!({ "max_sparse_runs": 1 })),
    )?;

    // A self-referential base: the descriptor's base equals the frame's own
    // content hash, so the DAG resolver re-enters the root frame.
    let zero_target = vec![0u8; 12];
    let cycle_hash = hash_domain(DOMAIN_CHUNK, &zero_target);
    let mut cycle_meta = meta.clone();
    cycle_meta.base = cycle_hash;
    cycle_meta.delta_root = hash_domain(tensor::DOMAIN_TENSOR_VALUES, &zero_target);
    let cycle_frame = sparse_delta_frame(
        &cycle_meta,
        &sparse_payload(0, &[], 2),
        zero_target.len() as u64,
        cycle_hash,
        base_mode,
        limits,
    )?;
    write_sparse_delta_reject(
        dir,
        "114_dependency_cycle",
        &cycle_frame,
        None,
        ErrorClass::DependencyCycle,
        "the descriptor base equals the frame's own content hash",
        true,
        None,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Mode 12 KV_CACHE
// ---------------------------------------------------------------------------

fn kv_words_bf16(offset: usize, count: usize) -> Vec<u8> {
    const PATTERN: [u16; 12] = [
        0x3F80, 0xC000, 0x0000, 0x8000, 0x7FC0, 0x7F81, 0x0001, 0x7F80, 0xBC00, 0xFF80, 0x0002,
        0x4000,
    ];
    let mut out = Vec::with_capacity(count * 2);
    for index in 0..count {
        let word = PATTERN[(offset + index) % PATTERN.len()];
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn kv_words_fp16(offset: usize, count: usize) -> Vec<u8> {
    const PATTERN: [u16; 8] = [
        0x3C00, 0xBC00, 0x0000, 0x8000, 0x7E00, 0x7C00, 0xFC00, 0x0001,
    ];
    let mut out = Vec::with_capacity(count * 2);
    for index in 0..count {
        let word = PATTERN[(offset + index) % PATTERN.len()];
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn kv_words_fp32(offset: usize, count: usize) -> Vec<u8> {
    const PATTERN: [u32; 10] = [
        0x7FC0_0001,
        0x8000_0000,
        0x0000_0001,
        0x7F80_0000,
        0xFF80_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x007F_FFFF,
        0x0000_0002,
        0x7F7F_FFFF,
    ];
    let mut out = Vec::with_capacity(count * 4);
    for index in 0..count {
        let word = PATTERN[(offset + index) % PATTERN.len()];
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn write_kv_valid(
    dir: &Path,
    name: &str,
    pages: &[(kv_cache::KvCacheInput, Vec<u8>)],
    limits: &Limits,
) -> Result<()> {
    let mut page_json = Vec::with_capacity(pages.len());
    for (input, elements) in pages {
        let frame = kv_cache::encode_kv_page(input, elements, limits)?;
        let parsed = Frame::parse(&frame, limits)?;
        let decoded = kv_cache::decode_kv_page(&parsed, limits)?;
        if decoded.bytes != *elements {
            return Err(Error::corrupt("KV_CACHE vector does not reconstruct"));
        }
        page_json.push(json!({
            "page_index": decoded.meta.page_index,
            "seq_start": decoded.meta.seq_start,
            "seq_end": decoded.meta.seq_end,
            "page_rows": decoded.meta.shape[0],
            "page_bytes": decoded.meta.page_bytes,
            "payload_len": decoded.meta.payload_len,
            "elements_hex": hexs(elements),
            "frame_hex": hexs(&frame),
        }));
    }
    let first = &pages[0].0;
    let vector = json!({
        "name": name,
        "schema": kv_cache::KV_CACHE_SCHEMA,
        "model": first.model,
        "layer": first.layer,
        "session": first.session,
        "dtype": first.dtype.as_str(),
        "head_count": first.head_count,
        "head_dim": first.head_dim,
        "page_count": first.page_count,
        "pages": page_json,
        "expect": { "ok": true },
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn kv_frame_value(
    value: Value,
    payload: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(&value, limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("KV_CACHE metadata does not fit u32"))?;
    let mut body = Vec::with_capacity(4 + meta_bytes.len() + payload.len());
    body.extend_from_slice(&meta_len.to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(payload);
    build_frame(
        Mode::KvCache,
        DOMAIN_PAYLOAD,
        0,
        decoded_len,
        content_hash,
        &[],
        &body,
        limits,
    )
}

fn write_kv_reject(
    dir: &Path,
    name: &str,
    frame: &[u8],
    expected: ErrorClass,
    note: &str,
) -> Result<()> {
    let vector = json!({
        "name": name,
        "expect_error": expected.as_str(),
        "note": note,
        "frame_hex": hexs(frame),
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_kv_cache_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    // 001: bf16, two pages of one layer.
    let bf16_input = |page_index: u64, seq_start: u64, seq_end: u64| kv_cache::KvCacheInput {
        model: "llama-4-8b".to_owned(),
        layer: 3,
        session: "sess-7".to_owned(),
        page_index,
        page_count: 2,
        seq_start,
        seq_end,
        dtype: kv_cache::KvDtype::Bf16,
        head_count: 2,
        head_dim: 4,
    };
    let bf16_pages = vec![
        (bf16_input(0, 0, 4), kv_words_bf16(0, 32)),
        (bf16_input(1, 4, 8), kv_words_bf16(32, 32)),
    ];
    write_kv_valid(dir, "001_bf16_two_pages", &bf16_pages, limits)?;

    // 002: fp32, two pages.
    let fp32_input = |page_index: u64, seq_start: u64, seq_end: u64| kv_cache::KvCacheInput {
        model: "llama-4-70b".to_owned(),
        layer: 17,
        session: "sess-fp32".to_owned(),
        page_index,
        page_count: 2,
        seq_start,
        seq_end,
        dtype: kv_cache::KvDtype::Fp32,
        head_count: 2,
        head_dim: 3,
    };
    let fp32_pages = vec![
        (fp32_input(0, 0, 2), kv_words_fp32(0, 12)),
        (fp32_input(1, 2, 4), kv_words_fp32(12, 12)),
    ];
    write_kv_valid(dir, "002_fp32_two_pages", &fp32_pages, limits)?;

    // 003: bf16 sequence boundary (nonzero seq_start, late page).
    let boundary_pages = vec![(
        kv_cache::KvCacheInput {
            model: "llama-4-8b".to_owned(),
            layer: 0,
            session: "sess-boundary".to_owned(),
            page_index: 3,
            page_count: 4,
            seq_start: 4096,
            seq_end: 4098,
            dtype: kv_cache::KvDtype::Bf16,
            head_count: 1,
            head_dim: 2,
        },
        kv_words_bf16(7, 4),
    )];
    write_kv_valid(dir, "003_bf16_seq_boundary", &boundary_pages, limits)?;

    // Rejects derived from the first bf16 page.
    let base_frame = kv_cache::encode_kv_page(&bf16_pages[0].0, &bf16_pages[0].1, limits)?;
    let parsed = Frame::parse(&base_frame, limits)?;
    let meta = kv_cache::kv_cache_metadata(&parsed, limits)?;
    let body = &parsed.body;
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let payload = body[4 + meta_len..].to_vec();
    let decoded_len = parsed.header.decoded_len;
    let content_hash = parsed.header.content_hash;

    let mut zero_dims = meta.clone();
    zero_dims.head_count = 0;
    zero_dims.shape = vec![zero_dims.shape[0], 0, zero_dims.shape[2]];
    write_kv_reject(
        dir,
        "102_zero_dimension",
        &kv_cache::encode_descriptor_frame(
            &zero_dims,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "head_count is zero",
    )?;

    let unknown_dtype_value = match kv_cache::descriptor_value(&meta) {
        Value::Map(mut map) => {
            map.insert("dtype".to_owned(), Value::Str("q8_0".to_owned()));
            Value::Map(map)
        }
        other => other,
    };
    write_kv_reject(
        dir,
        "101_unknown_dtype",
        &kv_frame_value(
            unknown_dtype_value,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "dtype q8_0 has no KV_CACHE wire value",
    )?;

    let mut index_out = meta.clone();
    index_out.page_index = index_out.page_count;
    write_kv_reject(
        dir,
        "103_page_index_out_of_range",
        &kv_cache::encode_descriptor_frame(
            &index_out,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "page_index equals page_count",
    )?;

    let mut bad_seq = meta.clone();
    bad_seq.seq_start = 10;
    bad_seq.seq_end = 10;
    bad_seq.shape = vec![0, bad_seq.shape[1], bad_seq.shape[2]];
    write_kv_reject(
        dir,
        "104_seq_range_invalid",
        &kv_cache::encode_descriptor_frame(&bad_seq, &payload, decoded_len, content_hash, limits)?,
        ErrorClass::Malformed,
        "seq_start is not less than seq_end",
    )?;

    let mut bad_shape = meta.clone();
    bad_shape.shape = vec![
        bad_shape.shape[0] + 1,
        bad_shape.shape[1],
        bad_shape.shape[2],
    ];
    write_kv_reject(
        dir,
        "105_shape_mismatch",
        &kv_cache::encode_descriptor_frame(
            &bad_shape,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "shape[0] differs from page_rows",
    )?;

    let mut bad_bytes = meta.clone();
    bad_bytes.page_bytes += 2;
    write_kv_reject(
        dir,
        "106_page_bytes_mismatch",
        &kv_cache::encode_descriptor_frame(
            &bad_bytes,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "page_bytes is not the shape product times word_bytes",
    )?;

    let mut overflow = meta.clone();
    overflow.seq_start = 0;
    overflow.seq_end = u64::MAX;
    overflow.shape = vec![
        u64::MAX,
        u64::from(overflow.head_count),
        u64::from(overflow.head_dim),
    ];
    overflow.page_bytes = 64;
    write_kv_reject(
        dir,
        "107_seq_overflow",
        &kv_cache::encode_descriptor_frame(&overflow, &payload, decoded_len, content_hash, limits)?,
        ErrorClass::LimitExceeded,
        "page_rows * head_count * head_dim overflows u64",
    )?;

    let mut bad_payload_len = meta.clone();
    bad_payload_len.payload_len = payload.len() as u64 + 1;
    write_kv_reject(
        dir,
        "108_payload_len_mismatch",
        &kv_cache::encode_descriptor_frame(
            &bad_payload_len,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "declared payload_len is one byte larger than the stored payload",
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Mode 13 ACTIVATION_STREAM
// ---------------------------------------------------------------------------

fn stream_words(dtype: activation_stream::StreamDtype, offset: usize, count: usize) -> Vec<u8> {
    match dtype.as_str() {
        "bf16" => kv_words_bf16(offset, count),
        "fp16" => kv_words_fp16(offset, count),
        _ => kv_words_fp32(offset, count),
    }
}

fn stream_head(
    stream: activation_stream::StreamKind,
    step: u64,
    rank: u32,
    world: u32,
    chunk_count: u64,
    dtype: activation_stream::StreamDtype,
    shape: &[u64],
) -> activation_stream::ActivationStreamHead {
    activation_stream::ActivationStreamHead {
        stream,
        step,
        rank,
        world,
        chunk_count,
        dtype,
        shape: shape.to_vec(),
    }
}

fn write_stream_valid(
    dir: &Path,
    name: &str,
    head: &activation_stream::ActivationStreamHead,
    chunks: &[Vec<u8>],
    limits: &Limits,
) -> Result<()> {
    let mut frames = Vec::with_capacity(chunks.len());
    let mut chunk_json = Vec::with_capacity(chunks.len());
    for (index, elements) in chunks.iter().enumerate() {
        let frame =
            activation_stream::encode_activation_chunk(head, index as u64, elements, limits)?;
        let parsed = Frame::parse(&frame, limits)?;
        let decoded = activation_stream::decode_activation_chunk(&parsed, limits)?;
        if decoded.bytes != *elements {
            return Err(Error::corrupt(
                "ACTIVATION_STREAM vector does not reconstruct",
            ));
        }
        chunk_json.push(json!({
            "chunk_index": decoded.meta.chunk_index,
            "digest_root": decoded.meta.digest_root.to_hex(),
            "payload_len": decoded.meta.payload_len,
            "elements_hex": hexs(elements),
            "frame_hex": hexs(&frame),
        }));
        frames.push(frame);
    }
    let refs: Vec<&[u8]> = frames.iter().map(|frame| frame.as_slice()).collect();
    let assembly = activation_stream::assemble_stream(&refs, limits)?;
    let element_count: u64 = head.shape.iter().product();
    let vector = json!({
        "name": name,
        "schema": activation_stream::ACTIVATION_STREAM_SCHEMA,
        "stream": head.stream.as_str(),
        "step": head.step,
        "rank": head.rank,
        "world": head.world,
        "chunk_count": head.chunk_count,
        "dtype": head.dtype.as_str(),
        "shape": head.shape,
        "element_count": element_count,
        "chunks": chunk_json,
        "stream_root": assembly.stream_root.to_hex(),
        "chunk_digests": assembly
            .chunk_digests
            .iter()
            .map(|digest| digest.to_hex())
            .collect::<Vec<_>>(),
        "expect": { "ok": true },
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn stream_frame_value(
    value: Value,
    payload: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let meta_bytes = wcb::encode(&value, limits)?;
    let meta_len = u32::try_from(meta_bytes.len())
        .map_err(|_| Error::limit_exceeded("ACTIVATION_STREAM metadata does not fit u32"))?;
    let mut body = Vec::with_capacity(4 + meta_bytes.len() + payload.len());
    body.extend_from_slice(&meta_len.to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(payload);
    build_frame(
        Mode::ActivationStream,
        DOMAIN_PAYLOAD,
        0,
        decoded_len,
        content_hash,
        &[],
        &body,
        limits,
    )
}

fn write_stream_reject(
    dir: &Path,
    name: &str,
    frame: &[u8],
    expected: ErrorClass,
    note: &str,
) -> Result<()> {
    let vector = json!({
        "name": name,
        "expect_error": expected.as_str(),
        "note": note,
        "frame_hex": hexs(frame),
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_activation_stream_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    use activation_stream::{StreamDtype, StreamKind};

    // 001: activation, bf16, two chunks of one step.
    let activation_head = stream_head(
        StreamKind::Activation,
        7,
        0,
        1,
        2,
        StreamDtype::Bf16,
        &[2, 4],
    );
    let activation_chunks = vec![
        stream_words(StreamDtype::Bf16, 0, 8),
        stream_words(StreamDtype::Bf16, 8, 8),
    ];
    write_stream_valid(
        dir,
        "001_activation_bf16_two_chunks",
        &activation_head,
        &activation_chunks,
        limits,
    )?;

    // 002: gradient, fp32, two chunks.
    let gradient_head = stream_head(StreamKind::Gradient, 9, 1, 4, 2, StreamDtype::Fp32, &[1, 3]);
    let gradient_chunks = vec![
        stream_words(StreamDtype::Fp32, 0, 3),
        stream_words(StreamDtype::Fp32, 3, 3),
    ];
    write_stream_valid(
        dir,
        "002_gradient_fp32_two_chunks",
        &gradient_head,
        &gradient_chunks,
        limits,
    )?;

    // 003: activation, fp16, one chunk (sequence boundary configuration).
    let single_head = stream_head(
        StreamKind::Activation,
        0,
        0,
        2,
        1,
        StreamDtype::Fp16,
        &[2, 2],
    );
    let single_chunks = vec![stream_words(StreamDtype::Fp16, 5, 4)];
    write_stream_valid(
        dir,
        "003_activation_fp16_single_chunk",
        &single_head,
        &single_chunks,
        limits,
    )?;

    // Rejects derived from the first activation chunk.
    let base_frame = activation_stream::encode_activation_chunk(
        &activation_head,
        0,
        &activation_chunks[0],
        limits,
    )?;
    let parsed = Frame::parse(&base_frame, limits)?;
    let meta = activation_stream::activation_chunk_metadata(&parsed, limits)?;
    let body = &parsed.body;
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let payload = body[4 + meta_len..].to_vec();
    let decoded_len = parsed.header.decoded_len;
    let content_hash = parsed.header.content_hash;

    let unknown_dtype_value = match activation_stream::descriptor_value(&meta) {
        Value::Map(mut map) => {
            map.insert("dtype".to_owned(), Value::Str("q8_0".to_owned()));
            Value::Map(map)
        }
        other => other,
    };
    write_stream_reject(
        dir,
        "101_unknown_dtype",
        &stream_frame_value(
            unknown_dtype_value,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "dtype q8_0 has no ACTIVATION_STREAM wire value",
    )?;

    let unknown_stream_value = match activation_stream::descriptor_value(&meta) {
        Value::Map(mut map) => {
            map.insert("stream".to_owned(), Value::Str("optimizer".to_owned()));
            Value::Map(map)
        }
        other => other,
    };
    write_stream_reject(
        dir,
        "102_unknown_stream",
        &stream_frame_value(
            unknown_stream_value,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "stream optimizer has no v1 wire value",
    )?;

    let mut rank_world = meta.clone();
    rank_world.rank = rank_world.world;
    write_stream_reject(
        dir,
        "103_rank_not_below_world",
        &activation_stream::encode_descriptor_frame(
            &rank_world,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "rank equals world",
    )?;

    let mut chunk_out = meta.clone();
    chunk_out.chunk_index = chunk_out.chunk_count;
    write_stream_reject(
        dir,
        "104_chunk_index_out_of_range",
        &activation_stream::encode_descriptor_frame(
            &chunk_out,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "chunk_index equals chunk_count",
    )?;

    let mut count_mismatch = meta.clone();
    count_mismatch.element_count += 1;
    write_stream_reject(
        dir,
        "105_element_count_mismatch",
        &activation_stream::encode_descriptor_frame(
            &count_mismatch,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "element_count is not the shape product",
    )?;

    let mut digest_mismatch = meta.clone();
    let mut flipped = digest_mismatch.digest_root.into_bytes();
    flipped[0] ^= 0x01;
    digest_mismatch.digest_root = Hash32::from_bytes(flipped);
    write_stream_reject(
        dir,
        "106_digest_root_mismatch",
        &activation_stream::encode_descriptor_frame(
            &digest_mismatch,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Corrupt,
        "declared digest_root does not match the reconstructed elements",
    )?;

    let mut payload_mismatch = meta.clone();
    payload_mismatch.payload_len = payload.len() as u64 + 1;
    write_stream_reject(
        dir,
        "107_payload_len_mismatch",
        &activation_stream::encode_descriptor_frame(
            &payload_mismatch,
            &payload,
            decoded_len,
            content_hash,
            limits,
        )?,
        ErrorClass::Malformed,
        "declared payload_len is one byte larger than the stored payload",
    )?;

    write_stream_reject(
        dir,
        "108_decoded_len_mismatch",
        &activation_stream::encode_descriptor_frame(
            &meta,
            &payload,
            decoded_len + 1,
            content_hash,
            limits,
        )?,
        ErrorClass::Corrupt,
        "decoded_len differs from element_count * word_bytes",
    )?;

    Ok(())
}
// ---------------------------------------------------------------------------
// Semantic roots (§15) and mode 14 SEMANTIC_DELTA
// ---------------------------------------------------------------------------

fn semantic_policy_json(policy: &SemanticPolicy) -> Json {
    match policy {
        SemanticPolicy::None => json!({ "scheme": "none" }),
        SemanticPolicy::MantissaTruncate { bits } => {
            json!({ "scheme": "mantissa_truncate", "bits": bits })
        }
        SemanticPolicy::BlockQuantize { block_size, levels } => json!({
            "scheme": "block_quantize",
            "block_size": block_size,
            "levels": levels
        }),
    }
}

fn words_bf16(words: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn words_fp32(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn write_semantic_root_valid(
    dir: &Path,
    name: &str,
    dtype: WeightDtype,
    shape: &[u64],
    bytes: &[u8],
    policy: SemanticPolicy,
    limits: &Limits,
) -> Result<()> {
    let record = SemanticRootRecord::from_row_major(dtype, shape, bytes, policy, limits)?;
    let canonical = record.canonical_bytes(limits)?;
    let vector = json!({
        "name": name,
        "kind": "semantic_root",
        "schema": SEMANTIC_SCHEMA,
        "dtype": record.dtype.as_str(),
        "shape": record.shape,
        "element_count": record.element_count,
        "policy": semantic_policy_json(&record.policy),
        "values_hex": hexs(bytes),
        "quantized_values_hash": record.quantized_values_hash.to_hex(),
        "canonical_hex": hexs(&canonical),
        "semantic_root": record.root(limits)?.to_hex(),
        "expect": { "ok": true },
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn semantic_root_record_bytes(
    dtype: WeightDtype,
    shape: &[u64],
    element_count: u64,
    policy: Value,
    limits: &Limits,
) -> Result<Vec<u8>> {
    wcb::encode(
        &Value::map([
            ("schema", Value::Str(SEMANTIC_SCHEMA.to_owned())),
            ("dtype", Value::Str(dtype.as_str().to_owned())),
            (
                "shape",
                Value::Array(shape.iter().map(|dim| Value::U64(*dim)).collect()),
            ),
            ("element_count", Value::U64(element_count)),
            ("policy", policy),
            ("quantized_values_hash", Value::Bytes(vec![0u8; 32])),
        ]),
        limits,
    )
}

fn write_semantic_root_reject(dir: &Path, name: &str, record: &[u8], note: &str) -> Result<()> {
    let vector = json!({
        "name": name,
        "kind": "semantic_root",
        "expect_error": "Malformed",
        "note": note,
        "record_hex": hexs(record),
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_semantic_root_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    let bf16_bytes = words_bf16(&[0x3F80, 0xC000, 0x0000, 0x7FC0, 0x8000, 0x0001]);
    write_semantic_root_valid(
        dir,
        "001_root_none_bf16",
        WeightDtype::Bf16,
        &[2, 3],
        &bf16_bytes,
        SemanticPolicy::None,
        limits,
    )?;
    write_semantic_root_valid(
        dir,
        "002_root_mantissa_bf16_bits3",
        WeightDtype::Bf16,
        &[2, 3],
        &bf16_bytes,
        SemanticPolicy::MantissaTruncate { bits: 3 },
        limits,
    )?;

    let fp32_bytes: Vec<u8> = (0..128u32)
        .flat_map(|index| (0x3F80_0000u32.wrapping_add(index * 7919)).to_le_bytes())
        .collect();
    write_semantic_root_valid(
        dir,
        "003_root_block_quantize_fp32_b128_l256",
        WeightDtype::Fp32,
        &[2, 64],
        &fp32_bytes,
        SemanticPolicy::BlockQuantize {
            block_size: 128,
            levels: 256,
        },
        limits,
    )?;

    // Rejects -------------------------------------------------------------
    write_semantic_root_reject(
        dir,
        "101_unknown_scheme",
        &semantic_root_record_bytes(
            WeightDtype::Bf16,
            &[2, 3],
            6,
            Value::map([("scheme", Value::Str("approximate".to_owned()))]),
            limits,
        )?,
        "the scheme string is not one of none, mantissa_truncate, block_quantize",
    )?;
    write_semantic_root_reject(
        dir,
        "102_bits_out_of_range",
        &semantic_root_record_bytes(
            WeightDtype::Bf16,
            &[2, 3],
            6,
            Value::map([
                ("bits", Value::U64(8)),
                ("scheme", Value::Str("mantissa_truncate".to_owned())),
            ]),
            limits,
        )?,
        "bf16 has 7 mantissa bits; bits 8 is outside 1..=7",
    )?;
    write_semantic_root_reject(
        dir,
        "103_block_size_not_divisor",
        &semantic_root_record_bytes(
            WeightDtype::Bf16,
            &[2, 3],
            6,
            Value::map([
                ("block_size", Value::U64(4)),
                ("levels", Value::U64(16)),
                ("scheme", Value::Str("block_quantize".to_owned())),
            ]),
            limits,
        )?,
        "block_size 4 does not divide element_count 6",
    )?;
    write_semantic_root_reject(
        dir,
        "104_shape_mismatch",
        &semantic_root_record_bytes(
            WeightDtype::Bf16,
            &[2, 3],
            7,
            Value::map([("scheme", Value::Str("none".to_owned()))]),
            limits,
        )?,
        "element_count 7 does not match the shape product 6",
    )?;
    write_semantic_root_reject(
        dir,
        "105_levels_zero",
        &semantic_root_record_bytes(
            WeightDtype::Bf16,
            &[2, 3],
            6,
            Value::map([
                ("block_size", Value::U64(3)),
                ("levels", Value::U64(0)),
                ("scheme", Value::Str("block_quantize".to_owned())),
            ]),
            limits,
        )?,
        "levels 0 cannot describe a quantizer",
    )?;
    write_semantic_root_reject(
        dir,
        "106_non_finite_policy",
        &semantic_root_record_bytes(
            WeightDtype::Bf16,
            &[2, 3],
            6,
            Value::map([
                ("block_size", Value::U64(3)),
                ("levels", Value::Null),
                ("scheme", Value::Str("block_quantize".to_owned())),
            ]),
            limits,
        )?,
        "levels is null (not a finite u64); WCB has no float/NaN type",
    )?;
    Ok(())
}

fn semantic_delta_bf16_case(limits: &Limits) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let base_words: [u16; 6] = [0x3F80, 0xC000, 0x0000, 0x7FC0, 0x8000, 0x0001];
    let mut target_words = base_words;
    target_words[1] ^= 0x0003;
    let base_bytes = words_bf16(&base_words);
    let target_bytes = words_bf16(&target_words);
    let base_frame = modes::encode_tensor(&base_bytes, modes::TensorDtype::Bf16, &[2, 3], limits)?;
    Ok((target_bytes, base_bytes, base_frame))
}

fn semantic_delta_fp32_case(limits: &Limits) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let base_words: [u32; 4] = [0x3F80_0000, 0xC000_0000, 0x0000_0001, 0x7FC0_0001];
    let mut target_words = base_words;
    target_words[2] = 0x0000_00FE;
    let base_bytes = words_fp32(&base_words);
    let target_bytes = words_fp32(&target_words);
    let base_frame = modes::encode_tensor(&base_bytes, modes::TensorDtype::F32, &[2, 2], limits)?;
    Ok((target_bytes, base_bytes, base_frame))
}

fn semantic_delta_q8_case(limits: &Limits) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let base: [u8; 8] = [0x80, 0xFF, 0x00, 0x01, 0x7F, 0x2A, 0xF9, 0x64];
    let mut target = base;
    target[0] = base[0].wrapping_add(1);
    target[1] = base[1].wrapping_sub(1);
    target[3] = base[3].wrapping_add(2);
    target[4] = base[4].wrapping_sub(1);
    target[6] = base[6].wrapping_add(1);
    target[7] = base[7].wrapping_sub(2);
    let base_frame = modes::encode_tensor(&base, modes::TensorDtype::I8, &[2, 4], limits)?;
    Ok((target.to_vec(), base.to_vec(), base_frame))
}

fn write_semantic_delta_valid(
    dir: &Path,
    name: &str,
    dtype: WeightDtype,
    shape: &[u64],
    base_frame: &[u8],
    target: &[u8],
    policy: SemanticPolicy,
    limits: &Limits,
) -> Result<()> {
    let delta_frame =
        semantic_delta::encode_semantic_delta(target, dtype, shape, base_frame, policy, limits)?;
    let parsed = Frame::parse(&delta_frame, limits)?;
    let meta = semantic_delta::semantic_delta_metadata(&parsed, limits)?;
    let decoded = semantic_delta::decode_with_base_frame(
        &parsed,
        base_frame,
        &crate::dependency::MapResolver::new(),
        limits,
    )?;
    if decoded != target {
        return Err(Error::corrupt("SEMANTIC_DELTA vector does not reconstruct"));
    }
    let vector = json!({
        "name": name,
        "schema": SEMANTIC_DELTA_SCHEMA,
        "dtype": meta.dtype.as_str(),
        "shape": meta.shape,
        "element_count": meta.element_count,
        "word_bits": meta.word_bits,
        "base": meta.base.to_hex(),
        "base_dtype": meta.base_dtype.as_str(),
        "base_shape": meta.base_shape,
        "base_semantic_root": meta.base_semantic_root.to_hex(),
        "policy": semantic_policy_json(&meta.policy),
        "planes": meta.dtype.planes(),
        "plane_codec": meta.plane_codec,
        "delta_root": meta.delta_root.to_hex(),
        "payload_len": meta.payload_len,
        "target_hash": blake3_hex(target),
        "target_hex": hexs(target),
        "base_frame_hex": hexs(base_frame),
        "delta_frame_hex": hexs(&delta_frame),
        "expect": { "ok": true },
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_semantic_delta_reject(
    dir: &Path,
    name: &str,
    delta_frame: &[u8],
    base_frame: Option<&[u8]>,
    expected: ErrorClass,
    note: &str,
    use_frame_source: bool,
) -> Result<()> {
    let vector = json!({
        "name": name,
        "expect_error": expected.as_str(),
        "note": note,
        "base_frame_hex": base_frame.map(hexs),
        "delta_frame_hex": hexs(delta_frame),
        "use_frame_source": use_frame_source,
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn semantic_delta_frame(
    meta: &SemanticDeltaMeta,
    payload: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    base_mode: u32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    semantic_delta::encode_descriptor_frame(
        meta,
        payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )
}

fn write_semantic_delta_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    // --- valid ------------------------------------------------------------
    let (bf16_target, _bf16_base, bf16_base_frame) = semantic_delta_bf16_case(limits)?;
    write_semantic_delta_valid(
        dir,
        "001_bf16_near_duplicate",
        WeightDtype::Bf16,
        &[2, 3],
        &bf16_base_frame,
        &bf16_target,
        SemanticPolicy::MantissaTruncate { bits: 3 },
        limits,
    )?;

    let (fp32_target, _fp32_base, fp32_base_frame) = semantic_delta_fp32_case(limits)?;
    write_semantic_delta_valid(
        dir,
        "002_fp32_small_change",
        WeightDtype::Fp32,
        &[2, 2],
        &fp32_base_frame,
        &fp32_target,
        SemanticPolicy::MantissaTruncate { bits: 8 },
        limits,
    )?;

    let (q8_target, _q8_base, q8_base_frame) = semantic_delta_q8_case(limits)?;
    write_semantic_delta_valid(
        dir,
        "003_q8_0_mixed_changes",
        WeightDtype::Q8_0,
        &[2, 4],
        &q8_base_frame,
        &q8_target,
        SemanticPolicy::None,
        limits,
    )?;

    // --- rejects derived from the bf16 case -------------------------------
    let delta_frame = semantic_delta::encode_semantic_delta(
        &bf16_target,
        WeightDtype::Bf16,
        &[2, 3],
        &bf16_base_frame,
        SemanticPolicy::MantissaTruncate { bits: 3 },
        limits,
    )?;
    let parsed = Frame::parse(&delta_frame, limits)?;
    let meta = semantic_delta::semantic_delta_metadata(&parsed, limits)?;
    let payload = {
        let body = &parsed.body;
        let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
        body[4 + meta_len..].to_vec()
    };
    let content_hash = parsed.header.content_hash;
    let decoded_len = parsed.header.decoded_len;
    let base_mode = 6u32;

    let mut base_root = meta.clone();
    let mut flipped = base_root.base_semantic_root.into_bytes();
    flipped[0] ^= 0x01;
    base_root.base_semantic_root = Hash32::from_bytes(flipped);
    let frame = semantic_delta_frame(
        &base_root,
        &payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )?;
    write_semantic_delta_reject(
        dir,
        "101_base_semantic_mismatch",
        &frame,
        Some(&bf16_base_frame),
        ErrorClass::DependencyMismatch,
        "declared base_semantic_root does not equal the root recomputed from the base",
        false,
    )?;

    let mut policy_mismatch = meta.clone();
    policy_mismatch.policy = SemanticPolicy::MantissaTruncate { bits: 1 };
    let frame = semantic_delta_frame(
        &policy_mismatch,
        &payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )?;
    write_semantic_delta_reject(
        dir,
        "102_policy_mismatch",
        &frame,
        Some(&bf16_base_frame),
        ErrorClass::DependencyMismatch,
        "policy bits=1 recomputes a different base semantic root than the descriptor",
        false,
    )?;

    let mut delta_root = meta.clone();
    let mut flipped = delta_root.delta_root.into_bytes();
    flipped[0] ^= 0x01;
    delta_root.delta_root = Hash32::from_bytes(flipped);
    let frame = semantic_delta_frame(
        &delta_root,
        &payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )?;
    write_semantic_delta_reject(
        dir,
        "103_delta_root_mismatch",
        &frame,
        Some(&bf16_base_frame),
        ErrorClass::Corrupt,
        "declared delta_root does not match the reconstructed delta words",
        false,
    )?;

    write_semantic_delta_reject(
        dir,
        "104_missing_base",
        &delta_frame,
        None,
        ErrorClass::DependencyMissing,
        "no resolver entry and no --base frame was supplied",
        false,
    )?;

    let other_base =
        modes::encode_tensor(&vec![0u8; 12], modes::TensorDtype::Bf16, &[3, 2], limits)?;
    let other_parsed = Frame::parse(&other_base, limits)?;
    let mut shape_mismatch = meta.clone();
    shape_mismatch.base = other_parsed.header.content_hash;
    let frame = semantic_delta_frame(
        &shape_mismatch,
        &payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )?;
    write_semantic_delta_reject(
        dir,
        "105_base_shape_mismatch",
        &frame,
        Some(&other_base),
        ErrorClass::DependencyMismatch,
        "supplied base frame shape [3,2] does not match the declared shape [2,3]",
        false,
    )?;

    let mut payload_mismatch = meta.clone();
    payload_mismatch.payload_len = payload.len() as u64 + 1;
    let frame = semantic_delta_frame(
        &payload_mismatch,
        &payload,
        decoded_len,
        content_hash,
        base_mode,
        limits,
    )?;
    write_semantic_delta_reject(
        dir,
        "106_payload_length_mismatch",
        &frame,
        Some(&bf16_base_frame),
        ErrorClass::Malformed,
        "declared payload_len is one byte larger than the stored payload",
        false,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Authenticated capability envelopes (WEFT-CAPABILITIES §9)
// ---------------------------------------------------------------------------

fn capability_seed() -> [u8; 32] {
    [0x07u8; 32]
}

fn capability_envelope(minimal: bool) -> Result<CapabilityEnvelope> {
    let seed = capability_seed();
    let grantor = capability::public_key_for_seed(&seed)?;
    let grantee = Hash32::from_bytes([0x42u8; 32]);
    let revocation = Hash32::from_bytes([0x33u8; 32]);
    let nonce = Hash32::from_bytes([0x44u8; 32]);
    let roots = if minimal {
        Vec::new()
    } else {
        vec![
            Hash32::from_bytes([0x11u8; 32]),
            Hash32::from_bytes([0x22u8; 32]),
        ]
    };
    let budget = if minimal {
        None
    } else {
        Some(Budget {
            max_results: Some(10),
            max_bytes: Some(4096),
            max_depth: Some(2),
            max_work: Some(100),
        })
    };
    CapabilityEnvelope::new(
        grantor,
        grantee,
        if minimal {
            "conformance-minimal"
        } else {
            "conformance-tenant"
        },
        roots,
        vec!["default".to_owned()],
        vec!["derived_from".to_owned()],
        if minimal {
            None
        } else {
            Some(2_000_000_000_000_000_000)
        },
        budget,
        revocation,
        1_700_000_000_000_000_000,
        nonce,
    )
}

fn write_capability_valid(dir: &Path, name: &str, description: &str, minimal: bool) -> Result<()> {
    let envelope = capability_envelope(minimal)?;
    let canonical = envelope.canonical_bytes()?;
    let envelope_id = envelope.id()?;
    let sign_input = envelope.sign_input()?;
    let blob = capability::sign_envelope(&envelope, &capability_seed())?;
    let blob_bytes = blob.encode()?;
    let vector = json!({
        "name": name,
        "description": description,
        "schema": capability::ENVELOPE_SCHEMA,
        "envelope_json": envelope.to_json(),
        "canonical_hex": hexs(&canonical),
        "envelope_id": envelope_id.to_hex(),
        "sign_input": sign_input.to_hex(),
        "blob_id": blob.id()?.to_hex(),
        "blob_hex": hexs(&blob_bytes),
        "seed_hex": hexs(&capability_seed()),
        "grantee": envelope.grantee.to_hex(),
        "now_ns": 1_700_000_000_000_000_001i64,
        "expect": { "ok": true },
    });
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_capability_reject(
    dir: &Path,
    name: &str,
    description: &str,
    blob: &[u8],
    now_ns: i64,
    grantee: Option<&str>,
    revocations: &[&str],
    expect_error: &str,
) -> Result<()> {
    let mut vector = json!({
        "name": name,
        "description": description,
        "blob_hex": hexs(blob),
        "now_ns": now_ns,
        "expect_error": expect_error,
    });
    if let Some(grantee) = grantee {
        vector["grantee"] = json!(grantee);
    }
    if !revocations.is_empty() {
        vector["revocations"] = json!(revocations);
    }
    write_json(&dir.join(format!("{name}.json")), &vector)
}

fn write_capability_vectors(dir: &Path, _limits: &Limits) -> Result<()> {
    write_capability_valid(
        dir,
        "001_valid_envelope",
        "Canonical signed envelope with roots, namespaces, predicates, budget, expiry, and revocation id.",
        false,
    )?;
    write_capability_valid(
        dir,
        "002_valid_minimal_envelope",
        "Canonical signed envelope with no roots, budget, or expiry (null fields are canonical).",
        true,
    )?;

    let envelope = capability_envelope(false)?;
    let blob = capability::sign_envelope(&envelope, &capability_seed())?;
    let blob_bytes = blob.encode()?;
    let other_grantee = Hash32::from_bytes([0xAAu8; 32]).to_hex();
    let revocation = envelope.revocation_id.to_hex();

    let mut bad_signature = blob_bytes.clone();
    let last = bad_signature
        .last_mut()
        .ok_or_else(|| Error::malformed("empty capability blob"))?;
    *last ^= 0xFF;
    write_capability_reject(
        dir,
        "101_bad_signature",
        "A well-formed blob whose Ed25519 signature was flipped at the last byte.",
        &bad_signature,
        1_700_000_000_000_000_001,
        None,
        &[],
        "BadSignature",
    )?;

    write_capability_reject(
        dir,
        "102_wrong_grantee",
        "A valid blob presented for a different grantee.",
        &blob_bytes,
        1_700_000_000_000_000_001,
        Some(&other_grantee),
        &[],
        "WrongGrantee",
    )?;

    write_capability_reject(
        dir,
        "103_expired",
        "A valid blob verified after its inclusive expiry.",
        &blob_bytes,
        2_000_000_000_000_000_001,
        None,
        &[],
        "Expired",
    )?;

    write_capability_reject(
        dir,
        "104_revoked",
        "A valid blob whose revocation id is present in the caller-supplied set.",
        &blob_bytes,
        1_700_000_000_000_000_001,
        None,
        &[revocation.as_str()],
        "Revoked",
    )?;

    let mut non_canonical = blob_bytes.clone();
    non_canonical.push(0);
    write_capability_reject(
        dir,
        "105_non_canonical",
        "A well-formed blob with one trailing zero byte; canonical re-encoding must fail.",
        &non_canonical,
        1_700_000_000_000_000_001,
        None,
        &[],
        "NonCanonical",
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// TENSOR_TILED (mode 15, WEFT-AI-CODEC §16)
// ---------------------------------------------------------------------------

/// Generate only the mode-15 tensor-tiled vectors into `<dir>/tensor-tiled`.
pub fn generate_tiled_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let tiled_dir = dir.join("tensor-tiled");
    fs::create_dir_all(&tiled_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", tiled_dir.display())))?;
    write_tiled_vectors(&tiled_dir, limits)
}

/// Write the mode-17 `FLOAT_TILED` conformance vectors into `<dir>/float-tiled`.
pub fn generate_float_tiled_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let tiled_dir = dir.join("float-tiled");
    fs::create_dir_all(&tiled_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", tiled_dir.display())))?;
    write_float_tiled_vectors(&tiled_dir, limits)
}

fn float_tiled_rebuild(
    meta: &Value,
    stored: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let body = ai_body(meta, stored, limits)?;
    build_frame(
        Mode::FloatTiled,
        DOMAIN_PAYLOAD,
        0,
        decoded_len,
        content_hash,
        &[],
        &body,
        limits,
    )
}

fn write_float_tiled_valid(
    dir: &Path,
    name: &str,
    dtype: modes::FloatDtype,
    shape: &[u64],
    tile_shape: &[u64],
    plaintext: &[u8],
    region: Option<(&[u64], &[u64])>,
    limits: &Limits,
) -> Result<()> {
    let frame = modes::encode_float_tiled(plaintext, dtype, shape, tile_shape, limits)?;
    let parsed = Frame::parse(&frame, limits)?;
    let descriptor = modes::parse_float_tiled(&parsed, limits)?;
    let decoded = modes::decode_float_tiled(&parsed, limits)?;
    if decoded != plaintext {
        return Err(Error::corrupt(format!(
            "{name}: float-tiled round trip is not exact"
        )));
    }
    let mut value = json!({
        "name": name,
        "mode": "float_tiled",
        "mode_value": 17,
        "dtype": dtype.as_str(),
        "shape": shape,
        "tile_shape": tile_shape,
        "tile_counts": descriptor.meta.tile_counts,
        "tile_count": descriptor.meta.tiles.len(),
        "logical_root": descriptor.meta.logical_root.to_hex(),
        "payload_len": descriptor.meta.payload_len,
        "decoded_len": plaintext.len(),
        "content_hash": parsed.header.content_hash.to_hex(),
        "plaintext_hex": hexs(plaintext),
        "frame_hex": hexs(&frame),
        "tiles": descriptor.meta.tiles.iter().map(|entry| json!({
            "index": entry.index,
            "offset": entry.offset,
            "compressed_len": entry.compressed_len,
            "exponent_plane_len": entry.exponent_plane_len,
            "mantissa_plane_len": entry.mantissa_plane_len,
            "element_len": entry.element_len,
            "tile_root": entry.tile_root.to_hex(),
        })).collect::<Vec<_>>(),
        "expect": { "ok": true },
    });
    if let Some((start, end)) = region {
        let region = tensor::Region::new(start.to_vec(), end.to_vec());
        let (bytes, tiles_touched) = modes::read_float_region(&descriptor, &region, limits)?;
        value["region"] = json!({
            "start": start,
            "end": end,
            "bytes_hex": hexs(&bytes),
            "tiles_touched": tiles_touched,
        });
    }
    write_json(&dir.join(format!("{name}.json")), &value)
}

fn write_float_tiled_reject(dir: &Path, name: &str, class: ErrorClass, frame: &[u8]) -> Result<()> {
    let parsed = Frame::parse(frame, &Limits::default())?;
    let value = json!({
        "name": name,
        "mode": "float_tiled",
        "mode_value": 17,
        "frame_hex": hexs(frame),
        "decoded_len": parsed.header.decoded_len,
        "content_hash": parsed.header.content_hash.to_hex(),
        "dependencies": [],
        "expect": { "error": class.as_str() },
    });
    write_json(&dir.join(format!("{name}.json")), &value)
}

fn write_float_tiled_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    let bf16_bytes = tiled_pattern(16 * 2, 0x51);
    write_float_tiled_valid(
        dir,
        "001_float_tiled_bf16_4x4",
        modes::FloatDtype::Bf16,
        &[4, 4],
        &[2, 2],
        &bf16_bytes,
        Some((&[0, 0], &[3, 3])),
        limits,
    )?;
    let fp32_bytes = tiled_pattern(8 * 4, 0x77);
    write_float_tiled_valid(
        dir,
        "002_float_tiled_fp32_2x2x2",
        modes::FloatDtype::F32,
        &[2, 2, 2],
        &[1, 2, 2],
        &fp32_bytes,
        Some((&[0, 0, 0], &[2, 1, 2])),
        limits,
    )?;
    let f16_bytes = tiled_pattern(6 * 2, 0x33);
    write_float_tiled_valid(
        dir,
        "003_float_tiled_f16_single_tile",
        modes::FloatDtype::F16,
        &[2, 3],
        &[2, 3],
        &f16_bytes,
        None,
        limits,
    )?;

    let base = modes::encode_float_tiled(
        &bf16_bytes,
        modes::FloatDtype::Bf16,
        &[4, 4],
        &[2, 2],
        limits,
    )?;
    let (meta, stored, decoded_len, content_hash) = tiled_body_parts(&base, limits)?;

    // 101: a tile extent that does not divide the shape extent.
    let mut map = tiled_meta_map(&meta)?;
    map.insert(
        "tile_shape".to_owned(),
        Value::Array(vec![Value::U64(3), Value::U64(2)]),
    );
    let frame = float_tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_float_tiled_reject(
        dir,
        "101_non_dividing_tile_shape",
        ErrorClass::Malformed,
        &frame,
    )?;

    // 102: tile rank differs from the shape rank.
    let mut map = tiled_meta_map(&meta)?;
    map.insert("tile_shape".to_owned(), Value::Array(vec![Value::U64(2)]));
    let frame = float_tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_float_tiled_reject(dir, "102_tile_rank_mismatch", ErrorClass::Malformed, &frame)?;

    // 103: a tile root that does not match its decoded tile.
    let mut map = tiled_meta_map(&meta)?;
    let mut first = tiled_tile_map(&map, 0)?;
    first.insert("tile_root".to_owned(), Value::Bytes(vec![0u8; 32]));
    let mut tiles = match map.get("tiles") {
        Some(Value::Array(items)) => items.clone(),
        _ => {
            return Err(Error::malformed(
                "float tiled descriptor has no tiles array",
            ))
        }
    };
    tiles[0] = Value::Map(first);
    map.insert("tiles".to_owned(), Value::Array(tiles));
    let frame = float_tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_float_tiled_reject(dir, "103_tile_root_mismatch", ErrorClass::Corrupt, &frame)?;

    // 104: truncated stored payload. The last tile loses its final byte and
    // its descriptor lengths shrink by one, so the frame is structurally
    // consistent but the tile no longer decodes to its recorded root.
    let mut map = tiled_meta_map(&meta)?;
    let last_position = match map.get("tiles") {
        Some(Value::Array(items)) => items.len() - 1,
        _ => {
            return Err(Error::malformed(
                "float tiled descriptor has no tiles array",
            ))
        }
    };
    let mut last = tiled_tile_map(&map, last_position)?;
    let shrink = |value: &mut BTreeMap<String, Value>, key: &str| -> Result<()> {
        match value.get(key) {
            Some(Value::U64(current)) if *current > 1 => {
                value.insert(key.to_owned(), Value::U64(*current - 1));
                Ok(())
            }
            _ => Err(Error::malformed(format!("float tile has no {key}"))),
        }
    };
    shrink(&mut last, "compressed_len")?;
    shrink(&mut last, "exponent_plane_len")?;
    let mut tiles = match map.get("tiles") {
        Some(Value::Array(items)) => items.clone(),
        _ => {
            return Err(Error::malformed(
                "float tiled descriptor has no tiles array",
            ))
        }
    };
    tiles[last_position] = Value::Map(last);
    map.insert("tiles".to_owned(), Value::Array(tiles));
    shrink(&mut map, "payload_len")?;
    let mut truncated_stored = stored.clone();
    truncated_stored.pop();
    let frame = float_tiled_rebuild(
        &Value::Map(map),
        &truncated_stored,
        decoded_len,
        content_hash,
        limits,
    )?;
    write_float_tiled_reject(dir, "104_truncated_payload", ErrorClass::Corrupt, &frame)?;

    // 105: a tile index outside canonical order.
    let mut map = tiled_meta_map(&meta)?;
    let mut second = tiled_tile_map(&map, 1)?;
    second.insert("index".to_owned(), Value::U64(99));
    let mut tiles = match map.get("tiles") {
        Some(Value::Array(items)) => items.clone(),
        _ => {
            return Err(Error::malformed(
                "float tiled descriptor has no tiles array",
            ))
        }
    };
    tiles[1] = Value::Map(second);
    map.insert("tiles".to_owned(), Value::Array(tiles));
    let frame = float_tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_float_tiled_reject(
        dir,
        "105_out_of_range_tile_index",
        ErrorClass::Malformed,
        &frame,
    )?;

    // 106: unknown exponent coder.
    let mut map = tiled_meta_map(&meta)?;
    map.insert(
        "exponent_coder".to_owned(),
        Value::Str("zstd-native-v1".to_owned()),
    );
    let frame = float_tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_float_tiled_reject(
        dir,
        "106_unknown_exponent_coder",
        ErrorClass::Malformed,
        &frame,
    )?;

    // 107: zero shape dimension.
    let mut map = tiled_meta_map(&meta)?;
    map.insert(
        "shape".to_owned(),
        Value::Array(vec![Value::U64(4), Value::U64(0)]),
    );
    let frame = float_tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_float_tiled_reject(dir, "107_zero_dimension", ErrorClass::Malformed, &frame)?;

    // 108: a logical root that does not match the assembled tensor.
    let mut map = tiled_meta_map(&meta)?;
    map.insert("logical_root".to_owned(), Value::Bytes(vec![0u8; 32]));
    let frame = float_tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_float_tiled_reject(
        dir,
        "108_logical_root_mismatch",
        ErrorClass::Corrupt,
        &frame,
    )?;
    Ok(())
}

fn tiled_pattern(len: usize, mut state: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state & 0xff) as u8);
    }
    out
}

fn tiled_body_parts(frame: &[u8], limits: &Limits) -> Result<(Value, Vec<u8>, u64, Hash32)> {
    let parsed = Frame::parse(frame, limits)?;
    let body = parsed.body;
    if body.len() < 4 {
        return Err(Error::malformed("tiled frame body is shorter than 4 bytes"));
    }
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let end = 4usize
        .checked_add(meta_len)
        .ok_or_else(|| Error::malformed("tiled metadata length overflows"))?;
    if end > body.len() {
        return Err(Error::malformed("tiled metadata extends past the body"));
    }
    let meta = wcb::decode(&body[4..end], limits)?;
    Ok((
        meta,
        body[end..].to_vec(),
        parsed.header.decoded_len,
        parsed.header.content_hash,
    ))
}

fn tiled_rebuild(
    meta: &Value,
    stored: &[u8],
    decoded_len: u64,
    content_hash: Hash32,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let body = ai_body(meta, stored, limits)?;
    build_frame(
        Mode::TensorTiled,
        DOMAIN_PAYLOAD,
        0,
        decoded_len,
        content_hash,
        &[],
        &body,
        limits,
    )
}

fn tiled_meta_map(meta: &Value) -> Result<BTreeMap<String, Value>> {
    match meta {
        Value::Map(map) => Ok(map.clone()),
        _ => Err(Error::malformed("tiled descriptor is not a WCB map")),
    }
}

fn tiled_tile_map(
    meta: &BTreeMap<String, Value>,
    position: usize,
) -> Result<BTreeMap<String, Value>> {
    let tiles = match meta.get("tiles") {
        Some(Value::Array(items)) => items,
        _ => return Err(Error::malformed("tiled descriptor has no tiles array")),
    };
    let entry = tiles
        .get(position)
        .ok_or_else(|| Error::malformed("tiled descriptor is missing a tile entry"))?;
    match entry {
        Value::Map(map) => Ok(map.clone()),
        _ => Err(Error::malformed("tile entry is not a WCB map")),
    }
}

fn write_tiled_valid(
    dir: &Path,
    name: &str,
    dtype: modes::TensorDtype,
    shape: &[u64],
    tile_shape: &[u64],
    plaintext: &[u8],
    region: Option<(&[u64], &[u64])>,
    limits: &Limits,
) -> Result<()> {
    let frame = modes::encode_tiled_tensor(plaintext, dtype, shape, tile_shape, limits)?;
    let parsed = Frame::parse(&frame, limits)?;
    let descriptor = modes::parse_tiled_tensor(&parsed, limits)?;
    let decoded = modes::decode_tiled_tensor(&parsed, limits)?;
    if decoded != plaintext {
        return Err(Error::corrupt(format!(
            "{name}: tiled round trip is not exact"
        )));
    }
    let mut value = json!({
        "name": name,
        "mode": "tensor_tiled",
        "mode_value": 15,
        "dtype": dtype.as_str(),
        "shape": shape,
        "tile_shape": tile_shape,
        "tile_counts": descriptor.meta.tile_counts,
        "tile_count": descriptor.meta.tiles.len(),
        "logical_root": descriptor.meta.logical_root.to_hex(),
        "payload_len": descriptor.meta.payload_len,
        "decoded_len": plaintext.len(),
        "content_hash": parsed.header.content_hash.to_hex(),
        "plaintext_hex": hexs(plaintext),
        "frame_hex": hexs(&frame),
        "tiles": descriptor.meta.tiles.iter().map(|entry| json!({
            "index": entry.index,
            "offset": entry.offset,
            "compressed_len": entry.compressed_len,
            "element_len": entry.element_len,
            "tile_root": entry.tile_root.to_hex(),
        })).collect::<Vec<_>>(),
        "expect": { "ok": true },
    });
    if let Some((start, end)) = region {
        let region = tensor::Region::new(start.to_vec(), end.to_vec());
        let (bytes, tiles_touched) = modes::read_region(&descriptor, &region, limits)?;
        value["region"] = json!({
            "start": start,
            "end": end,
            "bytes_hex": hexs(&bytes),
            "tiles_touched": tiles_touched,
        });
    }
    write_json(&dir.join(format!("{name}.json")), &value)
}

fn write_tiled_reject(dir: &Path, name: &str, class: ErrorClass, frame: &[u8]) -> Result<()> {
    let parsed = Frame::parse(frame, &Limits::default())?;
    let mut value = json!({
        "name": name,
        "mode": "tensor_tiled",
        "mode_value": 15,
        "frame_hex": hexs(frame),
        "decoded_len": parsed.header.decoded_len,
        "content_hash": parsed.header.content_hash.to_hex(),
        "dependencies": [],
        "expect": { "error": class.as_str() },
    });
    value["name"] = json!(name);
    write_json(&dir.join(format!("{name}.json")), &value)
}

fn write_tiled_vectors(dir: &Path, limits: &Limits) -> Result<()> {
    // Valid vectors: bf16 4x4 (2x2 tiles), fp32 2x2x2 (1x2x2 tiles), and a
    // single-tile f16 2x3 tensor.
    let bf16_bytes = tiled_pattern(16 * 2, 0x51);
    write_tiled_valid(
        dir,
        "001_tiled_bf16_4x4",
        modes::TensorDtype::Bf16,
        &[4, 4],
        &[2, 2],
        &bf16_bytes,
        Some((&[0, 0], &[3, 3])),
        limits,
    )?;
    let fp32_bytes = tiled_pattern(8 * 4, 0x77);
    write_tiled_valid(
        dir,
        "002_tiled_fp32_2x2x2",
        modes::TensorDtype::F32,
        &[2, 2, 2],
        &[1, 2, 2],
        &fp32_bytes,
        Some((&[0, 0, 0], &[2, 1, 2])),
        limits,
    )?;
    let f16_bytes = tiled_pattern(6 * 2, 0x33);
    write_tiled_valid(
        dir,
        "003_tiled_f16_single_tile",
        modes::TensorDtype::F16,
        &[2, 3],
        &[2, 3],
        &f16_bytes,
        None,
        limits,
    )?;

    // Reject vectors start from a valid mode-15 frame and change exactly one
    // descriptor field, so each vector isolates one rule.
    let base = modes::encode_tiled_tensor(
        &bf16_bytes,
        modes::TensorDtype::Bf16,
        &[4, 4],
        &[2, 2],
        limits,
    )?;
    let (meta, stored, decoded_len, content_hash) = tiled_body_parts(&base, limits)?;

    // 101: tile shape does not divide the extent.
    let mut map = tiled_meta_map(&meta)?;
    map.insert(
        "tile_shape".to_owned(),
        Value::Array(vec![Value::U64(3), Value::U64(2)]),
    );
    let frame = tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_tiled_reject(
        dir,
        "101_non_dividing_tile_shape",
        ErrorClass::Malformed,
        &frame,
    )?;

    // 102: tile_counts disagrees with shape / tile_shape.
    let mut map = tiled_meta_map(&meta)?;
    map.insert(
        "tile_counts".to_owned(),
        Value::Array(vec![Value::U64(3), Value::U64(2)]),
    );
    let frame = tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_tiled_reject(
        dir,
        "102_tile_count_mismatch",
        ErrorClass::Malformed,
        &frame,
    )?;

    // 103: overlapping / non-monotonic tile offsets.
    let mut map = tiled_meta_map(&meta)?;
    let mut second = tiled_tile_map(&map, 1)?;
    let first_offset = match tiled_tile_map(&map, 0)?.get("offset") {
        Some(Value::U64(value)) => *value,
        _ => return Err(Error::malformed("tile entry has no offset")),
    };
    second.insert("offset".to_owned(), Value::U64(first_offset));
    let mut tiles = match map.get("tiles") {
        Some(Value::Array(items)) => items.clone(),
        _ => return Err(Error::malformed("tiled descriptor has no tiles array")),
    };
    tiles[1] = Value::Map(second);
    map.insert("tiles".to_owned(), Value::Array(tiles));
    let frame = tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_tiled_reject(
        dir,
        "103_overlapping_tile_offsets",
        ErrorClass::Malformed,
        &frame,
    )?;

    // 104: a tile root that does not match its decoded tile.
    let mut map = tiled_meta_map(&meta)?;
    let mut first = tiled_tile_map(&map, 0)?;
    first.insert("tile_root".to_owned(), Value::Bytes(vec![0u8; 32]));
    let mut tiles = match map.get("tiles") {
        Some(Value::Array(items)) => items.clone(),
        _ => return Err(Error::malformed("tiled descriptor has no tiles array")),
    };
    tiles[0] = Value::Map(first);
    map.insert("tiles".to_owned(), Value::Array(tiles));
    let frame = tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_tiled_reject(dir, "104_tile_root_mismatch", ErrorClass::Corrupt, &frame)?;

    // 105: declared payload_len differs from the stored payload.
    let mut map = tiled_meta_map(&meta)?;
    let payload_len = match map.get("payload_len") {
        Some(Value::U64(value)) => *value,
        _ => return Err(Error::malformed("tiled descriptor has no payload_len")),
    };
    map.insert("payload_len".to_owned(), Value::U64(payload_len + 1));
    let frame = tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_tiled_reject(
        dir,
        "105_payload_length_mismatch",
        ErrorClass::Malformed,
        &frame,
    )?;

    // 106: unknown tile codec.
    let mut map = tiled_meta_map(&meta)?;
    map.insert(
        "tile_codec".to_owned(),
        Value::Str("zstd-native-v1".to_owned()),
    );
    let frame = tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_tiled_reject(dir, "106_unknown_tile_codec", ErrorClass::Malformed, &frame)?;

    // 107: zero shape dimension.
    let mut map = tiled_meta_map(&meta)?;
    map.insert(
        "shape".to_owned(),
        Value::Array(vec![Value::U64(4), Value::U64(0)]),
    );
    let frame = tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_tiled_reject(dir, "107_zero_dimension", ErrorClass::Malformed, &frame)?;

    // 108: logical root that does not match the assembled tensor.
    let mut map = tiled_meta_map(&meta)?;
    map.insert("logical_root".to_owned(), Value::Bytes(vec![0u8; 32]));
    let frame = tiled_rebuild(&Value::Map(map), &stored, decoded_len, content_hash, limits)?;
    write_tiled_reject(
        dir,
        "108_logical_root_mismatch",
        ErrorClass::Corrupt,
        &frame,
    )?;
    Ok(())
}
