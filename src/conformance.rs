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

use crate::context::{self, Budget, CausalClosure, ContextRoot, DEFAULT_RESOLVER};
use crate::error::{Error, ErrorClass, Result};
use crate::frame::{
    encode_frame, Dependency, Frame, FrameParts, Mode, DEP_DOMAIN_DICTIONARY, DEP_DOMAIN_PAYLOAD,
    DOMAIN_PAYLOAD, DOMAIN_STRUCTURED, FORMAT_MAJOR, FORMAT_MINOR,
};
use crate::hash::{
    hash_domain, Hash32, DOMAIN_CHUNK, DOMAIN_QUANTIZED_TENSOR_PAYLOAD, DOMAIN_SELECTOR,
    DOMAIN_TENSOR_PAYLOAD, DOMAIN_TOKENS_PAYLOAD,
};
use crate::limits::Limits;
use crate::modes;
use crate::selector;
use crate::varint;
use crate::wcb::{self, Value};

/// Generate every v1 vector into `dir` (creating `codec/`, `structured/`, and
/// `selectors/`).
pub fn generate_v1(dir: &Path, limits: &Limits) -> Result<()> {
    let codec_dir = dir.join("codec");
    let structured_dir = dir.join("structured");
    let selectors_dir = dir.join("selectors");
    let context_dir = dir.join("context");
    let causal_dir = dir.join("causal");
    for sub in [
        &codec_dir,
        &structured_dir,
        &selectors_dir,
        &context_dir,
        &causal_dir,
    ] {
        fs::create_dir_all(sub)
            .map_err(|e| Error::io(format!("cannot create {}: {e}", sub.display())))?;
    }

    write_codec_vectors(&codec_dir, limits)?;
    write_structured_vectors(&structured_dir, limits)?;
    write_selector_vectors(&selectors_dir, limits)?;
    write_context_vectors(&context_dir, limits)?;
    write_causal_vectors(&causal_dir, limits)?;
    Ok(())
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
    unknown_mode[7] = 9;
    refix_header_and_trailer(&mut unknown_mode);
    push_error(
        &mut vectors,
        &name("unknown_mode_reserved9"),
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
