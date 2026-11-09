//! Mode 19 `KV_CACHE_PROV`: provenance binding, exactness, and vector replay.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use weftcodec::frame::Frame;
use weftcodec::hash::Hash32;
use weftcodec::kv_cache::{KvCacheInput, KvDtype};
use weftcodec::kv_prov::{
    decode_kv_prov_page, encode_kv_prov_page, kv_prov_metadata, KvAttention, KvLayout,
    KvPositional, KvProvDescriptor, KvProvenanceInput, KV_PROV_SCHEMA,
};
use weftcodec::limits::Limits;
use weftcodec::wcb::Value;
use weftcodec::ErrorClass;

fn limits() -> Limits {
    Limits::default()
}

fn root(byte: u8) -> Hash32 {
    Hash32::from_bytes([byte; 32])
}

fn attention(head_count: u32, head_dim: u32) -> KvAttention {
    KvAttention {
        head_count,
        head_dim,
        sliding_window: 0,
        rope_theta_bits: 10_000.0f64.to_bits(),
        rope_scale_bits: 1.0f64.to_bits(),
    }
}

fn layout(dtype: KvDtype, rows: u64, layers: u64) -> KvLayout {
    KvLayout {
        dtype,
        page_rows: rows,
        layer_count: layers,
    }
}

fn input(layer: u32, page_index: u64, seq_start: u64, seq_end: u64) -> KvCacheInput {
    KvCacheInput {
        model: "prov-model".to_owned(),
        layer,
        session: "prov-session".to_owned(),
        page_index,
        page_count: 4,
        seq_start,
        seq_end,
        dtype: KvDtype::Bf16,
        head_count: 2,
        head_dim: 4,
    }
}

fn provenance(model_root: u8, adapters: &[u8], salt: u8) -> KvProvenanceInput {
    let mut input = KvProvenanceInput::new(
        root(model_root),
        "a-bpe",
        151_936,
        attention(2, 4),
        KvPositional::Rope,
        layout(KvDtype::Bf16, 4, 8),
        root(salt),
    );
    for adapter in adapters {
        input = input.with_adapter(root(*adapter));
    }
    input
}

fn elements() -> Vec<u8> {
    (0..64u16)
        .map(|value| (value.wrapping_mul(37) ^ 0x5A) as u8)
        .collect()
}

#[test]
fn roundtrip_is_bit_exact_and_deterministic() {
    let data = elements();
    let prov = provenance(0x11, &[], 0xA0);
    let frame = encode_kv_prov_page(&input(0, 0, 0, 4), &data, &prov, &limits()).expect("encode");
    let parsed = Frame::parse(&frame, &limits()).expect("parse");
    assert_eq!(parsed.header.mode.as_str(), "kv_cache_prov");
    let decoded = decode_kv_prov_page(&parsed, &limits()).expect("decode");
    assert_eq!(decoded.bytes, data);
    assert_eq!(decoded.descriptor.page.page_bytes, 64);
    assert_eq!(decoded.descriptor.provenance.adapter_roots, Vec::new());
    let again = encode_kv_prov_page(&input(0, 0, 0, 4), &data, &prov, &limits()).expect("again");
    assert_eq!(again, frame);
    assert_eq!(
        decoded.root,
        kv_prov_metadata(&parsed, &limits())
            .expect("metadata")
            .root(&limits())
            .expect("root")
    );
}

#[test]
fn identical_payload_with_different_adapter_roots_has_a_different_root() {
    let data = elements();
    let frame_a = encode_kv_prov_page(
        &input(0, 0, 0, 4),
        &data,
        &provenance(0x11, &[0x21], 0xA0),
        &limits(),
    )
    .expect("a");
    let frame_b = encode_kv_prov_page(
        &input(0, 0, 0, 4),
        &data,
        &provenance(0x11, &[0x22], 0xA0),
        &limits(),
    )
    .expect("b");
    let a = decode_kv_prov_page(&Frame::parse(&frame_a, &limits()).unwrap(), &limits()).unwrap();
    let b = decode_kv_prov_page(&Frame::parse(&frame_b, &limits()).unwrap(), &limits()).unwrap();
    assert_eq!(a.bytes, b.bytes, "payloads are byte-identical");
    assert_eq!(a.descriptor.payload_hash, b.descriptor.payload_hash);
    assert_ne!(a.root, b.root, "different adapter sets must differ");
    assert_ne!(
        a.descriptor.provenance.adapter_set_root,
        b.descriptor.provenance.adapter_set_root
    );
}

#[test]
fn adapter_set_is_order_independent_and_cardinality_sensitive() {
    let data = elements();
    let mut ab = provenance(0x11, &[0x21, 0x22], 0xA0);
    let mut ba = provenance(0x11, &[0x22, 0x21], 0xA0);
    let a_only = provenance(0x11, &[0x21], 0xA0);
    let frame_ab = encode_kv_prov_page(&input(0, 0, 0, 4), &data, &ab, &limits()).unwrap();
    let frame_ba = encode_kv_prov_page(&input(0, 0, 0, 4), &data, &ba, &limits()).unwrap();
    let frame_a = encode_kv_prov_page(&input(0, 0, 0, 4), &data, &a_only, &limits()).unwrap();
    assert_eq!(frame_ab, frame_ba, "{{A,B}} and {{B,A}} are one identity");
    assert_ne!(frame_ab, frame_a, "{{A}} and {{A,B}} differ");
    // Reversing the caller's list does not change the canonical set.
    ab.adapter_roots.reverse();
    assert_eq!(
        ab.canonicalize().unwrap().adapter_roots,
        vec![root(0x21), root(0x22)]
    );
    ba = provenance(0x11, &[0x21, 0x22], 0xA0);
    assert_eq!(
        ba.canonicalize().unwrap().adapter_roots,
        vec![root(0x21), root(0x22)]
    );
    assert_eq!(
        ba.canonicalize().unwrap().adapter_roots,
        vec![root(0x21), root(0x22)]
    );
}

#[test]
fn salt_changes_the_root_and_nothing_else() {
    let data = elements();
    let a = encode_kv_prov_page(
        &input(0, 0, 0, 4),
        &data,
        &provenance(0x11, &[], 0xB0),
        &limits(),
    )
    .unwrap();
    let b = encode_kv_prov_page(
        &input(0, 0, 0, 4),
        &data,
        &provenance(0x11, &[], 0xB1),
        &limits(),
    )
    .unwrap();
    let a = decode_kv_prov_page(&Frame::parse(&a, &limits()).unwrap(), &limits()).unwrap();
    let b = decode_kv_prov_page(&Frame::parse(&b, &limits()).unwrap(), &limits()).unwrap();
    assert_eq!(a.bytes, b.bytes);
    assert_eq!(a.descriptor.payload_hash, b.descriptor.payload_hash);
    assert_ne!(a.root, b.root);
    assert_ne!(a.descriptor.provenance.salt, b.descriptor.provenance.salt);
}

#[test]
fn duplicate_extension_names_are_rejected() {
    let prov = provenance(0x11, &[], 0xA0)
        .with_extension("weft.test", Value::U64(1))
        .with_extension("weft.test", Value::U64(2));
    let error = encode_kv_prov_page(&input(0, 0, 0, 4), &elements(), &prov, &limits())
        .expect_err("duplicate extension");
    assert_eq!(error.class(), ErrorClass::Malformed);
}

#[test]
fn zero_model_root_is_rejected() {
    let prov = provenance(0x00, &[], 0xA0);
    let error = encode_kv_prov_page(&input(0, 0, 0, 4), &elements(), &prov, &limits())
        .expect_err("zero model root");
    assert_eq!(error.class(), ErrorClass::Malformed);
}

#[test]
fn tampered_descriptor_dimensions_fail_closed() {
    let data = elements();
    let prov = provenance(0x11, &[0x21], 0xA0);
    let frame = encode_kv_prov_page(&input(0, 0, 0, 4), &data, &prov, &limits()).unwrap();
    let parsed = Frame::parse(&frame, &limits()).unwrap();
    let decoded = decode_kv_prov_page(&parsed, &limits()).unwrap();

    // A descriptor whose adapter_set_root does not hash its adapter roots.
    let mut forged = decoded.descriptor.clone();
    forged.provenance.adapter_set_root = root(0x5A);
    let bytes = forged.canonical_bytes(&limits()).unwrap();
    let error = KvProvDescriptor::from_canonical_bytes(&bytes, &limits()).expect_err("forged root");
    assert_eq!(error.class(), ErrorClass::Malformed);

    // A descriptor whose tokenizer_root does not hash its tokenizer.
    let mut forged = decoded.descriptor.clone();
    forged.provenance.tokenizer_root = root(0x5B);
    let bytes = forged.canonical_bytes(&limits()).unwrap();
    let error = KvProvDescriptor::from_canonical_bytes(&bytes, &limits()).expect_err("forged tok");
    assert_eq!(error.class(), ErrorClass::Malformed);

    // A descriptor that claims a different attention geometry than the page.
    let mut forged = decoded.descriptor.clone();
    forged.provenance.attention.head_count = 3;
    let bytes = forged.canonical_bytes(&limits()).unwrap();
    let error = KvProvDescriptor::from_canonical_bytes(&bytes, &limits()).expect_err("attention");
    assert_eq!(error.class(), ErrorClass::Malformed);

    // A descriptor that claims a different layout dtype than the page.
    let mut forged = decoded.descriptor.clone();
    forged.provenance.layout.dtype = KvDtype::Fp32;
    let bytes = forged.canonical_bytes(&limits()).unwrap();
    let error =
        KvProvDescriptor::from_canonical_bytes(&bytes, &limits()).expect_err("layout mismatch");
    assert_eq!(error.class(), ErrorClass::Malformed);
}

#[test]
fn a_swapped_payload_is_rejected_by_the_payload_hash() {
    let data = elements();
    let mut other = data.clone();
    other[0] ^= 0xFF;
    let prov = provenance(0x11, &[], 0xA0);
    let frame = encode_kv_prov_page(&input(0, 0, 0, 4), &data, &prov, &limits()).unwrap();
    let parsed = Frame::parse(&frame, &limits()).unwrap();
    let body = parsed.body;
    let descriptor_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let descriptor = &body[4..4 + descriptor_len];
    // Keep the descriptor (and therefore payload_hash) and replace the payload
    // with a different, self-consistent mode-12 payload.
    let other_frame = encode_kv_prov_page(&input(0, 0, 0, 4), &other, &prov, &limits()).unwrap();
    let other_parsed = Frame::parse(&other_frame, &limits()).unwrap();
    let other_body = other_parsed.body;
    let other_len =
        u32::from_le_bytes([other_body[0], other_body[1], other_body[2], other_body[3]]) as usize;
    let other_payload = &other_body[4 + other_len..];
    let mut swapped = Vec::with_capacity(4 + descriptor.len() + other_payload.len());
    swapped.extend_from_slice(&(descriptor.len() as u32).to_le_bytes());
    swapped.extend_from_slice(descriptor);
    swapped.extend_from_slice(other_payload);
    let frame = weftcodec::frame::encode_frame(
        &weftcodec::frame::FrameParts {
            format_major: weftcodec::frame::FORMAT_MAJOR,
            format_minor: weftcodec::frame::FORMAT_MINOR,
            domain: weftcodec::frame::DOMAIN_PAYLOAD,
            mode: weftcodec::frame::Mode::KvCacheProv,
            flags: 0,
            level: 0,
            decoded_len: parsed.header.decoded_len,
            content_hash: parsed.header.content_hash,
            dependencies: &[],
            chunks: &[],
            body: &swapped,
        },
        &limits(),
    )
    .unwrap();
    let parsed = Frame::parse(&frame, &limits()).unwrap();
    let error = decode_kv_prov_page(&parsed, &limits()).expect_err("swapped payload");
    assert_eq!(error.class(), ErrorClass::Corrupt);
}

// ---------------------------------------------------------------------------
// Committed vector replay
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct PageVector {
    page_index: u64,
    seq_start: u64,
    seq_end: u64,
    page_rows: u64,
    page_bytes: u64,
    payload_len: u64,
    descriptor_root: String,
    elements_hex: String,
    frame_hex: String,
}

#[derive(serde::Deserialize)]
struct AttentionVector {
    head_count: u32,
    head_dim: u32,
    sliding_window: u64,
    rope_theta_bits: u64,
    rope_scale_bits: u64,
}

#[derive(serde::Deserialize)]
struct TokenizerVector {
    scheme: String,
    vocab_size: u64,
}

#[derive(serde::Deserialize)]
struct LayoutVector {
    dtype: String,
    page_rows: u64,
    layer_count: u64,
}

#[derive(serde::Deserialize)]
struct ExtensionVector {
    name: String,
    value: Json,
}

#[derive(serde::Deserialize)]
struct ProvenanceVector {
    model_root: String,
    adapter_roots: Vec<String>,
    tokenizer: TokenizerVector,
    attention: AttentionVector,
    positional: String,
    layout: LayoutVector,
    salt: String,
    extensions: Vec<ExtensionVector>,
}

#[derive(serde::Deserialize)]
struct KvProvVector {
    name: String,
    schema: Option<String>,
    model: Option<String>,
    layer: Option<u32>,
    session: Option<String>,
    dtype: Option<String>,
    head_count: Option<u32>,
    head_dim: Option<u32>,
    page_count: Option<u64>,
    descriptor_root: Option<String>,
    provenance: Option<ProvenanceVector>,
    pages: Option<Vec<PageVector>>,
    expect: Option<Json>,
    expect_error: Option<String>,
    frame_hex: Option<String>,
}

fn json_to_wcb(value: &Json) -> Value {
    match value {
        Json::Null => Value::Null,
        Json::Bool(flag) => Value::Bool(*flag),
        Json::Number(number) => {
            if let Some(unsigned) = number.as_u64() {
                Value::U64(unsigned)
            } else {
                Value::I64(number.as_i64().expect("integer"))
            }
        }
        Json::String(text) => Value::Str(text.clone()),
        Json::Array(items) => Value::Array(items.iter().map(json_to_wcb).collect()),
        Json::Object(map) => {
            let mut out = std::collections::BTreeMap::new();
            for (key, item) in map {
                out.insert(key.clone(), json_to_wcb(item));
            }
            Value::Map(out)
        }
    }
}

fn vector_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1/kv-prov")
}

fn load(path: &Path) -> KvProvVector {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn committed_kv_prov_vectors_replay() {
    let directory = vector_dir();
    let mut files: Vec<PathBuf> = fs::read_dir(&directory)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", directory.display()))
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    files.sort();
    assert_eq!(files.len(), 20, "expected 8 valid and 12 reject vectors");

    let mut valid_vectors = 0usize;
    let mut valid_pages = 0usize;
    let mut rejects = 0usize;
    for path in files {
        let vector = load(&path);
        if let Some(expect_error) = vector.expect_error {
            let frame_hex = vector.frame_hex.expect("reject frame");
            let frame = hex::decode(frame_hex).expect("hex");
            // A reserved mode is rejected by the frame parser itself; every
            // other reject is rejected by the descriptor validator.
            match Frame::parse(&frame, &limits()) {
                Err(error) => assert_eq!(error.class().as_str(), expect_error, "{}", vector.name),
                Ok(parsed) => {
                    let error = decode_kv_prov_page(&parsed, &limits())
                        .expect_err(&format!("{} must be rejected", vector.name));
                    assert_eq!(error.class().as_str(), expect_error, "{}", vector.name);
                }
            }
            rejects += 1;
            continue;
        }
        assert_eq!(
            vector.schema.as_deref(),
            Some(KV_PROV_SCHEMA),
            "{}",
            vector.name
        );
        assert!(vector.expect.as_ref().and_then(|e| e.get("ok")).is_some());
        let provenance = vector.provenance.expect("provenance");
        let mut prov = KvProvenanceInput::new(
            Hash32::from_hex(&provenance.model_root).unwrap(),
            provenance.tokenizer.scheme,
            provenance.tokenizer.vocab_size,
            KvAttention {
                head_count: provenance.attention.head_count,
                head_dim: provenance.attention.head_dim,
                sliding_window: provenance.attention.sliding_window,
                rope_theta_bits: provenance.attention.rope_theta_bits,
                rope_scale_bits: provenance.attention.rope_scale_bits,
            },
            KvPositional::from_name(&provenance.positional).unwrap(),
            KvLayout {
                dtype: KvDtype::from_name(&provenance.layout.dtype).unwrap(),
                page_rows: provenance.layout.page_rows,
                layer_count: provenance.layout.layer_count,
            },
            Hash32::from_hex(&provenance.salt).unwrap(),
        );
        for adapter in provenance.adapter_roots {
            prov = prov.with_adapter(Hash32::from_hex(&adapter).unwrap());
        }
        for extension in provenance.extensions {
            prov = prov.with_extension(extension.name, json_to_wcb(&extension.value));
        }
        let first_root = vector.descriptor_root.expect("descriptor root");
        let pages = vector.pages.expect("pages");
        for page in &pages {
            let frame_bytes = hex::decode(&page.frame_hex).expect("frame hex");
            let expected_bytes = hex::decode(&page.elements_hex).expect("elements hex");
            let decoded_input = KvCacheInput {
                model: vector.model.clone().expect("model"),
                layer: vector.layer.expect("layer"),
                session: vector.session.clone().expect("session"),
                page_index: page.page_index,
                page_count: vector.page_count.expect("page_count"),
                seq_start: page.seq_start,
                seq_end: page.seq_end,
                dtype: KvDtype::from_name(&vector.dtype.clone().expect("dtype")).unwrap(),
                head_count: vector.head_count.expect("head_count"),
                head_dim: vector.head_dim.expect("head_dim"),
            };
            let reencoded =
                encode_kv_prov_page(&decoded_input, &expected_bytes, &prov, &limits()).unwrap();
            assert_eq!(
                reencoded, frame_bytes,
                "{} page {}",
                vector.name, page.page_index
            );
            let parsed = Frame::parse(&frame_bytes, &limits()).unwrap();
            let decoded = decode_kv_prov_page(&parsed, &limits()).unwrap();
            assert_eq!(decoded.bytes, expected_bytes, "{} bytes", vector.name);
            assert_eq!(
                decoded.root.to_hex(),
                page.descriptor_root,
                "{} root",
                vector.name
            );
            assert_eq!(decoded.descriptor.page.shape[0], page.page_rows);
            assert_eq!(decoded.descriptor.page.page_bytes, page.page_bytes);
            assert_eq!(decoded.descriptor.page.payload_len, page.payload_len);
            if valid_pages == 0 {
                assert_eq!(page.descriptor_root, first_root);
            }
            valid_pages += 1;
        }
        valid_vectors += 1;
    }
    assert_eq!(valid_vectors, 8, "valid vectors");
    assert_eq!(valid_pages, 9, "valid page frames");
    assert_eq!(rejects, 12, "rejects");
}
