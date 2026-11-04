//! Mode 12 `KV_CACHE` exactness, page independence, range reads, and vectors.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use weftcodec::frame::Frame;
use weftcodec::kv_cache::{
    decode_kv_page, decode_kv_page_range, encode_descriptor_frame, encode_kv_page,
    kv_cache_metadata, KvCacheInput, KvDtype, KvPageSet, KV_CACHE_SCHEMA,
};
use weftcodec::limits::Limits;
use weftcodec::ErrorClass;

fn limits() -> Limits {
    Limits::default()
}

fn input(dtype: KvDtype, page_index: u64, page_count: u64) -> KvCacheInput {
    KvCacheInput {
        model: "llama-test".to_owned(),
        layer: 5,
        session: "session-a".to_owned(),
        page_index,
        page_count,
        seq_start: page_index * 4,
        seq_end: page_index * 4 + 4,
        dtype,
        head_count: 2,
        head_dim: 4,
    }
}

#[test]
fn bf16_roundtrip_preserves_nan_denormal_and_signed_zero() {
    let words: [u16; 16] = [
        0x3F80, 0xC000, 0x0000, 0x8000, 0x7FC0, 0x7FC1, 0x7F81, 0x0001, 0x0002, 0x7F80, 0xFF80,
        0xBC00, 0x4000, 0x03FF, 0x8001, 0x7FFF,
    ];
    let mut elements = Vec::with_capacity(64);
    for index in 0..32 {
        elements.extend_from_slice(&words[index % words.len()].to_le_bytes());
    }
    let frame = encode_kv_page(&input(KvDtype::Bf16, 0, 1), &elements, &limits()).expect("encode");
    let parsed = Frame::parse(&frame, &limits()).expect("parse");
    let decoded = decode_kv_page(&parsed, &limits()).expect("decode");
    assert_eq!(decoded.bytes, elements);
    assert_eq!(decoded.meta.dtype, KvDtype::Bf16);
    assert_eq!(decoded.meta.page_bytes, 64);
    assert_eq!(decoded.meta.shape, vec![4, 2, 4]);
    assert_eq!(decoded.meta.plane_codec, "huffman-canonical-v1");
    assert_eq!(decoded.meta.payload_len, {
        let body = &parsed.body;
        let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
        (body.len() - 4 - meta_len) as u64
    });
    // Deterministic re-encoding.
    let again = encode_kv_page(&input(KvDtype::Bf16, 0, 1), &elements, &limits()).expect("again");
    assert_eq!(again, frame);
}

#[test]
fn fp32_roundtrip_is_bit_exact() {
    let words: [u32; 16] = [
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
        0x0000_0000,
        0x807F_FFFF,
        0x0000_0003,
        0x7F80_0001,
        0xFFC0_0000,
        0x3EAA_AAAB,
    ];
    let mut elements = Vec::with_capacity(128);
    for index in 0..32 {
        elements.extend_from_slice(&words[index % words.len()].to_le_bytes());
    }
    let frame = encode_kv_page(&input(KvDtype::Fp32, 0, 1), &elements, &limits()).expect("encode");
    let parsed = Frame::parse(&frame, &limits()).expect("parse");
    let decoded = decode_kv_page(&parsed, &limits()).expect("decode");
    assert_eq!(decoded.bytes, elements);
    assert_eq!(decoded.meta.page_bytes, 128);
}

#[test]
fn page_seven_of_eight_decodes_without_pages_zero_to_six() {
    let dtype = KvDtype::Bf16;
    let mut frames = Vec::new();
    let mut expected = Vec::new();
    for index in 0..8u64 {
        let elements: Vec<u8> = (0..64u16)
            .map(|offset| (index as u16 * 64 + offset) as u8)
            .collect();
        frames.push(encode_kv_page(&input(dtype, index, 8), &elements, &limits()).expect("page"));
        expected.push(elements);
    }
    // Corrupt every earlier page frame. Page 7 must still decode alone.
    for frame in frames.iter_mut().take(7) {
        frame.truncate(4);
    }
    let mut set = KvPageSet::new(frames);
    let page = set.decode_page(7, &limits()).expect("page 7");
    assert_eq!(page.meta.page_index, 7);
    assert_eq!(page.bytes, expected[7]);
    assert_eq!(set.decoded_pages(), 1);

    // Range reads inside one page touch only that page.
    let slice = set
        .read_page_range(7, 8, 24, &limits())
        .expect("page 7 range");
    assert_eq!(slice, expected[7][8..24].to_vec());
    assert_eq!(set.decoded_pages(), 2);
}

#[test]
fn range_decode_of_one_page_touches_only_that_page() {
    let elements: Vec<u8> = (0..64u16).map(|value| value as u8).collect();
    let frame = encode_kv_page(&input(KvDtype::Bf16, 1, 2), &elements, &limits()).expect("page");
    let parsed = Frame::parse(&frame, &limits()).expect("parse");
    let slice = decode_kv_page_range(&parsed, 16, 48, &limits()).expect("range");
    assert_eq!(slice, elements[16..48].to_vec());
    assert_eq!(
        decode_kv_page_range(&parsed, 0, 65, &limits())
            .expect_err("out of range")
            .class(),
        ErrorClass::Malformed
    );
}

fn vector_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1/kv-cache")
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    files.sort();
    files
}

fn load(path: &Path) -> Json {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hex_field(value: &Json, key: &str) -> Vec<u8> {
    hex::decode(value[key].as_str().expect("hex field")).expect("hex")
}

#[test]
fn conformance_vectors_replay_and_reencode() {
    let dir = vector_dir();
    let files = json_files(&dir);
    assert_eq!(files.len(), 11, "3 valid + 8 rejects");
    let mut valid = 0u64;
    let mut rejects = 0u64;
    for path in files {
        let vector = load(&path);
        let name = vector["name"].as_str().unwrap_or("?");
        if let Some(expected) = vector.get("expect_error").and_then(Json::as_str) {
            let frame = hex_field(&vector, "frame_hex");
            let parsed = Frame::parse(&frame, &limits()).unwrap_or_else(|e| panic!("{name}: {e}"));
            let error = decode_kv_page(&parsed, &limits()).expect_err(name);
            assert_eq!(error.class().as_str(), expected, "{name}: {error}");
            rejects += 1;
            continue;
        }
        let pages = vector["pages"].as_array().expect("pages");
        let dtype = KvDtype::from_name(vector["dtype"].as_str().unwrap()).unwrap();
        for page in pages {
            let page_frame = hex_field(page, "frame_hex");
            let elements = hex_field(page, "elements_hex");
            let parsed_page =
                Frame::parse(&page_frame, &limits()).unwrap_or_else(|e| panic!("{name}: {e}"));
            let decoded =
                decode_kv_page(&parsed_page, &limits()).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(decoded.bytes, elements, "{name}");
            let input = KvCacheInput {
                model: vector["model"].as_str().unwrap().to_owned(),
                layer: vector["layer"].as_u64().unwrap() as u32,
                session: vector["session"].as_str().unwrap().to_owned(),
                page_index: page["page_index"].as_u64().unwrap(),
                page_count: vector["page_count"].as_u64().unwrap(),
                seq_start: page["seq_start"].as_u64().unwrap(),
                seq_end: page["seq_end"].as_u64().unwrap(),
                dtype,
                head_count: vector["head_count"].as_u64().unwrap() as u32,
                head_dim: vector["head_dim"].as_u64().unwrap() as u32,
            };
            let reencoded = encode_kv_page(&input, &elements, &limits()).expect("re-encode");
            assert_eq!(
                reencoded, page_frame,
                "{name}: Go re-encode is not identical"
            );
            valid += 1;
        }
    }
    assert_eq!(rejects, 8);
    assert_eq!(valid, 5, "2 + 2 + 1 pages across three valid vectors");
}

#[test]
fn descriptor_rejects_are_typed() {
    // Low-level constructor sanity: an invalid descriptor frame can be built
    // and is rejected by the decoder only through validated paths.
    let elements = vec![0u8; 64];
    let frame = encode_kv_page(&input(KvDtype::Bf16, 0, 2), &elements, &limits()).expect("encode");
    let parsed = Frame::parse(&frame, &limits()).expect("parse");
    let meta = kv_cache_metadata(&parsed, &limits()).expect("meta");
    let mut bad = meta.clone();
    bad.page_bytes = 65;
    let body = &parsed.body;
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let payload = &body[4 + meta_len..];
    let frame = encode_descriptor_frame(
        &bad,
        payload,
        parsed.header.decoded_len,
        parsed.header.content_hash,
        &limits(),
    )
    .expect("frame");
    let parsed = Frame::parse(&frame, &limits()).expect("parse");
    assert_eq!(
        decode_kv_page(&parsed, &limits())
            .expect_err("bad page_bytes")
            .class(),
        ErrorClass::Malformed
    );
    assert_eq!(meta.payload_len, payload.len() as u64);
    let _ = KV_CACHE_SCHEMA;

    // A corrupted plane fails the content hash, not silently.
    let mut tampered = payload.to_vec();
    let tamper_index = tampered.len() - 1;
    tampered[tamper_index] ^= 0x01;
    let frame = encode_descriptor_frame(
        &meta,
        &tampered,
        parsed.header.decoded_len,
        parsed.header.content_hash,
        &limits(),
    )
    .expect("tampered frame");
    let parsed = Frame::parse(&frame, &limits()).expect("parse");
    let error = decode_kv_page(&parsed, &limits()).expect_err("tampered");
    assert!(matches!(
        error.class(),
        ErrorClass::Corrupt | ErrorClass::NonCanonical
    ));
}
