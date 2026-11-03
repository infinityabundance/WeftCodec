//! Exactness, validation, and limit tests for the AI-native modes
//! TENSOR (6) and TOKEN_SEQUENCE (7) from `WEFT-AI-CODEC.md`.

use weftcodec::dependency::MapResolver;
use weftcodec::error::ErrorClass;
use weftcodec::frame::{
    encode_frame, Frame, FrameParts, Mode, DOMAIN_PAYLOAD, FORMAT_MAJOR, FORMAT_MINOR,
};
use weftcodec::hash::{hash_domain, DOMAIN_CHUNK, DOMAIN_TENSOR_PAYLOAD, DOMAIN_TOKENS_PAYLOAD};
use weftcodec::limits::Limits;
use weftcodec::modes::{self, TensorDtype, TOKEN_ENCODING};
use weftcodec::wcb::{self, Value};

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

fn decode(frame: &[u8], limits: &Limits) -> Result<Vec<u8>, weftcodec::Error> {
    modes::decode_bytes(frame, &MapResolver::new(), limits)
}

fn tensor_meta(
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

fn token_meta(encoding: &str, count: u64, payload_bytes: u64, payload_hash: &[u8]) -> Value {
    Value::map([
        ("schema", Value::Str("weft.tokens.v1".to_owned())),
        ("vocab_schema", Value::Str("weft.test.vocab.v1".to_owned())),
        ("encoding", Value::Str(encoding.to_owned())),
        ("token_count", Value::U64(count)),
        ("payload_bytes", Value::U64(payload_bytes)),
        ("payload_hash", Value::Bytes(payload_hash.to_vec())),
    ])
}

fn raw_ai_frame(mode: Mode, meta: &Value, payload: &[u8], limits: &Limits) -> Vec<u8> {
    let meta_bytes = wcb::encode(meta, limits).expect("metadata");
    let mut body = Vec::new();
    body.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(payload);
    encode_frame(
        &FrameParts {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            domain: DOMAIN_PAYLOAD,
            mode,
            flags: 0,
            level: 0,
            decoded_len: payload.len() as u64,
            content_hash: hash_domain(DOMAIN_CHUNK, payload),
            dependencies: &[],
            chunks: &[],
            body: &body,
        },
        limits,
    )
    .expect("frame")
}

#[test]
fn tensor_roundtrip_all_dtypes_and_shapes() {
    let limits = Limits::default();
    let cases: &[(TensorDtype, &[u64])] = &[
        (TensorDtype::F16, &[]),
        (TensorDtype::Bf16, &[7]),
        (TensorDtype::F32, &[2, 3]),
        (TensorDtype::F64, &[2, 3, 4, 5]),
        (TensorDtype::I8, &[1, 255]),
        (TensorDtype::U8, &[0, 4]),
        (TensorDtype::I16, &[9, 9]),
        (TensorDtype::I32, &[4, 4, 4]),
        (TensorDtype::I64, &[3, 3, 3, 3, 3]),
    ];
    for (dtype, shape) in cases {
        let product: u64 = shape.iter().copied().product();
        let mut len = product * dtype.element_size();
        if shape.is_empty() {
            len = dtype.element_size();
        }
        let payload = pattern(len as usize, 0x1234_5678_9abc_def0 ^ len);
        let frame = modes::encode_tensor(&payload, *dtype, shape, &limits).expect("encode");
        let decoded = decode(&frame, &limits).expect("decode");
        assert_eq!(
            decoded,
            payload,
            "dtype {} shape {:?}",
            dtype.as_str(),
            shape
        );
        let parsed = Frame::parse(&frame, &limits).expect("parse");
        assert_eq!(parsed.header.mode, Mode::Tensor);
        let meta = modes::tensor_metadata(&parsed, &limits).expect("meta");
        assert_eq!(meta.dtype, *dtype);
        assert_eq!(meta.shape, shape.to_vec());
        assert_eq!(meta.payload_bytes, payload.len() as u64);
        // Re-encoding the decoded payload is byte-identical.
        let again = modes::encode_tensor(&decoded, *dtype, shape, &limits).expect("re-encode");
        assert_eq!(again, frame);
    }
}

#[test]
fn tensor_zero_element_is_exact() {
    let limits = Limits::default();
    for shape in [vec![0u64], vec![0, 0], vec![2, 0, 3]] {
        let frame = modes::encode_tensor(b"", TensorDtype::F32, &shape, &limits).expect("encode");
        assert_eq!(decode(&frame, &limits).expect("decode"), b"");
        let parsed = Frame::parse(&frame, &limits).expect("parse");
        assert_eq!(parsed.header.decoded_len, 0);
    }
}

#[test]
fn tensor_rejects_payload_shape_mismatch_on_encode() {
    let limits = Limits::default();
    let err = modes::encode_tensor(&[0u8; 23], TensorDtype::F32, &[2, 3], &limits)
        .expect_err("23 bytes is not 24");
    assert_eq!(err.class(), ErrorClass::Malformed);

    let err = modes::encode_tensor(&[0u8; 8], TensorDtype::I8, &[2, 3], &limits)
        .expect_err("8 bytes is not 6");
    assert_eq!(err.class(), ErrorClass::Malformed);
}

#[test]
fn tensor_rejects_bad_rank_and_overflow() {
    let limits = Limits::default();
    let nine = [1u64; 9];
    let err = modes::encode_tensor(&[0u8; 1], TensorDtype::U8, &nine, &limits).expect_err("rank 9");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);

    let err =
        modes::encode_tensor(b"", TensorDtype::U8, &[u64::MAX, 2], &limits).expect_err("overflow");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);

    let lowered = Limits::default().with_max_tensor_elements(5);
    let err = modes::encode_tensor(&[0u8; 6], TensorDtype::U8, &[2, 3], &lowered)
        .expect_err("element limit");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);

    let lowered = Limits::default().with_max_tensor_rank(2);
    let err = modes::encode_tensor(&[0u8; 1], TensorDtype::U8, &[1, 1, 1], &lowered)
        .expect_err("rank limit");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);
}

#[test]
fn tensor_decode_rejects_bad_metadata() {
    let limits = Limits::default();
    let payload = vec![1u8, 2, 3, 4, 5, 6];
    let hash = hash_domain(DOMAIN_TENSOR_PAYLOAD, &payload);

    // Shape product differs from element_count.
    let meta = tensor_meta("u8", &[2, 3], 5, 6, hash.as_bytes(), "row_major", "little");
    let frame = raw_ai_frame(Mode::Tensor, &meta, &payload, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("count").class(),
        ErrorClass::Malformed
    );

    // payload_bytes differs from element_count * element_size.
    let meta = tensor_meta("u8", &[2, 3], 6, 7, hash.as_bytes(), "row_major", "little");
    let frame = raw_ai_frame(Mode::Tensor, &meta, &payload, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("bytes").class(),
        ErrorClass::Malformed
    );

    // Payload hash does not bind this payload.
    let meta = tensor_meta("u8", &[2, 3], 6, 6, &[0u8; 32], "row_major", "little");
    let frame = raw_ai_frame(Mode::Tensor, &meta, &payload, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("hash").class(),
        ErrorClass::Corrupt
    );

    // Specified-but-not-implemented layout / endianness / dtype.
    for (layout, endian, dtype) in [
        ("column_major", "little", "u8"),
        ("row_major", "big", "u8"),
        ("row_major", "little", "q4"),
    ] {
        let meta = tensor_meta(dtype, &[2, 3], 6, 6, hash.as_bytes(), layout, endian);
        let frame = raw_ai_frame(Mode::Tensor, &meta, &payload, &limits);
        assert_eq!(
            decode(&frame, &limits).expect_err(layout).class(),
            ErrorClass::Malformed
        );
    }
}

#[test]
fn tokens_roundtrip_boundaries() {
    let limits = Limits::default();
    let cases: Vec<Vec<u64>> = vec![
        vec![],
        vec![0],
        vec![1],
        vec![u64::MAX],
        vec![0, 1, 2, 3, 4],
        vec![4, 3, 2, 1, 0],
        vec![0, u64::MAX, 0, u64::MAX - 1, 42, 42, 41],
        (0..4096u64).map(|i| (i * 7919) % 100_000).collect(),
    ];
    for ids in cases {
        let frame =
            modes::encode_token_sequence(&ids, "weft.test.vocab.v1", &limits).expect("encode");
        let payload = modes::encode_token_payload(&ids, &limits).expect("payload");
        let decoded = decode(&frame, &limits).expect("decode");
        assert_eq!(decoded, payload);
        let again =
            modes::encode_token_sequence(&ids, "weft.test.vocab.v1", &limits).expect("re-encode");
        assert_eq!(again, frame);
        let parsed = Frame::parse(&frame, &limits).expect("parse");
        let meta = modes::token_metadata(&parsed, &limits).expect("meta");
        assert_eq!(meta.token_count, ids.len() as u64);
        assert_eq!(meta.vocab_schema, "weft.test.vocab.v1");
    }
}

#[test]
fn token_payload_roundtrips_all_u64_edges() {
    let limits = Limits::default();
    let ids = vec![u64::MAX, 0, u64::MAX / 2, u64::MAX / 2 + 1, 0, 1];
    let payload = modes::encode_token_payload(&ids, &limits).expect("encode");
    let decoded = modes::decode_token_payload(&payload, ids.len() as u64, &limits).expect("decode");
    assert_eq!(decoded, ids);
    modes::validate_token_payload(&payload, ids.len() as u64, &limits).expect("validate");
}

#[test]
fn token_decode_rejects_malformed_payloads() {
    let limits = Limits::default();
    let ids = vec![1u64, 2, 3];
    let payload = modes::encode_token_payload(&ids, &limits).expect("payload");
    let hash = hash_domain(DOMAIN_TOKENS_PAYLOAD, &payload);

    // Count too small: one token remains, which is a trailing byte.
    let meta = token_meta(TOKEN_ENCODING, 2, payload.len() as u64, hash.as_bytes());
    let frame = raw_ai_frame(Mode::TokenSequence, &meta, &payload, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("count").class(),
        ErrorClass::Malformed
    );

    // Non-minimal varint.
    let non_minimal = vec![0x81u8, 0x00];
    let nm_hash = hash_domain(DOMAIN_TOKENS_PAYLOAD, &non_minimal);
    let meta = token_meta(TOKEN_ENCODING, 1, 2, nm_hash.as_bytes());
    let frame = raw_ai_frame(Mode::TokenSequence, &meta, &non_minimal, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("non-minimal").class(),
        ErrorClass::NonCanonical
    );

    // Trailing bytes after the declared token count.
    let trailing = vec![0x01u8, 0x02];
    let tr_hash = hash_domain(DOMAIN_TOKENS_PAYLOAD, &trailing);
    let meta = token_meta(TOKEN_ENCODING, 1, 2, tr_hash.as_bytes());
    let frame = raw_ai_frame(Mode::TokenSequence, &meta, &trailing, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("trailing").class(),
        ErrorClass::Malformed
    );

    // Payload hash mismatch.
    let meta = token_meta(TOKEN_ENCODING, 3, payload.len() as u64, &[0u8; 32]);
    let frame = raw_ai_frame(Mode::TokenSequence, &meta, &payload, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("hash").class(),
        ErrorClass::Corrupt
    );

    // Specified-but-not-implemented token encodings.
    for encoding in ["u32-fixed-little", "u16-fixed-little", "u32-delta"] {
        let meta = token_meta(encoding, 3, payload.len() as u64, hash.as_bytes());
        let frame = raw_ai_frame(Mode::TokenSequence, &meta, &payload, &limits);
        assert_eq!(
            decode(&frame, &limits).expect_err(encoding).class(),
            ErrorClass::Malformed
        );
    }
}

#[test]
fn token_limits_fail_closed() {
    let limits = Limits::default().with_max_tokens(2);
    let ids = vec![1u64, 2, 3];
    let payload = modes::encode_token_payload(&ids, &Limits::default()).expect("payload");
    let hash = hash_domain(DOMAIN_TOKENS_PAYLOAD, &payload);
    let meta = token_meta(TOKEN_ENCODING, 3, payload.len() as u64, hash.as_bytes());
    let frame = raw_ai_frame(Mode::TokenSequence, &meta, &payload, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("token limit").class(),
        ErrorClass::LimitExceeded
    );

    let err = modes::encode_token_sequence(&ids, "v", &limits).expect_err("encode limit");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);
}

#[test]
fn ai_domain_binding_is_mode_specific() {
    // The same payload binds under different domains for tensor and token
    // metadata; using the other domain's digest is a Corrupt mismatch.
    let limits = Limits::default();
    let payload = vec![1u8, 2, 3];
    let tensor_hash = hash_domain(DOMAIN_TENSOR_PAYLOAD, &payload);
    let token_hash = hash_domain(DOMAIN_TOKENS_PAYLOAD, &payload);
    assert_ne!(tensor_hash, token_hash);

    let mut meta = tensor_meta(
        "u8",
        &[3],
        3,
        3,
        token_hash.as_bytes(),
        "row_major",
        "little",
    );
    let frame = raw_ai_frame(Mode::Tensor, &meta, &payload, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("tensor domain").class(),
        ErrorClass::Corrupt
    );
    // Restoring the right digest makes the same body valid.
    meta = tensor_meta(
        "u8",
        &[3],
        3,
        3,
        tensor_hash.as_bytes(),
        "row_major",
        "little",
    );
    let frame = raw_ai_frame(Mode::Tensor, &meta, &payload, &limits);
    assert_eq!(decode(&frame, &limits).expect("valid"), payload);
}
