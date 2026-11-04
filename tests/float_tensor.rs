//! Exactness, validation, and limit tests for the FLOAT_TENSOR mode (9) from
//! `WEFT-AI-CODEC.md` §5.

use weftcodec::dependency::MapResolver;
use weftcodec::error::ErrorClass;
use weftcodec::frame::{
    encode_frame, Frame, FrameParts, Mode, DOMAIN_PAYLOAD, FORMAT_MAJOR, FORMAT_MINOR,
};
use weftcodec::hash::{hash_domain, DOMAIN_CHUNK, DOMAIN_FLOAT_TENSOR_PAYLOAD};
use weftcodec::limits::Limits;
use weftcodec::modes::{self, FloatDtype};
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

fn float_meta(
    dtype: &str,
    shape: &[u64],
    count: u64,
    sign: u64,
    exponent: u64,
    mantissa: u64,
    exp_len: u64,
    mant_len: u64,
    payload_hash: &[u8],
) -> Value {
    Value::map([
        ("schema", Value::Str(modes::FLOAT_TENSOR_SCHEMA.to_owned())),
        ("dtype", Value::Str(dtype.to_owned())),
        (
            "shape",
            Value::Array(shape.iter().map(|d| Value::U64(*d)).collect()),
        ),
        ("element_count", Value::U64(count)),
        ("sign_bits", Value::U64(sign)),
        ("exponent_bits", Value::U64(exponent)),
        ("mantissa_bits", Value::U64(mantissa)),
        ("exponent_plane_len", Value::U64(exp_len)),
        ("mantissa_plane_len", Value::U64(mant_len)),
        ("payload_hash", Value::Bytes(payload_hash.to_vec())),
    ])
}

fn raw_float_frame(meta: &Value, stored: &[u8], decoded_len: u64, limits: &Limits) -> Vec<u8> {
    let meta_bytes = wcb::encode(meta, limits).expect("metadata");
    let mut body = Vec::new();
    body.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    body.extend_from_slice(&meta_bytes);
    body.extend_from_slice(stored);
    let plaintext = vec![0u8; decoded_len as usize];
    encode_frame(
        &FrameParts {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            domain: DOMAIN_PAYLOAD,
            mode: Mode::FloatTensor,
            flags: 0,
            level: modes::FLOAT_TENSOR_ZSTD_LEVEL as u16,
            decoded_len,
            content_hash: hash_domain(DOMAIN_CHUNK, &plaintext),
            dependencies: &[],
            chunks: &[],
            body: &body,
        },
        limits,
    )
    .expect("frame")
}

fn words_to_bytes(words: &[u32], element_bytes: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for word in words {
        out.extend_from_slice(&word.to_le_bytes()[..element_bytes]);
    }
    out
}

#[test]
fn roundtrip_all_dtypes_and_shapes() {
    let limits = Limits::default();
    let cases: &[(FloatDtype, &[u64])] = &[
        (FloatDtype::Bf16, &[1]),
        (FloatDtype::Bf16, &[2, 3]),
        (FloatDtype::F16, &[3, 1, 2]),
        (FloatDtype::F32, &[2, 1, 1, 2]),
        (FloatDtype::Fp8E4m3, &[7]),
        (FloatDtype::Fp8E5m2, &[2, 5]),
        (FloatDtype::F32, &[4, 4, 4, 4, 4, 4, 4, 4]),
    ];
    for (dtype, shape) in cases {
        let count: u64 = shape.iter().copied().product();
        let decoded_bytes = count * dtype.element_size();
        let payload = pattern(
            decoded_bytes as usize,
            0xfeed_face_dead_beef ^ decoded_bytes,
        );
        let frame = modes::encode_float_tensor(&payload, *dtype, shape, &limits).expect("encode");
        let decoded = decode(&frame, &limits).expect("decode");
        assert_eq!(
            decoded,
            payload,
            "dtype {} shape {:?}",
            dtype.as_str(),
            shape
        );

        let parsed = Frame::parse(&frame, &limits).expect("parse");
        assert_eq!(parsed.header.mode, Mode::FloatTensor);
        assert_eq!(parsed.header.mode.as_str(), "float_tensor");
        assert_eq!(parsed.header.decoded_len, decoded_bytes);
        let meta = modes::float_tensor_metadata(&parsed, &limits).expect("metadata");
        assert_eq!(meta.dtype, *dtype);
        assert_eq!(meta.shape, shape.to_vec());
        assert_eq!(meta.element_count, count);
        assert_eq!(meta.sign_bits, dtype.sign_bits());
        assert_eq!(meta.exponent_bits, dtype.exponent_bits());
        assert_eq!(meta.mantissa_bits, dtype.mantissa_bits());

        let again =
            modes::encode_float_tensor(&decoded, *dtype, shape, &limits).expect("re-encode");
        assert_eq!(again, frame, "dtype {} shape {:?}", dtype.as_str(), shape);
    }
}

#[test]
fn special_values_are_bit_exact() {
    let limits = Limits::default();
    let cases: &[(FloatDtype, &[u32])] = &[
        (
            FloatDtype::Bf16,
            &[
                0x0000, // +0
                0x8000, // -0
                0x0001, // smallest positive denormal
                0x807f, // negative denormal, max magnitude
                0x7f80, // +inf
                0xff80, // -inf
                0x7fc1, // NaN, non-zero payload
                0xffef, // negative NaN, full payload
            ],
        ),
        (
            FloatDtype::F16,
            &[
                0x0000, // +0
                0x8000, // -0
                0x0001, // denormal
                0x03ff, // largest denormal
                0x7c00, // +inf
                0xfc00, // -inf
                0x7e01, // NaN, non-zero payload
                0xffff, // negative NaN, full payload
            ],
        ),
        (
            FloatDtype::F32,
            &[
                0x0000_0000, // +0
                0x8000_0000, // -0
                0x0000_0001, // denormal
                0x007f_ffff, // largest denormal
                0x7f80_0000, // +inf
                0xff80_0000, // -inf
                0x7f80_0001, // NaN, non-zero payload
                0xffc0_0001, // negative NaN with payload
            ],
        ),
        (
            FloatDtype::Fp8E4m3,
            &[
                0x00, // +0
                0x80, // -0
                0x01, // denormal
                0x07, // largest denormal
                0x7f, // all-ones pattern (NaN in e4m3)
                0xf0, // negative
                0x4f, // payload-bearing pattern
                0xff,
            ],
        ),
        (
            FloatDtype::Fp8E5m2,
            &[0x00, 0x80, 0x01, 0x03, 0x7c, 0xfc, 0x7d, 0xff],
        ),
    ];
    for (dtype, words) in cases {
        let payload = words_to_bytes(words, dtype.element_size() as usize);
        let frame = modes::encode_float_tensor(&payload, *dtype, &[words.len() as u64], &limits)
            .expect("encode");
        assert_eq!(
            decode(&frame, &limits).expect("decode"),
            payload,
            "dtype {}",
            dtype.as_str()
        );
    }
}

#[test]
fn skewed_exponent_plane_is_compressed_and_reconstructs() {
    let limits = Limits::default();
    // 64 bf16 values: 60 zeros (exponent 0) and four NaNs (exponent 0x7f).
    let mut words: Vec<u32> = vec![0x0000; 60];
    words.extend_from_slice(&[0x7fc1, 0x7fc2, 0x7fc3, 0x7fc4]);
    let payload = words_to_bytes(&words, 2);
    let frame =
        modes::encode_float_tensor(&payload, FloatDtype::Bf16, &[8, 8], &limits).expect("encode");
    let parsed = Frame::parse(&frame, &limits).expect("parse");
    let meta = modes::float_tensor_metadata(&parsed, &limits).expect("metadata");
    assert_eq!(meta.element_count, 64);
    assert_eq!(meta.mantissa_plane_len, 64); // 8 bits per element
    assert!(
        meta.exponent_plane_len < 64,
        "zstd must shrink a skewed 64-byte exponent plane, got {}",
        meta.exponent_plane_len
    );
    assert_eq!(decode(&frame, &limits).expect("decode"), payload);
}

#[test]
fn encode_rejects_bad_shape_and_length() {
    let limits = Limits::default();
    let err = modes::encode_float_tensor(&[0u8; 7], FloatDtype::F32, &[2], &limits)
        .expect_err("7 bytes is not 8");
    assert_eq!(err.class(), ErrorClass::Malformed);

    let err =
        modes::encode_float_tensor(&[0u8; 4], FloatDtype::F32, &[], &limits).expect_err("rank 0");
    assert_eq!(err.class(), ErrorClass::Malformed);

    let err = modes::encode_float_tensor(b"", FloatDtype::F32, &[2, 0], &limits)
        .expect_err("zero dimension");
    assert_eq!(err.class(), ErrorClass::Malformed);

    let err = modes::encode_float_tensor(&[0u8; 1], FloatDtype::Fp8E4m3, &[1; 9], &limits)
        .expect_err("rank 9");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);

    let lowered = Limits::default().with_max_tensor_elements(3);
    let err = modes::encode_float_tensor(&[0u8; 4], FloatDtype::Fp8E4m3, &[4], &lowered)
        .expect_err("element limit");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);

    let lowered = Limits::default().with_max_decoded_bytes(7);
    let err = modes::encode_float_tensor(&[0u8; 8], FloatDtype::F32, &[2], &lowered)
        .expect_err("decoded byte limit");
    assert_eq!(err.class(), ErrorClass::LimitExceeded);
}

#[test]
fn decode_rejects_bad_descriptors() {
    let limits = Limits::default();
    let payload = words_to_bytes(&[0x3f80_0000, 0x7f80_0001], 4);
    let (exp_plane, mant_plane) =
        modes::split_float_tensor_payload(&payload, FloatDtype::F32, 2).expect("split");
    let mut stored = exp_plane.clone();
    stored.extend_from_slice(&mant_plane);
    let hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &stored);

    // Schema mismatch.
    let mut meta = float_meta(
        "fp32",
        &[2],
        2,
        1,
        8,
        23,
        exp_plane.len() as u64,
        mant_plane.len() as u64,
        hash.as_bytes(),
    );
    if let Value::Map(map) = &mut meta {
        map.insert(
            "schema".to_owned(),
            Value::Str("weft.float-tensor.v1".to_owned()),
        );
    }
    let frame = raw_float_frame(&meta, &stored, 8, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("schema").class(),
        ErrorClass::Malformed
    );

    // Element count disagrees with the shape product.
    let meta = float_meta(
        "fp32",
        &[2],
        3,
        1,
        8,
        23,
        exp_plane.len() as u64,
        mant_plane.len() as u64,
        hash.as_bytes(),
    );
    let frame = raw_float_frame(&meta, &stored, 8, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("count").class(),
        ErrorClass::Malformed
    );

    // Field widths disagree with the dtype.
    let meta = float_meta(
        "fp16",
        &[2],
        2,
        1,
        8,
        7,
        exp_plane.len() as u64,
        mant_plane.len() as u64,
        hash.as_bytes(),
    );
    let frame = raw_float_frame(&meta, &stored, 8, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("widths").class(),
        ErrorClass::Malformed
    );

    // Plane lengths overflow u64 before any allocation.
    let meta = float_meta("fp8-e4m3", &[2], 2, 1, 4, 3, u64::MAX, 1, &[0u8; 32]);
    let frame = raw_float_frame(&meta, b"", 2, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("overflow").class(),
        ErrorClass::LimitExceeded
    );

    // Non-zstd exponent plane.
    let mut garbage = vec![0xffu8; 3];
    garbage.extend_from_slice(&mant_plane);
    let garbage_hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &garbage);
    let meta = float_meta(
        "fp32",
        &[2],
        2,
        1,
        8,
        23,
        3,
        mant_plane.len() as u64,
        garbage_hash.as_bytes(),
    );
    let frame = raw_float_frame(&meta, &garbage, 8, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("zstd").class(),
        ErrorClass::Corrupt
    );

    // Header decoded_len disagrees with the descriptor.
    let meta = float_meta(
        "fp32",
        &[2],
        2,
        1,
        8,
        23,
        exp_plane.len() as u64,
        mant_plane.len() as u64,
        hash.as_bytes(),
    );
    let frame = raw_float_frame(&meta, &stored, 7, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("decoded_len").class(),
        ErrorClass::Corrupt
    );

    // Payload hash does not bind the stored planes.
    let meta = float_meta(
        "fp32",
        &[2],
        2,
        1,
        8,
        23,
        exp_plane.len() as u64,
        mant_plane.len() as u64,
        &[0u8; 32],
    );
    let frame = raw_float_frame(&meta, &stored, 8, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("hash").class(),
        ErrorClass::Corrupt
    );
}

#[test]
fn mantissa_padding_bits_must_be_zero() {
    // fp16 has 11 sign+mantissa bits per element, so two elements need
    // ceil(22/8) = 3 bytes with two padding bits.
    let limits = Limits::default();
    let payload = words_to_bytes(&[0x7c00, 0x0001], 2);
    let (exp_plane, mut mant_plane) =
        modes::split_float_tensor_payload(&payload, FloatDtype::F16, 2).expect("split");
    assert_eq!(mant_plane.len(), 3);
    assert_eq!(mant_plane[2] & 0b11, 0);
    mant_plane[2] |= 0b01;
    let mut stored = exp_plane.clone();
    stored.extend_from_slice(&mant_plane);
    let hash = hash_domain(DOMAIN_FLOAT_TENSOR_PAYLOAD, &stored);
    let meta = float_meta(
        "fp16",
        &[2],
        2,
        1,
        5,
        10,
        exp_plane.len() as u64,
        mant_plane.len() as u64,
        hash.as_bytes(),
    );
    let frame = raw_float_frame(&meta, &stored, 4, &limits);
    assert_eq!(
        decode(&frame, &limits).expect_err("padding").class(),
        ErrorClass::Corrupt
    );
}
