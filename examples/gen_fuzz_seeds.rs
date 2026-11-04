//! Generate the shared differential-fuzz seed corpus.
//!
//! This is the exact generator for `conformance/fuzz/v1/seeds.txt` in the
//! WeftFS parent repository. It emits one lowercase hex WFT1 frame per line to
//! stdout. Every frame is decoded through the normal decoder with the fixed
//! dependency table before it is emitted, so a malformed seed aborts the
//! generator instead of poisoning the corpus.
//!
//! Run inside the Rust court:
//!
//! ```text
//! scripts/docker-court.sh rust \
//!   "cd /work/WeftCodec && cargo run --release --quiet --example gen_fuzz_seeds \
//!      > /work/.docker-cache/fuzz/seeds.txt"
//! ```
//!
//! The fixed bases here are duplicated verbatim in
//! `examples/fuzz_differential.rs` and in `WeftCodec-Go/cmd/fuzzdiff/main.go`.
//! They are part of the differential contract: seeds that reference a base
//! must decode identically in both languages.

use weftcodec::dependency::MapResolver;
use weftcodec::limits::Limits;
use weftcodec::modes::{self, TensorDtype};
use weftcodec::wcb::Value;

const MAX_SEED_LEN: usize = 4096;

fn base0() -> Vec<u8> {
    (0u16..=255).map(|b| b as u8).collect()
}

fn base1() -> Vec<u8> {
    b"weftcodec fuzz base payload one\n".repeat(8)
}

fn base2() -> Vec<u8> {
    Vec::new()
}

fn base3() -> Vec<u8> {
    b"delta-base-A".repeat(20)
}

fn build_resolver() -> MapResolver {
    let mut resolver = MapResolver::new();
    for base in [base0(), base1(), base2(), base3()] {
        resolver.insert_bytes(base);
    }
    resolver
}

fn push(frames: &mut Vec<Vec<u8>>, resolver: &MapResolver, frame: Vec<u8>, label: &str) {
    assert!(
        frame.len() <= MAX_SEED_LEN,
        "seed `{label}` is {} bytes, above {MAX_SEED_LEN}",
        frame.len()
    );
    modes::decode_bytes(&frame, resolver, &Limits::default())
        .unwrap_or_else(|error| panic!("seed `{label}` does not decode: {error}"));
    frames.push(frame);
}

fn main() {
    let limits = Limits::default();
    let resolver = build_resolver();
    let mut frames: Vec<Vec<u8>> = Vec::new();

    let payload_text = b"weftcodec differential fuzz seed payload\n".repeat(32);
    let payload_pattern: Vec<u8> = (0u32..1024).map(|i| ((i * 37) % 251) as u8).collect();

    // RAW, chunked and plain.
    push(
        &mut frames,
        &resolver,
        modes::encode_raw(b"", None, &limits).unwrap(),
        "raw-empty",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_raw(b"hello weft", None, &limits).unwrap(),
        "raw-hello",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_raw(&base0(), None, &limits).unwrap(),
        "raw-256",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_raw(&payload_text, None, &limits).unwrap(),
        "raw-text",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_raw(&payload_text, Some(256), &limits).unwrap(),
        "raw-chunked-256",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_raw(&base0(), Some(64), &limits).unwrap(),
        "raw-chunked-64",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_raw(&payload_pattern, Some(512), &limits).unwrap(),
        "raw-chunked-512",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_raw(&payload_pattern, Some(100), &limits).unwrap(),
        "raw-chunked-100",
    );

    // ZSTD, plain and chunked.
    push(
        &mut frames,
        &resolver,
        modes::encode_zstd(b"", 3, None, &limits).unwrap(),
        "zstd-empty",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_zstd(b"hello weft", 3, None, &limits).unwrap(),
        "zstd-hello",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_zstd(&payload_text, 1, None, &limits).unwrap(),
        "zstd-text-level1",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_zstd(&payload_text, 9, None, &limits).unwrap(),
        "zstd-text-level9",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_zstd(&payload_text, 3, Some(512), &limits).unwrap(),
        "zstd-chunked-512",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_zstd(&base0(), 5, Some(128), &limits).unwrap(),
        "zstd-chunked-128",
    );

    // REFERENCE against two fixed bases, including the empty base.
    push(
        &mut frames,
        &resolver,
        modes::encode_reference(&base0(), &base0(), &limits).unwrap(),
        "reference-256",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_reference(&base2(), &base2(), &limits).unwrap(),
        "reference-empty",
    );

    // DELTA against base3: same length, and extended.
    let mut edited = base3();
    edited[10..20].copy_from_slice(b"0123456789");
    push(
        &mut frames,
        &resolver,
        modes::encode_delta(&edited, &base3(), &limits).unwrap(),
        "delta-edit",
    );
    let mut extended = base3();
    extended.extend_from_slice(b"-tail");
    push(
        &mut frames,
        &resolver,
        modes::encode_delta(&extended, &base3(), &limits).unwrap(),
        "delta-extend",
    );

    // DICTIONARY against base1, including the empty payload.
    push(
        &mut frames,
        &resolver,
        modes::encode_dictionary(&payload_text, &base1(), 3, &limits).unwrap(),
        "dictionary-text",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_dictionary(&payload_pattern, &base1(), 1, &limits).unwrap(),
        "dictionary-pattern",
    );
    push(
        &mut frames,
        &resolver,
        modes::encode_dictionary(b"", &base1(), 3, &limits).unwrap(),
        "dictionary-empty",
    );

    // STRUCTURED: canonical WCB values of every tag shape.
    let structured: Vec<(&str, Value)> = vec![
        ("structured-null", Value::Null),
        (
            "structured-map",
            Value::map([
                ("a", Value::U64(42)),
                ("b", Value::Str("weft".to_owned())),
                ("c", Value::Bytes(vec![0, 1, 2, 255])),
                ("d", Value::Array(vec![Value::Null, Value::Bool(true)])),
                ("e", Value::Timestamp(1_725_000_000)),
            ]),
        ),
        (
            "structured-array",
            Value::Array(vec![
                Value::I64(-5),
                Value::U64(0),
                Value::Str("héllo".to_owned()),
                Value::Bytes(Vec::new()),
            ]),
        ),
        (
            "structured-nested",
            Value::Array(vec![Value::Array(vec![Value::Array(vec![
                Value::Bool(false),
                Value::Null,
            ])])]),
        ),
        ("structured-empty-map", Value::map::<_, &str>([])),
        ("structured-empty-array", Value::Array(Vec::new())),
    ];
    for (label, value) in structured {
        push(
            &mut frames,
            &resolver,
            modes::encode_structured(&value, &limits).unwrap(),
            label,
        );
    }

    // TENSOR with several dtypes, shapes, and a zero-element tensor.
    let tensor_cases: Vec<(&str, Vec<u8>, TensorDtype, Vec<u64>)> = vec![
        (
            "tensor-f32-2x3",
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23,
            ],
            TensorDtype::F32,
            vec![2, 3],
        ),
        (
            "tensor-f16-scalar",
            vec![0x00, 0x3c],
            TensorDtype::F16,
            vec![],
        ),
        ("tensor-u8-zero", Vec::new(), TensorDtype::U8, vec![0]),
        (
            "tensor-i64-1x2",
            (0u8..16).collect(),
            TensorDtype::I64,
            vec![1, 2],
        ),
        (
            "tensor-bf16-3",
            vec![0x80, 0x3f, 0x00, 0x00, 0xff, 0x7f],
            TensorDtype::Bf16,
            vec![3],
        ),
        (
            "tensor-f64-scalar",
            (0u8..8).collect(),
            TensorDtype::F64,
            vec![],
        ),
    ];
    for (label, payload, dtype, shape) in tensor_cases {
        push(
            &mut frames,
            &resolver,
            modes::encode_tensor(&payload, dtype, &shape, &limits).unwrap(),
            label,
        );
    }

    // TOKEN_SEQUENCE: empty, monotone, wrapping-delta boundaries, reversals.
    let token_cases: Vec<(&str, &[u64])> = vec![
        ("tokens-empty", &[]),
        ("tokens-single", &[0]),
        ("tokens-fib", &[0, 1, 2, 3, 5, 8, 13, 21, 34]),
        ("tokens-wrap", &[u64::MAX, 0, 1, 1 << 63, 42, u64::MAX - 1]),
        ("tokens-reverse", &[1000, 999, 1001, 0, 999]),
    ];
    for (label, ids) in token_cases {
        push(
            &mut frames,
            &resolver,
            modes::encode_token_sequence(ids, "weft.vocab.v1", &limits).unwrap(),
            label,
        );
    }

    // QUANTIZED_TENSOR: i8/u8, shapes, zero point boundaries.
    let quantized: Vec<(&str, Vec<u8>, TensorDtype, Vec<u64>, f64, i64)> = vec![
        (
            "quantized-i8-2x3",
            vec![0, 1, 2, 250, 255, 128],
            TensorDtype::I8,
            vec![2, 3],
            0.5,
            0,
        ),
        (
            "quantized-u8-zero",
            Vec::new(),
            TensorDtype::U8,
            vec![0],
            1.0,
            0,
        ),
        (
            "quantized-i8-scalar",
            vec![7],
            TensorDtype::I8,
            vec![],
            0.25,
            -3,
        ),
        (
            "quantized-u8-2x2",
            vec![0, 1, 2, 3],
            TensorDtype::U8,
            vec![2, 2],
            2.0,
            128,
        ),
        (
            "quantized-i8-4",
            vec![1, 2, 3, 4],
            TensorDtype::I8,
            vec![4],
            0.125,
            7,
        ),
        (
            "quantized-u8-1x3",
            vec![9, 8, 7],
            TensorDtype::U8,
            vec![1, 3],
            1.5,
            255,
        ),
    ];
    for (label, payload, dtype, shape, scale, zero_point) in quantized {
        push(
            &mut frames,
            &resolver,
            modes::encode_quantized_tensor(&payload, dtype, &shape, scale, zero_point, &limits)
                .unwrap(),
            label,
        );
    }

    assert_eq!(frames.len(), 44, "unexpected seed count");
    for frame in frames {
        println!("{}", hex::encode(frame));
    }
}
