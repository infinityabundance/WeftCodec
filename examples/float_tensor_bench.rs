//! FLOAT_TENSOR (mode 9) throughput: frame encode plus the fast word-at-a-time
//! plane join against the frozen per-bit reference join.
//!
//! Emits one JSON object per measured row and a summary object carrying the
//! per-configuration speedup ratios and their geometric mean.
//!
//! Methodology:
//! * Dtypes: bf16, fp16, fp32. Decoded sizes: 1 MiB and 16 MiB.
//! * Payloads are weight-like BF16/FP16/FP32 words: uniformly random mantissa
//!   and sign bits, exponent bytes drawn from a skewed halving distribution
//!   over the low half of the exponent range, so the Huffman coder sees the
//!   same kind of data as a real model, not uniform noise.
//! * Every row gets one warm-up call, then five measured calls; the reported
//!   value is the median. Throughput is decoded MiB per second on one thread
//!   with a warm in-process input.
//! * `encode` is `encode_float_tensor` (split + Huffman encode + frame).
//! * `join` is the plane join alone, with the fast and reference
//!   implementations measured on identical planes. Every measured output is
//!   asserted byte-identical to the source payload and to the other
//!   implementation, so the two rows are provably the same computation.
//! * The reference join below is a verbatim copy of the previous per-bit loop.
//!   It exists only for this comparison; the shipped decoder never calls it.
//!
//! This is a single-host, warm-cache microbenchmark of the coder only. It makes
//! no claim about end-to-end store throughput, cold cache, or other machines.

use std::time::Instant;

use serde_json::json;

use weftcodec::error::Result;
use weftcodec::huffman;
use weftcodec::limits::{check_len, Limits};
use weftcodec::modes::{
    encode_float_tensor, join_float_tensor_payload, split_float_tensor_payload, FloatDtype,
};

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

// ---------------------------------------------------------------------------
// Frozen reference join: a verbatim copy of the pre-accumulator per-bit loop.
// ---------------------------------------------------------------------------
fn take_bits(plane: &[u8], pos: &mut usize, count: usize) -> u64 {
    let mut value = 0u64;
    for _ in 0..count {
        let bit = (plane[*pos / 8] >> (7 - (*pos % 8))) & 1;
        value = (value << 1) | u64::from(bit);
        *pos += 1;
    }
    value
}

fn join_reference(
    exponent_plane: &[u8],
    mantissa_plane: &[u8],
    dtype: FloatDtype,
    element_count: u64,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let count = usize::try_from(element_count)
        .map_err(|_| weftcodec::Error::limit_exceeded("element count does not fit usize"))?;
    if exponent_plane.len() != count {
        return Err(weftcodec::Error::corrupt(
            "float tensor exponent plane length mismatch",
        ));
    }
    let element_bytes = dtype.element_size() as usize;
    let decoded_bytes = element_count * dtype.element_size();
    check_len(
        decoded_bytes,
        limits.max_decoded_bytes,
        "float tensor decoded bytes",
    )?;
    let sign_bits = dtype.sign_bits() as usize;
    let exponent_bits = dtype.exponent_bits() as usize;
    let mantissa_bits = dtype.mantissa_bits() as usize;
    let expected_mantissa_bytes = (element_count * (sign_bits + mantissa_bits) as u64 + 7) / 8;
    if mantissa_plane.len() as u64 != expected_mantissa_bytes {
        return Err(weftcodec::Error::corrupt(
            "float tensor mantissa plane length mismatch",
        ));
    }

    let capacity = decoded_bytes as usize;
    let mut out = Vec::with_capacity(capacity.min(1 << 20));
    let mut bit_pos = 0usize;
    for &exponent_byte in exponent_plane.iter() {
        if u64::from(exponent_byte) >> exponent_bits != 0 {
            return Err(weftcodec::Error::corrupt(
                "float tensor exponent byte exceeds the dtype exponent width",
            ));
        }
        let sign = take_bits(mantissa_plane, &mut bit_pos, sign_bits);
        let mantissa = take_bits(mantissa_plane, &mut bit_pos, mantissa_bits);
        let word = (sign << (exponent_bits + mantissa_bits))
            | (u64::from(exponent_byte) << mantissa_bits)
            | mantissa;
        out.extend_from_slice(&word.to_le_bytes()[..element_bytes]);
    }

    let used_bytes_bits = mantissa_plane.len() * 8;
    let padding = used_bytes_bits - bit_pos;
    if padding > 0 {
        let last = *mantissa_plane.last().expect("non-empty mantissa plane");
        let mask = (1u8 << padding) - 1;
        if last & mask != 0 {
            return Err(weftcodec::Error::corrupt(
                "float tensor mantissa padding bits must be zero",
            ));
        }
    }
    Ok(out)
}

type JoinFn = fn(&[u8], &[u8], FloatDtype, u64, &Limits) -> Result<Vec<u8>>;

// ---------------------------------------------------------------------------
// Weight-like payloads
// ---------------------------------------------------------------------------
fn weight_like_payload(state: &mut u64, dtype: FloatDtype, element_count: usize) -> Vec<u8> {
    let element_bytes = dtype.element_size() as usize;
    let exponent_symbols = 1usize << dtype.exponent_bits();
    // Halving weights over the low half of the exponent range, mirroring the
    // BF16/F16 exponent histogram of trained weights.
    let subset = (exponent_symbols / 2).max(1);
    let mut weights = vec![0u64; subset];
    let mut remaining = 1u64 << 20;
    for weight in weights.iter_mut() {
        let take = (remaining / 2).max(1);
        *weight = take;
        remaining = (remaining - take).max(1);
    }
    let total: u64 = weights.iter().sum();

    let mantissa_mask = (1u64 << dtype.mantissa_bits()) - 1;
    let mut payload = Vec::with_capacity(element_count * element_bytes);
    for _ in 0..element_count {
        let mut pick = xorshift(state) % total;
        let mut exponent = 0usize;
        for (index, weight) in weights.iter().enumerate() {
            if pick < *weight {
                exponent = index;
                break;
            }
            pick -= *weight;
        }
        let sign = (xorshift(state) >> 63) as u64;
        let mantissa = xorshift(state) & mantissa_mask;
        let word = (sign << (dtype.exponent_bits() + dtype.mantissa_bits()))
            | ((exponent as u64) << dtype.mantissa_bits())
            | mantissa;
        payload.extend_from_slice(&word.to_le_bytes()[..element_bytes]);
    }
    payload
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn round(value: f64, places: i32) -> f64 {
    let scale = 10f64.powi(places);
    (value * scale).round() / scale
}

/// One warm-up call then `reps` measured calls; returns the median milliseconds
/// and the last output for byte-exactness checks.
fn time<F: FnMut() -> Vec<u8>>(mut call: F, reps: usize) -> (f64, Vec<u8>) {
    let warm = call();
    let mut millis = Vec::with_capacity(reps);
    let mut last = warm;
    for _ in 0..reps {
        let start = Instant::now();
        last = call();
        millis.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    (median(millis), last)
}

fn main() {
    const REPS: usize = 5;
    const WARMUP: usize = 1;
    let limits = Limits::default();
    let dtypes = [FloatDtype::Bf16, FloatDtype::F16, FloatDtype::F32];
    let sizes: [usize; 2] = [1 << 20, 16 << 20];
    let mut seed = 0x243f_6a88_85a3_08d3u64;

    let mut speedups = Vec::new();
    let mut summary_rows = Vec::new();

    for dtype in dtypes {
        for size_bytes in sizes {
            let element_bytes = dtype.element_size() as usize;
            let element_count = size_bytes / element_bytes;
            let payload = weight_like_payload(&mut seed, dtype, element_count);
            let decoded_bytes = payload.len();
            let shape = [element_count as u64];

            // Prepare the planes exactly as the frame decoder sees them.
            let (compressed, mantissa_plane) =
                split_float_tensor_payload(&payload, dtype, element_count as u64).expect("split");
            let exponent_plane = huffman::decode(&compressed, element_count).expect("huffman");
            let reference_out = join_reference(
                &exponent_plane,
                &mantissa_plane,
                dtype,
                element_count as u64,
                &limits,
            )
            .expect("reference join");
            let fast_out = join_float_tensor_payload(
                &exponent_plane,
                &mantissa_plane,
                dtype,
                element_count as u64,
                &limits,
            )
            .expect("fast join");
            assert_eq!(reference_out, payload, "reference join mismatch");
            assert_eq!(fast_out, payload, "fast join mismatch");

            let frame_bytes = encode_float_tensor(&payload, dtype, &shape, &limits)
                .expect("encode")
                .len();

            // Encode throughput.
            let (encode_ms, encoded) = time(
                || encode_float_tensor(&payload, dtype, &shape, &limits).expect("encode"),
                REPS,
            );
            assert_eq!(encoded.len(), frame_bytes);
            println!(
                "{}",
                json!({
                    "row": "encode",
                    "dtype": dtype.as_str(),
                    "size_bytes": decoded_bytes,
                    "element_count": element_count,
                    "frame_bytes": frame_bytes,
                    "encoded_ratio": round(frame_bytes as f64 / decoded_bytes as f64, 6),
                    "median_ms": round(encode_ms, 3),
                    "mib_per_second": round(mib(decoded_bytes) / (encode_ms / 1000.0), 2),
                    "reps": REPS,
                    "warmup": WARMUP,
                })
            );

            // Join throughput, fast and reference, on identical planes.
            let mut join_rows = Vec::new();
            for (decoder, join) in [
                ("fast", join_float_tensor_payload as JoinFn),
                ("reference", join_reference as JoinFn),
            ] {
                let (ms, out) = time(
                    || {
                        join(
                            &exponent_plane,
                            &mantissa_plane,
                            dtype,
                            element_count as u64,
                            &limits,
                        )
                        .expect("join")
                    },
                    REPS,
                );
                assert_eq!(out, payload, "{decoder} join output mismatch");
                let mibs = mib(decoded_bytes) / (ms / 1000.0);
                let ns_elem = ms * 1_000_000.0 / element_count as f64;
                println!(
                    "{}",
                    json!({
                        "row": "join",
                        "decoder": decoder,
                        "dtype": dtype.as_str(),
                        "size_bytes": decoded_bytes,
                        "element_count": element_count,
                        "median_ms": round(ms, 3),
                        "mib_per_second": round(mibs, 2),
                        "ns_per_element": round(ns_elem, 3),
                        "reps": REPS,
                        "warmup": WARMUP,
                    })
                );
                join_rows.push((decoder, ms, mibs, ns_elem));
            }
            let reference_ms = join_rows[1].1;
            let fast_ms = join_rows[0].1;
            let speedup = reference_ms / fast_ms;
            speedups.push(speedup);
            summary_rows.push(json!({
                "dtype": dtype.as_str(),
                "size_bytes": decoded_bytes,
                "element_count": element_count,
                "speedup": round(speedup, 3),
                "reference_mib_per_second": round(join_rows[1].2, 2),
                "fast_mib_per_second": round(join_rows[0].2, 2),
                "reference_median_ms": round(reference_ms, 3),
                "fast_median_ms": round(fast_ms, 3),
                "reference_ns_per_element": round(join_rows[1].3, 3),
                "fast_ns_per_element": round(join_rows[0].3, 3),
            }));
            println!(
                "{}",
                json!({
                    "row": "join_speedup",
                    "dtype": dtype.as_str(),
                    "size_bytes": decoded_bytes,
                    "speedup": round(speedup, 3),
                    "reference_mib_per_second": round(join_rows[1].2, 2),
                    "fast_mib_per_second": round(join_rows[0].2, 2),
                    "reference_ns_per_element": round(join_rows[1].3, 3),
                    "fast_ns_per_element": round(join_rows[0].3, 3),
                })
            );
        }
    }

    let geometric_mean =
        (speedups.iter().map(|value| value.ln()).sum::<f64>() / speedups.len() as f64).exp();
    let min = speedups.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = speedups.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "{}",
        json!({
            "schema": "weft.float-tensor-bench.v1",
            "summary": true,
            "mode": "float_tensor",
            "reference_join": "frozen per-bit take_bits loop",
            "fast_join": "MSB-first 64-bit accumulator, whole-element reads, pre-sized output",
            "rows": summary_rows,
            "speedup_min": round(min, 3),
            "speedup_max": round(max, 3),
            "speedup_geometric_mean": round(geometric_mean, 3),
            "reps": REPS,
            "warmup": WARMUP,
            "threads": 1,
            "metric": "MiB per second of decoded logical bytes",
            "methodology": "single host, warm in-process planes, median of five join-only runs per row; every output asserted byte-identical to the source payload and to the other join",
        })
    );
}
