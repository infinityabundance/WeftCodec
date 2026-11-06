//! `huffman-canonical-v1` decode throughput: table-driven path vs the frozen
//! bit-at-a-time reference.
//!
//! Emits one JSON object per measured row and a summary object carrying the
//! per-configuration speedup ratios plus their geometric mean.
//!
//! Methodology:
//! * Decoded plane distributions: a BF16 exponent plane (skewed over 16
//!   symbols), a uniform 256-symbol plane, and a single-symbol plane.
//! * Decoded sizes: 1 MiB and 16 MiB.
//! * Each decoder gets one warm-up decode, then five measured decodes; the
//!   reported value is the median. Throughput is decoded MiB per second on one
//!   thread with a warm in-process input.
//! * Every measured decode is checked byte-for-byte against the source plane,
//!   and the reference output is checked against the fast output, so the two
//!   rows for a configuration are provably the same computation.
//! * The reference decoder below is a verbatim copy of the pre-table
//!   bit-at-a-time decoder. It exists only to make the comparison honest; the
//!   shipped `decode` never calls it.
//!
//! This is a single-host, warm-cache microbenchmark of the coder only. It makes
//! no claim about end-to-end FLOAT_TENSOR/WEIGHT_DELTA throughput, cold cache,
//! or other machines.

use std::time::Instant;

use serde_json::json;

use weftcodec::error::{Error, Result};
use weftcodec::huffman::{decode, encode, MAX_CODE_BITS, TABLE_BYTES};

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
// Frozen reference decoder: a verbatim copy of the pre-table `decode_prefix`
// loop. Kept only for measurement and cross-checking.
// ---------------------------------------------------------------------------
fn decode_prefix_reference(encoded: &[u8], expected_len: usize) -> Result<(Vec<u8>, usize)> {
    if encoded.len() < TABLE_BYTES {
        return Err(Error::corrupt(format!(
            "huffman-canonical-v1 stream is {} bytes, shorter than its {TABLE_BYTES}-byte code-length table",
            encoded.len()
        )));
    }
    let lengths = &encoded[..TABLE_BYTES];
    let stream = &encoded[TABLE_BYTES..];

    let mut counts = [0u32; MAX_CODE_BITS + 1];
    let mut used = 0usize;
    let mut max_len = 0usize;
    for (symbol, &length) in lengths.iter().enumerate() {
        let length = length as usize;
        if length > MAX_CODE_BITS {
            return Err(Error::non_canonical(format!(
                "huffman-canonical-v1 symbol {symbol} has code length {length} > {MAX_CODE_BITS}"
            )));
        }
        if length > 0 {
            used += 1;
            counts[length] += 1;
            max_len = max_len.max(length);
        }
    }

    if used == 0 {
        if expected_len != 0 {
            return Err(Error::corrupt(
                "huffman-canonical-v1 table has no symbols but the descriptor declares a non-empty plane",
            ));
        }
        return Ok((Vec::new(), TABLE_BYTES));
    }

    if used == 1 {
        if max_len != 1 {
            return Err(Error::non_canonical(
                "huffman-canonical-v1 single-symbol table must use code length 1",
            ));
        }
        if expected_len == 0 {
            return Err(Error::non_canonical(
                "huffman-canonical-v1 non-empty table for an empty plane",
            ));
        }
        let symbol = lengths
            .iter()
            .position(|&length| length > 0)
            .expect("one used symbol") as u8;
        let capacity = expected_len.min(1 << 20);
        let mut out = Vec::with_capacity(capacity);
        out.resize(expected_len, symbol);
        return Ok((out, TABLE_BYTES));
    }

    if expected_len == 0 {
        return Err(Error::non_canonical(
            "huffman-canonical-v1 non-empty table for an empty plane",
        ));
    }

    let space = 1u64 << MAX_CODE_BITS;
    let mut used_space = 0u64;
    for length in 1..=MAX_CODE_BITS {
        used_space += u64::from(counts[length]) << (MAX_CODE_BITS - length);
        if used_space > space {
            return Err(Error::non_canonical(
                "huffman-canonical-v1 code lengths are over-subscribed",
            ));
        }
    }
    if used_space != space {
        return Err(Error::non_canonical(
            "huffman-canonical-v1 code lengths are incomplete",
        ));
    }

    let mut order: Vec<(u8, u8)> = (0..256)
        .filter(|&symbol| lengths[symbol] > 0)
        .map(|symbol| (lengths[symbol], symbol as u8))
        .collect();
    order.sort_unstable();
    let mut first_code = [0u32; MAX_CODE_BITS + 1];
    let mut first_index = [0u32; MAX_CODE_BITS + 1];
    let mut assigned = [0u32; MAX_CODE_BITS + 1];
    let mut symbols = Vec::with_capacity(used);
    let mut code = 0u32;
    let mut previous = 0u8;
    for (length, symbol) in order {
        code <<= u32::from(length - previous);
        let slot = length as usize;
        if assigned[slot] == 0 {
            first_code[slot] = code;
            first_index[slot] = symbols.len() as u32;
        }
        assigned[slot] += 1;
        symbols.push(symbol);
        code += 1;
        previous = length;
    }

    let capacity = expected_len.min(1 << 20);
    let mut out = Vec::with_capacity(capacity);
    let mut bit_pos = 0usize;
    let mut current_code = 0u32;
    let mut current_len = 0usize;
    while out.len() < expected_len {
        if current_len > 0 {
            let slot = current_len;
            if current_code >= first_code[slot] {
                let offset = current_code - first_code[slot];
                if offset < counts[slot] {
                    let symbol = symbols[(first_index[slot] + offset) as usize];
                    out.push(symbol);
                    current_code = 0;
                    current_len = 0;
                    continue;
                }
            }
            if current_len >= max_len {
                return Err(Error::corrupt(
                    "huffman-canonical-v1 bitstream contains an invalid code",
                ));
            }
        }
        if bit_pos >= stream.len() * 8 {
            return Err(Error::corrupt(format!(
                "huffman-canonical-v1 bitstream ended after {} of {expected_len} bytes",
                out.len()
            )));
        }
        let bit = (stream[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1;
        current_code = (current_code << 1) | u32::from(bit);
        current_len += 1;
        bit_pos += 1;
    }

    let consumed_stream = bit_pos.div_ceil(8);
    let used_bits_in_last = bit_pos % 8;
    if used_bits_in_last != 0 {
        let last = stream[consumed_stream - 1];
        let mask = (1u8 << (8 - used_bits_in_last)) - 1;
        if last & mask != 0 {
            return Err(Error::corrupt(
                "huffman-canonical-v1 padding bits must be zero",
            ));
        }
    }
    Ok((out, TABLE_BYTES + consumed_stream))
}

fn decode_reference(encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let (out, consumed) = decode_prefix_reference(encoded, expected_len)?;
    if consumed != encoded.len() {
        return Err(Error::corrupt(
            "huffman-canonical-v1 bitstream has trailing bytes after the declared output",
        ));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------
fn bf16_exponent_plane(state: &mut u64, len: usize) -> Vec<u8> {
    // BF16 exponent bytes cluster in a narrow band and are dominated by a few
    // values. Weights halve across the lowest 16 symbols.
    let mut weights = [0u64; 16];
    let mut remaining = 1u64 << 20;
    for weight in weights.iter_mut() {
        let take = remaining / 2;
        *weight = take.max(1);
        remaining = (remaining - take).max(1);
    }
    let total: u64 = weights.iter().sum();
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let mut pick = xorshift(state) % total;
        let mut symbol = 0u8;
        for (index, &weight) in weights.iter().enumerate() {
            if pick < weight {
                symbol = index as u8;
                break;
            }
            pick -= weight;
        }
        out.push(symbol);
    }
    out
}

fn uniform_plane(state: &mut u64, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push((xorshift(state) & 0xff) as u8);
    }
    out
}

fn run_decoder(
    decoder: &str,
    distribution: &str,
    size_bytes: usize,
    encoded: &[u8],
    expected: &[u8],
    reference_output: &[u8],
) -> serde_json::Value {
    let decode_fn: fn(&[u8], usize) -> Result<Vec<u8>> = if decoder == "fast" {
        decode
    } else {
        decode_reference
    };

    // One warm-up, then five measured runs.
    let warm = decode_fn(encoded, expected.len()).expect("warm-up decode");
    assert_eq!(warm, expected, "{decoder} warm-up mismatch");
    let mut millis = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        let decoded = decode_fn(encoded, expected.len()).expect("decode");
        millis.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(decoded, expected, "{decoder} output mismatch");
        if decoder == "fast" {
            assert_eq!(decoded, reference_output, "fast vs reference mismatch");
        }
    }
    let median_ms = median(millis);
    let mib = size_bytes as f64 / (1024.0 * 1024.0);
    json!({
        "decoder": decoder,
        "distribution": distribution,
        "size_bytes": size_bytes,
        "encoded_bytes": encoded.len(),
        "median_ms": (median_ms * 1000.0).round() / 1000.0,
        "mib_per_second": ((mib / (median_ms / 1000.0)) * 100.0).round() / 100.0,
        "reps": 5,
    })
}

fn main() {
    let distributions: [(&str, fn(&mut u64, usize) -> Vec<u8>); 3] = [
        ("bf16-exponent-16", bf16_exponent_plane),
        ("uniform-256", uniform_plane),
        ("single-symbol", |_state, len| vec![0x3fu8; len]),
    ];
    let sizes: [usize; 2] = [1 << 20, 16 << 20];
    let mut seed = 0x243f_6a88_85a3_08d3u64;

    let mut rows = Vec::new();
    let mut speedups = Vec::new();
    for (distribution, build) in distributions {
        for size_bytes in sizes {
            let expected = build(&mut seed, size_bytes);
            let encoded = encode(&expected);
            let reference_output =
                decode_reference(&encoded, expected.len()).expect("reference decode");

            let reference = run_decoder(
                "reference",
                distribution,
                size_bytes,
                &encoded,
                &expected,
                &reference_output,
            );
            let fast = run_decoder(
                "fast",
                distribution,
                size_bytes,
                &encoded,
                &expected,
                &reference_output,
            );

            let reference_ms = reference["median_ms"].as_f64().expect("number");
            let fast_ms = fast["median_ms"].as_f64().expect("number");
            let reference_mibs = reference["mib_per_second"].as_f64().expect("number");
            let fast_mibs = fast["mib_per_second"].as_f64().expect("number");
            let speedup = ((reference_ms / fast_ms) * 100.0).round() / 100.0;

            println!("{reference}");
            println!("{fast}");
            println!(
                "{}",
                json!({
                    "speedup": speedup,
                    "distribution": distribution,
                    "size_bytes": size_bytes,
                    "reference_mib_per_second": reference_mibs,
                    "fast_mib_per_second": fast_mibs,
                    "reference_median_ms": reference_ms,
                    "fast_median_ms": fast_ms,
                })
            );

            rows.push((distribution, size_bytes, speedup));
            speedups.push(speedup);
        }
    }

    let geometric_mean =
        (speedups.iter().map(|value| value.ln()).sum::<f64>() / speedups.len() as f64).exp();
    println!(
        "{}",
        json!({
            "schema": "weft.huffman-decode-bench.v1",
            "summary": true,
            "coder": "huffman-canonical-v1",
            "reference_decoder": "frozen bit-at-a-time canonical loop",
            "fast_decoder": "12-bit primary table + MSB 64-bit accumulator",
            "rows": rows.iter().map(|(distribution, size_bytes, speedup)| json!({
                "distribution": distribution,
                "size_bytes": size_bytes,
                "speedup": speedup,
            })).collect::<Vec<_>>(),
            "speedup_min": ((speedups.iter().cloned().fold(f64::INFINITY, f64::min)) * 100.0).round() / 100.0,
            "speedup_max": ((speedups.iter().cloned().fold(f64::NEG_INFINITY, f64::max)) * 100.0).round() / 100.0,
            "speedup_geometric_mean": (geometric_mean * 100.0).round() / 100.0,
            "reps": 5,
            "warmup": 1,
            "threads": 1,
            "methodology": "single host, warm in-process input, median of five decode-only runs per row; outputs asserted byte-exact",
        })
    );
}
