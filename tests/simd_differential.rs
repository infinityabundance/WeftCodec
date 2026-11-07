//! Differential test: the runtime-dispatched wide-word kernels must produce
//! exactly the scalar result, byte for byte, on randomised inputs.
//!
//! 10,000 randomised Huffman streams and 10,000 randomised BF16 plane pairs
//! per run (both counts are fixed, not probabilistic). Streams include valid
//! encodes of several distributions plus single-byte mutations, so accept and
//! reject decisions are compared, not just successful decodes.

use weftcodec::huffman::{decode_prefix, decode_prefix_scalar, encode};
use weftcodec::simd;

fn xorshift64star(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn random_below(state: &mut u64, bound: usize) -> usize {
    if bound == 0 {
        return 0;
    }
    (xorshift64star(state) % bound as u64) as usize
}

/// Four distributions: uniform, skewed, single symbol, and two-symbol.
fn plane(kind: usize, len: usize, state: &mut u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    match kind {
        0 => {
            for _ in 0..len {
                out.push(xorshift64star(state) as u8);
            }
        }
        1 => {
            for _ in 0..len {
                let r = xorshift64star(state);
                out.push((r.leading_zeros() as u8).wrapping_mul(7));
            }
        }
        2 => out.resize(len, 0x2a),
        _ => {
            for _ in 0..len {
                out.push(if xorshift64star(state) & 1 == 0 {
                    0x01
                } else {
                    0xfe
                });
            }
        }
    }
    out
}

#[test]
fn huffman_kernels_agree_on_10000_randomised_streams() {
    let mut state = 0x1234_5678_9abc_def1u64;
    let mut compared = 0usize;
    let mut rejected = 0usize;
    for case in 0..10_000usize {
        let kind = case % 4;
        let len = random_below(&mut state, 2048);
        let source = plane(kind, len, &mut state);
        let mut encoded = encode(&source);
        // A third of the cases are mutated after encoding.
        if case % 3 == 0 && !encoded.is_empty() {
            let index = random_below(&mut state, encoded.len());
            let mask = 1u8 << random_below(&mut state, 8);
            encoded[index] ^= mask;
        }
        let fast = decode_prefix(&encoded, source.len());
        let slow = decode_prefix_scalar(&encoded, source.len());
        match (fast, slow) {
            (Ok((fast_bytes, fast_consumed)), Ok((slow_bytes, slow_consumed))) => {
                assert_eq!(fast_bytes, slow_bytes, "case {case} bytes differ");
                assert_eq!(fast_consumed, slow_consumed, "case {case} consumed differs");
                if case % 3 != 0 {
                    assert_eq!(fast_bytes, source, "case {case} round trip differs");
                }
                compared += 1;
            }
            (Err(fast_error), Err(slow_error)) => {
                assert_eq!(
                    fast_error.class(),
                    slow_error.class(),
                    "case {case} error class differs"
                );
                rejected += 1;
            }
            (fast, slow) => {
                panic!("case {case}: one kernel accepted, the other did not: {fast:?} vs {slow:?}")
            }
        }
    }
    assert_eq!(compared + rejected, 10_000);
    // The corpus must exercise both outcomes.
    assert!(compared > 5_000, "too few accepted cases: {compared}");
    assert!(rejected > 0, "mutations were all harmless");
}

#[test]
fn join_kernels_agree_on_10000_randomised_plane_pairs() {
    let mut state = 0x0f1e_2d3c_4b5a_6978u64;
    let path = simd::active_path();
    for case in 0..10_000usize {
        let elements = random_below(&mut state, 4096);
        let mut exponent = Vec::with_capacity(elements);
        let mut mantissa = Vec::with_capacity(elements);
        for _ in 0..elements {
            exponent.push(xorshift64star(&mut state) as u8);
            mantissa.push(xorshift64star(&mut state) as u8);
        }
        let mut fast = vec![0u8; elements * 2];
        let mut slow = vec![0u8; elements * 2];
        simd::join_bf16(&exponent, &mantissa, &mut fast);
        simd::join_bf16_scalar(&exponent, &mantissa, &mut slow);
        assert_eq!(fast, slow, "case {case} ({path}) bytes differ");
        let mut explicit = vec![0u8; elements * 2];
        simd::join_bf16_avx2_path(&exponent, &mantissa, &mut explicit);
        assert_eq!(explicit, slow, "case {case} explicit AVX2 differs");
        for index in 0..elements {
            let exponent = exponent[index];
            let plane = mantissa[index];
            assert_eq!(fast[index * 2], (plane & 0x7f) | ((exponent & 1) << 7));
            assert_eq!(
                fast[index * 2 + 1],
                ((exponent >> 1) & 0x7f) | (plane & 0x80)
            );
        }
    }
}

#[test]
fn active_path_is_named() {
    let path = simd::active_path();
    assert_eq!(path == "avx2", simd::avx2_available());
}
