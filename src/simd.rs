//! Runtime-dispatched wide-word decode kernels.
//!
//! This is the **only** module in `weftcodec` that is permitted to contain
//! `unsafe` code: it holds the AVX2 intrinsics and nothing else does (the crate
//! root is `deny(unsafe_code)` and `tests/no_unsafe_outside_simd.rs` fails if
//! any other source file mentions `unsafe`).
//!
//! What is dispatched at runtime:
//!
//! * [`unpack`] — the `huffman-canonical-v1` primary-table bit unpack loop.
//!   The AVX2 build uses `_mm_loadl_epi64` for eight-byte refills and decodes
//!   two symbols per table probe when both prefixes are buffered. The scalar
//!   build is byte-refill, one symbol per probe.
//! * [`join_bf16`] — the `FLOAT_TENSOR` plane join for BF16, where the
//!   sign+mantissa plane is exactly one byte per element. The AVX2 build
//!   interleaves two 16-byte planes with `_mm256_unpacklo_epi8`; the scalar
//!   build is the word-at-a-time loop.
//!
//! The wire format is unchanged by either kernel: the same bits are consumed
//! and the same bytes are produced. Both kernels are checked against each other
//! on 10,000 randomised inputs per language in the differential test.
//!
//! `unpack` deliberately does **not** own error handling. It decodes as many
//! symbols as the primary table can prove and returns; the caller keeps the
//! canonical path for long codes, stream tails, and every error class. That
//! keeps error parity exact without duplicating the validator.

#![allow(unsafe_code)]

/// Packed primary-table entry (`symbol | length << 8`); `length == 0` is the
/// "longer than the table" sentinel and is never consumed.
pub const ENTRY_LONG: u16 = 0;

/// Mutable bit-unpack state shared with the caller's canonical fallback.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnpackState {
    pub acc: u64,
    pub acc_bits: u32,
    pub next_byte: usize,
    pub bits_consumed: usize,
}

/// True when the CPU exposes AVX2 and the optimised kernels may run.
#[cfg(target_arch = "x86_64")]
pub fn avx2_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

/// Non-x86_64 targets run the scalar kernels.
#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_available() -> bool {
    false
}

/// Name of the path that will run on this CPU (`"avx2"` or `"scalar"`).
pub fn active_path() -> &'static str {
    if avx2_available() {
        "avx2"
    } else {
        "scalar"
    }
}

/// Decode as many primary-table symbols as possible into `out`.
///
/// Returns with `state` describing exactly how much of `stream` was consumed.
/// The caller resumes with the canonical decoder for long codes and tails, so
/// this routine may stop early for any reason without changing behaviour.
pub fn unpack(
    table: &[u16],
    primary_bits: u32,
    stream: &[u8],
    expected_len: usize,
    out: &mut Vec<u8>,
    state: &mut UnpackState,
) {
    if avx2_available() {
        // SAFETY: `avx2_available()` proved AVX2 is present, so the
        // `#[target_feature]` contract of `unpack_avx2` holds. The intrinsics
        // only read within `stream` (bounds are re-checked before each load)
        // and write only through the safe `Vec` API.
        unsafe { unpack_avx2(table, primary_bits, stream, expected_len, out, state) };
    } else {
        unpack_scalar(table, primary_bits, stream, expected_len, out, state);
    }
}

/// Force the scalar kernel. Used by the differential test and by callers who
/// need a reproducible path; never selected implicitly when AVX2 exists.
pub fn unpack_scalar_forced(
    table: &[u16],
    primary_bits: u32,
    stream: &[u8],
    expected_len: usize,
    out: &mut Vec<u8>,
    state: &mut UnpackState,
) {
    unpack_scalar(table, primary_bits, stream, expected_len, out, state);
}

/// Byte-refill, one-probe scalar kernel.
fn unpack_scalar(
    table: &[u16],
    primary_bits: u32,
    stream: &[u8],
    expected_len: usize,
    out: &mut Vec<u8>,
    state: &mut UnpackState,
) {
    let primary_shift = 64 - primary_bits;
    let stream_len = stream.len();
    while out.len() < expected_len {
        if state.acc_bits < primary_bits {
            while state.acc_bits <= 56 && state.next_byte < stream_len {
                state.acc |= u64::from(stream[state.next_byte]) << (56 - state.acc_bits);
                state.next_byte += 1;
                state.acc_bits += 8;
            }
        }
        if state.acc_bits < primary_bits {
            return;
        }
        let entry = table[(state.acc >> primary_shift) as usize];
        let length = u32::from(entry >> 8);
        if length == 0 {
            return;
        }
        out.push(entry as u8);
        state.acc <<= length;
        state.acc_bits -= length;
        state.bits_consumed += length as usize;
    }
}

/// Eight-byte-refill, two-symbol-unrolled AVX2 kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2(
    table: &[u16],
    primary_bits: u32,
    stream: &[u8],
    expected_len: usize,
    out: &mut Vec<u8>,
    state: &mut UnpackState,
) {
    use std::arch::x86_64::{__m128i, _mm_cvtsi128_si64, _mm_loadl_epi64};

    let primary_shift = 64 - primary_bits;
    let stream_len = stream.len();
    while out.len() < expected_len {
        // Bulk refill: one unaligned 8-byte load instead of up to eight byte
        // loads. `k` whole bytes fit in the accumulator's free space.
        if state.acc_bits < primary_bits {
            while state.acc_bits <= 56 && state.next_byte + 8 <= stream_len {
                let ptr = stream.as_ptr().add(state.next_byte) as *const __m128i;
                let raw = _mm_cvtsi128_si64(_mm_loadl_epi64(ptr)) as u64;
                let word = raw.swap_bytes();
                let k = ((64 - state.acc_bits) / 8) as u32;
                let bytes = k * 8;
                state.acc |= (word >> (64 - bytes)) << (64 - state.acc_bits - bytes);
                state.next_byte += (k as usize).min(8);
                state.acc_bits += bytes;
            }
            while state.acc_bits <= 56 && state.next_byte < stream_len {
                state.acc |= u64::from(stream[state.next_byte]) << (56 - state.acc_bits);
                state.next_byte += 1;
                state.acc_bits += 8;
            }
        }
        if state.acc_bits < primary_bits {
            return;
        }

        let first = table[(state.acc >> primary_shift) as usize];
        let first_len = u32::from(first >> 8);
        if first_len == 0 {
            return;
        }

        // Second symbol in the same iteration when both prefixes are buffered
        // and the declared output still has room for it.
        let after_first = state.acc << first_len;
        let bits_after_first = state.acc_bits - first_len;
        if out.len() + 1 < expected_len && bits_after_first >= primary_bits {
            let second = table[(after_first >> primary_shift) as usize];
            let second_len = u32::from(second >> 8);
            if second_len != 0 {
                out.push(first as u8);
                out.push(second as u8);
                state.acc = after_first << second_len;
                state.acc_bits = bits_after_first - second_len;
                state.bits_consumed += (first_len + second_len) as usize;
                continue;
            }
        }

        out.push(first as u8);
        state.acc = after_first;
        state.acc_bits = bits_after_first;
        state.bits_consumed += first_len as usize;
    }
}

/// Join BF16 planes (`exponent_plane.len() == mantissa_plane.len() == out/2`).
///
/// The **selected** kernel is the byte-wise scalar form. LLVM auto-vectorises
/// it, and the measured court shows it is at least as fast as the hand-written
/// AVX2 kernel at every tested size (see `bench/RESULTS-decode-simd.md`), so
/// there is no reason to dispatch. The explicit AVX2 kernel is retained behind
/// [`join_bf16_avx2_path`], differential-tested against this one.
pub fn join_bf16(exponent_plane: &[u8], mantissa_plane: &[u8], out: &mut [u8]) {
    debug_assert_eq!(exponent_plane.len(), mantissa_plane.len());
    debug_assert_eq!(out.len(), exponent_plane.len() * 2);
    join_bf16_scalar(exponent_plane, mantissa_plane, out);
}

/// Explicit AVX2 BF16 join with runtime detection.
///
/// Not the selected kernel: the explicit intrinsics measured a tie at 16 MiB
/// and a loss on small inputs against the compiler-vectorised scalar form.
/// Kept because it is a real, tested alternative and because the differential
/// test must compare the two.
pub fn join_bf16_avx2_path(exponent_plane: &[u8], mantissa_plane: &[u8], out: &mut [u8]) {
    if avx2_available() {
        // SAFETY: AVX2 is present. `join_bf16_avx2` only reads the two planes
        // (bounds-checked by chunking) and writes `out` within its length.
        unsafe { join_bf16_avx2(exponent_plane, mantissa_plane, out) };
    } else {
        join_bf16_scalar(exponent_plane, mantissa_plane, out);
    }
}

/// Scalar BF16 plane join.
///
/// BF16 does not store the exponent byte next to a `sign|mantissa` byte; the
/// 16-bit word is `sign<<15 | exponent<<7 | mantissa`, so the exponent crosses
/// the byte boundary. `mantissa_plane[i]` is `sign<<7 | mantissa` and the two
/// output bytes are:
///
/// * low  = `(plane & 0x7f) | ((exponent & 1) << 7)`
/// * high = `((exponent >> 1) & 0x7f) | (plane & 0x80)`
pub fn join_bf16_scalar(exponent_plane: &[u8], mantissa_plane: &[u8], out: &mut [u8]) {
    for (index, (&exponent, &plane)) in exponent_plane.iter().zip(mantissa_plane).enumerate() {
        out[index * 2] = (plane & 0x7f) | ((exponent & 1) << 7);
        out[index * 2 + 1] = ((exponent >> 1) & 0x7f) | (plane & 0x80);
    }
}

/// AVX2 BF16 plane join: 16 elements (32 output bytes) per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn join_bf16_avx2(exponent_plane: &[u8], mantissa_plane: &[u8], out: &mut [u8]) {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_set_m128i, _mm256_storeu_si256, _mm_and_si128, _mm_loadu_si128,
        _mm_or_si128, _mm_set1_epi8, _mm_slli_epi16, _mm_srli_epi16, _mm_unpackhi_epi8,
        _mm_unpacklo_epi8,
    };

    let elements = exponent_plane.len();
    let mask_low = _mm_set1_epi8(0x7f);
    let mask_high = _mm_set1_epi8(0x80u8 as i8);
    let mask_one = _mm_set1_epi8(0x01);
    let mut index = 0usize;
    while index + 16 <= elements {
        let exponent = _mm_loadu_si128(exponent_plane.as_ptr().add(index) as *const __m128i);
        let plane = _mm_loadu_si128(mantissa_plane.as_ptr().add(index) as *const __m128i);
        // low = mantissa | (exponent bit0 << 7); the 16-bit shift puts each
        // element's bit0 into its own byte's bit7, so no cross-byte
        // contamination survives.
        let low = _mm_or_si128(
            _mm_and_si128(plane, mask_low),
            _mm_slli_epi16(_mm_and_si128(exponent, mask_one), 7),
        );
        // high = (exponent >> 1) & 0x7f | sign. The 0x7f mask removes the
        // adjacent element's bit0 that the 16-bit shift places in bit7.
        let high = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(exponent, 1), mask_low),
            _mm_and_si128(plane, mask_high),
        );
        let lo = _mm_unpacklo_epi8(low, high);
        let hi = _mm_unpackhi_epi8(low, high);
        let joined = _mm256_set_m128i(hi, lo);
        let dst = out.as_mut_ptr().add(index * 2) as *mut __m256i;
        _mm256_storeu_si256(dst, joined);
        index += 16;
    }
    while index < elements {
        let exponent = exponent_plane[index];
        let plane = mantissa_plane[index];
        out[index * 2] = (plane & 0x7f) | ((exponent & 1) << 7);
        out[index * 2 + 1] = ((exponent >> 1) & 0x7f) | (plane & 0x80);
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_is_reported() {
        let path = active_path();
        assert!(path == "avx2" || path == "scalar");
    }

    #[test]
    fn bf16_join_matches_scalar() {
        let exponent: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let mantissa: Vec<u8> = (0..=255u8).rev().cycle().take(4096).collect();
        let mut fast = vec![0u8; 8192];
        let mut slow = vec![0u8; 8192];
        join_bf16(&exponent, &mantissa, &mut fast);
        join_bf16_scalar(&exponent, &mantissa, &mut slow);
        if fast != slow {
            let at = fast.iter().zip(&slow).position(|(a, b)| a != b).unwrap();
            panic!(
                "path={} first mismatch at {at}: fast={:#04x} slow={:#04x}",
                active_path(),
                fast[at],
                slow[at]
            );
        }
        assert_eq!(fast, slow);
        // Explicit AVX2 must agree too.
        let mut explicit = vec![0u8; 8192];
        join_bf16_avx2_path(&exponent, &mantissa, &mut explicit);
        assert_eq!(explicit, fast);
        // Independent check against the documented BF16 bit layout.
        for index in 0..4096usize {
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
