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

/// One stream's packed primary table for the multi-stream kernel.
pub struct StreamTable<'a> {
    pub table: &'a [u16],
    pub primary_bits: u32,
}

/// Decode several independently coded streams with a wide-word kernel.
///
/// Returns `None` whenever the fast path cannot prove every stream to its
/// exact end (a long code, a tail shorter than the primary table, a
/// single-symbol table, trailing bytes, or non-zero padding). The caller then
/// falls back to the canonical [`crate::huffman::decode`] per stream, so the
/// fast path can never change an error class or a decoded byte.
///
/// The AVX2 build walks four streams in lockstep with 32-bit accumulators and
/// one `_mm256_i32gather_epi32` per four symbols. The scalar build uses the
/// existing byte-refill kernel per stream.
pub fn unpack_multi(
    tables: &[StreamTable<'_>],
    streams: &[&[u8]],
    expected: &[usize],
) -> Option<Vec<Vec<u8>>> {
    if tables.len() != streams.len() || tables.len() != expected.len() {
        return None;
    }
    let mut out = Vec::with_capacity(streams.len());
    let mut index = 0usize;
    if avx2_available() {
        while index + 4 <= streams.len() {
            // SAFETY: `avx2_available()` proved AVX2 is present.
            let group = unsafe {
                unpack_multi_group4_avx2(
                    [
                        &tables[index],
                        &tables[index + 1],
                        &tables[index + 2],
                        &tables[index + 3],
                    ],
                    [
                        streams[index],
                        streams[index + 1],
                        streams[index + 2],
                        streams[index + 3],
                    ],
                    [
                        expected[index],
                        expected[index + 1],
                        expected[index + 2],
                        expected[index + 3],
                    ],
                )
            };
            let group = group?;
            out.extend(group);
            index += 4;
        }
    }
    while index < streams.len() {
        out.push(unpack_multi_one(
            &tables[index],
            streams[index],
            expected[index],
        )?);
        index += 1;
    }
    Some(out)
}

/// Strict single-stream primary-table decode used for the remainder of a
/// multi-stream group and as the scalar fallback.
fn unpack_multi_one(
    table: &StreamTable<'_>,
    stream: &[u8],
    expected_len: usize,
) -> Option<Vec<u8>> {
    if table.primary_bits == 0 || table.primary_bits > 16 {
        return None;
    }
    let mut out = Vec::with_capacity(expected_len.min(1 << 20));
    let mut state = UnpackState::default();
    unpack_scalar(
        table.table,
        table.primary_bits,
        stream,
        expected_len,
        &mut out,
        &mut state,
    );
    if out.len() != expected_len || !stream_consumed_exactly(stream, &state) {
        return None;
    }
    Some(out)
}

/// True when `state` consumed the whole stream and the padding is zero.
fn stream_consumed_exactly(stream: &[u8], state: &UnpackState) -> bool {
    let consumed_bytes = state.bits_consumed.div_ceil(8);
    if consumed_bytes != stream.len() {
        return false;
    }
    let used_bits_in_last = state.bits_consumed % 8;
    if used_bits_in_last == 0 {
        return true;
    }
    let last = stream[consumed_bytes - 1];
    let mask = (1u8 << (8 - used_bits_in_last)) - 1;
    last & mask == 0
}

/// AVX2 4-stream lockstep primary-table walk.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_multi_group4_avx2(
    tables: [&StreamTable<'_>; 4],
    streams: [&[u8]; 4],
    expected: [usize; 4],
) -> Option<[Vec<u8>; 4]> {
    use std::arch::x86_64::{
        __m256i, _mm256_i32gather_epi32, _mm256_setr_epi32, _mm256_storeu_si256,
    };

    // One concatenated i32 table so a single gather base covers all four
    // streams; each lane's index is biased by its own table offset.
    let mut combined: Vec<i32> = Vec::new();
    let mut offsets = [0i32; 4];
    let mut bits = [0u32; 4];
    for lane in 0..4 {
        bits[lane] = tables[lane].primary_bits;
        if bits[lane] == 0 || bits[lane] > 12 {
            return None;
        }
        offsets[lane] = i32::try_from(combined.len()).ok()?;
        combined.extend(tables[lane].table.iter().map(|&entry| i32::from(entry)));
    }

    let mut acc = [0u32; 4];
    let mut acc_bits = [0u32; 4];
    let mut next_byte = [0usize; 4];
    let mut bits_consumed = [0usize; 4];
    let mut done = [false; 4];
    let mut out: [Vec<u8>; 4] = [
        Vec::with_capacity(expected[0].min(1 << 20)),
        Vec::with_capacity(expected[1].min(1 << 20)),
        Vec::with_capacity(expected[2].min(1 << 20)),
        Vec::with_capacity(expected[3].min(1 << 20)),
    ];
    for lane in 0..4 {
        if expected[lane] == 0 {
            return None;
        }
        refill32(
            &mut acc[lane],
            &mut acc_bits[lane],
            &mut next_byte[lane],
            streams[lane],
        );
    }

    let base = combined.as_ptr();
    loop {
        if done.iter().all(|flag| *flag) {
            break;
        }
        let mut index = [0i32; 4];
        for lane in 0..4 {
            if done[lane] {
                index[lane] = offsets[lane];
                continue;
            }
            if acc_bits[lane] < bits[lane] {
                return None;
            }
            let shift = 32 - bits[lane];
            let mask = (1u32 << bits[lane]) - 1;
            index[lane] = offsets[lane] + ((acc[lane] >> shift) & mask) as i32;
        }
        let index_vec = _mm256_setr_epi32(index[0], index[1], index[2], index[3], 0, 0, 0, 0);
        let gathered: __m256i = _mm256_i32gather_epi32::<4>(base, index_vec);
        let mut entries = [0i32; 8];
        _mm256_storeu_si256(entries.as_mut_ptr() as *mut __m256i, gathered);
        for lane in 0..4 {
            if done[lane] {
                continue;
            }
            let entry = entries[lane] as u32;
            let length = entry >> 8;
            if length == 0 || length > acc_bits[lane] {
                return None;
            }
            out[lane].push((entry & 0xff) as u8);
            acc[lane] <<= length;
            acc_bits[lane] -= length;
            bits_consumed[lane] += length as usize;
            if out[lane].len() == expected[lane] {
                done[lane] = true;
            }
        }
        for lane in 0..4 {
            if !done[lane] {
                refill32(
                    &mut acc[lane],
                    &mut acc_bits[lane],
                    &mut next_byte[lane],
                    streams[lane],
                );
            }
        }
    }
    for lane in 0..4 {
        let state = UnpackState {
            acc: u64::from(acc[lane]),
            acc_bits: acc_bits[lane],
            next_byte: next_byte[lane],
            bits_consumed: bits_consumed[lane],
        };
        if !stream_consumed_exactly(streams[lane], &state) {
            return None;
        }
    }
    Some(out)
}

/// Refill a 32-bit MSB accumulator one byte at a time while eight bits fit.
#[inline]
fn refill32(acc: &mut u32, acc_bits: &mut u32, next_byte: &mut usize, stream: &[u8]) {
    while *acc_bits <= 24 && *next_byte < stream.len() {
        *acc |= u32::from(stream[*next_byte]) << (24 - *acc_bits);
        *next_byte += 1;
        *acc_bits += 8;
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
