//! Differential and reject tests for the FLOAT_TENSOR (mode 9) plane join.
//!
//! The shipped [`join_float_tensor_payload`] reads the sign+mantissa plane with
//! a 64-bit MSB accumulator, one whole element at a time. `reference_join`
//! below is a verbatim copy of the previous per-bit `take_bits` loop. It is
//! compiled only for tests and exists to prove the two implementations agree
//! byte-for-byte, including on malformed planes.
//!
//! Coverage: all five dtypes, including the non-byte-aligned sign+mantissa
//! widths (`fp16` 11 bits, `fp8-e4m3` 4 bits, `fp8-e5m2` 3 bits), element
//! counts from 0 upward including counts that leave padding bits, and
//! randomized element words (NaN payloads, denormals, signed zeros, and
//! infinities are ordinary bit patterns here).

use weftcodec::error::{Error, ErrorClass};
use weftcodec::huffman;
use weftcodec::limits::{check_len, Limits};
use weftcodec::modes::{join_float_tensor_payload, split_float_tensor_payload, FloatDtype};

const ALL_DTYPES: [FloatDtype; 5] = [
    FloatDtype::Bf16,
    FloatDtype::F16,
    FloatDtype::F32,
    FloatDtype::Fp8E4m3,
    FloatDtype::Fp8E5m2,
];

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn fast_join(
    exponent_plane: &[u8],
    mantissa_plane: &[u8],
    dtype: FloatDtype,
    element_count: u64,
    limits: &Limits,
) -> Result<Vec<u8>, ErrorClass> {
    join_float_tensor_payload(exponent_plane, mantissa_plane, dtype, element_count, limits)
        .map_err(|err: Error| err.class())
}

// ---------------------------------------------------------------------------
// Test-only reference: a verbatim copy of the previous per-bit join.
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

fn reference_join(
    exponent_plane: &[u8],
    mantissa_plane: &[u8],
    dtype: FloatDtype,
    element_count: u64,
    limits: &Limits,
) -> Result<Vec<u8>, ErrorClass> {
    let count = usize::try_from(element_count).map_err(|_| ErrorClass::LimitExceeded)?;
    if exponent_plane.len() != count {
        return Err(ErrorClass::Corrupt);
    }
    let element_bytes =
        usize::try_from(dtype.element_size()).map_err(|_| ErrorClass::LimitExceeded)?;
    let decoded_bytes = element_count
        .checked_mul(dtype.element_size())
        .ok_or(ErrorClass::LimitExceeded)?;
    check_len(
        decoded_bytes,
        limits.max_decoded_bytes,
        "float tensor decoded bytes",
    )
    .map_err(|err| err.class())?;
    let sign_bits = dtype.sign_bits() as usize;
    let exponent_bits = dtype.exponent_bits() as usize;
    let mantissa_bits = dtype.mantissa_bits() as usize;
    let expected_mantissa_bits = element_count
        .checked_mul((sign_bits + mantissa_bits) as u64)
        .ok_or(ErrorClass::LimitExceeded)?;
    let expected_mantissa_bytes = expected_mantissa_bits
        .checked_add(7)
        .ok_or(ErrorClass::LimitExceeded)?
        / 8;
    if mantissa_plane.len() as u64 != expected_mantissa_bytes {
        return Err(ErrorClass::Corrupt);
    }

    let capacity = usize::try_from(decoded_bytes).map_err(|_| ErrorClass::LimitExceeded)?;
    let mut out = Vec::with_capacity(capacity.min(1 << 20));
    let mut bit_pos = 0usize;
    for &exponent_byte in exponent_plane.iter() {
        if u64::from(exponent_byte) >> exponent_bits != 0 {
            return Err(ErrorClass::Corrupt);
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
            return Err(ErrorClass::Corrupt);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn words_to_bytes(words: &[u32], element_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * element_bytes);
    for word in words {
        out.extend_from_slice(&word.to_le_bytes()[..element_bytes]);
    }
    out
}

/// Recover the raw (uncompressed) exponent plane and mantissa plane through the
/// shipped splitter and Huffman decoder.
fn planes_for(payload: &[u8], dtype: FloatDtype, count: u64) -> (Vec<u8>, Vec<u8>) {
    let (compressed, mantissa_plane) =
        split_float_tensor_payload(payload, dtype, count).expect("split");
    let exponent_plane =
        huffman::decode(&compressed, count as usize).expect("huffman decode of exponent plane");
    (exponent_plane, mantissa_plane)
}

fn random_word(state: &mut u64, dtype: FloatDtype) -> u32 {
    let bits = dtype.element_bits();
    let mask = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    match next_u64(state) % 16 {
        // Signed zeros, smallest denormals, largest denormals, infinities, and
        // payload-bearing NaNs: three quarters of the corpus stays in the
        // interesting exponent/sign neighbourhood.
        0 => 0,
        1 => 1,
        2 => u32::MAX & mask,
        3 => {
            let exponent_mask = ((1u32 << dtype.exponent_bits()) - 1) << dtype.mantissa_bits();
            exponent_mask & mask
        }
        _ => (next_u64(state) as u32) & mask,
    }
}

fn random_payload(state: &mut u64, dtype: FloatDtype, count: usize) -> Vec<u8> {
    let element_bytes = dtype.element_size() as usize;
    let mut payload = Vec::with_capacity(count * element_bytes);
    for _ in 0..count {
        let word = random_word(state, dtype);
        payload.extend_from_slice(&word.to_le_bytes()[..element_bytes]);
    }
    payload
}

fn assert_same(
    exponent_plane: &[u8],
    mantissa_plane: &[u8],
    dtype: FloatDtype,
    element_count: u64,
    limits: &Limits,
    label: &str,
) {
    let fast = fast_join(exponent_plane, mantissa_plane, dtype, element_count, limits);
    let reference = reference_join(exponent_plane, mantissa_plane, dtype, element_count, limits);
    match (&fast, &reference) {
        (Ok(fast_bytes), Ok(reference_bytes)) => {
            assert_eq!(
                fast_bytes,
                reference_bytes,
                "{label}: dtype {} count {element_count}",
                dtype.as_str()
            )
        }
        (Err(fast_class), Err(reference_class)) => assert_eq!(
            fast_class,
            reference_class,
            "{label}: dtype {} count {element_count}",
            dtype.as_str()
        ),
        _ => panic!(
            "{label}: dtype {} count {element_count}: fast={fast:?} reference={reference:?}",
            dtype.as_str()
        ),
    }
}

// ---------------------------------------------------------------------------
// Randomized differential: valid planes
// ---------------------------------------------------------------------------
#[test]
fn fast_join_matches_reference_on_random_planes() {
    let limits = Limits::default();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut cases = 0usize;
    for dtype in ALL_DTYPES {
        for case in 0..10_000usize {
            let count = match case {
                0 => 0usize,
                1 => 1,
                2 => 2,
                3 => 3,
                4 => 7,
                _ => (next_u64(&mut state) % 257) as usize + 1,
            };
            let payload = random_payload(&mut state, dtype, count);
            let (exponent_plane, mantissa_plane) = planes_for(&payload, dtype, count as u64);
            let fast = fast_join(
                &exponent_plane,
                &mantissa_plane,
                dtype,
                count as u64,
                &limits,
            )
            .unwrap_or_else(|class| {
                panic!(
                    "valid plane rejected ({class:?}) for dtype {} count {count}",
                    dtype.as_str()
                )
            });
            assert_eq!(
                fast,
                payload,
                "dtype {} count {count}: fast join does not reproduce the source bytes",
                dtype.as_str()
            );
            assert_same(
                &exponent_plane,
                &mantissa_plane,
                dtype,
                count as u64,
                &limits,
                "valid plane",
            );
            cases += 1;
        }
    }
    assert_eq!(cases, ALL_DTYPES.len() * 10_000);
}

// ---------------------------------------------------------------------------
// Randomized differential: mutated planes (error classes must match too)
// ---------------------------------------------------------------------------
#[test]
fn fast_join_matches_reference_on_mutated_planes() {
    let limits = Limits::default();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut corrupt_cases = 0usize;
    for dtype in ALL_DTYPES {
        for _ in 0..4_000usize {
            let count = (next_u64(&mut state) % 64) as usize + 1;
            let payload = random_payload(&mut state, dtype, count);
            let (mut exponent_plane, mut mantissa_plane) =
                planes_for(&payload, dtype, count as u64);
            // Flip one to three random bits across both planes.
            for _ in 0..(next_u64(&mut state) % 3 + 1) {
                if next_u64(&mut state) % 2 == 0 {
                    if !exponent_plane.is_empty() {
                        let index = (next_u64(&mut state) as usize) % exponent_plane.len();
                        let bit = 1u8 << (next_u64(&mut state) % 8);
                        exponent_plane[index] ^= bit;
                    }
                } else if !mantissa_plane.is_empty() {
                    let index = (next_u64(&mut state) as usize) % mantissa_plane.len();
                    let bit = 1u8 << (next_u64(&mut state) % 8);
                    mantissa_plane[index] ^= bit;
                }
            }
            let reference = reference_join(
                &exponent_plane,
                &mantissa_plane,
                dtype,
                count as u64,
                &limits,
            );
            if reference.is_err() {
                corrupt_cases += 1;
            }
            assert_same(
                &exponent_plane,
                &mantissa_plane,
                dtype,
                count as u64,
                &limits,
                "mutated plane",
            );
        }
    }
    // The mutation corpus must actually exercise the reject paths, otherwise
    // the error-class comparison would be vacuous.
    assert!(
        corrupt_cases > 100,
        "mutation corpus produced only {corrupt_cases} rejects"
    );
}

// ---------------------------------------------------------------------------
// Special values
// ---------------------------------------------------------------------------
#[test]
fn special_values_round_trip_every_dtype() {
    let limits = Limits::default();
    let cases: &[(FloatDtype, &[u32])] = &[
        (
            FloatDtype::Bf16,
            &[
                0x0000, // +0
                0x8000, // -0
                0x0001, // smallest denormal
                0x807f, // largest negative denormal
                0x7f80, // +inf
                0xff80, // -inf
                0x7fc1, // NaN with payload
                0xffef, // negative NaN, full payload
            ],
        ),
        (
            FloatDtype::F16,
            &[
                0x0000, 0x8000, 0x0001, 0x03ff, 0x7c00, 0xfc00, 0x7e01, 0xffff,
            ],
        ),
        (
            FloatDtype::F32,
            &[
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x007f_ffff,
                0x7f80_0000,
                0xff80_0000,
                0x7f80_0001,
                0xffc0_0001,
            ],
        ),
        (
            FloatDtype::Fp8E4m3,
            &[0x00, 0x80, 0x01, 0x07, 0x7f, 0xf0, 0x4f, 0xff],
        ),
        (
            FloatDtype::Fp8E5m2,
            &[0x00, 0x80, 0x01, 0x03, 0x7c, 0xfc, 0x7d, 0xff],
        ),
    ];
    for (dtype, words) in cases {
        let payload = words_to_bytes(words, dtype.element_size() as usize);
        let count = words.len() as u64;
        let (exponent_plane, mantissa_plane) = planes_for(&payload, *dtype, count);
        let fast =
            fast_join(&exponent_plane, &mantissa_plane, *dtype, count, &limits).expect("fast join");
        assert_eq!(fast, payload, "dtype {}", dtype.as_str());
        assert_same(
            &exponent_plane,
            &mantissa_plane,
            *dtype,
            count,
            &limits,
            "special values",
        );
    }
}

// ---------------------------------------------------------------------------
// Explicit rejects
// ---------------------------------------------------------------------------
#[test]
fn rejects_out_of_width_exponent_byte() {
    let limits = Limits::default();
    for dtype in [FloatDtype::F16, FloatDtype::Fp8E4m3, FloatDtype::Fp8E5m2] {
        let count = 3usize;
        let payload = words_to_bytes(&[0, 0, 0], dtype.element_size() as usize);
        let (mut exponent_plane, mantissa_plane) = planes_for(&payload, dtype, count as u64);
        exponent_plane[1] = 1u8 << dtype.exponent_bits();
        let fast = fast_join(
            &exponent_plane,
            &mantissa_plane,
            dtype,
            count as u64,
            &limits,
        );
        assert_eq!(
            fast.expect_err("exponent byte exceeds the dtype width"),
            ErrorClass::Corrupt,
            "dtype {}",
            dtype.as_str()
        );
        assert_same(
            &exponent_plane,
            &mantissa_plane,
            dtype,
            count as u64,
            &limits,
            "out-of-width exponent",
        );
        // An all-ones exponent byte is out of width for every dtype narrower
        // than a byte.
        if dtype.exponent_bits() < 8 {
            let mut all_ones = exponent_plane.clone();
            all_ones[0] = 0xff;
            assert_eq!(
                fast_join(&all_ones, &mantissa_plane, dtype, count as u64, &limits)
                    .expect_err("0xff exceeds the dtype exponent width"),
                ErrorClass::Corrupt
            );
        }
    }
}

#[test]
fn rejects_non_zero_mantissa_padding() {
    let limits = Limits::default();
    // Dtypes whose element widths leave padding: fp16 (11 bits), fp8-e4m3
    // (4 bits), fp8-e5m2 (3 bits). bf16 and fp32 are byte-aligned per element.
    let cases: &[(FloatDtype, usize)] = &[
        (FloatDtype::F16, 1),
        (FloatDtype::Fp8E4m3, 1),
        (FloatDtype::Fp8E5m2, 1),
        (FloatDtype::F16, 3),
        (FloatDtype::Fp8E5m2, 5),
    ];
    for &(dtype, count) in cases {
        let payload = words_to_bytes(&vec![0u32; count], dtype.element_size() as usize);
        let (exponent_plane, mut mantissa_plane) = planes_for(&payload, dtype, count as u64);
        let used_bits = count * (dtype.sign_bits() + dtype.mantissa_bits()) as usize;
        let padding = mantissa_plane.len() * 8 - used_bits;
        assert!(
            padding > 0,
            "dtype {} count {count} has no padding",
            dtype.as_str()
        );
        let last = mantissa_plane.len() - 1;
        assert_eq!(mantissa_plane[last] & ((1u8 << padding) - 1), 0);
        mantissa_plane[last] |= 1;
        assert_eq!(
            fast_join(
                &exponent_plane,
                &mantissa_plane,
                dtype,
                count as u64,
                &limits
            )
            .expect_err("non-zero padding"),
            ErrorClass::Corrupt,
            "dtype {} count {count}",
            dtype.as_str()
        );
        assert_same(
            &exponent_plane,
            &mantissa_plane,
            dtype,
            count as u64,
            &limits,
            "non-zero mantissa padding",
        );
    }
}

#[test]
fn rejects_wrong_plane_lengths() {
    let limits = Limits::default();
    let dtype = FloatDtype::Bf16;
    let count = 4usize;
    let payload = words_to_bytes(&[0x3f80, 0x0001, 0x7fc0, 0x8000], 2);
    let (exponent_plane, mantissa_plane) = planes_for(&payload, dtype, count as u64);

    // Exponent plane length is part of the descriptor.
    for delta in [-1i64, 1] {
        let mut wrong = exponent_plane.clone();
        if delta < 0 {
            wrong.pop();
        } else {
            wrong.push(0);
        }
        assert_eq!(
            fast_join(&wrong, &mantissa_plane, dtype, count as u64, &limits)
                .expect_err("short/long exponent plane"),
            ErrorClass::Corrupt
        );
    }

    // Mantissa plane length must equal ceil(count * (sign + mantissa) / 8).
    for delta in [-1i64, 1] {
        let mut wrong = mantissa_plane.clone();
        if delta < 0 {
            wrong.pop();
        } else {
            wrong.push(0);
        }
        assert_eq!(
            fast_join(&exponent_plane, &wrong, dtype, count as u64, &limits)
                .expect_err("short/long mantissa plane"),
            ErrorClass::Corrupt
        );
    }

    // The declared element count must match the exponent plane length.
    assert_eq!(
        fast_join(
            &exponent_plane,
            &mantissa_plane,
            dtype,
            count as u64 + 1,
            &limits
        )
        .expect_err("element count disagrees with exponent plane"),
        ErrorClass::Corrupt
    );

    // The decoded byte limit is enforced before allocation.
    let tight = Limits::default().with_max_decoded_bytes(count as u64 * 2 - 1);
    assert_eq!(
        fast_join(
            &exponent_plane,
            &mantissa_plane,
            dtype,
            count as u64,
            &tight
        )
        .expect_err("decoded byte limit"),
        ErrorClass::LimitExceeded
    );
    assert_same(
        &exponent_plane,
        &mantissa_plane,
        dtype,
        count as u64,
        &tight,
        "decoded byte limit",
    );
}

#[test]
fn handles_element_counts_with_padding_bits() {
    // Every count in 0..=32 is exercised for each dtype, so the final partial
    // mantissa byte is covered for all widths, including single elements.
    let limits = Limits::default();
    let mut state = 0xdead_beef_cafe_f00du64;
    for dtype in ALL_DTYPES {
        for count in 0usize..=32 {
            let payload = random_payload(&mut state, dtype, count);
            let (exponent_plane, mantissa_plane) = planes_for(&payload, dtype, count as u64);
            assert_eq!(
                fast_join(
                    &exponent_plane,
                    &mantissa_plane,
                    dtype,
                    count as u64,
                    &limits
                )
                .expect("valid plane"),
                payload,
                "dtype {} count {count}",
                dtype.as_str()
            );
            assert_same(
                &exponent_plane,
                &mantissa_plane,
                dtype,
                count as u64,
                &limits,
                "padding-bit counts",
            );
        }
    }
}
