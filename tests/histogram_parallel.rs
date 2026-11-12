//! R22.2 byte-identity court for the parallel exponent histogram.
//!
//! `float_tensor_exponent_histogram_with` splits a payload into whole-element
//! windows, accumulates each window's counts on its own thread and adds them.
//! Integer addition is associative and commutative, so the sum must equal the
//! single-pass histogram exactly — and because the R18 mode-9 size prediction is
//! a pure function of the histogram, the predicted length must be identical
//! too. This court proves both over every dtype, every thread count, window
//! boundaries at element granularity, and degenerate payloads.

use weftcodec::limits::Limits;
use weftcodec::modes::{
    self, float_tensor_encoded_len_from_histogram, float_tensor_exponent_histogram,
    float_tensor_exponent_histogram_with, FloatDtype,
};

fn dtypes() -> [FloatDtype; 3] {
    [FloatDtype::Bf16, FloatDtype::F16, FloatDtype::F32]
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// A payload whose exponent distribution is skewed like real weights.
fn payload(elements: usize, element_bytes: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed | 1);
    let mut bytes = Vec::with_capacity(elements * element_bytes);
    for index in 0..elements {
        // A few exponent values dominate, exactly the shape the histogram is
        // for.
        let exponent: u8 = match index % 16 {
            0 => 134,
            1..=11 => 133,
            12..=13 => 132,
            14 => 0,
            _ => 135,
        };
        for byte in 0..element_bytes {
            let value = if byte == element_bytes - 1 {
                exponent
            } else {
                (rng.next() & 0xFF) as u8
            };
            bytes.push(value);
        }
    }
    bytes
}

#[test]
fn threaded_histogram_equals_one_pass_for_every_dtype_and_thread_count() {
    for dtype in dtypes() {
        let element_bytes = dtype.element_size() as usize;
        for elements in [
            0usize, 1, 2, 3, 7, 8, 9, 255, 256, 257, 1000, 4095, 4096, 4097, 65_536, 65_537,
        ] {
            let bytes = payload(elements, element_bytes, 0x1234_5678 ^ elements as u64);
            let sequential = float_tensor_exponent_histogram(&bytes, dtype).expect("sequential");
            for threads in [1usize, 2, 3, 4, 7, 8, 16, 64] {
                for min_parallel in [0usize, 4096, 1 << 20] {
                    let parallel =
                        float_tensor_exponent_histogram_with(&bytes, dtype, threads, min_parallel)
                            .expect("parallel");
                    assert_eq!(
                        sequential, parallel,
                        "{dtype:?} {elements} elements, {threads} threads, min {min_parallel}"
                    );
                }
            }
            // The counts must also sum to the element count, so a window can
            // neither lose nor double-count an element.
            let total: u64 = sequential.iter().sum();
            assert_eq!(total, elements as u64, "{dtype:?} {elements}");
        }
    }
}

#[test]
fn the_mode9_prediction_is_unchanged_by_the_window_split() {
    let limits = Limits::default();
    for dtype in dtypes() {
        let element_bytes = dtype.element_size() as usize;
        let elements = 131_072usize;
        let bytes = payload(elements, element_bytes, 0xDEAD_BEEF);
        let shape = [elements as u64];
        let sequential = float_tensor_exponent_histogram(&bytes, dtype).expect("sequential");
        let one = float_tensor_encoded_len_from_histogram(dtype, &shape, &sequential, &limits)
            .expect("sequential prediction");
        for threads in [2usize, 3, 4, 8] {
            let parallel =
                float_tensor_exponent_histogram_with(&bytes, dtype, threads, 0).expect("parallel");
            let predicted =
                float_tensor_encoded_len_from_histogram(dtype, &shape, &parallel, &limits)
                    .expect("parallel prediction");
            assert_eq!(
                one, predicted,
                "the predicted mode-9 length must not depend on the thread count ({dtype:?})"
            );
        }
        // And the prediction is the truth: encode and compare the length.
        let frame = modes::encode_float_tensor(&bytes, dtype, &shape, &limits).expect("encode");
        assert_eq!(
            frame.len() as u64,
            one,
            "the histogram prediction must be exact for {dtype:?}"
        );
    }
}

#[test]
fn a_partial_element_payload_is_refused() {
    let err = float_tensor_exponent_histogram_with(&[0u8; 3], FloatDtype::Bf16, 4, 0)
        .expect_err("3 bytes is not a whole number of bf16 elements");
    assert_eq!(err.class(), weftcodec::ErrorClass::Malformed);
}
