//! **EXPERIMENTAL — not part of the WeftCodec wire contract.**
//!
//! This module tree exists only to measure candidate entropy coders for a
//! future format revision. Nothing in [`crate`] calls into it from a
//! persistent encode or decode path: the frozen `huffman-canonical-v1` coder
//! remains the only coder used by modes 9–15, and no new wire mode is defined
//! here.
//!
//! The code is additive and may change or disappear without a format bump.
//! Do not use it to produce or consume committed `WFT1` frames.

pub mod entropy;
