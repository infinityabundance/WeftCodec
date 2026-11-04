//! WeftCodec v1 — deterministic, lossless, context-conditioned codec.
//!
//! This crate implements the `weftcodec-format 1.0` wire format from
//! `spec/WEFTCODEC-FORMAT.md` and the `weft-selectors 1.0` language from
//! `spec/WEFT-SELECTORS.md`:
//!
//! * [`hash`] — domain-separated BLAKE3-256 digests.
//! * [`varint`] — minimal LEB128 with strict canonical validation.
//! * [`wcb`] — Weft Canonical Bytes, a strict deterministic structured format.
//! * [`frame`] — the 96-byte WFT1 header, dependency and chunk tables, trailer.
//! * [`modes`] — RAW, ZSTD, REFERENCE, DELTA, DICTIONARY, STRUCTURED.
//! * [`dependency`] — resolvers, frame sources, recursive DAG decoding.
//! * [`archive`] — the self-contained WFA1 transport container.
//! * [`selector`] — WeftQL parser, normalization, canonical selector identity.
//! * [`context`] — Context Root, causal closure, capability scope, `weft://`.
//! * [`stream`] — chunked encoding and random-range reads.
//!
//! All parsing of untrusted input returns `Result`; the crate forbids unsafe
//! code and has no panic paths on malformed input.

#![forbid(unsafe_code)]

pub mod adversarial;
pub mod archive;
pub mod conformance;
pub mod context;
pub mod dependency;
pub mod error;
pub mod frame;
pub mod generated;
pub mod hash;
pub mod huffman;
pub mod limits;
pub mod modes;
pub mod selector;
pub mod stream;
pub mod varint;
pub mod wcb;

pub use error::{Error, ErrorClass, Result};
pub use frame::{Dependency, Frame, FrameHeader, Mode};
pub use hash::Hash32;
pub use limits::Limits;
