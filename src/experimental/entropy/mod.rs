//! Experimental entropy coders for AI byte planes.
//!
//! **This is a measurement module, not a wire-format module.** No persistent
//! encoder or decoder in `weftcodec` reaches it. The frozen
//! `huffman-canonical-v1` implementation is imported unchanged as the
//! baseline; the Tunstall and rANS coders below are experimental candidates
//! whose framing is private to this module.
//!
//! Every coder exposes `encode(&[u8]) -> Vec<u8>` and
//! `decode(&[u8], expected_len) -> Result<Vec<u8>>`, plus a
//! caller-buffer decode where it is meaningful. All coders are exactly
//! lossless on every input; malformed streams fail with a typed
//! [`ErrorClass`].
//!
//! ## Modules
//!
//! * [`huffman`] — the shipped `huffman-canonical-v1` coder (baseline).
//! * [`tunstall`] — variable-to-fixed coding with `k`-bit codewords.
//! * [`rans`] — static order-0 rANS with a 32-bit state.
//! * [`zstd`] — levels 3 and 19 via the existing dependency.
//! * [`raw`] — no coding.
//! * [`planes`] — extracts the exact plane bytes that the shipped modes 9,
//!   11, 12, and 15 carry, using the shipped encoders rather than a
//!   reimplementation.

use crate::error::Result;

pub mod huffman;
pub mod planes;
pub mod rans;
pub mod raw;
pub mod tunstall;
pub mod zstd;

/// A single experimental coder.
pub trait Coder {
    /// Stable identifier used in the court receipt.
    fn id(&self) -> &'static str;

    /// Encode one byte plane. The output is self-describing: it carries every
    /// table or dictionary the decoder needs.
    fn encode(&self, input: &[u8]) -> Vec<u8>;

    /// Decode exactly `expected_len` bytes. A wrong length, trailing bytes,
    /// or any malformed table is a typed error.
    fn decode(&self, encoded: &[u8], expected_len: usize) -> Result<Vec<u8>>;

    /// Decode into a caller-supplied buffer. The default delegates to
    /// [`Coder::decode`]; coders override it where a direct write is natural.
    fn decode_into(&self, encoded: &[u8], out: &mut [u8]) -> Result<()> {
        let decoded = self.decode(encoded, out.len())?;
        out.copy_from_slice(&decoded);
        Ok(())
    }
}

/// Upper bound on a caller-declared decoded length for the experimental
/// decoders (1 GiB). The court never exceeds ~103 MiB; the bound exists so a
/// mistaken `expected_len` cannot request an unbounded allocation. The caller
/// owns the declared length, exactly as with the shipped `huffman-canonical-v1`
/// decoder.
pub const MAX_EXPECTED_LEN: usize = 1 << 30;

/// Check a caller-supplied decoded length before allocating.
pub(crate) fn check_expected_len(expected_len: usize) -> Result<usize> {
    if expected_len > MAX_EXPECTED_LEN {
        return Err(crate::error::Error::limit_exceeded(format!(
            "experimental decoder expected_len {expected_len} exceeds the {MAX_EXPECTED_LEN}-byte experimental bound"
        )));
    }
    Ok(expected_len)
}

/// Append `length` bits of `code` (most significant bit first) to `out`.
pub(crate) fn append_bits(
    out: &mut Vec<u8>,
    buffer: &mut u64,
    pending: &mut u32,
    code: u32,
    length: u32,
) {
    debug_assert!(length <= 32);
    *buffer = (*buffer << length) | u64::from(code);
    *pending += length;
    while *pending >= 8 {
        *pending -= 8;
        out.push((*buffer >> *pending) as u8);
    }
    if *pending == 0 {
        *buffer = 0;
    } else {
        *buffer &= (1u64 << *pending) - 1;
    }
}
