//! Domain-separated BLAKE3 hashing.
//!
//! v1 uses exactly one construction:
//!
//! ```text
//! H(domain, data) = BLAKE3-256( u32le(|domain|) || domain || data )
//! ```
//!
//! Domain strings are ASCII, contain no NUL, and are versioned. The domain
//! tag is length-prefixed so no two domain/data splits can collide.

use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

/// Length of a v1 digest in bytes.
pub const HASH_LEN: usize = 32;

/// Raw chunk/plaintext bytes.
pub const DOMAIN_CHUNK: &str = "weft/chunk/v1";
/// Whole-file content manifest.
pub const DOMAIN_MANIFEST: &str = "weft/manifest/v1";
/// Structured frame value.
pub const DOMAIN_STRUCTURED: &str = "weft/structured/v1";
/// Selector canonical expression.
///
/// The identity-layer domains are generated from
/// `spec/schema/identity-v1.schema`; do not edit them here.
pub use crate::generated::{
    DOMAIN_CAPABILITY, DOMAIN_CAUSAL, DOMAIN_CONTEXT_ROOT, DOMAIN_SELECTOR,
};
/// Snapshot record.
pub const DOMAIN_SNAPSHOT: &str = "weft/snapshot/v1";
/// Commit record.
pub const DOMAIN_COMMIT: &str = "weft/store/commit/v1";
/// Relation assertion.
pub const DOMAIN_RELATION: &str = "weft/relation/v1";
/// Object version record.
pub const DOMAIN_VERSION: &str = "weft/version/v1";
/// Object identity seed.
pub const DOMAIN_OBJECT: &str = "weft/object/v1";
/// Tensor element payload binding (WEFT-AI-CODEC §2).
pub const DOMAIN_TENSOR_PAYLOAD: &str = "weft/tensor/payload/v1";
/// Token-sequence payload binding (WEFT-AI-CODEC §3).
pub const DOMAIN_TOKENS_PAYLOAD: &str = "weft/tokens/payload/v1";
/// Quantized-tensor payload binding (WEFT-AI-CODEC §4).
pub const DOMAIN_QUANTIZED_TENSOR_PAYLOAD: &str = "weft/quantized-tensor/payload/v1";
/// Float-tensor stored-payload binding (WEFT-AI-CODEC §5).
pub const DOMAIN_FLOAT_TENSOR_PAYLOAD: &str = "weft/float-tensor/payload/v1";
/// Sharded quantized tensor stored-payload binding (WEFT-AI-CODEC §11).
pub const DOMAIN_TENSOR_SHARDED_PAYLOAD: &str = "weft/tensor-sharded/payload/v1";
/// Independently decodable tensor-tile identity (WEFT-AI-CODEC §16).
pub const DOMAIN_TENSOR_TILE: &str = "weft/tensor/tile/v1";

/// A 32-byte Weft digest.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Hash32([u8; HASH_LEN]);

impl Hash32 {
    /// The all-zero digest. Useful only as a placeholder; it is never a valid
    /// content address for a value in practice.
    pub const ZERO: Hash32 = Hash32([0u8; HASH_LEN]);

    /// Wrap 32 raw bytes.
    pub const fn from_bytes(bytes: [u8; HASH_LEN]) -> Self {
        Hash32(bytes)
    }

    /// Borrow the raw bytes.
    pub const fn as_bytes(&self) -> &[u8; HASH_LEN] {
        &self.0
    }

    /// Consume into raw bytes.
    pub const fn into_bytes(self) -> [u8; HASH_LEN] {
        self.0
    }

    /// Lowercase hex rendering (the canonical display and directory filename).
    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }

    /// Parse exactly 32 bytes from a slice.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != HASH_LEN {
            return Err(Error::malformed(format!(
                "hash must be {HASH_LEN} bytes, got {}",
                bytes.len()
            )));
        }
        let mut out = [0u8; HASH_LEN];
        out.copy_from_slice(bytes);
        Ok(Hash32(out))
    }

    /// Parse 64 hex digits (either case).
    pub fn from_hex(text: &str) -> Result<Self> {
        if text.len() != HASH_LEN * 2 {
            return Err(Error::malformed(format!(
                "hash hex must be {} characters, got {}",
                HASH_LEN * 2,
                text.len()
            )));
        }
        if !text.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::malformed("hash hex contains non-hex characters"));
        }
        let decoded =
            hex::decode(text).map_err(|e| Error::malformed(format!("invalid hash hex: {e}")))?;
        Self::from_slice(&decoded)
    }

    /// Parse 64 lowercase hex digits, as required for directory filenames.
    pub fn from_lower_hex(text: &str) -> Result<Self> {
        if !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::malformed(
                "hash filename must be lowercase hexadecimal",
            ));
        }
        Self::from_hex(text)
    }
}

impl fmt::Display for Hash32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for Hash32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hash32({})", hex::encode(self.0))
    }
}

impl FromStr for Hash32 {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Hash32::from_hex(s)
    }
}

/// Domain-separated BLAKE3-256 digest.
///
/// Domain strings used by v1 are small ASCII constants. The `u32` length
/// prefix is encoded with a saturating conversion: a domain longer than
/// `u32::MAX` bytes cannot be represented by the format and is unreachable for
/// real domains, but this function must not panic.
pub fn hash_domain(domain: &str, data: &[u8]) -> Hash32 {
    let len = u32::try_from(domain.len()).unwrap_or(u32::MAX);
    let mut hasher = blake3::Hasher::new();
    hasher.update(&len.to_le_bytes());
    hasher.update(domain.as_bytes());
    hasher.update(data);
    Hash32(*hasher.finalize().as_bytes())
}

/// Checked variant of [`hash_domain`] that rejects domains which are not
/// representable in v1 (longer than `u32::MAX`) with `LimitExceeded`, and
/// rejects non-ASCII domains or domains containing NUL with `Malformed`.
pub fn try_hash_domain(domain: &str, data: &[u8]) -> Result<Hash32> {
    if domain.len() > u32::MAX as usize {
        return Err(Error::limit_exceeded("hash domain exceeds u32 length"));
    }
    if !domain.is_ascii() || domain.as_bytes().contains(&0) {
        return Err(Error::malformed("hash domain must be ASCII without NUL"));
    }
    Ok(hash_domain(domain, data))
}

/// Convenience: chunk/plaintext digest.
pub fn hash_bytes(data: &[u8]) -> Hash32 {
    hash_domain(DOMAIN_CHUNK, data)
}

/// Convenience: canonical selector bytes digest.
pub fn hash_selector(bytes: &[u8]) -> Hash32 {
    hash_domain(DOMAIN_SELECTOR, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_separation_changes_digest() {
        let a = hash_domain("a", b"data");
        let b = hash_domain("b", b"data");
        assert_ne!(a, b);
        let c = hash_domain("a", b"datb");
        assert_ne!(a, c);
    }

    #[test]
    fn hex_roundtrip() {
        let h = hash_domain(DOMAIN_CHUNK, b"hello");
        assert_eq!(h.to_hex().len(), 64);
        assert_eq!(Hash32::from_hex(&h.to_hex()).unwrap(), h);
        assert_eq!(Hash32::from_hex(&h.to_hex().to_uppercase()).unwrap(), h);
        assert!(Hash32::from_hex("00").is_err());
        assert!(Hash32::from_hex(&"zz".repeat(32)).is_err());
    }

    #[test]
    fn domain_validation() {
        assert!(try_hash_domain(DOMAIN_CHUNK, b"x").is_ok());
        assert_eq!(
            try_hash_domain("bad\0domain", b"x").unwrap_err().class(),
            crate::ErrorClass::Malformed
        );
        assert_eq!(
            try_hash_domain("non-ascii-\u{e9}", b"x")
                .unwrap_err()
                .class(),
            crate::ErrorClass::Malformed
        );
    }

    #[test]
    fn hand_computed_domain_prefix() {
        // The prefix bytes are exactly u32le(10) || "weft/chunk/v1".
        let mut manual = Vec::new();
        manual.extend_from_slice(&13u32.to_le_bytes());
        manual.extend_from_slice(b"weft/chunk/v1");
        manual.extend_from_slice(b"x");
        let expected = blake3::hash(&manual);
        assert_eq!(
            hash_domain(DOMAIN_CHUNK, b"x").as_bytes(),
            expected.as_bytes()
        );
    }
}
