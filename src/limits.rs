//! Decoder/encoder resource limits (WEFTCODEC-FORMAT §10).
//!
//! Every limit is configurable *downward* through builder methods. Encoding and
//! decoding check limits before allocating; a frame that would exceed a limit
//! yields `LimitExceeded` even when otherwise well formed.

use crate::error::{Error, Result};

/// Default maximum encoded frame size (1 GiB).
pub const DEFAULT_MAX_FRAME_BYTES: u64 = 1 << 30;
/// Default maximum decoded plaintext size (4 GiB).
pub const DEFAULT_MAX_DECODED_BYTES: u64 = 4 << 30;
/// Default maximum dependency chain depth (8).
pub const DEFAULT_MAX_DEPENDENCY_DEPTH: u64 = 8;
/// Default maximum dependency table entries (64).
pub const DEFAULT_MAX_DEPENDENCY_COUNT: u64 = 64;
/// Default maximum bytes fetched while resolving dependencies (8 GiB).
pub const DEFAULT_MAX_TOTAL_FETCHED_BYTES: u64 = 8 << 30;
/// Default maximum chunk table entries (1,048,576).
pub const DEFAULT_MAX_CHUNKS: u64 = 1_048_576;
/// Default maximum dictionary size (64 MiB).
pub const DEFAULT_MAX_DICTIONARY_BYTES: u64 = 64 << 20;
/// Default maximum WCB nesting depth (64).
pub const DEFAULT_MAX_STRUCTURED_DEPTH: u64 = 64;
/// Default maximum WCB value/item count (1,000,000).
pub const DEFAULT_MAX_STRUCTURED_ITEMS: u64 = 1_000_000;
/// Default maximum string/bytes length (256 MiB).
pub const DEFAULT_MAX_STRING_BYTES: u64 = 256 << 20;
/// Default maximum delta operations (16,777,216).
pub const DEFAULT_MAX_DELTA_OPS: u64 = 16_777_216;
/// Default maximum tensor element count (2^32).
pub const DEFAULT_MAX_TENSOR_ELEMENTS: u64 = 1 << 32;
/// Default maximum tensor rank (8 dimensions, WEFT-AI-CODEC §2.1).
pub const DEFAULT_MAX_TENSOR_RANK: u64 = 8;
/// Default maximum token count in a token sequence (2^24 = 16,777,216).
/// Chosen so an in-memory `Vec<u64>` of ids is at most 128 MiB.
pub const DEFAULT_MAX_TOKENS: u64 = 1 << 24;
/// Default maximum selector source length (64 KiB).
pub const DEFAULT_MAX_SELECTOR_BYTES: u64 = 64 << 10;
/// Default maximum selector AST nesting depth (64).
pub const DEFAULT_MAX_SELECTOR_DEPTH: u64 = 64;
/// Default maximum selector field-path segments (16).
pub const DEFAULT_MAX_FIELD_SEGMENTS: u64 = 16;
/// Default maximum selector AST nodes (4,096).
pub const DEFAULT_MAX_SELECTOR_OPERANDS: u64 = 4_096;
/// Default maximum archive entries (1,000,000).
pub const DEFAULT_MAX_ARCHIVE_ENTRIES: u64 = 1_000_000;
/// Default maximum causal closure entries (4,096).
pub const DEFAULT_MAX_CAUSAL_ENTRIES: u64 = 4_096;
/// Default maximum SPARSE_DELTA non-zero runs (16,777,216).
///
/// A run costs one varint plus one element word, so the bound also caps the
/// mode-16 payload at roughly `2^24 * (storage_bytes + 1)` bytes. Mode 11 is the
/// dense representation; a delta with more non-zero words than this is
/// `LimitExceeded` under mode 16 by design.
pub const DEFAULT_MAX_SPARSE_RUNS: u64 = 16_777_216;
/// Default maximum elements in one engram row group (2^26 = 67,108,864).
///
/// One group is decoded in a single allocation; at 8-byte elements this bounds
/// a group at 512 MiB. Group **count** is bounded by `max_structured_items`.
pub const DEFAULT_MAX_ENGRAM_GROUP_ELEMENTS: u64 = 1 << 26;
/// Default maximum capability grant-list entries (4,096).
pub const DEFAULT_MAX_CAPABILITY_ENTRIES: u64 = 4_096;
/// Default maximum `weft://` URI length (64 KiB).
pub const DEFAULT_MAX_URI_BYTES: u64 = 64 << 10;

/// Resource limits for encoding and decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_frame_bytes: u64,
    pub max_decoded_bytes: u64,
    pub max_dependency_depth: u64,
    pub max_dependency_count: u64,
    pub max_total_fetched_bytes: u64,
    pub max_chunks: u64,
    pub max_dictionary_bytes: u64,
    pub max_structured_depth: u64,
    pub max_structured_items: u64,
    pub max_string_bytes: u64,
    pub max_delta_ops: u64,
    /// Tensor element count (WEFT-AI-CODEC §2).
    pub max_tensor_elements: u64,
    /// Tensor rank / shape dimensions (WEFT-AI-CODEC §2).
    pub max_tensor_rank: u64,
    /// Token count in a token sequence (WEFT-AI-CODEC §3).
    pub max_tokens: u64,
    /// Selector source bytes (WEFT-SELECTORS §1).
    pub max_selector_bytes: u64,
    /// Selector AST nesting depth (WEFT-SELECTORS §1).
    pub max_selector_depth: u64,
    /// Field path segments (WEFT-SELECTORS §1).
    pub max_field_segments: u64,
    /// Selector AST operands/nodes (WEFT-SELECTORS §5).
    pub max_selector_operands: u64,
    /// Archive container entry count (WEFTCODEC-FORMAT §7).
    pub max_archive_entries: u64,
    /// Causal closure entries (WEFT-CAUSAL §5).
    pub max_causal_entries: u64,
    /// SPARSE_DELTA non-zero runs (WEFT-AI-CODEC §17).
    pub max_sparse_runs: u64,
    /// Capability grant-list entries (WEFT-CAPABILITIES §1).
    pub max_capability_entries: u64,
    /// `weft://` URI length (WEFT-CONTEXT-ROOT §3.3).
    pub max_uri_bytes: u64,
    /// Elements in one engram row group (WEFT-ENGRAM §10).
    pub max_engram_group_elements: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            max_decoded_bytes: DEFAULT_MAX_DECODED_BYTES,
            max_dependency_depth: DEFAULT_MAX_DEPENDENCY_DEPTH,
            max_dependency_count: DEFAULT_MAX_DEPENDENCY_COUNT,
            max_total_fetched_bytes: DEFAULT_MAX_TOTAL_FETCHED_BYTES,
            max_chunks: DEFAULT_MAX_CHUNKS,
            max_dictionary_bytes: DEFAULT_MAX_DICTIONARY_BYTES,
            max_structured_depth: DEFAULT_MAX_STRUCTURED_DEPTH,
            max_structured_items: DEFAULT_MAX_STRUCTURED_ITEMS,
            max_string_bytes: DEFAULT_MAX_STRING_BYTES,
            max_delta_ops: DEFAULT_MAX_DELTA_OPS,
            max_tensor_elements: DEFAULT_MAX_TENSOR_ELEMENTS,
            max_tensor_rank: DEFAULT_MAX_TENSOR_RANK,
            max_tokens: DEFAULT_MAX_TOKENS,
            max_selector_bytes: DEFAULT_MAX_SELECTOR_BYTES,
            max_selector_depth: DEFAULT_MAX_SELECTOR_DEPTH,
            max_field_segments: DEFAULT_MAX_FIELD_SEGMENTS,
            max_selector_operands: DEFAULT_MAX_SELECTOR_OPERANDS,
            max_archive_entries: DEFAULT_MAX_ARCHIVE_ENTRIES,
            max_causal_entries: DEFAULT_MAX_CAUSAL_ENTRIES,
            max_sparse_runs: DEFAULT_MAX_SPARSE_RUNS,
            max_capability_entries: DEFAULT_MAX_CAPABILITY_ENTRIES,
            max_uri_bytes: DEFAULT_MAX_URI_BYTES,
            max_engram_group_elements: DEFAULT_MAX_ENGRAM_GROUP_ELEMENTS,
        }
    }
}

impl Limits {
    /// Fresh default limits.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_frame_bytes(mut self, value: u64) -> Self {
        self.max_frame_bytes = value;
        self
    }
    pub fn with_max_decoded_bytes(mut self, value: u64) -> Self {
        self.max_decoded_bytes = value;
        self
    }
    pub fn with_max_dependency_depth(mut self, value: u64) -> Self {
        self.max_dependency_depth = value;
        self
    }
    pub fn with_max_dependency_count(mut self, value: u64) -> Self {
        self.max_dependency_count = value;
        self
    }
    pub fn with_max_total_fetched_bytes(mut self, value: u64) -> Self {
        self.max_total_fetched_bytes = value;
        self
    }
    pub fn with_max_chunks(mut self, value: u64) -> Self {
        self.max_chunks = value;
        self
    }
    pub fn with_max_dictionary_bytes(mut self, value: u64) -> Self {
        self.max_dictionary_bytes = value;
        self
    }
    pub fn with_max_structured_depth(mut self, value: u64) -> Self {
        self.max_structured_depth = value;
        self
    }
    pub fn with_max_structured_items(mut self, value: u64) -> Self {
        self.max_structured_items = value;
        self
    }
    pub fn with_max_string_bytes(mut self, value: u64) -> Self {
        self.max_string_bytes = value;
        self
    }
    pub fn with_max_delta_ops(mut self, value: u64) -> Self {
        self.max_delta_ops = value;
        self
    }
    pub fn with_max_tensor_elements(mut self, value: u64) -> Self {
        self.max_tensor_elements = value;
        self
    }
    pub fn with_max_tensor_rank(mut self, value: u64) -> Self {
        self.max_tensor_rank = value;
        self
    }
    pub fn with_max_tokens(mut self, value: u64) -> Self {
        self.max_tokens = value;
        self
    }
    pub fn with_max_selector_bytes(mut self, value: u64) -> Self {
        self.max_selector_bytes = value;
        self
    }
    pub fn with_max_selector_depth(mut self, value: u64) -> Self {
        self.max_selector_depth = value;
        self
    }
    pub fn with_max_field_segments(mut self, value: u64) -> Self {
        self.max_field_segments = value;
        self
    }
    pub fn with_max_selector_operands(mut self, value: u64) -> Self {
        self.max_selector_operands = value;
        self
    }
    pub fn with_max_archive_entries(mut self, value: u64) -> Self {
        self.max_archive_entries = value;
        self
    }
    pub fn with_max_causal_entries(mut self, value: u64) -> Self {
        self.max_causal_entries = value;
        self
    }
    pub fn with_max_sparse_runs(mut self, value: u64) -> Self {
        self.max_sparse_runs = value;
        self
    }
    pub fn with_max_capability_entries(mut self, value: u64) -> Self {
        self.max_capability_entries = value;
        self
    }
    pub fn with_max_uri_bytes(mut self, value: u64) -> Self {
        self.max_uri_bytes = value;
        self
    }
    pub fn with_max_engram_group_elements(mut self, value: u64) -> Self {
        self.max_engram_group_elements = value;
        self
    }
    /// Alias for [`Limits::with_max_string_bytes`] used by some callers.
    pub fn with_max_bytes(mut self, value: u64) -> Self {
        self.max_string_bytes = value;
        self
    }
}

/// Check that a byte length is within a limit.
pub fn check_len(len: u64, max: u64, what: &str) -> Result<()> {
    if len > max {
        return Err(Error::limit_exceeded(format!(
            "{what} length {len} exceeds limit {max}"
        )));
    }
    Ok(())
}

/// Convert a `u64` length into `usize`, or report `LimitExceeded`.
pub fn to_usize(value: u64, what: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::limit_exceeded(format!("{what} length {value} does not fit usize")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorClass;

    #[test]
    fn defaults_match_spec() {
        let l = Limits::default();
        assert_eq!(l.max_frame_bytes, 1_073_741_824);
        assert_eq!(l.max_decoded_bytes, 4_294_967_296);
        assert_eq!(l.max_dependency_depth, 8);
        assert_eq!(l.max_dependency_count, 64);
        assert_eq!(l.max_total_fetched_bytes, 8_589_934_592);
        assert_eq!(l.max_chunks, 1_048_576);
        assert_eq!(l.max_dictionary_bytes, 67_108_864);
        assert_eq!(l.max_structured_depth, 64);
        assert_eq!(l.max_structured_items, 1_000_000);
        assert_eq!(l.max_string_bytes, 268_435_456);
        assert_eq!(l.max_delta_ops, 16_777_216);
        assert_eq!(l.max_tensor_elements, 4_294_967_296);
        assert_eq!(l.max_tensor_rank, 8);
        assert_eq!(l.max_tokens, 16_777_216);
        assert_eq!(l.max_sparse_runs, 16_777_216);
    }

    #[test]
    fn builders_and_checks() {
        let l = Limits::new().with_max_decoded_bytes(10);
        assert_eq!(l.max_decoded_bytes, 10);
        assert!(check_len(10, l.max_decoded_bytes, "x").is_ok());
        assert_eq!(
            check_len(11, l.max_decoded_bytes, "x").unwrap_err().class(),
            ErrorClass::LimitExceeded
        );
    }
}
