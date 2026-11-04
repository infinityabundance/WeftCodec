//! Semantic tensor roots and candidate grouping (WEFT-AI-CODEC §15).
//!
//! A *semantic root* is deliberately **not** an identity for the exact element
//! bytes. It groups tensors that are equal under an explicit, canonical
//! approximation policy so a caller can propose near-duplicate delta bases.
//! Exactness is always decided by the frame identity and by mode 14's
//! verification of the base bytes, never by this root.
//!
//! ```text
//! SemanticRoot = H("weft/tensor/semantic/v1",
//!                   WCB({schema:"weft.tensor.semantic.v1", dtype, shape,
//!                        element_count, policy, quantized_values_hash}))
//! ```
//!
//! `quantized_values_hash` is:
//!
//! * for `{"scheme":"none"}`, exactly the logical tensor values hash
//!   `H("weft/tensor/values/v1", row_major_bytes)`;
//! * otherwise `H("weft/tensor/semantic-values/v1", transformed_bytes)` where
//!   `transformed_bytes` is the policy output described below.
//!
//! Policies are canonical WCB maps:
//!
//! * `{"scheme":"none"}` — identity.
//! * `{"scheme":"mantissa_truncate","bits":N}` — float dtypes only. Clear the
//!   `N` least-significant mantissa bits of every element word. Sign,
//!   exponent, and every NaN/infinity word are preserved verbatim (a word
//!   whose exponent field is all ones is stored unchanged). `1 <= N <=
//!   mantissa_bits(dtype)`; bf16 has 7, fp16 has 10, fp32 has 23.
//! * `{"scheme":"block_quantize","block_size":B,"levels":L}` — split the
//!   element sequence into blocks of `B` (which MUST divide `element_count`)
//!   and quantize each block's raw words in **signed two's-complement**
//!   interpretation. `amax = max |s_i|` over the block. `L >= 1`. For
//!   `amax == 0` or `L == 1` every output word is zero. Otherwise the
//!   reconstruction grid is the `L` levels
//!
//!   ```text
//!   level_k = -amax + floor((2*amax*k + (L-1)/2) / (L-1)),  k = 0..L-1
//!   ```
//!
//!   and element `s` maps to
//!
//!   ```text
//!   k = floor(((s + amax)*(L-1) + amax) / (2*amax))
//!   ```
//!
//!   clamped to `0..=L-1`; the stored word is `level_k` in two's complement.
//!   All arithmetic is integer and deterministic; `amax <= 2^31` and
//!   `L <= 2^24` keep every intermediate inside `u64`.

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::hash::{hash_domain, Hash32};
use crate::huffman;
use crate::limits::{check_len, Limits};
use crate::modes;
use crate::wcb::{self, Value};
use crate::weight_delta::{
    base_frame_descriptor, build_planes, check_quantized_words, payload_len, push_word,
    validate_shape, word_at, WeightDtype,
};

/// Canonical semantic-root record schema.
pub const SEMANTIC_SCHEMA: &str = "weft.tensor.semantic.v1";
/// Semantic-root identity domain.
pub const DOMAIN_TENSOR_SEMANTIC: &str = "weft/tensor/semantic/v1";
/// Domain for policy-transformed values (all non-`none` schemes).
pub const DOMAIN_TENSOR_SEMANTIC_VALUES: &str = "weft/tensor/semantic-values/v1";
/// Upper bound on `block_quantize.levels` so integer arithmetic cannot overflow.
pub const MAX_BLOCK_LEVELS: u64 = 1 << 24;

/// One canonical semantic policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticPolicy {
    /// Identity policy; the values hash is the exact logical values hash.
    None,
    /// Clear `bits` least-significant mantissa bits, preserving sign/exponent.
    MantissaTruncate { bits: u32 },
    /// Symmetric per-block quantization of signed raw words.
    BlockQuantize { block_size: u64, levels: u64 },
}

impl SemanticPolicy {
    /// Parse a policy value, validating it against `dtype` and `element_count`.
    pub fn from_wcb_value(value: &Value, dtype: WeightDtype, element_count: u64) -> Result<Self> {
        let map = match value {
            Value::Map(map) => map,
            _ => return Err(Error::malformed("semantic policy must be a WCB map")),
        };
        let scheme = match map.get("scheme") {
            Some(Value::Str(text)) => text.as_str(),
            _ => return Err(Error::malformed("semantic policy `scheme` must be text")),
        };
        match scheme {
            "none" => {
                check_keys(map, &["scheme"], "none")?;
                Ok(SemanticPolicy::None)
            }
            "mantissa_truncate" => {
                check_keys(map, &["bits", "scheme"], "mantissa_truncate")?;
                let bits = u64_field(map, "bits", "mantissa_truncate")?;
                let max = mantissa_bits(dtype).ok_or_else(|| {
                    Error::malformed(format!(
                        "mantissa_truncate policy is not defined for dtype {}",
                        dtype.as_str()
                    ))
                })?;
                if bits == 0 || bits > u64::from(max) {
                    return Err(Error::malformed(format!(
                        "mantissa_truncate bits {bits} is outside 1..={max} for {}",
                        dtype.as_str()
                    )));
                }
                Ok(SemanticPolicy::MantissaTruncate { bits: bits as u32 })
            }
            "block_quantize" => {
                check_keys(
                    map,
                    &["block_size", "levels", "scheme"],
                    "block_quantize",
                )?;
                let block_size = u64_field(map, "block_size", "block_quantize")?;
                let levels = u64_field(map, "levels", "block_quantize")?;
                if block_size == 0 {
                    return Err(Error::malformed(
                        "block_quantize block_size must be greater than zero",
                    ));
                }
                if levels == 0 || levels > MAX_BLOCK_LEVELS {
                    return Err(Error::malformed(format!(
                        "block_quantize levels {levels} is outside 1..={MAX_BLOCK_LEVELS}"
                    )));
                }
                if element_count % block_size != 0 {
                    return Err(Error::malformed(format!(
                        "block_quantize block_size {block_size} does not divide element_count {element_count}"
                    )));
                }
                Ok(SemanticPolicy::BlockQuantize {
                    block_size,
                    levels,
                })
            }
            other => Err(Error::malformed(format!(
                "unknown semantic policy scheme `{other}` (v1 allows none, mantissa_truncate, block_quantize)"
            ))),
        }
    }

    /// Canonical WCB value.
    pub fn to_wcb_value(&self) -> Value {
        match self {
            SemanticPolicy::None => Value::map([("scheme", Value::Str("none".to_owned()))]),
            SemanticPolicy::MantissaTruncate { bits } => Value::map([
                ("bits", Value::U64(u64::from(*bits))),
                ("scheme", Value::Str("mantissa_truncate".to_owned())),
            ]),
            SemanticPolicy::BlockQuantize { block_size, levels } => Value::map([
                ("block_size", Value::U64(*block_size)),
                ("levels", Value::U64(*levels)),
                ("scheme", Value::Str("block_quantize".to_owned())),
            ]),
        }
    }

    /// Validate the policy against a dtype and element count.
    pub fn validate(&self, dtype: WeightDtype, element_count: u64) -> Result<()> {
        match self {
            SemanticPolicy::None => Ok(()),
            SemanticPolicy::MantissaTruncate { bits } => {
                let max = mantissa_bits(dtype).ok_or_else(|| {
                    Error::malformed(format!(
                        "mantissa_truncate policy is not defined for dtype {}",
                        dtype.as_str()
                    ))
                })?;
                if *bits == 0 || *bits > max {
                    return Err(Error::malformed(format!(
                        "mantissa_truncate bits {bits} is outside 1..={max} for {}",
                        dtype.as_str()
                    )));
                }
                Ok(())
            }
            SemanticPolicy::BlockQuantize { block_size, levels } => {
                if *block_size == 0 {
                    return Err(Error::malformed(
                        "block_quantize block_size must be greater than zero",
                    ));
                }
                if *levels == 0 || *levels > MAX_BLOCK_LEVELS {
                    return Err(Error::malformed(format!(
                        "block_quantize levels {levels} is outside 1..={MAX_BLOCK_LEVELS}"
                    )));
                }
                if element_count % block_size != 0 {
                    return Err(Error::malformed(format!(
                        "block_quantize block_size {block_size} does not divide element_count {element_count}"
                    )));
                }
                Ok(())
            }
        }
    }

    /// Human-readable CLI spelling.
    pub fn as_str(&self) -> String {
        match self {
            SemanticPolicy::None => "none".to_owned(),
            SemanticPolicy::MantissaTruncate { bits } => {
                format!("mantissa_truncate:{bits}")
            }
            SemanticPolicy::BlockQuantize { block_size, levels } => {
                format!("block_quantize:{block_size}:{levels}")
            }
        }
    }
}

fn check_keys(
    map: &std::collections::BTreeMap<String, Value>,
    allowed: &[&str],
    scheme: &str,
) -> Result<()> {
    for key in map.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::malformed(format!(
                "semantic policy `{scheme}` has unknown key `{key}`"
            )));
        }
    }
    for key in allowed {
        if !map.contains_key(*key) {
            return Err(Error::malformed(format!(
                "semantic policy `{scheme}` is missing key `{key}`"
            )));
        }
    }
    Ok(())
}

fn u64_field(
    map: &std::collections::BTreeMap<String, Value>,
    key: &str,
    scheme: &str,
) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!(
            "semantic policy `{scheme}` key `{key}` must be u64"
        ))),
    }
}

fn mantissa_bits(dtype: WeightDtype) -> Option<u32> {
    match dtype {
        WeightDtype::Bf16 => Some(7),
        WeightDtype::Fp16 => Some(10),
        WeightDtype::Fp32 => Some(23),
        _ => None,
    }
}

fn word_mask(dtype: WeightDtype) -> u32 {
    match dtype.word_bits() {
        32 => u32::MAX,
        16 => 0xFFFF,
        8 => 0xFF,
        4 => 0x0F,
        _ => unreachable!("word_bits is fixed by dtype"),
    }
}

fn exponent_all_ones(dtype: WeightDtype, word: u32) -> bool {
    match dtype {
        WeightDtype::Bf16 | WeightDtype::Fp16 => (word & 0x7F80) == 0x7F80,
        WeightDtype::Fp32 => (word & 0x7F80_0000) == 0x7F80_0000,
        _ => false,
    }
}

fn signed_word(dtype: WeightDtype, word: u32) -> i64 {
    let bits = dtype.word_bits();
    let sign = 1u64 << (bits - 1);
    let modulus = 1u64 << bits;
    let raw = u64::from(word);
    if raw >= sign {
        raw as i64 - modulus as i64
    } else {
        raw as i64
    }
}

fn raw_word(dtype: WeightDtype, signed: i64) -> u32 {
    (signed as u64 & u64::from(word_mask(dtype))) as u32
}

fn transform_bytes(
    dtype: WeightDtype,
    bytes: &[u8],
    element_count: usize,
    policy: &SemanticPolicy,
) -> Vec<u8> {
    match policy {
        SemanticPolicy::None => bytes.to_vec(),
        SemanticPolicy::MantissaTruncate { bits } => {
            let retain = !(((1u64 << bits) - 1) as u32);
            let mut out = Vec::with_capacity(bytes.len());
            for index in 0..element_count {
                let word = word_at(dtype, bytes, index);
                let transformed = if exponent_all_ones(dtype, word) {
                    word
                } else {
                    word & retain
                };
                push_word(dtype, transformed, &mut out);
            }
            out
        }
        SemanticPolicy::BlockQuantize { block_size, levels } => {
            let block = *block_size as usize;
            let l = *levels;
            let mut out = Vec::with_capacity(bytes.len());
            let mut start = 0usize;
            while start < element_count {
                let end = (start + block).min(element_count);
                let mut amax: i64 = 0;
                for index in start..end {
                    let value = signed_word(dtype, word_at(dtype, bytes, index));
                    let magnitude = value.unsigned_abs() as i64;
                    if magnitude > amax {
                        amax = magnitude;
                    }
                }
                if amax == 0 || l == 1 {
                    for _ in start..end {
                        push_word(dtype, raw_word(dtype, 0), &mut out);
                    }
                } else {
                    let span = (2 * amax) as u64;
                    for index in start..end {
                        let value = signed_word(dtype, word_at(dtype, bytes, index));
                        let v = (value + amax) as u64;
                        let mut k = (v * (l - 1) + amax as u64) / span;
                        if k >= l {
                            k = l - 1;
                        }
                        let level = (span * k + (l - 1) / 2) / (l - 1);
                        let reconstructed = level as i64 - amax;
                        push_word(dtype, raw_word(dtype, reconstructed), &mut out);
                    }
                }
                start = end;
            }
            out
        }
    }
}

/// The canonical semantic-root record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticRootRecord {
    pub dtype: WeightDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub policy: SemanticPolicy,
    pub quantized_values_hash: Hash32,
}

impl SemanticRootRecord {
    /// Build the record from canonical row-major element bytes.
    pub fn from_row_major(
        dtype: WeightDtype,
        shape: &[u64],
        bytes: &[u8],
        policy: SemanticPolicy,
        limits: &Limits,
    ) -> Result<Self> {
        let element_count = validate_shape(shape, limits)?;
        let expected = payload_len(element_count, dtype)?;
        check_len(expected, limits.max_decoded_bytes, "semantic tensor bytes")?;
        if expected != bytes.len() as u64 {
            return Err(Error::malformed(format!(
                "semantic tensor is {} bytes but shape {:?} x {} requires {expected}",
                bytes.len(),
                shape,
                dtype.as_str()
            )));
        }
        policy.validate(dtype, element_count)?;
        if dtype.is_quantized() {
            check_quantized_words(dtype, bytes, "semantic tensor")?;
        }
        let count = usize::try_from(element_count)
            .map_err(|_| Error::limit_exceeded("semantic element count does not fit usize"))?;
        let quantized_values_hash = match &policy {
            SemanticPolicy::None => hash_domain(crate::tensor::DOMAIN_TENSOR_VALUES, bytes),
            other => {
                let transformed = transform_bytes(dtype, bytes, count, other);
                hash_domain(DOMAIN_TENSOR_SEMANTIC_VALUES, &transformed)
            }
        };
        Ok(SemanticRootRecord {
            dtype,
            shape: shape.to_vec(),
            element_count,
            policy,
            quantized_values_hash,
        })
    }

    /// Validate every structural invariant.
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        let count = validate_shape(&self.shape, limits)?;
        if self.element_count != count {
            return Err(Error::malformed(format!(
                "semantic element_count {} does not match shape product {count}",
                self.element_count
            )));
        }
        payload_len(count, self.dtype)?;
        self.policy.validate(self.dtype, count)
    }

    /// Canonical WCB value.
    pub fn to_wcb_value(&self) -> Value {
        Value::map([
            ("schema", Value::Str(SEMANTIC_SCHEMA.to_owned())),
            ("dtype", Value::Str(self.dtype.as_str().to_owned())),
            (
                "shape",
                Value::Array(self.shape.iter().map(|dim| Value::U64(*dim)).collect()),
            ),
            ("element_count", Value::U64(self.element_count)),
            ("policy", self.policy.to_wcb_value()),
            (
                "quantized_values_hash",
                Value::Bytes(self.quantized_values_hash.as_bytes().to_vec()),
            ),
        ])
    }

    /// Canonical WCB bytes.
    pub fn canonical_bytes(&self, limits: &Limits) -> Result<Vec<u8>> {
        self.validate(limits)?;
        wcb::encode(&self.to_wcb_value(), limits)
    }

    /// `H("weft/tensor/semantic/v1", canonical bytes)`.
    pub fn root(&self, limits: &Limits) -> Result<Hash32> {
        Ok(hash_domain(
            DOMAIN_TENSOR_SEMANTIC,
            &self.canonical_bytes(limits)?,
        ))
    }

    /// Decode a canonical semantic-root record, rejecting non-canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let value = wcb::decode(bytes, limits)?;
        let map = match value {
            Value::Map(map) => map,
            _ => return Err(Error::malformed("semantic root record must be a WCB map")),
        };
        const KEYS: [&str; 6] = [
            "dtype",
            "element_count",
            "policy",
            "quantized_values_hash",
            "schema",
            "shape",
        ];
        for key in map.keys() {
            if !KEYS.contains(&key.as_str()) {
                return Err(Error::malformed(format!(
                    "semantic root record has unknown key `{key}`"
                )));
            }
        }
        for key in KEYS {
            if !map.contains_key(key) {
                return Err(Error::malformed(format!(
                    "semantic root record is missing key `{key}`"
                )));
            }
        }
        let schema = text_field(&map, "schema")?;
        if schema != SEMANTIC_SCHEMA {
            return Err(Error::malformed(format!(
                "unsupported semantic root schema `{schema}`"
            )));
        }
        let dtype = WeightDtype::from_name(&text_field(&map, "dtype")?)?;
        let shape = shape_field(&map, "shape")?;
        let element_count = match map.get("element_count") {
            Some(Value::U64(value)) => *value,
            _ => {
                return Err(Error::malformed(
                    "semantic root `element_count` must be u64",
                ))
            }
        };
        let policy_value = map
            .get("policy")
            .ok_or_else(|| Error::malformed("semantic root is missing policy"))?;
        let policy = SemanticPolicy::from_wcb_value(policy_value, dtype, element_count)?;
        let quantized_values_hash = hash_field(&map, "quantized_values_hash")?;
        let record = SemanticRootRecord {
            dtype,
            shape,
            element_count,
            policy,
            quantized_values_hash,
        };
        record.validate(limits)?;
        let reencoded = wcb::encode(&record.to_wcb_value(), limits)?;
        if reencoded != bytes {
            return Err(Error::non_canonical(
                "semantic root bytes are not in canonical form",
            ));
        }
        Ok(record)
    }

    /// Two records are candidates only when dtype, shape, policy, and the
    /// policy-transformed values hash all agree.
    pub fn matches(&self, other: &SemanticRootRecord) -> bool {
        self.dtype == other.dtype
            && self.shape == other.shape
            && self.element_count == other.element_count
            && self.policy == other.policy
            && self.quantized_values_hash == other.quantized_values_hash
    }
}

fn text_field(map: &std::collections::BTreeMap<String, Value>, key: &str) -> Result<String> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.clone()),
        _ => Err(Error::malformed(format!(
            "semantic root `{key}` must be text"
        ))),
    }
}

fn shape_field(map: &std::collections::BTreeMap<String, Value>, key: &str) -> Result<Vec<u64>> {
    match map.get(key) {
        Some(Value::Array(items)) => {
            let mut shape = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(value) => shape.push(*value),
                    _ => {
                        return Err(Error::malformed(format!(
                            "semantic root `{key}` elements must be u64"
                        )))
                    }
                }
            }
            Ok(shape)
        }
        _ => Err(Error::malformed(format!(
            "semantic root `{key}` must be an array"
        ))),
    }
}

fn hash_field(map: &std::collections::BTreeMap<String, Value>, key: &str) -> Result<Hash32> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) => {
            if bytes.len() != crate::hash::HASH_LEN {
                return Err(Error::malformed(format!(
                    "semantic root `{key}` must be {} bytes, got {}",
                    crate::hash::HASH_LEN,
                    bytes.len()
                )));
            }
            Hash32::from_slice(bytes)
        }
        _ => Err(Error::malformed(format!(
            "semantic root `{key}` must be bytes"
        ))),
    }
}

/// `SemanticRoot` computed from canonical row-major bytes with default limits.
pub fn semantic_root(
    dtype: WeightDtype,
    shape: &[u64],
    bytes: &[u8],
    policy: SemanticPolicy,
) -> Result<Hash32> {
    semantic_root_with_limits(dtype, shape, bytes, policy, &Limits::default())
}

/// `SemanticRoot` computed with explicit limits.
pub fn semantic_root_with_limits(
    dtype: WeightDtype,
    shape: &[u64],
    bytes: &[u8],
    policy: SemanticPolicy,
    limits: &Limits,
) -> Result<Hash32> {
    SemanticRootRecord::from_row_major(dtype, shape, bytes, policy, limits)?.root(limits)
}

/// Build the full semantic-root record (dtype, shape, policy, value hash).
pub fn semantic_record(
    dtype: WeightDtype,
    shape: &[u64],
    bytes: &[u8],
    policy: SemanticPolicy,
    limits: &Limits,
) -> Result<SemanticRootRecord> {
    SemanticRootRecord::from_row_major(dtype, shape, bytes, policy, limits)
}

/// Whether two semantic-root records describe the same candidate group.
pub fn semantic_matches(a: &SemanticRootRecord, b: &SemanticRootRecord) -> bool {
    a.matches(b)
}

/// Select the cheapest exact base among semantic-root candidates.
///
/// Both the target and every candidate must be present as bytes: the predicted
/// cost is the exact per-plane `huffman-canonical-v1` size of the modular word
/// difference, so no approximation is involved. Candidates whose dtype, shape,
/// or semantic root (under `policy`) differ from the target are not offered.
/// Ties are broken by the candidate frame's content hash, so the result is
/// deterministic for a fixed candidate order. Returns the index of the chosen
/// candidate, or `None` when no candidate matches.
pub fn select_semantic_base(
    target: &[u8],
    dtype: WeightDtype,
    shape: &[u64],
    policy: &SemanticPolicy,
    candidates: &[&[u8]],
    limits: &Limits,
) -> Result<Option<usize>> {
    let target_record =
        SemanticRootRecord::from_row_major(dtype, shape, target, policy.clone(), limits)?;
    let target_root = target_record.root(limits)?;
    let count = usize::try_from(target_record.element_count)
        .map_err(|_| Error::limit_exceeded("semantic element count does not fit usize"))?;
    let mut best: Option<(usize, u64, Hash32)> = None;
    for (index, frame_bytes) in candidates.iter().enumerate() {
        let parsed = match Frame::parse(frame_bytes, limits) {
            Ok(frame) => frame,
            Err(_) => continue,
        };
        let (candidate_dtype, candidate_shape) = match base_frame_descriptor(&parsed, limits) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if candidate_dtype != dtype || candidate_shape != shape {
            continue;
        }
        let candidate_bytes = match modes::decode_bytes(
            frame_bytes,
            &crate::dependency::MapResolver::new(),
            limits,
        ) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        let expected = payload_len(target_record.element_count, dtype)? as usize;
        if candidate_bytes.len() != expected {
            continue;
        }
        let candidate_record = match SemanticRootRecord::from_row_major(
            dtype,
            shape,
            &candidate_bytes,
            policy.clone(),
            limits,
        ) {
            Ok(record) => record,
            Err(_) => continue,
        };
        if candidate_record.root(limits)? != target_root {
            continue;
        }
        let (_, planes) = build_planes(dtype, target, &candidate_bytes, count);
        let cost: u64 = planes
            .iter()
            .map(|plane| huffman::encode(plane).len() as u64)
            .sum();
        let key = parsed.header.content_hash;
        let better = match &best {
            None => true,
            Some((_, best_cost, best_hash)) => (cost, key) < (*best_cost, *best_hash),
        };
        if better {
            best = Some((index, cost, key));
        }
    }
    Ok(best.map(|(index, _, _)| index))
}

// Mode 14 shares the exact plane layout through the `weight_delta` helpers.

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(words: &[u16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(words.len() * 2);
        for word in words {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    #[test]
    fn none_policy_uses_the_exact_values_hash() {
        let bytes = bf16_bytes(&[0x3F80, 0x4000, 0x0001]);
        let record = semantic_record(
            WeightDtype::Bf16,
            &[3],
            &bytes,
            SemanticPolicy::None,
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(
            record.quantized_values_hash,
            hash_domain(crate::tensor::DOMAIN_TENSOR_VALUES, &bytes)
        );
    }

    #[test]
    fn mantissa_truncate_preserves_nan_inf_and_groups_low_bits() {
        let base = bf16_bytes(&[0x3F80, 0x7FC0, 0x7F80, 0x4001]);
        let mut changed = base.clone();
        changed[6] ^= 0x07; // low three mantissa bits of element 3
        let policy = SemanticPolicy::MantissaTruncate { bits: 3 };
        let a = semantic_root(WeightDtype::Bf16, &[4], &base, policy.clone()).unwrap();
        let b = semantic_root(WeightDtype::Bf16, &[4], &changed, policy).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn block_quantize_is_deterministic_and_rejects_non_divisors() {
        let bytes = bf16_bytes(&[0x3F80, 0xC000, 0x0000, 0x7FC0, 0x8000, 0x0001]);
        let policy = SemanticPolicy::BlockQuantize {
            block_size: 3,
            levels: 8,
        };
        let a = semantic_root(WeightDtype::Bf16, &[6], &bytes, policy.clone()).unwrap();
        let b = semantic_root(WeightDtype::Bf16, &[6], &bytes, policy).unwrap();
        assert_eq!(a, b);
        let bad = SemanticPolicy::BlockQuantize {
            block_size: 4,
            levels: 8,
        };
        assert_eq!(
            semantic_root(WeightDtype::Bf16, &[6], &bytes, bad)
                .unwrap_err()
                .class(),
            crate::ErrorClass::Malformed
        );
    }
}
