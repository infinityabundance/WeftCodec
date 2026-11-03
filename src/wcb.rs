//! WCB — Weft Canonical Bytes (WEFTCODEC-FORMAT §8).
//!
//! WCB is a strict, deterministic subset used for structured frame bodies and
//! selector canonicalization. It is *not* CBOR: tags are fixed one-byte values,
//! integers are minimal varints, map keys are text and are ordered by their
//! **full encoded key bytes** (tag + length + bytes) ascending.
//!
//! Canonical bytes are unique for a value. Decoders reject non-canonical input
//! (non-minimal varints, unsorted or duplicate keys, non-text keys, trailing
//! bytes, unknown tags) with `NonCanonical`; truncated input is `Malformed`.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::limits::Limits;
use crate::varint;

/// A WCB value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Null,
    Bool(bool),
    U64(u64),
    I64(i64),
    Timestamp(i64),
    Str(String),
    Bytes(Vec<u8>),
    Array(Vec<Value>),
    /// Keys are text. Canonical ordering is by encoded key bytes, which is
    /// *not* the same as Rust `String` ordering; encoding sorts explicitly.
    Map(BTreeMap<String, Value>),
}

pub const TAG_NULL: u8 = 0x00;
pub const TAG_FALSE: u8 = 0x01;
pub const TAG_TRUE: u8 = 0x02;
pub const TAG_U64: u8 = 0x03;
pub const TAG_I64: u8 = 0x04;
pub const TAG_TEXT: u8 = 0x05;
pub const TAG_BYTES: u8 = 0x06;
pub const TAG_ARRAY: u8 = 0x07;
pub const TAG_MAP: u8 = 0x08;
pub const TAG_TIMESTAMP: u8 = 0x09;

impl Value {
    /// Build a map value from pairs. Later duplicate keys replace earlier ones;
    /// callers that must reject duplicates should validate before construction.
    pub fn map<I, K>(pairs: I) -> Value
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let mut map = BTreeMap::new();
        for (key, value) in pairs {
            map.insert(key.into(), value);
        }
        Value::Map(map)
    }

    /// Convenience for text values.
    pub fn text(value: impl Into<String>) -> Value {
        Value::Str(value.into())
    }
}

/// Encode a value with default limits.
pub fn encode_default(value: &Value) -> Result<Vec<u8>> {
    encode(value, &Limits::default())
}

/// Encode a value to canonical WCB bytes.
pub fn encode(value: &Value, limits: &Limits) -> Result<Vec<u8>> {
    let mut encoder = Encoder {
        out: Vec::new(),
        limits,
        items: 0,
    };
    encoder.value(value, 0)?;
    Ok(encoder.out)
}

/// Decode canonical WCB bytes with default limits.
pub fn decode_default(bytes: &[u8]) -> Result<Value> {
    decode(bytes, &Limits::default())
}

/// Decode canonical WCB bytes, rejecting every non-canonical form.
pub fn decode(bytes: &[u8], limits: &Limits) -> Result<Value> {
    if bytes.is_empty() {
        return Err(Error::malformed("empty WCB value"));
    }
    let mut decoder = Decoder {
        input: bytes,
        pos: 0,
        limits,
        items: 0,
    };
    let value = decoder.value(0)?;
    if decoder.pos != bytes.len() {
        return Err(Error::non_canonical(format!(
            "trailing bytes after WCB value: {} of {} unconsumed",
            bytes.len() - decoder.pos,
            bytes.len()
        )));
    }
    Ok(value)
}

struct Encoder<'a> {
    out: Vec<u8>,
    limits: &'a Limits,
    items: u64,
}

impl Encoder<'_> {
    fn bump_item(&mut self) -> Result<()> {
        self.items = self
            .items
            .checked_add(1)
            .ok_or_else(|| Error::limit_exceeded("WCB item count overflow"))?;
        if self.items > self.limits.max_structured_items {
            return Err(Error::limit_exceeded(format!(
                "WCB item count {} exceeds limit {}",
                self.items, self.limits.max_structured_items
            )));
        }
        Ok(())
    }

    fn check_text(&self, len: usize, what: &str) -> Result<()> {
        if len as u64 > self.limits.max_string_bytes {
            return Err(Error::limit_exceeded(format!(
                "{what} length {len} exceeds limit {}",
                self.limits.max_string_bytes
            )));
        }
        Ok(())
    }

    fn enter(&self, depth: u32) -> Result<u32> {
        let next = depth
            .checked_add(1)
            .ok_or_else(|| Error::limit_exceeded("WCB depth overflow"))?;
        if u64::from(next) > self.limits.max_structured_depth {
            return Err(Error::limit_exceeded(format!(
                "WCB nesting depth {next} exceeds limit {}",
                self.limits.max_structured_depth
            )));
        }
        Ok(next)
    }

    fn value(&mut self, value: &Value, depth: u32) -> Result<()> {
        self.bump_item()?;
        match value {
            Value::Null => self.out.push(TAG_NULL),
            Value::Bool(false) => self.out.push(TAG_FALSE),
            Value::Bool(true) => self.out.push(TAG_TRUE),
            Value::U64(v) => {
                self.out.push(TAG_U64);
                varint::encode(*v, &mut self.out);
            }
            Value::I64(v) => {
                self.out.push(TAG_I64);
                varint::encode(varint::zigzag_encode(*v), &mut self.out);
            }
            Value::Timestamp(v) => {
                self.out.push(TAG_TIMESTAMP);
                varint::encode(varint::zigzag_encode(*v), &mut self.out);
            }
            Value::Str(s) => {
                self.check_text(s.len(), "text")?;
                self.out.push(TAG_TEXT);
                varint::encode(s.len() as u64, &mut self.out);
                self.out.extend_from_slice(s.as_bytes());
            }
            Value::Bytes(b) => {
                self.check_text(b.len(), "bytes")?;
                self.out.push(TAG_BYTES);
                varint::encode(b.len() as u64, &mut self.out);
                self.out.extend_from_slice(b);
            }
            Value::Array(items) => {
                let child = self.enter(depth)?;
                if items.len() as u64 > self.limits.max_structured_items {
                    return Err(Error::limit_exceeded("WCB array too large"));
                }
                self.out.push(TAG_ARRAY);
                varint::encode(items.len() as u64, &mut self.out);
                for item in items {
                    self.value(item, child)?;
                }
            }
            Value::Map(map) => {
                let child = self.enter(depth)?;
                if map.len() as u64 > self.limits.max_structured_items {
                    return Err(Error::limit_exceeded("WCB map too large"));
                }
                // Sort by full encoded key bytes, not by Rust string order.
                let mut entries: Vec<(&String, Vec<u8>)> = Vec::with_capacity(map.len());
                for key in map.keys() {
                    self.check_text(key.len(), "map key")?;
                    let mut encoded = Vec::with_capacity(1 + 10 + key.len());
                    encoded.push(TAG_TEXT);
                    varint::encode(key.len() as u64, &mut encoded);
                    encoded.extend_from_slice(key.as_bytes());
                    entries.push((key, encoded));
                }
                entries.sort_by(|a, b| a.1.cmp(&b.1));
                self.out.push(TAG_MAP);
                varint::encode(entries.len() as u64, &mut self.out);
                for (key, encoded_key) in entries {
                    self.out.extend_from_slice(&encoded_key);
                    let value = map
                        .get(key)
                        .ok_or_else(|| Error::malformed("map key vanished during encoding"))?;
                    self.value(value, child)?;
                }
            }
        }
        Ok(())
    }
}

struct Decoder<'a> {
    input: &'a [u8],
    pos: usize,
    limits: &'a Limits,
    items: u64,
}

impl<'a> Decoder<'a> {
    fn bump_item(&mut self) -> Result<()> {
        self.items = self
            .items
            .checked_add(1)
            .ok_or_else(|| Error::limit_exceeded("WCB item count overflow"))?;
        if self.items > self.limits.max_structured_items {
            return Err(Error::limit_exceeded(format!(
                "WCB item count {} exceeds limit {}",
                self.items, self.limits.max_structured_items
            )));
        }
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8> {
        let byte = *self
            .input
            .get(self.pos)
            .ok_or_else(|| Error::malformed("truncated WCB value"))?;
        self.pos += 1;
        Ok(byte)
    }

    fn read_varint(&mut self) -> Result<u64> {
        varint::decode(self.input, &mut self.pos)
    }

    fn remaining(&self) -> usize {
        self.input.len() - self.pos
    }

    fn take(&mut self, len: u64, what: &str) -> Result<&'a [u8]> {
        if len > self.limits.max_string_bytes {
            return Err(Error::limit_exceeded(format!(
                "{what} length {len} exceeds limit {}",
                self.limits.max_string_bytes
            )));
        }
        let len = usize::try_from(len)
            .map_err(|_| Error::limit_exceeded(format!("{what} length does not fit usize")))?;
        if len > self.remaining() {
            return Err(Error::malformed(format!("truncated WCB {what}")));
        }
        let start = self.pos;
        self.pos += len;
        let input = self.input;
        Ok(&input[start..self.pos])
    }

    fn enter(&self, depth: u32) -> Result<u32> {
        let next = depth
            .checked_add(1)
            .ok_or_else(|| Error::limit_exceeded("WCB depth overflow"))?;
        if u64::from(next) > self.limits.max_structured_depth {
            return Err(Error::limit_exceeded(format!(
                "WCB nesting depth {next} exceeds limit {}",
                self.limits.max_structured_depth
            )));
        }
        Ok(next)
    }

    fn check_count(&self, count: u64, min_bytes_each: u64, what: &str) -> Result<usize> {
        if count > self.limits.max_structured_items {
            return Err(Error::limit_exceeded(format!(
                "WCB {what} count {count} exceeds limit {}",
                self.limits.max_structured_items
            )));
        }
        let min_bytes = count
            .checked_mul(min_bytes_each)
            .ok_or_else(|| Error::limit_exceeded("WCB count overflow"))?;
        if min_bytes > self.remaining() as u64 {
            return Err(Error::malformed(format!(
                "truncated WCB {what}: count {count} cannot fit in {} remaining bytes",
                self.remaining()
            )));
        }
        usize::try_from(count).map_err(|_| Error::limit_exceeded("WCB count does not fit usize"))
    }

    fn value(&mut self, depth: u32) -> Result<Value> {
        self.bump_item()?;
        let tag = self.read_u8()?;
        match tag {
            TAG_NULL => Ok(Value::Null),
            TAG_FALSE => Ok(Value::Bool(false)),
            TAG_TRUE => Ok(Value::Bool(true)),
            TAG_U64 => Ok(Value::U64(self.read_varint()?)),
            TAG_I64 => Ok(Value::I64(varint::zigzag_decode(self.read_varint()?))),
            TAG_TIMESTAMP => Ok(Value::Timestamp(varint::zigzag_decode(self.read_varint()?))),
            TAG_TEXT => {
                let len = self.read_varint()?;
                let bytes = self.take(len, "text")?;
                let text = std::str::from_utf8(bytes)
                    .map_err(|_| Error::non_canonical("WCB text is not valid UTF-8"))?;
                Ok(Value::Str(text.to_owned()))
            }
            TAG_BYTES => {
                let len = self.read_varint()?;
                let bytes = self.take(len, "bytes")?;
                Ok(Value::Bytes(bytes.to_vec()))
            }
            TAG_ARRAY => {
                let child = self.enter(depth)?;
                let count = self.read_varint()?;
                let count = self.check_count(count, 1, "array")?;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.value(child)?);
                }
                Ok(Value::Array(items))
            }
            TAG_MAP => {
                let child = self.enter(depth)?;
                let count = self.read_varint()?;
                let count = self.check_count(count, 2, "map")?;
                let mut map: BTreeMap<String, Value> = BTreeMap::new();
                let mut prev_key: Option<(usize, usize)> = None;
                for _ in 0..count {
                    let key_start = self.pos;
                    let key_tag = self.read_u8()?;
                    if key_tag != TAG_TEXT {
                        return Err(Error::non_canonical(
                            "WCB map key is not a text value (tag 0x05)",
                        ));
                    }
                    let key_len = self.read_varint()?;
                    let key_bytes = self.take(key_len, "map key")?;
                    let key_end = self.pos;
                    if let Some((prev_start, prev_end)) = prev_key {
                        let previous = &self.input[prev_start..prev_end];
                        let current = &self.input[key_start..key_end];
                        if current <= previous {
                            return Err(Error::non_canonical(
                                "WCB map keys are duplicated or not sorted by encoded key bytes",
                            ));
                        }
                    }
                    prev_key = Some((key_start, key_end));
                    let key = std::str::from_utf8(key_bytes)
                        .map_err(|_| Error::non_canonical("WCB map key is not valid UTF-8"))?;
                    if map.contains_key(key) {
                        return Err(Error::non_canonical("duplicate WCB map key"));
                    }
                    let value = self.value(child)?;
                    map.insert(key.to_owned(), value);
                }
                Ok(Value::Map(map))
            }
            _ => Err(Error::non_canonical(format!("unknown WCB tag 0x{tag:02x}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorClass;

    fn map(pairs: &[(&str, Value)]) -> Value {
        Value::map(pairs.iter().map(|(k, v)| (*k, v.clone())))
    }

    #[test]
    fn scalar_roundtrip() {
        let values = vec![
            Value::Null,
            Value::Bool(false),
            Value::Bool(true),
            Value::U64(0),
            Value::U64(u64::MAX),
            Value::I64(-1),
            Value::I64(i64::MIN),
            Value::Timestamp(1_700_000_000_000_000_000),
            Value::Str("héllo".into()),
            Value::Bytes(vec![0, 1, 2, 0xff]),
        ];
        for value in values {
            let bytes = encode_default(&value).unwrap();
            assert_eq!(decode_default(&bytes).unwrap(), value);
        }
    }

    #[test]
    fn map_sorted_by_encoded_key_bytes() {
        // Byte length is encoded first, so "a" (len 1) sorts before "bb"
        // (len 2) even though lexicographically both orders agree here.
        // Use keys of length 1 and 128: the 128-byte key encodes with a
        // two-byte length (0x80 0x01), which sorts *after* any single-byte
        // length prefix.
        let long = "z".repeat(128);
        let value = map(&[(&long, Value::U64(1)), ("a", Value::U64(2))]);
        let bytes = encode_default(&value).unwrap();
        let decoded = decode_default(&bytes).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn rejects_non_minimal_varint() {
        // u64 with value 0 encoded in two bytes.
        let err = decode_default(&[TAG_U64, 0x80, 0x00]).unwrap_err();
        assert_eq!(err.class(), ErrorClass::NonCanonical);
    }

    #[test]
    fn rejects_unsorted_and_duplicate_keys() {
        // map 2 { "b": 1, "a": 2 }
        let unsorted = [
            TAG_MAP, 2, TAG_TEXT, 1, b'b', TAG_U64, 1, TAG_TEXT, 1, b'a', TAG_U64, 2,
        ];
        assert_eq!(
            decode_default(&unsorted).unwrap_err().class(),
            ErrorClass::NonCanonical
        );
        let duplicate = [
            TAG_MAP, 2, TAG_TEXT, 1, b'a', TAG_U64, 1, TAG_TEXT, 1, b'a', TAG_U64, 2,
        ];
        assert_eq!(
            decode_default(&duplicate).unwrap_err().class(),
            ErrorClass::NonCanonical
        );
    }

    #[test]
    fn rejects_non_text_key_and_trailing_and_unknown_tag() {
        let non_text = [TAG_MAP, 1, TAG_U64, 1, TAG_U64, 2];
        assert_eq!(
            decode_default(&non_text).unwrap_err().class(),
            ErrorClass::NonCanonical
        );
        assert_eq!(
            decode_default(&[TAG_NULL, TAG_NULL]).unwrap_err().class(),
            ErrorClass::NonCanonical
        );
        assert_eq!(
            decode_default(&[0x0a]).unwrap_err().class(),
            ErrorClass::NonCanonical
        );
    }

    #[test]
    fn rejects_truncation() {
        assert_eq!(
            decode_default(&[TAG_TEXT, 5, b'a']).unwrap_err().class(),
            ErrorClass::Malformed
        );
        assert_eq!(
            decode_default(&[]).unwrap_err().class(),
            ErrorClass::Malformed
        );
    }

    #[test]
    fn limits() {
        let limits = Limits::new().with_max_structured_depth(2);
        let nested = Value::Array(vec![Value::Array(vec![Value::Array(vec![Value::Null])])]);
        assert_eq!(
            encode(&nested, &limits).unwrap_err().class(),
            ErrorClass::LimitExceeded
        );
        let limits = Limits::new().with_max_string_bytes(2);
        assert_eq!(
            encode(&Value::Str("abc".into()), &limits)
                .unwrap_err()
                .class(),
            ErrorClass::LimitExceeded
        );
        // Depth rejection on decode.
        let deep_bytes = encode_default(&nested).unwrap();
        assert_eq!(
            decode(&deep_bytes, &Limits::new().with_max_structured_depth(2))
                .unwrap_err()
                .class(),
            ErrorClass::LimitExceeded
        );
    }
}
