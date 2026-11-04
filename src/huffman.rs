//! `huffman-canonical-v1`: the deterministic order-0 canonical Huffman coder.
//!
//! This coder exists so that Rust and Go emit **bit-identical** bytes for the
//! `FLOAT_TENSOR` exponent plane (WEFT-AI-CODEC §5). It is deliberately
//! integer-only and has no library-dependent choices.
//!
//! ## Serialized form
//!
//! ```text
//! u8[256] code lengths (0 = symbol absent)
//! || bitstream, MSB-first, zero-padded to a byte boundary
//! ```
//!
//! The uncompressed length is carried by the caller (the `FLOAT_TENSOR`
//! descriptor's `element_count`). A zero-symbol table is valid only for an
//! empty plane; a one-symbol table uses length 1 and an empty bitstream (the
//! decoder repeats that symbol), which keeps the canonical form total without
//! a zero-length code.
//!
//! ## Encoder
//!
//! 1. Count byte frequencies.
//! 2. Build a Huffman tree with the deterministic tie-break
//!    `(weight, minimum symbol byte, node index)`. Leaves are inserted in
//!    ascending symbol order, so even the final fallback is deterministic.
//! 3. Canonicalise: sort `(code length, symbol)`, assign increasing codes,
//!    MSB-first within each byte.
//! 4. Code lengths are limited to 24 bits. If the naive tree exceeds that,
//!    every non-zero frequency `f` becomes `max(1, f >> 1)` and the tree is
//!    rebuilt; the process repeats until every length is at most 24. With all
//!    weights equal the tree has depth at most 9 for a 256-symbol alphabet, so
//!    the loop always terminates.
//!
//! ## Decoder
//!
//! Rebuilds the canonical table, rejects lengths > 24, over-subscribed and
//! incomplete tables (except the single-symbol form), truncated bitstreams,
//! trailing bytes, and non-zero padding. Table-shape failures are
//! [`ErrorClass::NonCanonical`]; stream/length failures are
//! [`ErrorClass::Corrupt`]. Both implementations share this mapping.

use crate::error::{Error, Result};

/// Normative coder identifier carried in the `FLOAT_TENSOR` descriptor.
pub const CODER_ID: &str = "huffman-canonical-v1";
/// Number of serialized code-length bytes.
pub const TABLE_BYTES: usize = 256;
/// Maximum code length in bits.
pub const MAX_CODE_BITS: usize = 24;

#[derive(Clone, Copy, Debug)]
struct Node {
    weight: u64,
    min_symbol: u8,
    left: i32,
    right: i32,
    symbol: i32,
    active: bool,
}

fn node_key(node: &Node, index: usize) -> (u64, u8, usize) {
    (node.weight, node.min_symbol, index)
}

fn min_active(nodes: &[Node], exclude: Option<usize>) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, node) in nodes.iter().enumerate() {
        if !node.active || exclude == Some(index) {
            continue;
        }
        match best {
            None => best = Some(index),
            Some(current) => {
                if node_key(node, index) < node_key(&nodes[current], current) {
                    best = Some(index);
                }
            }
        }
    }
    best
}

/// Build the deterministic Huffman code lengths for the given frequencies.
fn huffman_lengths(weights: &[u64; 256], present: &[usize]) -> [u8; 256] {
    let mut nodes: Vec<Node> = present
        .iter()
        .map(|&symbol| Node {
            weight: weights[symbol],
            min_symbol: symbol as u8,
            left: -1,
            right: -1,
            symbol: symbol as i32,
            active: true,
        })
        .collect();
    let mut active = present.len();
    while active > 1 {
        let first = min_active(&nodes, None).expect("two active nodes");
        let second = min_active(&nodes, Some(first)).expect("second active node");
        nodes[first].active = false;
        nodes[second].active = false;
        let weight = nodes[first]
            .weight
            .checked_add(nodes[second].weight)
            .expect("frequency sum fits u64");
        let min_symbol = nodes[first].min_symbol.min(nodes[second].min_symbol);
        nodes.push(Node {
            weight,
            min_symbol,
            left: first as i32,
            right: second as i32,
            symbol: -1,
            active: true,
        });
        active -= 1;
    }

    let root = nodes
        .iter()
        .position(|node| node.active)
        .expect("one active root");
    let mut lengths = [0u8; 256];
    let mut stack = vec![(root, 0u16)];
    while let Some((index, depth)) = stack.pop() {
        let node = nodes[index];
        if node.symbol >= 0 {
            lengths[node.symbol as usize] = depth as u8;
        } else {
            stack.push((node.left as usize, depth + 1));
            stack.push((node.right as usize, depth + 1));
        }
    }
    lengths
}

/// Compute canonical code lengths, applying deterministic length limiting.
fn build_lengths(weights: &[u64; 256]) -> [u8; 256] {
    let present: Vec<usize> = (0..256).filter(|&symbol| weights[symbol] > 0).collect();
    let mut lengths = [0u8; 256];
    if present.is_empty() {
        return lengths;
    }
    if present.len() == 1 {
        // A zero-length code is not representable; use the canonical
        // single-symbol form (length 1, empty bitstream).
        lengths[present[0]] = 1;
        return lengths;
    }
    let mut scaled = *weights;
    loop {
        let candidate = huffman_lengths(&scaled, &present);
        let max_len = candidate.iter().copied().max().unwrap_or(0) as usize;
        if max_len <= MAX_CODE_BITS {
            return candidate;
        }
        for &symbol in &present {
            let halved = scaled[symbol] >> 1;
            scaled[symbol] = if halved == 0 { 1 } else { halved };
        }
    }
}

/// Assign canonical codes to the present symbols.
fn canonical_codes(lengths: &[u8; 256]) -> [u32; 256] {
    let mut order: Vec<(u8, u8)> = (0..256)
        .filter(|&symbol| lengths[symbol] > 0)
        .map(|symbol| (lengths[symbol], symbol as u8))
        .collect();
    order.sort_unstable();
    let mut codes = [0u32; 256];
    let mut code = 0u32;
    let mut previous = 0u8;
    for (length, symbol) in order {
        code <<= u32::from(length - previous);
        codes[symbol as usize] = code;
        code += 1;
        previous = length;
    }
    codes
}

fn append_bits(out: &mut Vec<u8>, buffer: &mut u64, pending: &mut u32, code: u32, length: u32) {
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

/// Encode a byte plane with `huffman-canonical-v1`.
///
/// Never fails: the output is always the 256-byte length table plus a
/// zero-padded canonical bitstream.
pub fn encode(input: &[u8]) -> Vec<u8> {
    let mut weights = [0u64; 256];
    for &byte in input {
        weights[byte as usize] += 1;
    }
    let lengths = build_lengths(&weights);
    let used = lengths.iter().filter(|&&length| length > 0).count();
    let mut out = Vec::with_capacity(TABLE_BYTES + input.len());
    out.extend_from_slice(&lengths);
    if used < 2 {
        // Empty plane or one symbol: canonical form has no bitstream.
        return out;
    }
    let codes = canonical_codes(&lengths);
    let mut buffer = 0u64;
    let mut pending = 0u32;
    for &byte in input {
        let length = u32::from(lengths[byte as usize]);
        debug_assert!(length > 0);
        append_bits(
            &mut out,
            &mut buffer,
            &mut pending,
            codes[byte as usize],
            length,
        );
    }
    if pending > 0 {
        out.push((buffer << (8 - pending)) as u8);
    }
    out
}

/// Decode a `huffman-canonical-v1` stream into exactly `expected_len` bytes.
pub fn decode(encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    if encoded.len() < TABLE_BYTES {
        return Err(Error::corrupt(format!(
            "huffman-canonical-v1 stream is {} bytes, shorter than its {TABLE_BYTES}-byte code-length table",
            encoded.len()
        )));
    }
    let lengths = &encoded[..TABLE_BYTES];
    let stream = &encoded[TABLE_BYTES..];

    let mut counts = [0u32; MAX_CODE_BITS + 1];
    let mut used = 0usize;
    let mut max_len = 0usize;
    for (symbol, &length) in lengths.iter().enumerate() {
        let length = length as usize;
        if length > MAX_CODE_BITS {
            return Err(Error::non_canonical(format!(
                "huffman-canonical-v1 symbol {symbol} has code length {length} > {MAX_CODE_BITS}"
            )));
        }
        if length > 0 {
            used += 1;
            counts[length] += 1;
            max_len = max_len.max(length);
        }
    }

    if used == 0 {
        if expected_len != 0 {
            return Err(Error::corrupt(
                "huffman-canonical-v1 table has no symbols but the descriptor declares a non-empty plane",
            ));
        }
        if !stream.is_empty() {
            return Err(Error::corrupt(
                "huffman-canonical-v1 empty plane has trailing bitstream bytes",
            ));
        }
        return Ok(Vec::new());
    }

    if used == 1 {
        if max_len != 1 {
            return Err(Error::non_canonical(
                "huffman-canonical-v1 single-symbol table must use code length 1",
            ));
        }
        if expected_len == 0 {
            return Err(Error::non_canonical(
                "huffman-canonical-v1 non-empty table for an empty plane",
            ));
        }
        if !stream.is_empty() {
            return Err(Error::corrupt(
                "huffman-canonical-v1 single-symbol stream must be empty",
            ));
        }
        let symbol = lengths
            .iter()
            .position(|&length| length > 0)
            .expect("one used symbol") as u8;
        let capacity = expected_len.min(1 << 20);
        let mut out = Vec::with_capacity(capacity);
        out.resize(expected_len, symbol);
        return Ok(out);
    }

    if expected_len == 0 {
        return Err(Error::non_canonical(
            "huffman-canonical-v1 non-empty table for an empty plane",
        ));
    }

    // Kraft equality proves a complete prefix code without building a table.
    let space = 1u64 << MAX_CODE_BITS;
    let mut used_space = 0u64;
    for length in 1..=MAX_CODE_BITS {
        used_space += u64::from(counts[length]) << (MAX_CODE_BITS - length);
        if used_space > space {
            return Err(Error::non_canonical(
                "huffman-canonical-v1 code lengths are over-subscribed",
            ));
        }
    }
    if used_space != space {
        return Err(Error::non_canonical(
            "huffman-canonical-v1 code lengths are incomplete",
        ));
    }

    // Canonical decode arrays: for each length, the first code and the index
    // of its first symbol in the `(length, symbol)`-sorted symbol list.
    let mut order: Vec<(u8, u8)> = (0..256)
        .filter(|&symbol| lengths[symbol] > 0)
        .map(|symbol| (lengths[symbol], symbol as u8))
        .collect();
    order.sort_unstable();
    let mut first_code = [0u32; MAX_CODE_BITS + 1];
    let mut first_index = [0u32; MAX_CODE_BITS + 1];
    let mut assigned = [0u32; MAX_CODE_BITS + 1];
    let mut symbols = Vec::with_capacity(used);
    let mut code = 0u32;
    let mut previous = 0u8;
    for (length, symbol) in order {
        code <<= u32::from(length - previous);
        let slot = length as usize;
        if assigned[slot] == 0 {
            first_code[slot] = code;
            first_index[slot] = symbols.len() as u32;
        }
        assigned[slot] += 1;
        symbols.push(symbol);
        code += 1;
        previous = length;
    }

    let capacity = expected_len.min(1 << 20);
    let mut out = Vec::with_capacity(capacity);
    let mut bit_pos = 0usize;
    let mut current_code = 0u32;
    let mut current_len = 0usize;
    while out.len() < expected_len {
        if current_len > 0 {
            let slot = current_len;
            if current_code >= first_code[slot] {
                let offset = current_code - first_code[slot];
                if offset < counts[slot] {
                    let symbol = symbols[(first_index[slot] + offset) as usize];
                    out.push(symbol);
                    current_code = 0;
                    current_len = 0;
                    continue;
                }
            }
            if current_len >= max_len {
                return Err(Error::corrupt(
                    "huffman-canonical-v1 bitstream contains an invalid code",
                ));
            }
        }
        if bit_pos >= stream.len() * 8 {
            return Err(Error::corrupt(format!(
                "huffman-canonical-v1 bitstream ended after {} of {expected_len} bytes",
                out.len()
            )));
        }
        let bit = (stream[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1;
        current_code = (current_code << 1) | u32::from(bit);
        current_len += 1;
        bit_pos += 1;
    }

    let remaining = stream.len() * 8 - bit_pos;
    if remaining >= 8 {
        return Err(Error::corrupt(
            "huffman-canonical-v1 bitstream has trailing bytes after the declared output",
        ));
    }
    if remaining > 0 {
        let last = *stream.last().expect("non-empty trailing byte");
        let mask = (1u8 << remaining) - 1;
        if last & mask != 0 {
            return Err(Error::corrupt(
                "huffman-canonical-v1 padding bits must be zero",
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(len: usize, mut state: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push((state & 0xff) as u8);
        }
        out
    }

    fn roundtrip(input: &[u8]) {
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).expect("decode");
        assert_eq!(decoded, input);
        assert_eq!(encode(input), encoded, "encoder is deterministic");
    }

    #[test]
    fn empty_plane() {
        let encoded = encode(b"");
        assert_eq!(encoded.len(), TABLE_BYTES);
        assert!(encoded.iter().all(|&byte| byte == 0));
        assert_eq!(decode(&encoded, 0).expect("decode"), b"");
    }

    #[test]
    fn single_distinct_byte() {
        let input = vec![0x5au8; 4096];
        let encoded = encode(&input);
        assert_eq!(encoded.len(), TABLE_BYTES);
        assert_eq!(encoded[0x5a], 1);
        assert_eq!(encoded.iter().filter(|&&b| b > 0).count(), 1);
        assert_eq!(decode(&encoded, input.len()).expect("decode"), input);
    }

    #[test]
    fn all_256_distinct() {
        let input: Vec<u8> = (0..256u32).map(|value| value as u8).collect();
        roundtrip(&input);
        let encoded = encode(&input);
        assert!(encoded[..TABLE_BYTES].iter().all(|&length| length > 0));
    }

    #[test]
    fn highly_skewed() {
        let mut input = vec![0u8; 1_000_000];
        for (index, byte) in input.iter_mut().enumerate().skip(990_000) {
            *byte = (index % 255 + 1) as u8;
        }
        roundtrip(&input);
    }

    #[test]
    fn random_uniform() {
        let input = pattern(250_000, 0x1234_5678_9abc_def0);
        roundtrip(&input);
    }

    #[test]
    fn two_megabyte_plane() {
        let mut input = pattern(2_000_000, 0xdead_beef_cafe_babe);
        // Mix in a skewed tail so the code is non-trivial.
        for byte in input.iter_mut().take(1_500_000) {
            *byte &= 0x0f;
        }
        roundtrip(&input);
    }

    #[test]
    fn length_limiting_is_applied_and_bounded() {
        // Fibonacci-like weights force a deep naive tree.
        let mut fib = [0u64; 32];
        fib[0] = 1;
        fib[1] = 1;
        for index in 2..fib.len() {
            fib[index] = fib[index - 1] + fib[index - 2];
        }
        let total: u64 = fib.iter().sum();
        let mut input = Vec::with_capacity(total as usize);
        for (symbol, &count) in fib.iter().enumerate() {
            input.extend(std::iter::repeat(symbol as u8).take(count as usize));
        }
        let encoded = encode(&input);
        assert!(encoded[..TABLE_BYTES]
            .iter()
            .all(|&length| (length as usize) <= MAX_CODE_BITS));
        assert_eq!(decode(&encoded, input.len()).expect("decode"), input);
    }

    #[test]
    fn rejects_code_length_too_long() {
        let mut encoded = vec![0u8; TABLE_BYTES];
        encoded[0] = 25;
        assert_eq!(
            decode(&encoded, 1).expect_err("too long").class(),
            crate::error::ErrorClass::NonCanonical
        );
    }

    #[test]
    fn rejects_over_subscribed_table() {
        let mut encoded = vec![1u8; TABLE_BYTES];
        encoded.extend_from_slice(&[0, 0]);
        assert_eq!(
            decode(&encoded, 2).expect_err("over").class(),
            crate::error::ErrorClass::NonCanonical
        );
    }

    #[test]
    fn rejects_incomplete_table() {
        let mut encoded = vec![0u8; TABLE_BYTES];
        encoded[0] = 2;
        encoded[1] = 2;
        assert_eq!(
            decode(&encoded, 2).expect_err("incomplete").class(),
            crate::error::ErrorClass::NonCanonical
        );
    }

    #[test]
    fn rejects_single_symbol_with_extra_bits() {
        let mut encoded = vec![0u8; TABLE_BYTES];
        encoded[7] = 1;
        encoded.push(0);
        assert_eq!(
            decode(&encoded, 4).expect_err("single stream").class(),
            crate::error::ErrorClass::Corrupt
        );
    }

    #[test]
    fn rejects_truncation() {
        let mut encoded = vec![0u8; TABLE_BYTES];
        encoded[0] = 1;
        encoded[1] = 1;
        assert_eq!(
            decode(&encoded, 4).expect_err("truncated").class(),
            crate::error::ErrorClass::Corrupt
        );
    }

    #[test]
    fn rejects_non_zero_padding() {
        let mut encoded = vec![0u8; TABLE_BYTES];
        encoded[0] = 1;
        encoded[1] = 1;
        // One symbol (0) followed by a non-zero padding bit.
        encoded.push(0b0100_0000);
        assert_eq!(
            decode(&encoded, 1).expect_err("padding").class(),
            crate::error::ErrorClass::Corrupt
        );
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut encoded = vec![0u8; TABLE_BYTES];
        encoded[0] = 1;
        encoded[1] = 1;
        encoded.push(0x00);
        encoded.push(0x00);
        assert_eq!(
            decode(&encoded, 1).expect_err("trailing").class(),
            crate::error::ErrorClass::Corrupt
        );
    }

    #[test]
    fn rejects_short_table() {
        assert_eq!(
            decode(&[0u8; 4], 1).expect_err("short").class(),
            crate::error::ErrorClass::Corrupt
        );
    }

    #[test]
    fn deterministic_bytes_for_fixed_input() {
        let input = b"canonical huffman determinism";
        assert_eq!(encode(input), encode(input));
        roundtrip(input);
    }
}
