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
///
/// The whole slice is one stream: any trailing byte after the declared output
/// is `Corrupt`.
pub fn decode(encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let (out, consumed) = decode_prefix(encoded, expected_len)?;
    if consumed != encoded.len() {
        return Err(Error::corrupt(
            "huffman-canonical-v1 bitstream has trailing bytes after the declared output",
        ));
    }
    Ok(out)
}

/// Largest primary lookup-table width. Codes no longer than this are decoded
/// with one table probe; longer codes fall back to the exact canonical loop.
const PRIMARY_BITS_MAX: usize = 12;

/// MSB-aligned 64-bit bit reader over the encoded stream.
///
/// Bits live in the high end of `acc`: the next unread bit is bit 63. Bytes are
/// loaded only from the stream, so the accumulator never contains invented
/// bits; `bits_consumed + acc_bits == 8 * next_byte` always holds.
struct BitReader<'a> {
    stream: &'a [u8],
    next_byte: usize,
    acc: u64,
    acc_bits: u32,
}

impl<'a> BitReader<'a> {
    /// Load whole bytes while at least eight bits still fit.
    #[inline]
    fn refill(&mut self) {
        while self.acc_bits <= 56 && self.next_byte < self.stream.len() {
            self.acc |= u64::from(self.stream[self.next_byte]) << (56 - self.acc_bits);
            self.next_byte += 1;
            self.acc_bits += 8;
        }
    }

    /// Read one bit. The caller has already proved a bit remains.
    #[inline]
    fn next_bit(&mut self) -> u32 {
        if self.acc_bits == 0 {
            self.refill();
        }
        let bit = (self.acc >> 63) as u32;
        self.acc <<= 1;
        self.acc_bits -= 1;
        bit
    }
}

/// Canonical `(first_code, counts, first_index, symbols)` arrays for the rare
/// bit-at-a-time path.
struct CanonicalTables {
    counts: [u32; MAX_CODE_BITS + 1],
    first_code: [u32; MAX_CODE_BITS + 1],
    first_index: [u32; MAX_CODE_BITS + 1],
    symbols: Vec<u8>,
    max_len: usize,
}

/// Decode exactly one symbol with the canonical arrays.
///
/// This is the original bit-at-a-time loop, kept off the hot path: it runs only
/// for codes longer than the primary table covers and for stream tails with
/// fewer than `primary_bits` buffered bits. Its checks and error classes are
/// exactly the pre-table decoder's.
#[cold]
#[inline(never)]
fn decode_one_canonical(
    tables: &CanonicalTables,
    reader: &mut BitReader<'_>,
    bits_consumed: &mut usize,
    decoded: usize,
    expected_len: usize,
) -> Result<u8> {
    let mut current_code = 0u32;
    let mut current_len = 0usize;
    loop {
        if current_len > 0 {
            let slot = current_len;
            if current_code >= tables.first_code[slot] {
                let offset = current_code - tables.first_code[slot];
                if offset < tables.counts[slot] {
                    return Ok(tables.symbols[(tables.first_index[slot] + offset) as usize]);
                }
            }
            if current_len >= tables.max_len {
                return Err(Error::corrupt(
                    "huffman-canonical-v1 bitstream contains an invalid code",
                ));
            }
        }
        if *bits_consumed >= reader.stream.len() * 8 {
            return Err(Error::corrupt(format!(
                "huffman-canonical-v1 bitstream ended after {decoded} of {expected_len} bytes"
            )));
        }
        let bit = reader.next_bit();
        current_code = (current_code << 1) | bit;
        current_len += 1;
        *bits_consumed += 1;
    }
}

/// Decode one `huffman-canonical-v1` stream from the start of `encoded` and
/// return the bytes it consumed (the 256-byte table plus its own padded
/// bitstream).
///
/// Used where several independently coded streams are concatenated (mode 11
/// byte planes). The padding bits of the last consumed byte must be zero; any
/// bytes after the consumed prefix belong to the caller, not to this stream.
pub fn decode_prefix(encoded: &[u8], expected_len: usize) -> Result<(Vec<u8>, usize)> {
    decode_prefix_impl(encoded, expected_len, false)
}

/// Force the scalar bit-unpack kernel.
///
/// This exists so the differential test can compare the runtime-dispatched
/// kernel with the scalar one on the same input. It is not part of the public
/// API contract and never changes the wire format or the decoded bytes.
#[doc(hidden)]
pub fn decode_prefix_scalar(encoded: &[u8], expected_len: usize) -> Result<(Vec<u8>, usize)> {
    decode_prefix_impl(encoded, expected_len, true)
}

fn decode_prefix_impl(
    encoded: &[u8],
    expected_len: usize,
    force_scalar: bool,
) -> Result<(Vec<u8>, usize)> {
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
        return Ok((Vec::new(), TABLE_BYTES));
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
        let symbol = lengths
            .iter()
            .position(|&length| length > 0)
            .expect("one used symbol") as u8;
        let capacity = expected_len.min(1 << 20);
        let mut out = Vec::with_capacity(capacity);
        out.resize(expected_len, symbol);
        return Ok((out, TABLE_BYTES));
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
    // of its first symbol in the `(length, symbol)`-sorted symbol list. The
    // same pass fills the packed primary lookup table: every code no longer
    // than `primary_bits` expands into its whole primary-prefix range.
    //
    // Each entry is `symbol | length << 8`; zero is the sentinel for a code
    // longer than the table and is never consumed. The packed form is what the
    // SIMD kernel reads, so there is exactly one table layout.
    let primary_bits = max_len.min(PRIMARY_BITS_MAX);
    let mut table = vec![0u16; 1usize << primary_bits];
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
        if slot <= primary_bits {
            let shift = primary_bits - slot;
            let start = (code as usize) << shift;
            let width = 1usize << shift;
            let packed = u16::from(symbol) | (u16::from(length) << 8);
            for entry in &mut table[start..start + width] {
                *entry = packed;
            }
        }
        code += 1;
        previous = length;
    }

    let tables = CanonicalTables {
        counts,
        first_code,
        first_index,
        symbols,
        max_len,
    };

    let capacity = expected_len.min(1 << 20);
    let mut out = Vec::with_capacity(capacity);
    let primary_bits_u32 = primary_bits as u32;
    let mut state = crate::simd::UnpackState::default();
    loop {
        // Fast path: the runtime-dispatched kernel decodes every symbol the
        // primary table can prove, then returns. AVX2 is selected when the CPU
        // exposes it; the scalar kernel is byte-refill, single probe.
        if force_scalar {
            crate::simd::unpack_scalar_forced(
                &table,
                primary_bits_u32,
                stream,
                expected_len,
                &mut out,
                &mut state,
            );
        } else {
            crate::simd::unpack(
                &table,
                primary_bits_u32,
                stream,
                expected_len,
                &mut out,
                &mut state,
            );
        }
        if out.len() >= expected_len {
            break;
        }

        // Rare exact path: codes longer than the table covers and stream tails
        // with fewer than `primary_bits` buffered bits.
        let mut reader = BitReader {
            stream,
            next_byte: state.next_byte,
            acc: state.acc,
            acc_bits: state.acc_bits,
        };
        let symbol = decode_one_canonical(
            &tables,
            &mut reader,
            &mut state.bits_consumed,
            out.len(),
            expected_len,
        )?;
        state.next_byte = reader.next_byte;
        state.acc = reader.acc;
        state.acc_bits = reader.acc_bits;
        out.push(symbol);
    }
    let bits_consumed = state.bits_consumed;

    // Padding is checked only inside the bytes this stream consumed; any
    // further bytes belong to the caller (the next plane).
    let consumed_stream = bits_consumed.div_ceil(8);
    let used_bits_in_last = bits_consumed % 8;
    if used_bits_in_last != 0 {
        let last = stream[consumed_stream - 1];
        let mask = (1u8 << (8 - used_bits_in_last)) - 1;
        if last & mask != 0 {
            return Err(Error::corrupt(
                "huffman-canonical-v1 padding bits must be zero",
            ));
        }
    }
    Ok((out, TABLE_BYTES + consumed_stream))
}

/// Decode several independent `huffman-canonical-v1` streams.
///
/// This is the multi-stream entry point used by mode 18. It first tries the
/// wide-word kernel in [`crate::simd::unpack_multi`], which only succeeds when
/// every stream is fully proved by the primary table and consumes exactly its
/// whole byte range with zero padding. Otherwise it decodes each stream with
/// [`decode`], in index order, so the decoded bytes and the error class are
/// identical to N independent calls.
pub fn decode_streams(encoded: &[&[u8]], expected: &[usize]) -> Result<Vec<Vec<u8>>> {
    if encoded.len() != expected.len() {
        return Err(Error::malformed(
            "multi-stream decode length tables disagree",
        ));
    }
    if encoded.len() > 1 {
        if let Some(out) = decode_streams_fast(encoded, expected) {
            return Ok(out);
        }
    }
    decode_streams_scalar(encoded, expected)
}

/// Force the per-stream canonical decoder (the scalar fallback).
#[doc(hidden)]
pub fn decode_streams_scalar(encoded: &[&[u8]], expected: &[usize]) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(encoded.len());
    for (bytes, len) in encoded.iter().zip(expected) {
        out.push(decode(bytes, *len)?);
    }
    Ok(out)
}

/// Try the primary-table fast path for all streams at once.
///
/// Returns `None` on anything the frozen canonical decoder must decide.
fn decode_streams_fast(encoded: &[&[u8]], expected: &[usize]) -> Option<Vec<Vec<u8>>> {
    let mut owned: Vec<(Vec<u16>, u32)> = Vec::with_capacity(encoded.len());
    for bytes in encoded {
        if bytes.len() < TABLE_BYTES {
            return None;
        }
        let lengths = &bytes[..TABLE_BYTES];
        let mut counts = [0u32; MAX_CODE_BITS + 1];
        let mut used = 0usize;
        let mut max_len = 0usize;
        for (symbol, &length) in lengths.iter().enumerate() {
            let length = length as usize;
            if length > MAX_CODE_BITS {
                let _ = symbol;
                return None;
            }
            if length > 0 {
                used += 1;
                counts[length] += 1;
                max_len = max_len.max(length);
            }
        }
        if used < 2 {
            return None;
        }
        let space = 1u64 << MAX_CODE_BITS;
        let mut used_space = 0u64;
        for length in 1..=MAX_CODE_BITS {
            used_space += u64::from(counts[length]) << (MAX_CODE_BITS - length);
            if used_space > space {
                return None;
            }
        }
        if used_space != space {
            return None;
        }
        let primary_bits = max_len.min(PRIMARY_BITS_MAX);
        let mut table = vec![0u16; 1usize << primary_bits];
        let mut order: Vec<(u8, u8)> = (0..256)
            .filter(|&symbol| lengths[symbol] > 0)
            .map(|symbol| (lengths[symbol], symbol as u8))
            .collect();
        order.sort_unstable();
        let mut code = 0u32;
        let mut previous = 0u8;
        for (length, symbol) in order {
            code <<= u32::from(length - previous);
            let slot = length as usize;
            if slot <= primary_bits {
                let shift = primary_bits - slot;
                let start = (code as usize) << shift;
                let width = 1usize << shift;
                let packed = u16::from(symbol) | (u16::from(length) << 8);
                for entry in &mut table[start..start + width] {
                    *entry = packed;
                }
            }
            code += 1;
            previous = length;
        }
        owned.push((table, primary_bits as u32));
    }
    let tables: Vec<crate::simd::StreamTable<'_>> = owned
        .iter()
        .map(|(table, primary_bits)| crate::simd::StreamTable {
            table,
            primary_bits: *primary_bits,
        })
        .collect();
    crate::simd::unpack_multi(&tables, encoded, expected)
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

    // ------------------------------------------------------------------
    // Test-only reference decoder.
    //
    // This is a verbatim copy of the pre-table bit-at-a-time decoder. It is
    // never compiled into the library (or the release example) and exists only
    // so the table-driven path can be proved equivalent on every input shape.
    // ------------------------------------------------------------------
    fn decode_prefix_reference(encoded: &[u8], expected_len: usize) -> Result<(Vec<u8>, usize)> {
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
            return Ok((Vec::new(), TABLE_BYTES));
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
            let symbol = lengths
                .iter()
                .position(|&length| length > 0)
                .expect("one used symbol") as u8;
            let capacity = expected_len.min(1 << 20);
            let mut out = Vec::with_capacity(capacity);
            out.resize(expected_len, symbol);
            return Ok((out, TABLE_BYTES));
        }

        if expected_len == 0 {
            return Err(Error::non_canonical(
                "huffman-canonical-v1 non-empty table for an empty plane",
            ));
        }

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

        let consumed_stream = bit_pos.div_ceil(8);
        let used_bits_in_last = bit_pos % 8;
        if used_bits_in_last != 0 {
            let last = stream[consumed_stream - 1];
            let mask = (1u8 << (8 - used_bits_in_last)) - 1;
            if last & mask != 0 {
                return Err(Error::corrupt(
                    "huffman-canonical-v1 padding bits must be zero",
                ));
            }
        }
        Ok((out, TABLE_BYTES + consumed_stream))
    }

    fn decode_reference(encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
        let (out, consumed) = decode_prefix_reference(encoded, expected_len)?;
        if consumed != encoded.len() {
            return Err(Error::corrupt(
                "huffman-canonical-v1 bitstream has trailing bytes after the declared output",
            ));
        }
        Ok(out)
    }

    type Outcome = (
        std::result::Result<Vec<u8>, crate::error::ErrorClass>,
        Option<usize>,
    );

    fn outcome_prefix(encoded: &[u8], expected_len: usize) -> Outcome {
        match decode_prefix(encoded, expected_len) {
            Ok((bytes, consumed)) => (Ok(bytes), Some(consumed)),
            Err(err) => (Err(err.class()), None),
        }
    }

    fn outcome_prefix_reference(encoded: &[u8], expected_len: usize) -> Outcome {
        match decode_prefix_reference(encoded, expected_len) {
            Ok((bytes, consumed)) => (Ok(bytes), Some(consumed)),
            Err(err) => (Err(err.class()), None),
        }
    }

    fn outcome_whole(encoded: &[u8], expected_len: usize) -> Outcome {
        match decode(encoded, expected_len) {
            Ok(bytes) => (Ok(bytes), None),
            Err(err) => (Err(err.class()), None),
        }
    }

    fn outcome_whole_reference(encoded: &[u8], expected_len: usize) -> Outcome {
        match decode_reference(encoded, expected_len) {
            Ok(bytes) => (Ok(bytes), None),
            Err(err) => (Err(err.class()), None),
        }
    }

    fn assert_same(encoded: &[u8], expected_len: usize, label: &str) {
        let fast = outcome_prefix(encoded, expected_len);
        let reference = outcome_prefix_reference(encoded, expected_len);
        assert_eq!(
            fast, reference,
            "prefix decoder differs from reference for {label} (expected_len={expected_len}, encoded_len={})",
            encoded.len()
        );
        let fast = outcome_whole(encoded, expected_len);
        let reference = outcome_whole_reference(encoded, expected_len);
        assert_eq!(
            fast, reference,
            "whole decoder differs from reference for {label} (expected_len={expected_len}, encoded_len={})",
            encoded.len()
        );
    }

    /// xorshift64* — the PRNG family the other WeftFS differential courts use.
    fn xorshift64star(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn random_below(state: &mut u64, bound: usize) -> usize {
        (xorshift64star(state) % bound as u64) as usize
    }

    fn skewed_plane(state: &mut u64, symbol_count: usize, len: usize) -> Vec<u8> {
        // Geometric weights over the lowest `symbol_count` symbols, the shape
        // of a BF16 exponent plane.
        let mut weights = Vec::with_capacity(symbol_count);
        let mut remaining = 1u64 << 20;
        for _ in 0..symbol_count {
            let weight = remaining / 2;
            weights.push(weight.max(1));
            remaining = (remaining - weight).max(1);
        }
        let total: u64 = weights.iter().sum();
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let mut pick = xorshift64star(state) % total;
            let mut symbol = 0usize;
            for (index, &weight) in weights.iter().enumerate() {
                if pick < weight {
                    symbol = index;
                    break;
                }
                pick -= weight;
            }
            out.push(symbol as u8);
        }
        out
    }

    fn fibonacci_plane(max_symbols: usize) -> Vec<u8> {
        let mut fib = [0u64; 32];
        fib[0] = 1;
        fib[1] = 1;
        for index in 2..max_symbols {
            fib[index] = fib[index - 1] + fib[index - 2];
        }
        let mut input = Vec::new();
        for (symbol, &count) in fib.iter().enumerate().take(max_symbols) {
            input.extend(std::iter::repeat(symbol as u8).take(count as usize));
        }
        input
    }

    #[test]
    fn reference_decoder_matches_before_any_change() {
        // Sanity: the frozen reference implementation still round-trips.
        for input in [
            Vec::new(),
            vec![0x00u8; 1024],
            (0..=255u8).collect::<Vec<_>>(),
            pattern(4096, 0xfeed_face_dead_beef),
            skewed_plane(&mut 0x1234_5678_9abc_def0, 12, 4096),
            fibonacci_plane(16),
        ] {
            let encoded = encode(&input);
            assert_eq!(
                decode_reference(&encoded, input.len()).expect("reference decode"),
                input
            );
            assert_same(&encoded, input.len(), "roundtrip");
        }
    }

    #[test]
    fn differential_against_reference_decoder() {
        const CASES: usize = 10_000;
        const KINDS: usize = 12;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for case in 0..CASES {
            match case % KINDS {
                0 => {
                    // Empty plane.
                    let encoded = encode(&[]);
                    assert_same(&encoded, 0, "empty plane");
                }
                1 => {
                    // Single symbol, random symbol and length.
                    let symbol = (xorshift64star(&mut state) & 0xff) as u8;
                    let len = random_below(&mut state, 4096) + 1;
                    let input = vec![symbol; len];
                    let encoded = encode(&input);
                    assert_same(&encoded, input.len(), "single symbol");
                }
                2 => {
                    // Uniform over all 256 symbols.
                    let len = random_below(&mut state, 4096) + 1;
                    let input = pattern(len, xorshift64star(&mut state));
                    let encoded = encode(&input);
                    assert_same(&encoded, input.len(), "uniform 256");
                }
                3 => {
                    // Highly skewed over 8 symbols.
                    let len = random_below(&mut state, 4096) + 1;
                    let input = skewed_plane(&mut state, 8, len);
                    let encoded = encode(&input);
                    assert_same(&encoded, input.len(), "skewed 8");
                }
                4 => {
                    // Highly skewed over 16 symbols.
                    let len = random_below(&mut state, 4096) + 1;
                    let input = skewed_plane(&mut state, 16, len);
                    let encoded = encode(&input);
                    assert_same(&encoded, input.len(), "skewed 16");
                }
                5 => {
                    // Fibonacci weights force the length limiter.
                    let symbols = random_below(&mut state, 24) + 2;
                    let input = fibonacci_plane(symbols);
                    let encoded = encode(&input);
                    assert_same(&encoded, input.len(), "fibonacci");
                }
                6 => {
                    // Truncated valid stream.
                    let len = random_below(&mut state, 1024) + 1;
                    let input = skewed_plane(&mut state, 12, len);
                    let encoded = encode(&input);
                    let cut = random_below(&mut state, encoded.len() - TABLE_BYTES + 1);
                    let truncated = encoded[..TABLE_BYTES + cut].to_vec();
                    assert_same(&truncated, input.len(), "truncated");
                }
                7 => {
                    // Appended trailing bytes.
                    let len = random_below(&mut state, 1024) + 1;
                    let input = pattern(len, xorshift64star(&mut state));
                    let mut encoded = encode(&input);
                    let extra = random_below(&mut state, 4) + 1;
                    for _ in 0..extra {
                        encoded.push((xorshift64star(&mut state) & 0xff) as u8);
                    }
                    assert_same(&encoded, input.len(), "trailing bytes");
                }
                8 => {
                    // Non-zero padding: flip a bit in the final byte.
                    let len = random_below(&mut state, 1024) + 1;
                    let input = skewed_plane(&mut state, 6, len);
                    let mut encoded = encode(&input);
                    if encoded.len() > TABLE_BYTES {
                        let last = encoded.len() - 1;
                        encoded[last] |= 1;
                    }
                    assert_same(&encoded, input.len(), "non-zero padding");
                }
                9 => {
                    // Arbitrary length table plus arbitrary stream.
                    let mut encoded = vec![0u8; TABLE_BYTES];
                    let entries = random_below(&mut state, 256) + 1;
                    for _ in 0..entries {
                        let symbol = random_below(&mut state, 256);
                        encoded[symbol] = (xorshift64star(&mut state) % 25) as u8;
                    }
                    let stream_len = random_below(&mut state, 64);
                    for _ in 0..stream_len {
                        encoded.push((xorshift64star(&mut state) & 0xff) as u8);
                    }
                    let expected_len = random_below(&mut state, 128);
                    assert_same(&encoded, expected_len, "arbitrary table");
                }
                10 => {
                    // Over-long code length (> 24).
                    let mut encoded = vec![0u8; TABLE_BYTES];
                    let symbol = random_below(&mut state, 256);
                    encoded[symbol] = (xorshift64star(&mut state) % 231 + 25) as u8;
                    let expected_len = random_below(&mut state, 32);
                    assert_same(&encoded, expected_len, "over-long length");
                }
                _ => {
                    // Over-subscribed and incomplete tables from short lengths.
                    let mut encoded = vec![0u8; TABLE_BYTES];
                    let entries = random_below(&mut state, 8) + 2;
                    for _ in 0..entries {
                        let symbol = random_below(&mut state, 256);
                        encoded[symbol] = (xorshift64star(&mut state) % 4 + 1) as u8;
                    }
                    let stream_len = random_below(&mut state, 16);
                    for _ in 0..stream_len {
                        encoded.push((xorshift64star(&mut state) & 0xff) as u8);
                    }
                    let expected_len = random_below(&mut state, 64);
                    assert_same(&encoded, expected_len, "kraft-mismatch table");
                }
            }
        }
    }

    #[test]
    fn reference_decoder_agrees_on_prefix_consumption() {
        // Concatenated planes (modes 11-15) depend on the exact consumed byte
        // count, so check it directly over a mixed corpus.
        let mut state = 0xa5a5_5a5a_0123_4567u64;
        let mut packed = Vec::new();
        let mut lengths = Vec::new();
        for _ in 0..64 {
            let len = random_below(&mut state, 2048) + 1;
            let input = skewed_plane(&mut state, 10, len);
            packed.extend_from_slice(&encode(&input));
            lengths.push(input.len());
        }
        let mut offset = 0usize;
        for &expected_len in &lengths {
            let (fast, fast_consumed) =
                decode_prefix(&packed[offset..], expected_len).expect("fast");
            let (reference, reference_consumed) =
                decode_prefix_reference(&packed[offset..], expected_len).expect("reference");
            assert_eq!(fast, reference);
            assert_eq!(fast_consumed, reference_consumed);
            offset += fast_consumed;
        }
        assert_eq!(offset, packed.len());
    }

    /// 10,000 randomised multi-stream sets: the wide-word multi-stream kernel
    /// must agree with N independent canonical decodes, in bytes and in error
    /// class.
    #[test]
    fn multi_stream_kernel_matches_independent_decoders() {
        let mut state = 0x18_18_18_18_dead_beefu64;
        for case in 0..10_000usize {
            let n = 2 + random_below(&mut state, 7); // 2..=8
            let mut planes: Vec<Vec<u8>> = Vec::with_capacity(n);
            let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(n);
            for stream in 0..n {
                let len = random_below(&mut state, 600);
                let plane = match (case + stream) % 3 {
                    0 => skewed_plane(&mut state, 4 + stream, len),
                    1 => skewed_plane(&mut state, 32, len),
                    _ => skewed_plane(&mut state, 200, len),
                };
                encoded.push(encode(&plane));
                planes.push(plane);
            }
            let expected: Vec<usize> = planes.iter().map(Vec::len).collect();
            let refs: Vec<&[u8]> = encoded.iter().map(Vec::as_slice).collect();
            let fast = decode_streams(&refs, &expected).expect("multi-stream");
            let slow = decode_streams_scalar(&refs, &expected).expect("scalar");
            assert_eq!(fast, slow, "case {case}");
            assert_eq!(fast, planes, "case {case}");

            // Corrupt one stream in three of the cases: both paths must
            // return the same error class.
            if case % 3 == 0 {
                let victim = random_below(&mut state, n);
                let mut mutated = encoded.clone();
                if mutated[victim].len() > TABLE_BYTES + 1 {
                    mutated[victim].pop();
                } else {
                    mutated[victim].push(0xff);
                }
                let refs: Vec<&[u8]> = mutated.iter().map(Vec::as_slice).collect();
                let fast = decode_streams(&refs, &expected).err().map(|e| e.class());
                let slow = decode_streams_scalar(&refs, &expected)
                    .err()
                    .map(|e| e.class());
                assert_eq!(fast, slow, "corrupt case {case}");
            }
        }
    }
}
