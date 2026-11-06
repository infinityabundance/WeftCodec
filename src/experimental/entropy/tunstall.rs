//! Experimental Tunstall (variable-to-fixed) coder with `k`-bit codewords.
//!
//! ## Dictionary construction (normative for this experiment)
//!
//! The alphabet is always the **full 256-byte alphabet**. The initial
//! dictionary is the 256 one-byte phrases `[0] … [255]`; symbol `s` carries
//! the exact probability `count[s] / total`. A symbol that does not occur has
//! probability 0 and is never expanded, but its one-byte phrase remains in the
//! dictionary, so every byte is always encodable.
//!
//! Construction repeatedly performs exactly one operation:
//!
//! > choose the phrase with the highest probability, breaking exact ties by
//! > `(phrase bytes ascending, insertion index ascending)`, remove it, and
//! > insert its 256 one-byte extensions `phrase || [s]` for `s = 0 … 255`
//! > with probability `p(phrase) * p(s)`.
//!
//! A split therefore replaces 1 entry with 256 entries: the dictionary grows
//! by 255. Splitting stops when the **next** split would push the dictionary
//! above `2^k` entries, or when the serialised dictionary would exceed
//! `MAX_DICT_BYTES` (1 MiB), or immediately when the histogram has no
//! observations. At `k = 8` no split fits (`256 + 255 > 256`); at `k = 10`
//! three splits fit (1021 entries); at `k = 12` fifteen splits fit (4081
//! entries). This arithmetic is a property of a 256-symbol alphabet with
//! fixed-width codewords and is reported by the court.
//!
//! Because phrase bytes are unique, the insertion-index tie-break is never
//! actually consulted; it is implemented anyway so the rule is total.
//!
//! The final dictionary is sorted by phrase bytes ascending and codeword
//! `i` is the `i`-th entry of that sorted list. Dictionary serialisation is
//! canonical: `u16le count || count * (u16le len || bytes)`.
//!
//! ## Partial final phrase and unused codewords
//!
//! Encoding is greedy longest-match. A phrase can only fail to match at the
//! very end of the input: every split covers all 256 one-byte extensions, so
//! any position with at least one following byte is covered. The remaining
//! suffix (a strict prefix of a split phrase) is emitted **literally** in a
//! tail section. Nothing is ambiguous: the codeword count and tail length are
//! explicit in the stream, and the decoder requires
//! `decoded_codewords + tail == expected_len` exactly.
//!
//! Codewords in `[dict_count, 2^k)` are **invalid** and rejected as
//! [`ErrorClass::Corrupt`]. Bits beyond the final codeword in the last byte
//! must be zero.
//!
//! ## Stream form
//!
//! Self-contained (`encode`):
//!
//! ```text
//! u8 version = 1
//! u8 k
//! u32le dictionary_len
//! dictionary bytes
//! u32le codeword_count
//! u32le tail_len
//! tail bytes
//! k-bit codewords, MSB-first, zero-padded
//! ```
//!
//! Scoped payload (`encode_with_dict`, used by the per-tensor and family
//! dictionary scopes) is the same stream without the first four fields; the
//! dictionary is carried separately. The court charges the full serialised
//! dictionary exactly once per scope.
//!
//! [`ErrorClass::Corrupt`]: crate::error::ErrorClass::Corrupt

use crate::error::{Error, Result};

use super::{append_bits, check_expected_len, Coder};

/// Stream version.
pub const VERSION: u8 = 1;
/// Supported codeword widths.
pub const SUPPORTED_K: [u8; 3] = [8, 10, 12];
/// Largest serialised dictionary this construction will build.
pub const MAX_DICT_BYTES: usize = 1 << 20;
/// Largest literal tail (a phrase length is at most 65535 bytes).
pub const MAX_TAIL_BYTES: usize = 1 << 16;

/// Experimental Tunstall coder with `k`-bit codewords.
#[derive(Clone, Copy, Debug)]
pub struct Tunstall {
    k: u8,
}

/// One dictionary phrase and its construction metadata.
#[derive(Clone, Debug)]
struct Leaf {
    bytes: Vec<u8>,
    probability: f64,
    insertion: u64,
}

fn better(a: &Leaf, b: &Leaf) -> bool {
    if a.probability != b.probability {
        return a.probability > b.probability;
    }
    if a.bytes != b.bytes {
        return a.bytes < b.bytes;
    }
    a.insertion < b.insertion
}

#[derive(Clone, Debug, Default)]
struct TrieNode {
    children: Vec<(u8, usize)>,
    terminal: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    fn build(entries: &[Vec<u8>]) -> Result<Self> {
        let mut trie = Trie {
            nodes: vec![TrieNode::default()],
        };
        for (index, phrase) in entries.iter().enumerate() {
            if phrase.is_empty() {
                return Err(Error::non_canonical(
                    "Tunstall dictionary has an empty phrase",
                ));
            }
            let mut node = 0usize;
            for &byte in phrase {
                let position = trie.nodes[node]
                    .children
                    .binary_search_by_key(&byte, |&(key, _)| key);
                node = match position {
                    Ok(found) => trie.nodes[node].children[found].1,
                    Err(insert) => {
                        let child = trie.nodes.len();
                        trie.nodes.push(TrieNode::default());
                        trie.nodes[node].children.insert(insert, (byte, child));
                        child
                    }
                };
            }
            if trie.nodes[node].terminal.is_some() {
                return Err(Error::non_canonical(
                    "Tunstall dictionary contains a duplicate phrase",
                ));
            }
            trie.nodes[node].terminal = Some(index);
        }
        // Phrases must be prefix-free. A terminal node with any child means a
        // longer phrase extends a shorter one.
        for node in &trie.nodes {
            if node.terminal.is_some() && !node.children.is_empty() {
                return Err(Error::non_canonical(
                    "Tunstall dictionary is not prefix-free",
                ));
            }
        }
        Ok(trie)
    }

    fn longest_match(&self, input: &[u8]) -> Option<(usize, usize)> {
        let mut node = 0usize;
        let mut best: Option<(usize, usize)> = None;
        for (offset, &byte) in input.iter().enumerate() {
            let position = self.nodes[node]
                .children
                .binary_search_by_key(&byte, |&(key, _)| key);
            let next = match position {
                Ok(found) => self.nodes[node].children[found].1,
                Err(_) => break,
            };
            node = next;
            if let Some(index) = self.nodes[node].terminal {
                best = Some((offset + 1, index));
            }
        }
        best
    }
}

/// A canonical Tunstall dictionary.
#[derive(Clone, Debug)]
pub struct Dictionary {
    k: u8,
    entries: Vec<Vec<u8>>,
    trie: Trie,
}

impl Dictionary {
    /// Codeword width.
    pub fn k(&self) -> u8 {
        self.k
    }

    /// Number of phrases (`<= 2^k`).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the dictionary has no entries (only possible for a zero-entry
    /// dictionary parsed from bytes; construction always has 256 entries).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Canonical serialised bytes: `u16le count || count * (u16le len || bytes)`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.serialized_len());
        out.extend_from_slice(&(self.entries.len() as u16).to_le_bytes());
        for phrase in &self.entries {
            out.extend_from_slice(&(phrase.len() as u16).to_le_bytes());
            out.extend_from_slice(phrase);
        }
        out
    }

    /// Exact canonical serialised length.
    pub fn serialized_len(&self) -> usize {
        2 + self
            .entries
            .iter()
            .map(|phrase| 2 + phrase.len())
            .sum::<usize>()
    }

    /// Parse and validate a canonical dictionary.
    pub fn from_bytes(bytes: &[u8], k: u8) -> Result<Self> {
        if !SUPPORTED_K.contains(&k) {
            return Err(Error::unsupported_feature(format!(
                "experimental Tunstall supports k in {SUPPORTED_K:?}, got {k}"
            )));
        }
        if bytes.len() < 2 {
            return Err(Error::malformed(
                "Tunstall dictionary is shorter than its 2-byte entry count",
            ));
        }
        let count = usize::from(u16::from_le_bytes([bytes[0], bytes[1]]));
        let cap = 1usize << k;
        if count > cap {
            return Err(Error::malformed(format!(
                "Tunstall dictionary has {count} phrases but k={k} allows at most {cap}"
            )));
        }
        let mut entries = Vec::with_capacity(count);
        let mut position = 2usize;
        for _ in 0..count {
            if position + 2 > bytes.len() {
                return Err(Error::malformed(
                    "Tunstall dictionary ends inside a phrase length",
                ));
            }
            let length = usize::from(u16::from_le_bytes([bytes[position], bytes[position + 1]]));
            position += 2;
            if length == 0 {
                return Err(Error::non_canonical(
                    "Tunstall dictionary contains an empty phrase",
                ));
            }
            let end = position
                .checked_add(length)
                .ok_or_else(|| Error::malformed("Tunstall phrase length overflows usize"))?;
            if end > bytes.len() {
                return Err(Error::malformed("Tunstall dictionary ends inside a phrase"));
            }
            entries.push(bytes[position..end].to_vec());
            position = end;
        }
        if position != bytes.len() {
            return Err(Error::malformed(
                "Tunstall dictionary has trailing bytes after its phrases",
            ));
        }
        for window in entries.windows(2) {
            if window[0] >= window[1] {
                return Err(Error::non_canonical(
                    "Tunstall dictionary phrases are not strictly ascending",
                ));
            }
        }
        let trie = Trie::build(&entries)?;
        Ok(Dictionary { k, entries, trie })
    }

    /// Build a dictionary from a 256-symbol histogram with the documented
    /// deterministic split rule.
    pub fn build(k: u8, counts: &[u64; 256]) -> Result<Self> {
        if !SUPPORTED_K.contains(&k) {
            return Err(Error::unsupported_feature(format!(
                "experimental Tunstall supports k in {SUPPORTED_K:?}, got {k}"
            )));
        }
        let cap = 1usize << k;
        let total: u64 = counts.iter().sum();
        let denominator = if total == 0 { 1.0 } else { total as f64 };
        let probability: [f64; 256] =
            std::array::from_fn(|symbol| counts[symbol] as f64 / denominator);

        let mut leaves: Vec<Leaf> = (0..256)
            .map(|symbol| Leaf {
                bytes: vec![symbol as u8],
                probability: probability[symbol],
                insertion: symbol as u64,
            })
            .collect();
        let mut next_insertion: u64 = 256;
        // 2-byte count + 256 * (2-byte length + 1-byte phrase).
        let mut dictionary_bytes = 2 + 256 * 3;

        if total > 0 {
            loop {
                // A split replaces one phrase with 256 and therefore adds 255.
                if leaves.len() + 255 > cap {
                    break;
                }
                let mut best = 0usize;
                for index in 1..leaves.len() {
                    if better(&leaves[index], &leaves[best]) {
                        best = index;
                    }
                }
                let phrase = leaves[best].bytes.clone();
                let parent_probability = leaves[best].probability;
                let removed = 2 + phrase.len();
                let added = 256usize * (2 + phrase.len() + 1);
                if dictionary_bytes + added - removed > MAX_DICT_BYTES {
                    break;
                }
                leaves.remove(best);
                for symbol in 0..256usize {
                    let mut child = Vec::with_capacity(phrase.len() + 1);
                    child.extend_from_slice(&phrase);
                    child.push(symbol as u8);
                    leaves.push(Leaf {
                        bytes: child,
                        probability: parent_probability * probability[symbol],
                        insertion: next_insertion,
                    });
                    next_insertion += 1;
                }
                dictionary_bytes = dictionary_bytes + added - removed;
            }
        }

        leaves.sort_by(|a, b| a.bytes.cmp(&b.bytes));
        let entries: Vec<Vec<u8>> = leaves.into_iter().map(|leaf| leaf.bytes).collect();
        for window in entries.windows(2) {
            if window[0] >= window[1] {
                return Err(Error::non_canonical(
                    "Tunstall construction produced duplicate phrases",
                ));
            }
        }
        let trie = Trie::build(&entries)?;
        Ok(Dictionary { k, entries, trie })
    }

    /// The canonical dictionary entry at `index`.
    pub fn phrase(&self, index: usize) -> Option<&[u8]> {
        self.entries.get(index).map(|phrase| phrase.as_slice())
    }
}

impl Tunstall {
    /// Construct a Tunstall coder with `k`-bit codewords.
    pub fn new(k: u8) -> Result<Self> {
        if !SUPPORTED_K.contains(&k) {
            return Err(Error::unsupported_feature(format!(
                "experimental Tunstall supports k in {SUPPORTED_K:?}, got {k}"
            )));
        }
        Ok(Tunstall { k })
    }

    /// Codeword width.
    pub fn k(&self) -> u8 {
        self.k
    }

    /// Build a dictionary from one histogram.
    pub fn build_dict(&self, counts: &[u64; 256]) -> Result<Dictionary> {
        Dictionary::build(self.k, counts)
    }

    /// Build a dictionary from the summed histogram of several planes.
    pub fn build_dict_from_planes(&self, planes: &[&[u8]]) -> Result<Dictionary> {
        let mut counts = [0u64; 256];
        for plane in planes {
            for &byte in *plane {
                counts[byte as usize] += 1;
            }
        }
        self.build_dict(&counts)
    }

    /// Encode `input` with an externally supplied dictionary. The returned
    /// bytes are the scoped payload (no dictionary and no four-field header).
    pub fn encode_with_dict(&self, input: &[u8], dictionary: &Dictionary) -> Result<Vec<u8>> {
        if dictionary.k != self.k {
            return Err(Error::malformed(format!(
                "Tunstall dictionary is k={} but the coder is k={}",
                dictionary.k, self.k
            )));
        }
        let mut codewords: Vec<usize> = Vec::new();
        let mut position = 0usize;
        while position < input.len() {
            match dictionary.trie.longest_match(&input[position..]) {
                Some((length, index)) if length > 0 => {
                    codewords.push(index);
                    position += length;
                }
                _ => break,
            }
        }
        if codewords.len() > u32::MAX as usize {
            return Err(Error::limit_exceeded(
                "Tunstall codeword count does not fit u32",
            ));
        }
        let tail = &input[position..];
        if tail.len() > MAX_TAIL_BYTES {
            return Err(Error::limit_exceeded(format!(
                "Tunstall literal tail is {} bytes (limit {MAX_TAIL_BYTES})",
                tail.len()
            )));
        }
        let total_bits = codewords
            .len()
            .checked_mul(self.k as usize)
            .ok_or_else(|| Error::limit_exceeded("Tunstall bit count overflows usize"))?;
        let bit_bytes = total_bits / 8 + usize::from(total_bits % 8 != 0);
        let mut out = Vec::with_capacity(8 + tail.len() + bit_bytes);
        out.extend_from_slice(&(codewords.len() as u32).to_le_bytes());
        out.extend_from_slice(&(tail.len() as u32).to_le_bytes());
        out.extend_from_slice(tail);
        let mut buffer = 0u64;
        let mut pending = 0u32;
        for &index in &codewords {
            append_bits(
                &mut out,
                &mut buffer,
                &mut pending,
                index as u32,
                u32::from(self.k),
            );
        }
        if pending > 0 {
            out.push((buffer << (8 - pending)) as u8);
        }
        Ok(out)
    }

    /// Decode a scoped payload with an externally supplied dictionary.
    pub fn decode_with_dict(
        &self,
        payload: &[u8],
        expected_len: usize,
        dictionary: &Dictionary,
    ) -> Result<Vec<u8>> {
        check_expected_len(expected_len)?;
        if dictionary.k != self.k {
            return Err(Error::malformed(format!(
                "Tunstall dictionary is k={} but the coder is k={}",
                dictionary.k, self.k
            )));
        }
        if payload.len() < 8 {
            return Err(Error::corrupt(
                "Tunstall payload is shorter than its two u32 counters",
            ));
        }
        let count = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        let tail_len =
            u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
        if tail_len > MAX_TAIL_BYTES {
            return Err(Error::corrupt(format!(
                "Tunstall payload declares a {tail_len}-byte literal tail (limit {MAX_TAIL_BYTES})"
            )));
        }
        let tail_start = 8usize;
        let tail_end = tail_start
            .checked_add(tail_len)
            .ok_or_else(|| Error::corrupt("Tunstall tail length overflows usize"))?;
        if tail_end > payload.len() {
            return Err(Error::corrupt(
                "Tunstall payload ends inside its literal tail",
            ));
        }
        let tail = &payload[tail_start..tail_end];
        let bitstream = &payload[tail_end..];
        let total_bits = count
            .checked_mul(self.k as usize)
            .ok_or_else(|| Error::corrupt("Tunstall codeword bit count overflows usize"))?;
        let bit_bytes = total_bits / 8 + usize::from(total_bits % 8 != 0);
        if bitstream.len() != bit_bytes {
            return Err(Error::corrupt(format!(
                "Tunstall bitstream is {} bytes but {count} x {} bits requires {bit_bytes}",
                bitstream.len(),
                self.k
            )));
        }
        if total_bits % 8 != 0 {
            let padding = 8 - (total_bits % 8);
            let last = *bitstream.last().expect("nonzero bit length");
            if last & ((1u8 << padding) - 1) != 0 {
                return Err(Error::corrupt(
                    "Tunstall bitstream has non-zero padding bits",
                ));
            }
        }

        let mut out = Vec::with_capacity(expected_len);
        let mut reader = BitReader::new(bitstream);
        for _ in 0..count {
            let index = reader
                .read(self.k)
                .ok_or_else(|| Error::corrupt("Tunstall bitstream ended inside a codeword"))?
                as usize;
            let phrase = dictionary.entries.get(index).ok_or_else(|| {
                Error::corrupt(format!(
                    "Tunstall codeword {index} is unused (dictionary has {} phrases)",
                    dictionary.entries.len()
                ))
            })?;
            let new_len = out
                .len()
                .checked_add(phrase.len())
                .ok_or_else(|| Error::corrupt("Tunstall decoded length overflows usize"))?;
            if new_len + tail_len > expected_len {
                return Err(Error::corrupt(format!(
                    "Tunstall output would reach {new_len} + {tail_len} tail bytes, beyond expected_len {expected_len}"
                )));
            }
            out.extend_from_slice(phrase);
        }
        if out.len() + tail_len != expected_len {
            return Err(Error::corrupt(format!(
                "Tunstall codewords decode to {} bytes + {tail_len} tail bytes, not expected_len {expected_len}",
                out.len()
            )));
        }
        out.extend_from_slice(tail);
        Ok(out)
    }

    /// Decode a scoped payload directly into `out`.
    pub fn decode_into_with_dict(
        &self,
        payload: &[u8],
        out: &mut [u8],
        dictionary: &Dictionary,
    ) -> Result<()> {
        let decoded = self.decode_with_dict(payload, out.len(), dictionary)?;
        out.copy_from_slice(&decoded);
        Ok(())
    }
}

/// MSB-first bit reader over a fixed byte slice.
struct BitReader<'a> {
    stream: &'a [u8],
    next_byte: usize,
    accumulator: u64,
    bits: u32,
}

impl<'a> BitReader<'a> {
    fn new(stream: &'a [u8]) -> Self {
        BitReader {
            stream,
            next_byte: 0,
            accumulator: 0,
            bits: 0,
        }
    }

    fn refill(&mut self) {
        while self.bits <= 56 && self.next_byte < self.stream.len() {
            self.accumulator |= u64::from(self.stream[self.next_byte]) << (56 - self.bits);
            self.next_byte += 1;
            self.bits += 8;
        }
    }

    fn read(&mut self, width: u8) -> Option<u32> {
        let width = u32::from(width);
        if self.bits < width {
            self.refill();
        }
        if self.bits < width {
            return None;
        }
        let shift = 64 - width;
        let value = (self.accumulator >> shift) as u32;
        self.accumulator <<= width;
        self.bits -= width;
        Some(value)
    }
}

impl Coder for Tunstall {
    fn id(&self) -> &'static str {
        match self.k {
            8 => "tunstall-8",
            10 => "tunstall-10",
            12 => "tunstall-12",
            _ => "tunstall",
        }
    }

    fn encode(&self, input: &[u8]) -> Vec<u8> {
        let mut counts = [0u64; 256];
        for &byte in input {
            counts[byte as usize] += 1;
        }
        let dictionary = Dictionary::build(self.k, &counts)
            .expect("Tunstall construction is total for byte histograms");
        let payload = self
            .encode_with_dict(input, &dictionary)
            .expect("a constructed dictionary always covers a byte input");
        let dictionary_bytes = dictionary.to_bytes();
        let mut out = Vec::with_capacity(6 + dictionary_bytes.len() + payload.len());
        out.push(VERSION);
        out.push(self.k);
        out.extend_from_slice(&(dictionary_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&dictionary_bytes);
        out.extend_from_slice(&payload);
        out
    }

    fn decode(&self, encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
        check_expected_len(expected_len)?;
        if encoded.len() < 6 {
            return Err(Error::malformed(
                "Tunstall stream is shorter than its 6-byte header",
            ));
        }
        if encoded[0] != VERSION {
            return Err(Error::malformed(format!(
                "unsupported Tunstall stream version {} (expected {VERSION})",
                encoded[0]
            )));
        }
        let k = encoded[1];
        if k != self.k {
            return Err(Error::malformed(format!(
                "Tunstall stream is k={k} but the coder is k={}",
                self.k
            )));
        }
        let dictionary_len =
            u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]) as usize;
        let dictionary_end = 6usize
            .checked_add(dictionary_len)
            .ok_or_else(|| Error::malformed("Tunstall dictionary length overflows usize"))?;
        if dictionary_end > encoded.len() {
            return Err(Error::malformed(
                "Tunstall stream ends inside its dictionary",
            ));
        }
        let dictionary = Dictionary::from_bytes(&encoded[6..dictionary_end], k)?;
        self.decode_with_dict(&encoded[dictionary_end..], expected_len, &dictionary)
    }

    fn decode_into(&self, encoded: &[u8], out: &mut [u8]) -> Result<()> {
        let decoded = self.decode(encoded, out.len())?;
        out.copy_from_slice(&decoded);
        Ok(())
    }
}
