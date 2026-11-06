//! Tests for the **experimental** entropy coders.
//!
//! These tests do not touch any persistent format. They prove exact
//! round-trips, the documented Tunstall dictionary rule, and differential
//! agreement between the optimised decoders and deliberately simple reference
//! decoders on 5,000 random inputs plus 5,000 mutated inputs per coder.

use std::collections::{BTreeSet, HashSet};

use weftcodec::error::ErrorClass;
use weftcodec::experimental::entropy::huffman::Huffman;
use weftcodec::experimental::entropy::rans::Rans;
use weftcodec::experimental::entropy::raw::Raw;
use weftcodec::experimental::entropy::tunstall::{Dictionary, Tunstall, SUPPORTED_K};
use weftcodec::experimental::entropy::zstd::Zstd;
use weftcodec::experimental::entropy::Coder;

// ---------------------------------------------------------------------------
// Deterministic random source
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next() % bound as u64) as usize
        }
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

fn skewed_input(rng: &mut Rng, len: usize, alphabet: usize) -> Vec<u8> {
    let alphabet = alphabet.clamp(1, 256);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        // Geometric-ish skew: the first symbols dominate.
        let mut index = rng.below(alphabet);
        if index > 0 && rng.below(4) != 0 {
            index = rng.below(index.min(4).max(1));
        }
        out.push(index as u8);
    }
    out
}

// ---------------------------------------------------------------------------
// Reference decoders (deliberately simple, independent of the fast paths)
// ---------------------------------------------------------------------------

fn ref_read_bits(stream: &[u8], bit: usize, width: u32) -> u32 {
    let mut value = 0u32;
    for offset in 0..width {
        let position = bit + offset as usize;
        let byte = stream[position >> 3];
        let bit_value = (byte >> (7 - (position & 7))) & 1;
        value = (value << 1) | u32::from(bit_value);
    }
    value
}

fn ref_tunstall_decode(encoded: &[u8], expected_len: usize, k: u8) -> Result<Vec<u8>, ErrorClass> {
    if encoded.len() < 6 {
        return Err(ErrorClass::Malformed);
    }
    if encoded[0] != 1 {
        return Err(ErrorClass::Malformed);
    }
    if encoded[1] != k {
        return Err(ErrorClass::Malformed);
    }
    let dictionary_len =
        u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]) as usize;
    let dictionary_end = match 6usize.checked_add(dictionary_len) {
        Some(end) if end <= encoded.len() => end,
        _ => return Err(ErrorClass::Malformed),
    };
    let dictionary = match ref_dictionary(&encoded[6..dictionary_end], k) {
        Ok(entries) => entries,
        Err(class) => return Err(class),
    };
    let payload = &encoded[dictionary_end..];
    if payload.len() < 8 {
        return Err(ErrorClass::Corrupt);
    }
    let count = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let tail_len = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
    if tail_len > 1 << 16 {
        return Err(ErrorClass::Corrupt);
    }
    let tail_end = match 8usize.checked_add(tail_len) {
        Some(end) if end <= payload.len() => end,
        _ => return Err(ErrorClass::Corrupt),
    };
    let tail = &payload[8..tail_end];
    let bitstream = &payload[tail_end..];
    let total_bits = match count.checked_mul(k as usize) {
        Some(bits) => bits,
        None => return Err(ErrorClass::Corrupt),
    };
    let bit_bytes = total_bits / 8 + usize::from(total_bits % 8 != 0);
    if bitstream.len() != bit_bytes {
        return Err(ErrorClass::Corrupt);
    }
    if total_bits % 8 != 0 {
        let padding = 8 - total_bits % 8;
        let last = *bitstream.last().unwrap();
        if last & ((1u8 << padding) - 1) != 0 {
            return Err(ErrorClass::Corrupt);
        }
    }
    let mut out: Vec<u8> = Vec::new();
    for index in 0..count {
        let code = ref_read_bits(bitstream, index * k as usize, u32::from(k)) as usize;
        let phrase = match dictionary.get(code) {
            Some(phrase) => phrase,
            None => return Err(ErrorClass::Corrupt),
        };
        if out.len() + phrase.len() + tail_len > expected_len {
            return Err(ErrorClass::Corrupt);
        }
        out.extend_from_slice(phrase);
    }
    if out.len() + tail_len != expected_len {
        return Err(ErrorClass::Corrupt);
    }
    out.extend_from_slice(tail);
    Ok(out)
}

fn ref_dictionary(bytes: &[u8], k: u8) -> Result<Vec<Vec<u8>>, ErrorClass> {
    if bytes.len() < 2 {
        return Err(ErrorClass::Malformed);
    }
    let count = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    if count > (1usize << k) {
        return Err(ErrorClass::Malformed);
    }
    let mut entries: Vec<Vec<u8>> = Vec::with_capacity(count);
    let mut position = 2usize;
    for _ in 0..count {
        if position + 2 > bytes.len() {
            return Err(ErrorClass::Malformed);
        }
        let length = u16::from_le_bytes([bytes[position], bytes[position + 1]]) as usize;
        position += 2;
        if length == 0 {
            return Err(ErrorClass::NonCanonical);
        }
        if position + length > bytes.len() {
            return Err(ErrorClass::Malformed);
        }
        entries.push(bytes[position..position + length].to_vec());
        position += length;
    }
    if position != bytes.len() {
        return Err(ErrorClass::Malformed);
    }
    let mut seen: BTreeSet<Vec<u8>> = BTreeSet::new();
    for entry in &entries {
        if !seen.insert(entry.clone()) {
            return Err(ErrorClass::NonCanonical);
        }
    }
    // Sorted order is required, and sorted order plus prefix-freedom is the
    // canonical dictionary form.
    for window in entries.windows(2) {
        if window[0] >= window[1] {
            return Err(ErrorClass::NonCanonical);
        }
    }
    let set: HashSet<&[u8]> = entries.iter().map(|entry| entry.as_slice()).collect();
    for entry in &entries {
        for prefix_len in 1..entry.len() {
            if set.contains(&entry[..prefix_len]) {
                return Err(ErrorClass::NonCanonical);
            }
        }
    }
    Ok(entries)
}

fn ref_rans_decode(encoded: &[u8], expected_len: usize) -> Result<Vec<u8>, ErrorClass> {
    if encoded.len() < 4 {
        return Err(ErrorClass::Malformed);
    }
    if encoded[0] != 1 || encoded[1] != 12 {
        return Err(ErrorClass::Malformed);
    }
    let symbol_count = u16::from_le_bytes([encoded[2], encoded[3]]) as usize;
    if symbol_count > 256 {
        return Err(ErrorClass::Malformed);
    }
    let table_end = 4 + symbol_count * 3;
    if encoded.len() < table_end + 4 {
        return Err(ErrorClass::Malformed);
    }
    let mut pairs: Vec<(u8, u16)> = Vec::with_capacity(symbol_count);
    let mut sum: u32 = 0;
    for index in 0..symbol_count {
        let offset = 4 + index * 3;
        let symbol = encoded[offset];
        let frequency = u16::from_le_bytes([encoded[offset + 1], encoded[offset + 2]]);
        if let Some(&(last, _)) = pairs.last() {
            if symbol <= last {
                return Err(ErrorClass::Malformed);
            }
        }
        if frequency == 0 {
            return Err(ErrorClass::Malformed);
        }
        sum += u32::from(frequency);
        pairs.push((symbol, frequency));
    }
    if symbol_count > 0 && sum != 4096 {
        return Err(ErrorClass::Malformed);
    }
    let state_offset = table_end;
    let mut state = u32::from_le_bytes([
        encoded[state_offset],
        encoded[state_offset + 1],
        encoded[state_offset + 2],
        encoded[state_offset + 3],
    ]);
    let renorm = &encoded[state_offset + 4..];
    if pairs.is_empty() {
        if expected_len != 0 || state != 1 << 23 || !renorm.is_empty() {
            return Err(ErrorClass::Corrupt);
        }
        return Ok(Vec::new());
    }
    let mut slots = [0u8; 4096];
    let mut next = 0usize;
    for &(symbol, frequency) in &pairs {
        for _ in 0..frequency {
            slots[next] = symbol;
            next += 1;
        }
    }
    if next != 4096 {
        return Err(ErrorClass::Malformed);
    }
    let mut position = 0usize;
    let mut out = Vec::with_capacity(expected_len);
    for _ in 0..expected_len {
        let slot = (state & 4095) as usize;
        let symbol = slots[slot];
        let mut cumulative = 0u32;
        let mut frequency = 0u32;
        for &(candidate, value) in &pairs {
            if candidate == symbol {
                frequency = u32::from(value);
                break;
            }
            cumulative += u32::from(value);
        }
        let next_state =
            u64::from(frequency) * u64::from(state >> 12) + slot as u64 - u64::from(cumulative);
        state = match u32::try_from(next_state) {
            Ok(value) => value,
            Err(_) => return Err(ErrorClass::Corrupt),
        };
        while state < 1 << 23 {
            if position >= renorm.len() {
                return Err(ErrorClass::Corrupt);
            }
            state = (state << 8) | u32::from(renorm[position]);
            position += 1;
        }
        out.push(symbol);
    }
    if state != 1 << 23 {
        return Err(ErrorClass::Corrupt);
    }
    if position != renorm.len() {
        return Err(ErrorClass::Corrupt);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Round trips and edge cases
// ---------------------------------------------------------------------------

fn coders() -> Vec<Box<dyn Coder>> {
    let mut all: Vec<Box<dyn Coder>> = vec![
        Box::new(Raw),
        Box::new(Zstd::new(3)),
        Box::new(Zstd::new(19)),
        Box::new(Huffman),
        Box::new(Rans),
    ];
    for k in SUPPORTED_K {
        all.push(Box::new(Tunstall::new(k).expect("supported k")));
    }
    all
}

fn edge_inputs() -> Vec<(&'static str, Vec<u8>)> {
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let mut cases: Vec<(&'static str, Vec<u8>)> = Vec::new();
    cases.push(("empty", Vec::new()));
    cases.push(("one_zero", vec![0]));
    cases.push(("one_ff", vec![0xff]));
    cases.push(("two_symbols", (0..1000).map(|i| (i % 2) as u8).collect()));
    cases.push(("all_256", (0..256 * 4).map(|i| (i % 256) as u8).collect()));
    cases.push((
        "highly_skewed",
        (0..65_536)
            .map(|i| if i % 100 < 99 { 0 } else { (i / 100) as u8 })
            .collect(),
    ));
    cases.push(("incompressible", (0..262_144).map(|_| rng.byte()).collect()));
    cases.push(("long_run", vec![0u8; 100_000]));
    for len in 1..=17 {
        let mut input = Vec::with_capacity(len);
        for _ in 0..len {
            input.push(rng.byte());
        }
        cases.push(("non_granular", input));
    }
    cases
}

#[test]
fn every_coder_round_trips_every_edge_case() {
    for (label, input) in edge_inputs() {
        for coder in coders() {
            let encoded = coder.encode(&input);
            let decoded = coder
                .decode(&encoded, input.len())
                .unwrap_or_else(|error| panic!("{} failed on {label}: {error}", coder.id()));
            assert_eq!(decoded, input, "{} on {label}", coder.id());
            let mut buffer = vec![0u8; input.len()];
            coder
                .decode_into(&encoded, &mut buffer)
                .unwrap_or_else(|error| {
                    panic!("{} decode_into failed on {label}: {error}", coder.id())
                });
            assert_eq!(buffer, input, "{} decode_into on {label}", coder.id());
            // The decoded length is caller-supplied for the shipped Huffman and
            // for rANS. Huffman padding can itself form valid short codes, and a
            // one-symbol rANS table has no renormalisation bytes, so a wrong
            // length is not always detectable. Those two are excluded from the
            // wrong-length assertions; raw, zstd, and Tunstall carry an exact
            // length and must reject it.
            let length_check = coder.id() != "huffman-canonical-v1" && coder.id() != "rans";
            if input.len() < (1 << 20) && length_check {
                assert!(
                    coder.decode(&encoded, input.len() + 1).is_err(),
                    "{} accepted an oversized expected_len on {label}",
                    coder.id()
                );
                if input.len() > 0 {
                    assert!(
                        coder.decode(&encoded, input.len() - 1).is_err(),
                        "{} accepted an undersized expected_len on {label}",
                        coder.id()
                    );
                }
            }
        }
    }
}

#[test]
fn deterministic_bytes_for_fixed_input() {
    let rng = &mut Rng::new(7);
    let input = skewed_input(rng, 4096, 16);
    for coder in coders() {
        let first = coder.encode(&input);
        let second = coder.encode(&input);
        assert_eq!(first, second, "{} is not deterministic", coder.id());
    }
}

// ---------------------------------------------------------------------------
// Tunstall dictionary rule
// ---------------------------------------------------------------------------

#[test]
fn tunstall_construction_is_deterministic_and_sorted() {
    let mut counts = [0u64; 256];
    for (symbol, count) in counts.iter_mut().enumerate() {
        *count = ((symbol * 37 + 11) % 101) as u64 + 1;
    }
    for k in SUPPORTED_K {
        let first = Tunstall::new(k).unwrap().build_dict(&counts).unwrap();
        let second = Tunstall::new(k).unwrap().build_dict(&counts).unwrap();
        assert_eq!(first.to_bytes(), second.to_bytes(), "k={k}");
        let mut previous: Option<Vec<u8>> = None;
        for index in 0..first.len() {
            let phrase = first.phrase(index).unwrap().to_vec();
            if let Some(last) = previous {
                assert!(last < phrase, "k={k} dictionary is not ascending");
            }
            previous = Some(phrase);
        }
    }
}

#[test]
fn tunstall_split_budget_matches_the_full_alphabet_arithmetic() {
    let mut counts = [0u64; 256];
    for count in counts.iter_mut() {
        *count = 1;
    }
    for (k, expected) in [(8u8, 256usize), (10, 1021), (12, 4081)] {
        let dict = Tunstall::new(k).unwrap().build_dict(&counts).unwrap();
        assert_eq!(dict.len(), expected, "k={k}");
        assert!(dict.len() <= 1usize << k);
    }
}

#[test]
fn tunstall_splits_the_most_probable_phrase_first() {
    let mut counts = [0u64; 256];
    counts[7] = 1_000_000;
    for symbol in 0..256 {
        if symbol != 7 {
            counts[symbol] = 1;
        }
    }
    let dict = Tunstall::new(12).unwrap().build_dict(&counts).unwrap();
    let has_one = (0..dict.len())
        .find(|&index| dict.phrase(index) == Some(&[7u8]))
        .is_some();
    let deepest = vec![7u8; 16];
    let has_deep_chain = (0..dict.len())
        .find(|&index| dict.phrase(index) == Some(deepest.as_slice()))
        .is_some();
    assert!(
        !has_one,
        "the dominant one-byte phrase should have been split"
    );
    assert!(
        has_deep_chain,
        "the dominant chain should have been split repeatedly"
    );
}

#[test]
fn tunstall_shared_dictionary_covers_symbols_it_never_saw() {
    let tunstall = Tunstall::new(12).unwrap();
    let dictionary = tunstall
        .build_dict_from_planes(&[vec![0u8; 4096].as_slice()])
        .unwrap();
    let input: Vec<u8> = (0..256u16).map(|value| value as u8).collect();
    let encoded = tunstall.encode_with_dict(&input, &dictionary).unwrap();
    let decoded = tunstall
        .decode_with_dict(&encoded, input.len(), &dictionary)
        .unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn tunstall_rejects_an_unknown_width() {
    assert_eq!(
        Tunstall::new(9).unwrap_err().class(),
        ErrorClass::UnsupportedFeature
    );
    assert_eq!(
        Tunstall::new(16).unwrap_err().class(),
        ErrorClass::UnsupportedFeature
    );
    assert_eq!(
        Dictionary::from_bytes(&[0, 0], 9).unwrap_err().class(),
        ErrorClass::UnsupportedFeature
    );
}

// ---------------------------------------------------------------------------
// Differential tests
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Ok(Vec<u8>),
    Err(ErrorClass),
}

fn fast_tunstall(encoded: &[u8], expected_len: usize, k: u8) -> Outcome {
    match Tunstall::new(k).unwrap().decode(encoded, expected_len) {
        Ok(bytes) => Outcome::Ok(bytes),
        Err(error) => Outcome::Err(error.class()),
    }
}

fn reference_tunstall(encoded: &[u8], expected_len: usize, k: u8) -> Outcome {
    match ref_tunstall_decode(encoded, expected_len, k) {
        Ok(bytes) => Outcome::Ok(bytes),
        Err(class) => Outcome::Err(class),
    }
}

fn fast_rans(encoded: &[u8], expected_len: usize) -> Outcome {
    match Rans.decode(encoded, expected_len) {
        Ok(bytes) => Outcome::Ok(bytes),
        Err(error) => Outcome::Err(error.class()),
    }
}

fn reference_rans(encoded: &[u8], expected_len: usize) -> Outcome {
    match ref_rans_decode(encoded, expected_len) {
        Ok(bytes) => Outcome::Ok(bytes),
        Err(class) => Outcome::Err(class),
    }
}

const CASES: usize = 5_000;

#[test]
fn tunstall_matches_the_reference_decoder_on_random_and_malformed_input() {
    let rng = &mut Rng::new(0xD1CE_5EED_1234_5678);
    for case in 0..CASES {
        let len = rng.below(4096);
        let alphabet = 1 + rng.below(32);
        let input = skewed_input(rng, len, alphabet);
        let k = SUPPORTED_K[rng.below(SUPPORTED_K.len())];
        let encoded = Tunstall::new(k).unwrap().encode(&input);
        let expected = input.len();
        let fast = fast_tunstall(&encoded, expected, k);
        let reference = reference_tunstall(&encoded, expected, k);
        assert_eq!(fast, reference, "valid case {case} k={k} len={len}");
        assert_eq!(fast, Outcome::Ok(input), "valid case {case} bytes");

        // Mutate the stream: flip, truncate, append, or overwrite a byte.
        let mut mutated = encoded.clone();
        match rng.below(4) {
            0 if !mutated.is_empty() => {
                let index = rng.below(mutated.len());
                mutated[index] ^= 1 << rng.below(8);
            }
            1 if !mutated.is_empty() => {
                let keep = rng.below(mutated.len());
                mutated.truncate(keep);
            }
            2 => mutated.push(rng.byte()),
            _ if !mutated.is_empty() => {
                let index = rng.below(mutated.len());
                mutated[index] = rng.byte();
            }
            _ => {}
        }
        let fast = fast_tunstall(&mutated, expected, k);
        let reference = reference_tunstall(&mutated, expected, k);
        assert_eq!(
            fast,
            reference,
            "mutated case {case} k={k} len={len} encoded_len={}",
            encoded.len()
        );
    }
}

#[test]
fn rans_matches_the_reference_decoder_on_random_and_malformed_input() {
    let rng = &mut Rng::new(0x5EED_0BAD_C0FF_EE01);
    for case in 0..CASES {
        let len = rng.below(4096);
        let alphabet = 1 + rng.below(256);
        let input = skewed_input(rng, len, alphabet);
        let encoded = Rans.encode(&input);
        let expected = input.len();
        let fast = fast_rans(&encoded, expected);
        let reference = reference_rans(&encoded, expected);
        assert_eq!(fast, reference, "valid case {case} len={len}");
        assert_eq!(fast, Outcome::Ok(input), "valid case {case} bytes");

        let mut mutated = encoded.clone();
        match rng.below(4) {
            0 if !mutated.is_empty() => {
                let index = rng.below(mutated.len());
                mutated[index] ^= 1 << rng.below(8);
            }
            1 if !mutated.is_empty() => {
                let keep = rng.below(mutated.len());
                mutated.truncate(keep);
            }
            2 => mutated.push(rng.byte()),
            _ if !mutated.is_empty() => {
                let index = rng.below(mutated.len());
                mutated[index] = rng.byte();
            }
            _ => {}
        }
        let fast = fast_rans(&mutated, expected);
        let reference = reference_rans(&mutated, expected);
        assert_eq!(
            fast,
            reference,
            "mutated case {case} len={len} encoded_len={}",
            encoded.len()
        );
    }
}

#[test]
fn rans_rejects_table_and_stream_corruption() {
    let input: Vec<u8> = (0..4096).map(|i| (i % 7) as u8).collect();
    let encoded = Rans.encode(&input);
    let cases: Vec<(Vec<u8>, ErrorClass)> = vec![
        (vec![0u8; 4], ErrorClass::Malformed),
        (vec![1, 11, 0, 0, 0, 0, 0, 0], ErrorClass::Malformed),
        (vec![1, 12, 0, 0], ErrorClass::Malformed),
        (
            {
                let mut copy = encoded.clone();
                copy[2] = 200;
                copy
            },
            ErrorClass::Malformed,
        ),
        (
            {
                let mut copy = encoded.clone();
                let last = copy.len() - 1;
                copy.truncate(last);
                copy
            },
            ErrorClass::Corrupt,
        ),
    ];
    for (bytes, expected) in cases {
        let error = Rans.decode(&bytes, input.len()).unwrap_err();
        assert_eq!(error.class(), expected, "bytes={bytes:?}");
    }
}
