//! Static order-0 rANS with a 32-bit state.
//!
//! ## Normalisation (deterministic, integer-only)
//!
//! Frequencies sum to `2^12 = 4096` over the symbols that occur in the input.
//! For every present symbol `s` with count `c_s` and total `T`:
//!
//! ```text
//! f_s = max(1, floor(c_s * 4096 / T))
//! r_s = (c_s * 4096) mod T
//! ```
//!
//! Let `F = sum f_s`.
//!
//! * If `F < 4096`, the deficit is added one at a time to the symbols with the
//!   largest `r_s`, ties by ascending symbol byte.
//! * If `F > 4096`, the excess is removed one at a time from symbols with
//!   `f_s > 1`, smallest `r_s` first, ties by ascending symbol byte.
//!
//! The result always sums to exactly 4096 (the arithmetic is asserted).
//! There are no floating-point operations anywhere in this module.
//!
//! ## Stream form
//!
//! ```text
//! u8  version = 1
//! u8  prob_bits = 12
//! u16le symbol_count
//! symbol_count * (u8 symbol, u16le frequency)   # sorted by symbol
//! u32le final_state
//! renormalisation bytes
//! ```
//!
//! The state is 32 bits. Renormalisation emits **one byte at a time**
//! (`state >>= 8`) while `state >= x_max`, with
//! `x_max = ((RANS_L >> 12) << 8) * f` and `RANS_L = 1 << 23`. The encoder
//! walks the input backwards; the emitted bytes are reversed once at the end
//! so the decoder consumes them forwards.
//!
//! A correct stream decodes to `expected_len` bytes, returns the state to
//! exactly `RANS_L`, and consumes every renormalisation byte. Both are
//! checked; a mismatch is [`ErrorClass::Corrupt`]. Header and table shape
//! failures are [`ErrorClass::Malformed`].
//!
//! The decoded length is caller-supplied, exactly as with the shipped
//! `huffman-canonical-v1` coder. A table with one symbol emits no
//! renormalisation bytes and is therefore length-elastic: it decodes to any
//! declared length, and the caller is responsible for supplying the true
//! length. Multi-symbol streams normally fail the state/consumption checks for
//! a wrong length, but this module does not claim that every wrong length is
//! detectable.
//!
//! [`ErrorClass::Corrupt`]: crate::error::ErrorClass::Corrupt
//! [`ErrorClass::Malformed`]: crate::error::ErrorClass::Malformed

use crate::error::{Error, Result};

use super::{check_expected_len, Coder};

/// Probability precision.
pub const PROB_BITS: u32 = 12;
/// Normalised frequency total.
pub const SCALE: u32 = 1 << PROB_BITS;
/// Lower renormalisation bound.
pub const RANS_L: u32 = 1 << 23;
/// Stream version.
pub const VERSION: u8 = 1;

/// Static order-0 rANS.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rans;

/// Normalised `(symbol, frequency)` pairs sorted by symbol.
fn normalize(counts: &[u64; 256]) -> Result<Vec<(u8, u16)>> {
    let used: Vec<u8> = (0..256)
        .filter(|&symbol| counts[symbol] > 0)
        .map(|symbol| symbol as u8)
        .collect();
    if used.is_empty() {
        return Ok(Vec::new());
    }
    let total: u64 = used.iter().map(|&s| counts[s as usize]).sum();
    let mut freq = [0u32; 256];
    let mut remainder = [0u64; 256];
    let mut sum: u32 = 0;
    for &symbol in &used {
        let scaled = counts[symbol as usize]
            .checked_mul(u64::from(SCALE))
            .ok_or_else(|| Error::limit_exceeded("rANS frequency scaling overflows u64"))?;
        let floor = scaled / total;
        let value = if floor == 0 { 1 } else { floor };
        if value > u64::from(SCALE) {
            return Err(Error::limit_exceeded(
                "rANS normalised frequency exceeds the 4096-slot table",
            ));
        }
        freq[symbol as usize] = value as u32;
        remainder[symbol as usize] = scaled % total;
        sum = sum
            .checked_add(value as u32)
            .ok_or_else(|| Error::limit_exceeded("rANS frequency sum overflows u32"))?;
    }

    if sum < SCALE {
        let deficit = (SCALE - sum) as usize;
        let mut order = used.clone();
        order.sort_by(|&a, &b| {
            remainder[b as usize]
                .cmp(&remainder[a as usize])
                .then(a.cmp(&b))
        });
        for &symbol in order.iter().take(deficit) {
            freq[symbol as usize] += 1;
            sum += 1;
        }
    } else if sum > SCALE {
        let mut remaining = (sum - SCALE) as usize;
        let mut order: Vec<u8> = used
            .iter()
            .copied()
            .filter(|&symbol| freq[symbol as usize] > 1)
            .collect();
        order.sort_by(|&a, &b| {
            remainder[a as usize]
                .cmp(&remainder[b as usize])
                .then(a.cmp(&b))
        });
        if order.is_empty() {
            return Err(Error::limit_exceeded(
                "rANS normalisation cannot remove the frequency excess without a zero frequency",
            ));
        }
        // Remove one slot at a time, cycling through the candidates in
        // (remainder ascending, symbol ascending) order. The total slack above
        // 1 is always at least the excess, so this terminates.
        let mut index = 0usize;
        let mut guard = 0usize;
        while remaining > 0 {
            let symbol = order[index % order.len()];
            if freq[symbol as usize] > 1 {
                freq[symbol as usize] -= 1;
                remaining -= 1;
            }
            index += 1;
            guard += 1;
            if guard > order.len() + (sum - SCALE) as usize * 2 {
                return Err(Error::limit_exceeded(
                    "rANS normalisation did not converge while removing the frequency excess",
                ));
            }
        }
        sum = SCALE;
    }

    if sum != SCALE {
        return Err(Error::limit_exceeded(
            "rANS normalisation did not reach the 4096-slot total",
        ));
    }
    Ok(used
        .into_iter()
        .map(|symbol| (symbol, freq[symbol as usize] as u16))
        .collect())
}

/// Cumulative/frequency tables indexed by symbol byte.
fn tables(pairs: &[(u8, u16)]) -> ([u32; 256], [u32; 256]) {
    let mut freq = [0u32; 256];
    let mut cum = [0u32; 256];
    let mut next = 0u32;
    for &(symbol, value) in pairs {
        freq[symbol as usize] = u32::from(value);
        cum[symbol as usize] = next;
        next += u32::from(value);
    }
    (freq, cum)
}

/// Parsed header shared by the fast decoder and the test reference decoder.
struct Header<'a> {
    pairs: Vec<(u8, u16)>,
    state: u32,
    renorm: &'a [u8],
}

fn parse_header(encoded: &[u8]) -> Result<Header<'_>> {
    if encoded.len() < 4 {
        return Err(Error::malformed(format!(
            "rANS stream is {} bytes, shorter than its 4-byte header",
            encoded.len()
        )));
    }
    if encoded[0] != VERSION {
        return Err(Error::malformed(format!(
            "unsupported rANS stream version {} (expected {VERSION})",
            encoded[0]
        )));
    }
    let prob_bits = u32::from(encoded[1]);
    if prob_bits != PROB_BITS {
        return Err(Error::malformed(format!(
            "rANS stream declares {prob_bits} probability bits (only {PROB_BITS} is supported)"
        )));
    }
    let symbol_count = usize::from(u16::from_le_bytes([encoded[2], encoded[3]]));
    if symbol_count > 256 {
        return Err(Error::malformed(format!(
            "rANS stream declares {symbol_count} symbols (the alphabet has 256)"
        )));
    }
    let table_end = 4usize
        .checked_add(
            symbol_count
                .checked_mul(3)
                .ok_or_else(|| Error::malformed("rANS symbol table length overflows usize"))?,
        )
        .ok_or_else(|| Error::malformed("rANS symbol table length overflows usize"))?;
    if encoded.len() < table_end + 4 {
        return Err(Error::malformed(
            "rANS stream ends inside its symbol table or final state",
        ));
    }
    let mut pairs = Vec::with_capacity(symbol_count);
    let mut previous: Option<u8> = None;
    let mut sum: u32 = 0;
    for index in 0..symbol_count {
        let offset = 4 + index * 3;
        let symbol = encoded[offset];
        let frequency = u16::from_le_bytes([encoded[offset + 1], encoded[offset + 2]]);
        if let Some(last) = previous {
            if symbol <= last {
                return Err(Error::malformed(
                    "rANS symbol table is not strictly ascending",
                ));
            }
        }
        if frequency == 0 {
            return Err(Error::malformed(
                "rANS symbol table contains a zero frequency",
            ));
        }
        previous = Some(symbol);
        sum = sum
            .checked_add(u32::from(frequency))
            .ok_or_else(|| Error::malformed("rANS frequency sum overflows u32"))?;
        pairs.push((symbol, frequency));
    }
    if symbol_count > 0 && sum != SCALE {
        return Err(Error::malformed(format!(
            "rANS frequencies sum to {sum}, not {SCALE}"
        )));
    }
    let state_offset = table_end;
    let state = u32::from_le_bytes([
        encoded[state_offset],
        encoded[state_offset + 1],
        encoded[state_offset + 2],
        encoded[state_offset + 3],
    ]);
    Ok(Header {
        pairs,
        state,
        renorm: &encoded[state_offset + 4..],
    })
}

fn build_slot_table(pairs: &[(u8, u16)]) -> Result<Vec<u8>> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    let mut table = vec![0u8; SCALE as usize];
    let mut next = 0u32;
    for &(symbol, frequency) in pairs {
        let frequency = u32::from(frequency);
        let end = next
            .checked_add(frequency)
            .ok_or_else(|| Error::malformed("rANS slot table overflows u32"))?;
        if end > SCALE {
            return Err(Error::malformed(
                "rANS frequencies exceed the 4096-slot table",
            ));
        }
        for slot in next..end {
            table[slot as usize] = symbol;
        }
        next = end;
    }
    if next != SCALE {
        return Err(Error::malformed(
            "rANS frequencies do not fill the 4096-slot table",
        ));
    }
    Ok(table)
}

impl Coder for Rans {
    fn id(&self) -> &'static str {
        "rans"
    }

    fn encode(&self, input: &[u8]) -> Vec<u8> {
        let mut counts = [0u64; 256];
        for &byte in input {
            counts[byte as usize] += 1;
        }
        let pairs = normalize(&counts).expect("rANS normalisation is total for byte counts");
        let (freq, cum) = tables(&pairs);

        let mut state: u32 = RANS_L;
        let mut renorm: Vec<u8> = Vec::new();
        for &byte in input.iter().rev() {
            let f = freq[byte as usize];
            debug_assert!(f > 0);
            let c = cum[byte as usize];
            let x_max = ((RANS_L >> PROB_BITS) << 8)
                .checked_mul(f)
                .expect("x_max fits u32");
            while state >= x_max {
                renorm.push((state & 0xff) as u8);
                state >>= 8;
            }
            state = (state / f) * SCALE + (state % f) + c;
        }
        renorm.reverse();

        let mut out = Vec::with_capacity(4 + pairs.len() * 3 + 4 + renorm.len());
        out.push(VERSION);
        out.push(PROB_BITS as u8);
        out.extend_from_slice(&(pairs.len() as u16).to_le_bytes());
        for &(symbol, frequency) in &pairs {
            out.push(symbol);
            out.extend_from_slice(&frequency.to_le_bytes());
        }
        out.extend_from_slice(&state.to_le_bytes());
        out.extend_from_slice(&renorm);
        out
    }

    fn decode(&self, encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
        check_expected_len(expected_len)?;
        let mut out = vec![0u8; expected_len];
        self.decode_into(encoded, &mut out)?;
        Ok(out)
    }

    fn decode_into(&self, encoded: &[u8], out: &mut [u8]) -> Result<()> {
        let header = parse_header(encoded)?;
        if header.pairs.is_empty() {
            if !out.is_empty() {
                return Err(Error::corrupt(format!(
                    "rANS stream has no symbols but the caller expects {} bytes",
                    out.len()
                )));
            }
            if header.state != RANS_L {
                return Err(Error::corrupt(
                    "empty rANS stream does not carry the initial state",
                ));
            }
            if !header.renorm.is_empty() {
                return Err(Error::corrupt(
                    "empty rANS stream carries unexpected renormalisation bytes",
                ));
            }
            return Ok(());
        }
        let slot_table = build_slot_table(&header.pairs)?;
        let (freq, cum) = tables(&header.pairs);
        let mut state = header.state;
        let mut position = 0usize;
        for destination in out.iter_mut() {
            let slot = state & (SCALE - 1);
            let symbol = slot_table[slot as usize];
            *destination = symbol;
            let f = u64::from(freq[symbol as usize]);
            let c = u64::from(cum[symbol as usize]);
            let next = f * u64::from(state >> PROB_BITS) + u64::from(slot) - c;
            state = u32::try_from(next)
                .map_err(|_| Error::corrupt("rANS state left the 32-bit range during decoding"))?;
            while state < RANS_L {
                if position >= header.renorm.len() {
                    return Err(Error::corrupt(
                        "rANS renormalisation bytes ended before the declared output",
                    ));
                }
                state = (state << 8) | u32::from(header.renorm[position]);
                position += 1;
            }
        }
        if state != RANS_L {
            return Err(Error::corrupt(format!(
                "rANS state is {state} after decoding, not the initial state {RANS_L}"
            )));
        }
        if position != header.renorm.len() {
            return Err(Error::corrupt(format!(
                "rANS stream has {} trailing renormalisation bytes",
                header.renorm.len() - position
            )));
        }
        Ok(())
    }
}
