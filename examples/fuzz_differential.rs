//! Deterministic cross-language differential fuzz runner (Rust side).
//!
//! This example is the Rust half of `scripts/fuzz-diff.sh` in the WeftFS parent
//! repository. It reads `conformance/fuzz/v1/seeds.txt`, generates exactly the
//! same N inputs as `WeftCodec-Go/cmd/fuzzdiff`, decodes each one through the
//! crate's normal decode entry points with `Limits::default()`, and prints one
//! tab-separated line per input:
//!
//! ```text
//! <i>\tok\t<content_root_hex>
//! <i>\terr\t<ErrorClass>
//! <i>\tpanic\t<message>
//! ```
//!
//! At the end it prints metadata lines that are *not* per-input results:
//!
//! ```text
//! corpus_hash\t<blake3-hex>
//! inputs\t<N>
//! elapsed_ms\t<milliseconds>
//! peak_rss_kib\t<kilobytes>
//! ```
//!
//! # Generator specification (shared verbatim with the Go implementation)
//!
//! PRNG: xorshift64* with initial state `0x9E3779B97F4A7C15`.
//!
//! ```text
//! next_u64():
//!     x = state
//!     x ^= x >> 12
//!     x ^= x << 25
//!     x ^= x >> 27
//!     state = x
//!     return x * 0x2545F4914F6CDD1D   (wrapping 64-bit)
//! ```
//!
//! `below(bound)` is `next_u64() % bound` for `bound > 0`.
//!
//! `next_byte()` returns the eight little-endian bytes of successive
//! `next_u64()` outputs, exhausting a buffer before drawing again.
//!
//! For input index `i` in `[0, N)`, draw `selector = below(8)`:
//!
//! * `0` — **raw random** (1/8): `len = below(2049)` bytes, each from
//!   `next_byte()`.
//! * `1` — **un-repaired mutation** (1/8): choose a seed, apply `1 + below(8)`
//!   of the legacy mutations below, no checksum repair.
//! * `2..=5` — **repaired mutation** (4/8): choose a seed, apply `1 + below(4)`
//!   mutations to its body and/or dependency/chunk table, then recompute the
//!   header CRC32C, body length, and trailing CRC32C.
//! * `6` — **semantic fault** (1/8): a structurally checksum-valid frame with a
//!   deliberately wrong dependency hash, wrong chunk extent, wrong mode byte,
//!   out-of-range declared size, unknown critical flag, non-canonical structured
//!   body, wrong content hash, or a self-referential dependency cycle.
//! * `7` — **clean seed** (1/8): an unmodified valid seed, which populates the
//!   successful-decode / content-root comparison path.
//!
//! Legacy mutation operation `below(5)` (branch 1): bit flip, byte replace,
//! insert up to 32 bytes (bounded total 4096), delete a range, truncate.
//!
//! Repaired-body mutation operation `below(4)` (branches 2..=5): body bit flip,
//! body byte replace, body insert/delete (bounded total 4096), or a table byte
//! flip/replace. After the mutations the runner parses the header layout from
//! the normative offsets in `spec/WEFTCODEC-FORMAT.md` §2.1–§2.3, sets
//! `body_len` to the actual post-mutation body length, and recomputes
//! `header_crc32c` over `[0,72) || [76,96)` and `frame_crc32c` over everything
//! before the trailing four bytes. It deliberately does **not** repair the
//! content hash, so repaired body mutations exercise the decoder's real
//! content-hash, WCB, chunk, dependency, and mode checks rather than the CRC
//! gate.
//!
//! Semantic fault operation `below(10)` (branch 6):
//!
//! 0,1,2 — replace dependency 0's hash with `0xFF..FF` (missing / lying /
//!          denying resolver, respectively).
//! 3     — rewrite a seed as a valid REFERENCE frame whose dependency hash is
//!          its own content hash, decoded through the DAG entry point so the
//!          cycle detector runs.
//! 4     — set the mode byte to one of `9, 127, 128, 255`.
//! 5     — declare `decoded_len = u64::MAX`.
//! 6     — set chunk 0's `encoded_off` past the body.
//! 7     — flip one content-hash byte.
//! 8     — set unknown critical flag bit 4.
//! 9     — append a byte to a STRUCTURED body and bump `decoded_len`.
//!
//! Every semantic fault is followed by the same CRC repair, so the frame is
//! structurally checksum-valid unless the fault intentionally violates a
//! declared length limit.
//!
//! The corpus hash is BLAKE3-256 over each generated input in order, framed
//! unambiguously as `LE64(len(input)) || input`. The two runners compare this
//! hash before any per-input comparison.
//!
//! Inputs are never retained: the runner streams generation, decoding, and
//! output so peak memory stays bounded regardless of N.
//!
//! The resolver is a fixed in-memory table shared with the Go runner:
//! `BASE0 = bytes 0x00..=0xFF`, `BASE1 = "weftcodec fuzz base payload one\n"`
//! repeated 8 times, `BASE2 = empty`, `BASE3 = "delta-base-A"` repeated 20
//! times. Seeds that use REFERENCE, DELTA, or DICTIONARY reference exactly
//! these bases so they decode instead of failing with `DependencyMissing`.
//!
//! Panics are caught with `catch_unwind`; a panic is reported as a `panic`
//! line and forces a non-zero exit. The panic hook is silenced so stderr stays
//! clean and the per-input line is the only evidence.
//!
//! This is not coverage-guided fuzzing. It is a deterministic, synthetic,
//! CPU-only differential conformance court.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::ExitCode;

use weftcodec::dependency::{decode_frame_dag, FrameSource, MapResolver, Resolver};
use weftcodec::error::Error;
use weftcodec::frame::{CHUNK_ENTRY_LEN, DEP_ENTRY_LEN, HEADER_LEN};
use weftcodec::hash::{hash_domain, Hash32, DOMAIN_CHUNK};
use weftcodec::limits::Limits;
use weftcodec::modes;

const PRNG_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const MAX_RANDOM_LEN: u64 = 2048;
const MAX_INPUT_LEN: usize = 4096;
const MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;

const TRAILER_LEN: usize = 4;

/// Fixed dependency bases shared with the Go runner and the seed generator.
pub fn base0() -> Vec<u8> {
    (0u16..=255).map(|b| b as u8).collect()
}
pub fn base1() -> Vec<u8> {
    b"weftcodec fuzz base payload one\n".repeat(8)
}
pub fn base2() -> Vec<u8> {
    Vec::new()
}
pub fn base3() -> Vec<u8> {
    b"delta-base-A".repeat(20)
}

/// Build the fixed resolver shared by the seed generator and this runner.
pub fn build_resolver() -> MapResolver {
    let mut resolver = MapResolver::new();
    for base in [base0(), base1(), base2(), base3()] {
        resolver.insert_bytes(base);
    }
    resolver
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InputKind {
    Standard,
    Lying,
    Denied,
    Cycle,
}

#[derive(Clone, Copy)]
enum FixedMode {
    Lying,
    Denied,
}

struct FixedResolver {
    mode: FixedMode,
}

impl Resolver for FixedResolver {
    fn resolve(
        &self,
        _hash: &Hash32,
        _decoded_len: u64,
        _dep_domain: u8,
    ) -> weftcodec::Result<Vec<u8>> {
        match self.mode {
            FixedMode::Lying => Ok(b"lying!!".to_vec()),
            FixedMode::Denied => Err(Error::dependency_denied("fuzz policy denies dependency")),
        }
    }
}

struct SelfFrameSource {
    frame: Vec<u8>,
}

impl FrameSource for SelfFrameSource {
    fn get_frame(&self, _content_hash: &Hash32) -> weftcodec::Result<Option<Vec<u8>>> {
        Ok(Some(self.frame.clone()))
    }
}

struct Rng {
    state: u64,
    buffer: [u8; 8],
    position: usize,
}

impl Rng {
    fn new() -> Self {
        Rng {
            state: PRNG_SEED,
            buffer: [0u8; 8],
            position: 8,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(MULTIPLIER)
    }

    fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        self.next_u64() % bound
    }

    fn next_byte(&mut self) -> u8 {
        if self.position == self.buffer.len() {
            self.buffer = self.next_u64().to_le_bytes();
            self.position = 0;
        }
        let byte = self.buffer[self.position];
        self.position += 1;
        byte
    }
}

fn le_u16(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}

fn le_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn le_u64(bytes: &[u8], off: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[off..off + 8]);
    u64::from_le_bytes(buf)
}

#[derive(Clone, Copy)]
struct FrameLayout {
    dep_count: u32,
    chunk_count: u32,
    body_start: usize,
    body_len: usize,
}

/// Locate the body and tables using the normative WFT1 offsets. The body
/// length is taken from the actual frame length so an insert/delete mutation
/// is repaired rather than misread.
fn frame_layout(data: &[u8]) -> Option<FrameLayout> {
    if data.len() < HEADER_LEN + TRAILER_LEN || &data[0..4] != b"WFT1" {
        return None;
    }
    let dep_count = le_u32(data, 28);
    let chunk_count = le_u32(data, 32);
    let table_len = le_u64(data, 76);
    let expected = (u64::from(dep_count))
        .checked_mul(DEP_ENTRY_LEN as u64)?
        .checked_add((u64::from(chunk_count)).checked_mul(CHUNK_ENTRY_LEN as u64)?)?;
    if table_len != expected {
        return None;
    }
    let body_start = HEADER_LEN.checked_add(usize::try_from(table_len).ok()?)?;
    if body_start.checked_add(TRAILER_LEN)? > data.len() {
        return None;
    }
    Some(FrameLayout {
        dep_count,
        chunk_count,
        body_start,
        body_len: data.len() - TRAILER_LEN - body_start,
    })
}

/// Recompute `body_len`, the header CRC32C, and the trailing CRC32C.
fn repair_frame(data: &mut Vec<u8>) -> bool {
    let Some(layout) = frame_layout(data) else {
        return false;
    };
    let body_len = data.len() - TRAILER_LEN - layout.body_start;
    data[20..28].copy_from_slice(&(body_len as u64).to_le_bytes());

    let mut header_crc_input = Vec::with_capacity(92);
    header_crc_input.extend_from_slice(&data[0..72]);
    header_crc_input.extend_from_slice(&data[76..96]);
    let header_crc = crc32c::crc32c(&header_crc_input);
    data[72..76].copy_from_slice(&header_crc.to_le_bytes());

    let trailer_at = data.len() - TRAILER_LEN;
    let trailer_crc = crc32c::crc32c(&data[..trailer_at]);
    data[trailer_at..].copy_from_slice(&trailer_crc.to_le_bytes());
    true
}

/// Legacy, un-repaired mutation (checksums are deliberately left stale).
fn legacy_mutate(rng: &mut Rng, data: &mut Vec<u8>) {
    match rng.below(5) {
        0 => {
            if !data.is_empty() {
                let pos = rng.below(data.len() as u64) as usize;
                let bit = rng.below(8) as u32;
                data[pos] ^= 1u8 << bit;
            }
        }
        1 => {
            if !data.is_empty() {
                let pos = rng.below(data.len() as u64) as usize;
                let value = rng.below(256) as u8;
                data[pos] = value;
            }
        }
        2 => {
            if data.len() < MAX_INPUT_LEN {
                let room = MAX_INPUT_LEN - data.len();
                let n = (1 + rng.below(32)) as usize;
                let n = n.min(room);
                let pos = rng.below(data.len() as u64 + 1) as usize;
                let mut inserted = Vec::with_capacity(n);
                for _ in 0..n {
                    inserted.push(rng.next_byte());
                }
                data.splice(pos..pos, inserted);
            }
        }
        3 => {
            if !data.is_empty() {
                let start = rng.below(data.len() as u64) as usize;
                let remaining = data.len() - start;
                let n = (1 + rng.below(remaining as u64)) as usize;
                data.drain(start..start + n);
            }
        }
        4 => {
            if !data.is_empty() {
                let new_len = rng.below(data.len() as u64 + 1) as usize;
                data.truncate(new_len);
            }
        }
        _ => unreachable!("below(5) is always in range"),
    }
}

fn random_bytes(rng: &mut Rng) -> Vec<u8> {
    let len = rng.below(MAX_RANDOM_LEN + 1) as usize;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(rng.next_byte());
    }
    data
}

/// Body/table mutation followed by checksum repair.
fn repaired_input(rng: &mut Rng, seeds: &[Vec<u8>]) -> (Vec<u8>, InputKind) {
    let index = rng.below(seeds.len() as u64) as usize;
    let mut data = seeds[index].clone();
    let Some(mut layout) = frame_layout(&data) else {
        return (data, InputKind::Standard);
    };
    let count = 1 + rng.below(4);
    for _ in 0..count {
        match rng.below(4) {
            0 => {
                if layout.body_len > 0 {
                    let pos = layout.body_start + rng.below(layout.body_len as u64) as usize;
                    let bit = rng.below(8) as u32;
                    data[pos] ^= 1u8 << bit;
                }
            }
            1 => {
                if layout.body_len > 0 {
                    let pos = layout.body_start + rng.below(layout.body_len as u64) as usize;
                    data[pos] = rng.below(256) as u8;
                }
            }
            2 => {
                if rng.below(2) == 0 {
                    if data.len() < MAX_INPUT_LEN {
                        let room = MAX_INPUT_LEN - data.len();
                        let n = ((1 + rng.below(16)) as usize).min(room);
                        let pos =
                            layout.body_start + rng.below(layout.body_len as u64 + 1) as usize;
                        let mut inserted = Vec::with_capacity(n);
                        for _ in 0..n {
                            inserted.push(rng.next_byte());
                        }
                        data.splice(pos..pos, inserted);
                    }
                } else if layout.body_len > 0 {
                    let start = layout.body_start + rng.below(layout.body_len as u64) as usize;
                    let remaining = layout.body_len - (start - layout.body_start);
                    let n = (1 + rng.below(remaining as u64)) as usize;
                    data.drain(start..start + n);
                }
                if let Some(updated) = frame_layout(&data) {
                    layout = updated;
                }
            }
            3 => {
                let entries = u64::from(layout.dep_count) + u64::from(layout.chunk_count);
                if entries > 0 {
                    let table_len = layout.body_start - HEADER_LEN;
                    let off = HEADER_LEN + rng.below(table_len as u64) as usize;
                    if rng.below(2) == 0 {
                        let bit = rng.below(8) as u32;
                        data[off] ^= 1u8 << bit;
                    } else {
                        data[off] = rng.below(256) as u8;
                    }
                }
            }
            _ => unreachable!("below(4) is always in range"),
        }
    }
    repair_frame(&mut data);
    (data, InputKind::Standard)
}

fn corrupt_content_hash(rng: &mut Rng, seeds: &[Vec<u8>]) -> (Vec<u8>, InputKind) {
    let index = rng.below(seeds.len() as u64) as usize;
    let mut data = seeds[index].clone();
    data[40] ^= 0x01;
    repair_frame(&mut data);
    (data, InputKind::Standard)
}

/// Build a structurally valid REFERENCE frame whose dependency 0 hash equals
/// its own content hash. The DAG entry point detects the cycle.
fn make_cycle(seed: &[u8]) -> (Vec<u8>, InputKind) {
    let mut data = vec![0u8; HEADER_LEN + DEP_ENTRY_LEN + TRAILER_LEN];
    data[0..4].copy_from_slice(b"WFT1");
    data[4] = 1;
    data[5] = 0;
    data[6] = 0;
    data[7] = 2; // REFERENCE
    data[8..10].copy_from_slice(&0u16.to_le_bytes());
    data[10..12].copy_from_slice(&0u16.to_le_bytes());
    let decoded_len = le_u64(seed, 12);
    data[12..20].copy_from_slice(&decoded_len.to_le_bytes());
    data[20..28].copy_from_slice(&0u64.to_le_bytes()); // body_len
    data[28..32].copy_from_slice(&1u32.to_le_bytes()); // dep_count
    data[32..36].copy_from_slice(&0u32.to_le_bytes()); // chunk_count
    data[36..38].copy_from_slice(&1u16.to_le_bytes()); // DEPENDENCIES
    data[40..72].copy_from_slice(&seed[40..72]); // content_hash
    data[76..84].copy_from_slice(&(DEP_ENTRY_LEN as u64).to_le_bytes());
    data[96..128].copy_from_slice(&seed[40..72]); // self-referential hash
    data[128..136].copy_from_slice(&decoded_len.to_le_bytes());
    data[136] = 0; // payload domain
    data[137] = 1; // REQUIRED
    repair_frame(&mut data);
    (data, InputKind::Cycle)
}

fn semantic_input(
    rng: &mut Rng,
    seeds: &[Vec<u8>],
    dep_seeds: &[usize],
    chunk_seeds: &[usize],
    structured_seeds: &[usize],
) -> (Vec<u8>, InputKind) {
    let fault = rng.below(10);
    match fault {
        0 | 1 | 2 => {
            if dep_seeds.is_empty() {
                return corrupt_content_hash(rng, seeds);
            }
            let index = dep_seeds[rng.below(dep_seeds.len() as u64) as usize];
            let mut data = seeds[index].clone();
            data[96..128].fill(0xFF);
            repair_frame(&mut data);
            let kind = match fault {
                0 => InputKind::Standard,
                1 => InputKind::Lying,
                _ => InputKind::Denied,
            };
            (data, kind)
        }
        3 => {
            if dep_seeds.is_empty() {
                let index = rng.below(seeds.len() as u64) as usize;
                return make_cycle(&seeds[index]);
            }
            let index = dep_seeds[rng.below(dep_seeds.len() as u64) as usize];
            make_cycle(&seeds[index])
        }
        4 => {
            let index = rng.below(seeds.len() as u64) as usize;
            let mut data = seeds[index].clone();
            let values = [9u8, 127, 128, 255];
            data[7] = values[rng.below(4) as usize];
            repair_frame(&mut data);
            (data, InputKind::Standard)
        }
        5 => {
            let index = rng.below(seeds.len() as u64) as usize;
            let mut data = seeds[index].clone();
            data[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
            repair_frame(&mut data);
            (data, InputKind::Standard)
        }
        6 => {
            if chunk_seeds.is_empty() {
                return corrupt_content_hash(rng, seeds);
            }
            let index = chunk_seeds[rng.below(chunk_seeds.len() as u64) as usize];
            let mut data = seeds[index].clone();
            let Some(layout) = frame_layout(&data) else {
                return (data, InputKind::Standard);
            };
            let chunk_start = HEADER_LEN + layout.dep_count as usize * DEP_ENTRY_LEN;
            let bad_offset = (layout.body_len as u64).saturating_add(8);
            data[chunk_start..chunk_start + 8].copy_from_slice(&bad_offset.to_le_bytes());
            repair_frame(&mut data);
            (data, InputKind::Standard)
        }
        7 => corrupt_content_hash(rng, seeds),
        8 => {
            let index = rng.below(seeds.len() as u64) as usize;
            let mut data = seeds[index].clone();
            let flags = le_u16(&data, 8) | 0x0010;
            data[8..10].copy_from_slice(&flags.to_le_bytes());
            repair_frame(&mut data);
            (data, InputKind::Standard)
        }
        9 => {
            if structured_seeds.is_empty() {
                return corrupt_content_hash(rng, seeds);
            }
            let index = structured_seeds[rng.below(structured_seeds.len() as u64) as usize];
            let mut data = seeds[index].clone();
            let Some(layout) = frame_layout(&data) else {
                return (data, InputKind::Standard);
            };
            let insert_at = layout.body_start + layout.body_len;
            data.insert(insert_at, 0x00);
            let decoded_len = le_u64(&data, 12).wrapping_add(1);
            data[12..20].copy_from_slice(&decoded_len.to_le_bytes());
            repair_frame(&mut data);
            (data, InputKind::Standard)
        }
        _ => unreachable!("below(10) is always in range"),
    }
}

fn generate(
    rng: &mut Rng,
    seeds: &[Vec<u8>],
    dep_seeds: &[usize],
    chunk_seeds: &[usize],
    structured_seeds: &[usize],
) -> (Vec<u8>, InputKind) {
    match rng.below(8) {
        0 => (random_bytes(rng), InputKind::Standard),
        1 => {
            let index = rng.below(seeds.len() as u64) as usize;
            let mut data = seeds[index].clone();
            let mutations = 1 + rng.below(8);
            for _ in 0..mutations {
                legacy_mutate(rng, &mut data);
            }
            (data, InputKind::Standard)
        }
        2..=5 => repaired_input(rng, seeds),
        6 => semantic_input(rng, seeds, dep_seeds, chunk_seeds, structured_seeds),
        7 => {
            let index = rng.below(seeds.len() as u64) as usize;
            (seeds[index].clone(), InputKind::Standard)
        }
        _ => unreachable!("below(8) is always in range"),
    }
}

fn seed_class_lists(seeds: &[Vec<u8>]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut dep_seeds = Vec::new();
    let mut chunk_seeds = Vec::new();
    let mut structured_seeds = Vec::new();
    for (index, seed) in seeds.iter().enumerate() {
        if let Some(layout) = frame_layout(seed) {
            if layout.dep_count > 0 {
                dep_seeds.push(index);
            }
            if layout.chunk_count > 0 {
                chunk_seeds.push(index);
            }
        }
        if seed.get(7) == Some(&5u8) {
            structured_seeds.push(index);
        }
    }
    (dep_seeds, chunk_seeds, structured_seeds)
}

fn load_seeds(path: &str) -> Result<Vec<Vec<u8>>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open seeds `{path}`: {e}"))?;
    let mut seeds = Vec::new();
    for (number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("cannot read seeds line {}: {e}", number + 1))?;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if text.len() % 2 != 0 {
            return Err(format!("seed line {} has odd hex length", number + 1));
        }
        let bytes =
            hex::decode(text).map_err(|e| format!("seed line {} is not hex: {e}", number + 1))?;
        if bytes.is_empty() {
            return Err(format!("seed line {} is empty", number + 1));
        }
        if bytes.len() > MAX_INPUT_LEN {
            return Err(format!(
                "seed line {} is {} bytes, above the {MAX_INPUT_LEN}-byte mutation bound",
                number + 1,
                bytes.len()
            ));
        }
        seeds.push(bytes);
    }
    if seeds.is_empty() {
        return Err(format!("seed file `{path}` contains no seeds"));
    }
    Ok(seeds)
}

fn peak_rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                let rest = line.strip_prefix("VmHWM:")?;
                rest.split_whitespace().next()?.parse::<u64>().ok()
            })
        })
        .unwrap_or(0)
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    let raw = if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string panic payload".to_owned()
    };
    raw.replace('\n', " ").replace('\r', " ").replace('\t', " ")
}

fn usage() -> ! {
    eprintln!("usage: fuzz_differential --seeds <path> [--n <count>] [--dump-index <index>]");
    std::process::exit(64)
}

fn main() -> ExitCode {
    let mut seeds_path: Option<String> = None;
    let mut n: u64 = 1_000_000;
    let mut dump_index: Option<u64> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seeds" => seeds_path = Some(args.next().unwrap_or_else(|| usage())),
            "--n" => {
                let value = args.next().unwrap_or_else(|| usage());
                n = value.parse().unwrap_or_else(|_| usage());
            }
            "--dump-index" => {
                let value = args.next().unwrap_or_else(|| usage());
                dump_index = Some(value.parse().unwrap_or_else(|_| usage()));
            }
            "-h" | "--help" => usage(),
            _ => {
                eprintln!("unknown argument `{arg}`");
                usage();
            }
        }
    }
    let Some(seeds_path) = seeds_path else {
        usage();
    };

    let seeds = match load_seeds(&seeds_path) {
        Ok(seeds) => seeds,
        Err(message) => {
            eprintln!("fuzz_differential: {message}");
            return ExitCode::from(2);
        }
    };
    let resolver = build_resolver();
    let lying = FixedResolver {
        mode: FixedMode::Lying,
    };
    let denied = FixedResolver {
        mode: FixedMode::Denied,
    };
    let limits = Limits::default();
    let (dep_seeds, chunk_seeds, structured_seeds) = seed_class_lists(&seeds);

    if let Some(target) = dump_index {
        let mut rng = Rng::new();
        let mut selected = Vec::new();
        for index in 0..=target {
            let (input, _kind) = generate(
                &mut rng,
                &seeds,
                &dep_seeds,
                &chunk_seeds,
                &structured_seeds,
            );
            if index == target {
                selected = input;
            }
        }
        println!("{}", hex::encode(selected));
        return ExitCode::SUCCESS;
    }

    // Suppress the default panic printer; the per-input `panic` line is the
    // evidence, and the runner must keep producing lines after a panic so the
    // comparator can see exactly where it happened.
    std::panic::set_hook(Box::new(|_| {}));

    let started = std::time::Instant::now();
    let stdout = std::io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());
    let mut hasher = blake3::Hasher::new();
    let mut rng = Rng::new();
    let mut panicked = false;

    for index in 0..n {
        let (input, kind) = generate(
            &mut rng,
            &seeds,
            &dep_seeds,
            &chunk_seeds,
            &structured_seeds,
        );
        hasher.update(&(input.len() as u64).to_le_bytes());
        hasher.update(&input);

        let outcome = catch_unwind(AssertUnwindSafe(|| match kind {
            InputKind::Standard => modes::decode_bytes(&input, &resolver, &limits),
            InputKind::Lying => modes::decode_bytes(&input, &lying, &limits),
            InputKind::Denied => modes::decode_bytes(&input, &denied, &limits),
            InputKind::Cycle => {
                let source = SelfFrameSource {
                    frame: input.clone(),
                };
                decode_frame_dag(&source, &input, &limits)
            }
        }));

        let written = match outcome {
            Ok(Ok(plaintext)) => writeln!(
                out,
                "{index}\tok\t{}",
                hash_domain(DOMAIN_CHUNK, &plaintext).to_hex()
            ),
            Ok(Err(error)) => writeln!(out, "{index}\terr\t{}", error.class().as_str()),
            Err(payload) => {
                panicked = true;
                writeln!(out, "{index}\tpanic\t{}", panic_message(&*payload))
            }
        };
        if let Err(error) = written {
            eprintln!("fuzz_differential: cannot write output: {error}");
            return ExitCode::from(2);
        }
    }

    let corpus_hash = hasher.finalize().to_hex();
    let elapsed_ms = started.elapsed().as_millis();
    let rss = peak_rss_kib();
    let metadata = writeln!(out, "corpus_hash\t{corpus_hash}")
        .and_then(|_| writeln!(out, "inputs\t{n}"))
        .and_then(|_| writeln!(out, "elapsed_ms\t{elapsed_ms}"))
        .and_then(|_| writeln!(out, "peak_rss_kib\t{rss}"))
        .and_then(|_| out.flush());
    if let Err(error) = metadata {
        eprintln!("fuzz_differential: cannot write metadata: {error}");
        return ExitCode::from(2);
    }

    if panicked {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
