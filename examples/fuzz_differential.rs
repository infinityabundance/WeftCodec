//! Deterministic cross-language differential fuzz runner (Rust side).
//!
//! This example is the Rust half of `scripts/fuzz-diff.sh` in the WeftFS parent
//! repository. It reads `conformance/fuzz/v1/seeds.txt`, generates exactly the
//! same N inputs as `WeftCodec-Go/cmd/fuzzdiff`, decodes each one through the
//! crate's normal [`weftcodec::modes::decode_bytes`] entry point with
//! `Limits::default()`, and prints one tab-separated line per input:
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
//! For input index `i` in `[0, N)`:
//!
//! * Draw `choice = next_u64()`. If `choice & 1 == 0` (probability 1/2):
//!   choose `seed_index = below(len(seeds))`, clone that seed, then apply
//!   `1 + below(8)` mutations in order.
//! * Otherwise emit `len = below(2049)` pure random bytes, each from
//!   `next_byte()`.
//!
//! Mutation operation `below(5)`:
//!
//! * 0 bit flip: if non-empty, `pos = below(len)`, `bit = below(8)`,
//!   `data[pos] ^= 1 << bit`.
//! * 1 byte replace: if non-empty, `pos = below(len)`, `value = below(256)`,
//!   `data[pos] = value`.
//! * 2 insert: if `len(data) < 4096`, `room = 4096 - len(data)`,
//!   `n = min(1 + below(32), room)`, `pos = below(len(data) + 1)`, insert
//!   `n` bytes from `next_byte()` at `pos`.
//! * 3 delete range: if non-empty, `start = below(len)`,
//!   `n = 1 + below(len - start)`, remove `[start, start + n)`.
//! * 4 truncate: if non-empty, `new_len = below(len + 1)`, truncate to it.
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

use weftcodec::dependency::MapResolver;
use weftcodec::hash::{hash_domain, DOMAIN_CHUNK};
use weftcodec::limits::Limits;
use weftcodec::modes;

const PRNG_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const MAX_RANDOM_LEN: u64 = 2048;
const MAX_INPUT_LEN: usize = 4096;
const MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;

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

fn mutate(rng: &mut Rng, data: &mut Vec<u8>) {
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

fn generate(rng: &mut Rng, seeds: &[Vec<u8>]) -> Vec<u8> {
    if rng.next_u64() & 1 == 0 {
        let index = rng.below(seeds.len() as u64) as usize;
        let mut data = seeds[index].clone();
        let mutations = 1 + rng.below(8);
        for _ in 0..mutations {
            mutate(rng, &mut data);
        }
        data
    } else {
        let len = rng.below(MAX_RANDOM_LEN + 1) as usize;
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(rng.next_byte());
        }
        data
    }
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
    eprintln!("usage: fuzz_differential --seeds <path> [--n <count>]");
    std::process::exit(64)
}

fn main() -> ExitCode {
    let mut seeds_path: Option<String> = None;
    let mut n: u64 = 1_000_000;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seeds" => seeds_path = Some(args.next().unwrap_or_else(|| usage())),
            "--n" => {
                let value = args.next().unwrap_or_else(|| usage());
                n = value.parse().unwrap_or_else(|_| usage());
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
    let limits = Limits::default();

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
        let input = generate(&mut rng, &seeds);
        hasher.update(&(input.len() as u64).to_le_bytes());
        hasher.update(&input);

        let outcome = catch_unwind(AssertUnwindSafe(|| {
            modes::decode_bytes(&input, &resolver, &limits)
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
