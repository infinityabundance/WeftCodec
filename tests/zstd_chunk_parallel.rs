//! R22 byte-identity court for chunk-parallel ZSTD encoding.
//!
//! A CHUNKED ZSTD frame is a concatenation of independent per-chunk frames
//! (`WEFTCODEC-FORMAT` §3). `encode_zstd_with` encodes those chunks on several
//! threads in **ordered batches** and concatenates them in index order, so the
//! claim is exact byte identity with `encode_zstd`, not a tolerance:
//!
//! 1. **Boundary matrix** — for two codec chunk sizes, every input length in
//!    `{k·chunk - 1, k·chunk, k·chunk + 1}` for `k = 0..=6` plus a set of
//!    awkward lengths, at levels 3 and 19, with 1, 2, 3, 4 and 8 threads and
//!    with a batch bound that forces several batches (1 chunk, 2, 4, 16 and the
//!    default).
//! 2. **Randomised differential** — 2000 pseudo-random inputs (compressible,
//!    incompressible and degenerate) at random lengths, chunk sizes, levels and
//!    thread counts.
//! 3. **Whole-frame check** — every parallel frame parses, decodes to the input
//!    through the normal decoder, and carries the same chunk table, content
//!    hash and chunk hashes as the sequential frame.
//! 4. **Fallback equality** — an input below the parallel threshold, an
//!    unchunked call and a one-thread bound must all be the sequential path's
//!    bytes.

use weftcodec::dependency::MapResolver;
use weftcodec::limits::Limits;
use weftcodec::modes::{self, ZstdEncodeOptions};
use weftcodec::Frame;

fn limits() -> Limits {
    Limits::default()
}

/// Deterministic xorshift64* — no external crates, reproducible across hosts.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            0
        } else {
            self.next() % bound
        }
    }
}

fn options(threads: usize, batch: usize, min_bytes: usize) -> ZstdEncodeOptions {
    ZstdEncodeOptions {
        threads,
        parallel_min_bytes: min_bytes,
        batch_decoded_bytes: batch,
    }
}

/// Compare the parallel frame with the sequential frame, byte for byte, and
/// re-parse both.
fn assert_identical(data: &[u8], level: i32, chunk: usize, opts: &ZstdEncodeOptions) -> usize {
    let limits = limits();
    let sequential =
        modes::encode_zstd(data, level, Some(chunk), &limits).expect("sequential encode");
    let parallel =
        modes::encode_zstd_with(data, level, Some(chunk), &limits, opts).expect("parallel encode");
    assert_eq!(
        sequential.len(),
        parallel.len(),
        "frame length differs (len {}, level {level}, chunk {chunk}, threads {})",
        data.len(),
        opts.threads
    );
    if sequential != parallel {
        let first = sequential
            .iter()
            .zip(parallel.iter())
            .position(|(a, b)| a != b)
            .expect("same length but different bytes");
        panic!(
            "frame bytes differ at offset {first} (len {}, level {level}, chunk {chunk}, threads {}, batch {})",
            data.len(),
            opts.threads,
            opts.batch_decoded_bytes
        );
    }
    // The same table and the same decoded bytes through the ordinary decoder.
    let frame = Frame::parse(&parallel, &limits).expect("parallel frame parses");
    assert_eq!(frame.header.mode, weftcodec::Mode::Zstd);
    // An empty input is never chunked (the chunk table cannot be empty when
    // CHUNKED); every non-empty chunked input is.
    assert_eq!(frame.header.is_chunked(), !data.is_empty());
    let plain = modes::decode_bytes(&parallel, &MapResolver::new(), &limits)
        .expect("parallel frame decodes");
    assert_eq!(plain, data, "parallel frame does not decode to its input");
    let sequential_frame = Frame::parse(&sequential, &limits).expect("sequential frame parses");
    assert_eq!(
        frame.chunks.len(),
        sequential_frame.chunks.len(),
        "chunk count differs"
    );
    for (a, b) in frame.chunks.iter().zip(sequential_frame.chunks.iter()) {
        assert_eq!(a, b, "chunk entry differs");
    }
    assert_eq!(
        frame.header.content_hash,
        sequential_frame.header.content_hash
    );
    frame.chunks.len()
}

/// Compressible-ish payload with a tunable alphabet.
fn payload(len: usize, alphabet: u8) -> Vec<u8> {
    let mut rng = Rng(0x00C0_FFEE_1234_5678 ^ len as u64 ^ u64::from(alphabet));
    (0..len)
        .map(|index| {
            if alphabet == 0 {
                0u8
            } else {
                // A repeated ramp plus noise keeps the chunks compressible but
                // not identical, so a batch/order bug changes the bytes.
                ((index as u64 % u64::from(alphabet.max(1))) as u8)
                    .wrapping_add((rng.next() & 0x03) as u8)
            }
        })
        .collect()
}

#[test]
fn every_chunk_boundary_and_thread_count_is_byte_identical() {
    let limits = limits();
    for chunk in [4096usize, 65536usize] {
        let mut lengths: Vec<usize> = Vec::new();
        for k in 0usize..=6 {
            let base = k * chunk;
            lengths.extend([base.saturating_sub(1), base, base + 1]);
        }
        lengths.extend([
            1,
            7,
            4095,
            4097,
            chunk * 2 + 4095,
            chunk * 3 + 12345,
            chunk * 5 - 7,
        ]);
        for level in [3i32, 19i32] {
            for threads in [1usize, 2, 3, 4, 8] {
                for batch in [chunk, chunk * 2, chunk * 4, chunk * 16, 32 * 1024 * 1024] {
                    for len in &lengths {
                        let data = payload(*len, 24);
                        let opts = options(threads, batch, 0);
                        let count = assert_identical(&data, level, chunk, &opts);
                        assert_eq!(count, len.div_ceil(chunk), "chunk count");
                    }
                }
            }
        }
    }
    // The empty input has no chunk table at all: the parallel path must not
    // engage and the frames must still match.
    let empty = modes::encode_zstd(&[], 3, Some(4096), &limits).expect("sequential empty");
    let parallel = modes::encode_zstd_with(&[], 3, Some(4096), &limits, &options(4, 1 << 20, 0))
        .expect("parallel empty");
    assert_eq!(empty, parallel);
}

#[test]
fn randomised_differential_is_byte_identical() {
    let mut rng = Rng(0x5EED_0BAD_CAFE_1234);
    let mut compared = 0usize;
    for case in 0..2000u32 {
        let len = match case % 4 {
            0 => rng.below(3 * 4096) as usize,
            1 => rng.below(200_000) as usize,
            2 => rng.below(1_000_000) as usize,
            _ => rng.below(40_000) as usize,
        };
        let alphabet = match case % 5 {
            0 => 0u8,
            1 => 2,
            2 => 16,
            3 => 64,
            _ => 255,
        };
        let data = payload(len, alphabet);
        let chunk = match case % 3 {
            0 => 4096usize,
            1 => 8192,
            _ => 65536,
        };
        let level = if case % 7 == 0 { 19 } else { 3 };
        let threads = 1 + (case as usize % 8);
        let batch = chunk * (1 + (case as usize % 8));
        let opts = options(threads, batch, 0);
        assert_identical(&data, level, chunk, &opts);
        compared += 1;
    }
    assert_eq!(compared, 2000);
}

#[test]
fn incompressible_and_degenerate_inputs_are_byte_identical() {
    let chunk = 4096usize;
    let opts = options(4, chunk * 3, 0);
    // Incompressible: every byte pattern present, no runs.
    let mut rng = Rng(0xDEAD_BEEF_0000_0001);
    let noise: Vec<u8> = (0..200_000).map(|_| (rng.next() & 0xFF) as u8).collect();
    assert_identical(&noise, 3, chunk, &opts);
    // Degenerate: one byte repeated, and a run of zeros past the last boundary.
    for value in [0u8, 0xFFu8, b'A'] {
        let flat = vec![value; 100_003];
        assert_identical(&flat, 3, chunk, &opts);
        assert_identical(&flat, 19, chunk, &opts);
    }
}

#[test]
fn the_sequential_paths_are_unchanged() {
    let limits = limits();
    let data = payload(300_000, 32);
    let sequential = modes::encode_zstd(&data, 3, Some(4096), &limits).expect("sequential");
    // A bound that forces the sequential branch (input under the threshold).
    let below = modes::encode_zstd_with(
        &data,
        3,
        Some(4096),
        &limits,
        &options(4, 1 << 20, data.len() + 1),
    )
    .expect("sequential fallback");
    assert_eq!(sequential, below);
    // One thread is the sequential encoder.
    let one = modes::encode_zstd_with(&data, 3, Some(4096), &limits, &options(1, 1 << 20, 0))
        .expect("one thread");
    assert_eq!(sequential, one);
    // No chunking is one frame: the parallel schedule cannot apply.
    let unchunked = modes::encode_zstd(&data, 3, None, &limits).expect("unchunked");
    let unchunked_parallel =
        modes::encode_zstd_with(&data, 3, None, &limits, &options(8, 1 << 20, 0))
            .expect("unchunked parallel");
    assert_eq!(unchunked, unchunked_parallel);
}

#[test]
fn a_zero_batch_bound_is_refused() {
    let limits = limits();
    let err = modes::encode_zstd_with(b"some bytes", 3, Some(4096), &limits, &options(4, 0, 0))
        .expect_err("a zero byte bound is unbounded memory");
    assert_eq!(err.class(), weftcodec::ErrorClass::Malformed);
}
