//! Synthetic benchmark for modes 12 (`KV_CACHE`) and 13
//! (`ACTIVATION_STREAM`).
//!
//! Run in the rust court:
//!
//! ```sh
//! scripts/docker-court.sh rust \
//!   "cd /work/WeftCodec && WEFT_BENCH_TIMESTAMP=\$(date -u +%Y%m%dT%H%M%SZ) \
//!    cargo run --release --quiet --example bench_kv_activation -- \
//!    /work/bench/receipts/kv-activation-\$WEFT_BENCH_TIMESTAMP.json"
//! ```
//!
//! The receipt is synthetic single-host timing only. It makes no comparison or
//! superiority claim.

use std::env;
use std::fs;
use std::time::Instant;

use serde_json::json;
use weftcodec::activation_stream::{
    assemble_stream, decode_activation_chunk, encode_activation_chunk, ActivationStreamHead,
    StreamDtype, StreamKind,
};
use weftcodec::frame::Frame;
use weftcodec::kv_cache::{decode_kv_page, encode_kv_page, KvCacheInput, KvDtype, KvPageSet};
use weftcodec::limits::Limits;

const MIB: u64 = 1 << 20;

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

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let middle = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn env_or(name: &str, fallback: &str) -> String {
    env::var(name).unwrap_or_else(|_| fallback.to_owned())
}

fn main() {
    let output = env::args()
        .nth(1)
        .expect("usage: bench_kv_activation <receipt.json>");
    let timestamp = env_or("WEFT_BENCH_TIMESTAMP", "unset");
    let limits = Limits::default();

    // --- KV_CACHE: 16 MiB in 16 pages of 1 MiB (bf16, 512 x 16 x 64). ---
    let kv_page_rows = 512u64;
    let kv_head_count = 16u32;
    let kv_head_dim = 64u32;
    let kv_page_bytes = kv_page_rows * u64::from(kv_head_count) * u64::from(kv_head_dim) * 2;
    let kv_page_count = 16u64;
    let mut kv_frames = Vec::new();
    let mut kv_elements = Vec::new();
    let mut kv_encode_ms = Vec::new();
    let mut kv_payload_bytes = 0u64;
    for index in 0..kv_page_count {
        let elements = pattern(kv_page_bytes as usize, 0x51ED_2701 + index);
        let input = KvCacheInput {
            model: "synthetic-ai-v1".to_owned(),
            layer: 7,
            session: "bench-session".to_owned(),
            page_index: index,
            page_count: kv_page_count,
            seq_start: index * kv_page_rows,
            seq_end: (index + 1) * kv_page_rows,
            dtype: KvDtype::Bf16,
            head_count: kv_head_count,
            head_dim: kv_head_dim,
        };
        let start = Instant::now();
        let frame = encode_kv_page(&input, &elements, &limits).expect("encode kv page");
        kv_encode_ms.push(elapsed_ms(start));
        let parsed = Frame::parse(&frame, &limits).expect("parse kv page");
        let meta = weftcodec::kv_cache::kv_cache_metadata(&parsed, &limits).expect("kv meta");
        kv_payload_bytes += meta.payload_len;
        kv_frames.push(frame);
        kv_elements.push(elements);
    }
    let kv_frame_bytes: u64 = kv_frames.iter().map(|frame| frame.len() as u64).sum();
    let kv_element_bytes: u64 = kv_elements
        .iter()
        .map(|elements| elements.len() as u64)
        .sum();
    let parsed_kv: Vec<Frame<'_>> = kv_frames
        .iter()
        .map(|frame| Frame::parse(frame, &limits).expect("parse kv"))
        .collect();

    let mut kv_decode_all_ms = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        for (index, frame) in parsed_kv.iter().enumerate() {
            let page = decode_kv_page(frame, &limits).expect("decode kv");
            assert_eq!(page.meta.page_index, index as u64);
        }
        kv_decode_all_ms.push(elapsed_ms(start));
    }
    let mut kv_single_ms = Vec::new();
    for _ in 0..101 {
        let start = Instant::now();
        let _ = decode_kv_page(&parsed_kv[7], &limits).expect("decode kv single");
        kv_single_ms.push(elapsed_ms(start));
    }
    let mut set = KvPageSet::new(kv_frames.clone());
    let _ = set.decode_page(7, &limits).expect("set decode");
    assert_eq!(set.decoded_pages(), 1);

    // --- ACTIVATION_STREAM: 64 MiB in 64 chunks of 1 MiB (bf16, 1 x 524288). ---
    let stream_chunk_bytes = 1 * MIB;
    let stream_chunk_count = 64u64;
    let stream_shape = vec![1u64, 524_288u64];
    let stream_head = ActivationStreamHead {
        stream: StreamKind::Activation,
        step: 1,
        rank: 0,
        world: 1,
        chunk_count: stream_chunk_count,
        dtype: StreamDtype::Bf16,
        shape: stream_shape,
    };
    let mut stream_frames = Vec::new();
    let mut stream_encode_ms = Vec::new();
    let mut stream_payload_bytes = 0u64;
    for index in 0..stream_chunk_count {
        let elements = pattern(stream_chunk_bytes as usize, 0xA11C_E001 + index);
        let start = Instant::now();
        let frame = encode_activation_chunk(&stream_head, index, &elements, &limits)
            .expect("encode stream chunk");
        stream_encode_ms.push(elapsed_ms(start));
        let parsed = Frame::parse(&frame, &limits).expect("parse stream chunk");
        let meta = weftcodec::activation_stream::activation_chunk_metadata(&parsed, &limits)
            .expect("stream meta");
        stream_payload_bytes += meta.payload_len;
        stream_frames.push(frame);
    }
    let stream_frame_bytes: u64 = stream_frames.iter().map(|frame| frame.len() as u64).sum();
    let parsed_stream: Vec<Frame<'_>> = stream_frames
        .iter()
        .map(|frame| Frame::parse(frame, &limits).expect("parse stream"))
        .collect();

    let mut stream_decode_all_ms = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for frame in &parsed_stream {
            let _ = decode_activation_chunk(frame, &limits).expect("decode stream chunk");
        }
        stream_decode_all_ms.push(elapsed_ms(start));
    }
    let stream_refs: Vec<&[u8]> = stream_frames.iter().map(|frame| frame.as_slice()).collect();
    let mut stream_assemble_ms = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let assembly = assemble_stream(&stream_refs, &limits).expect("assemble stream");
        assert_eq!(
            assembly.bytes.len() as u64,
            stream_chunk_bytes * stream_chunk_count
        );
        stream_assemble_ms.push(elapsed_ms(start));
    }
    let mut stream_single_ms = Vec::new();
    for _ in 0..101 {
        let start = Instant::now();
        let _ = decode_activation_chunk(&parsed_stream[31], &limits).expect("decode single");
        stream_single_ms.push(elapsed_ms(start));
    }

    let receipt = json!({
        "schema": "weft.bench.kv-activation.v1",
        "timestamp": timestamp,
        "label": "synthetic, single host",
        "revision": {
            "weftfs": env_or("WEFT_REPO_WEFTFS", "unset"),
            "weftcodec": env_or("WEFT_REPO_WEFTCODEC", "unset"),
            "weftcodec_go": env_or("WEFT_REPO_WEFTCODEC_GO", "unset"),
        },
        "court": {
            "image": env_or("WEFT_COURT_IMAGE", "unset"),
            "image_id": env_or("WEFT_COURT_IMAGE_ID", "unset"),
            "kernel": env_or("WEFT_COURT_KERNEL", "unset"),
            "memory": env_or("WEFT_COURT_MEMORY", "unset"),
            "cpus": env_or("WEFT_COURT_CPUS", "unset"),
            "pids": env_or("WEFT_COURT_PIDS", "unset"),
            "docker_server": env_or("WEFT_DOCKER_SERVER", "unset"),
        },
        "limits": {
            "max_decoded_bytes": limits.max_decoded_bytes,
            "max_frame_bytes": limits.max_frame_bytes,
            "max_tensor_elements": limits.max_tensor_elements,
            "max_tensor_rank": limits.max_tensor_rank,
            "max_structured_items": limits.max_structured_items,
        },
        "kv_cache": {
            "dtype": "bf16",
            "pages": kv_page_count,
            "page_bytes": kv_page_bytes,
            "element_bytes": kv_element_bytes,
            "frame_bytes": kv_frame_bytes,
            "payload_bytes": kv_payload_bytes,
            "encode_ms_median": median_ms(&mut kv_encode_ms),
            "decode_all_ms_median": median_ms(&mut kv_decode_all_ms),
            "decode_single_page_ms_median": median_ms(&mut kv_single_ms),
            "decode_all_pages_per_second": 1000.0 * kv_page_count as f64
                / median_ms(&mut kv_decode_all_ms.clone()),
        },
        "activation_stream": {
            "dtype": "bf16",
            "chunks": stream_chunk_count,
            "chunk_bytes": stream_chunk_bytes,
            "element_bytes": stream_chunk_bytes * stream_chunk_count,
            "frame_bytes": stream_frame_bytes,
            "payload_bytes": stream_payload_bytes,
            "encode_ms_median": median_ms(&mut stream_encode_ms),
            "decode_all_ms_median": median_ms(&mut stream_decode_all_ms),
            "assemble_ms_median": median_ms(&mut stream_assemble_ms),
            "decode_single_chunk_ms_median": median_ms(&mut stream_single_ms),
            "decode_all_chunks_per_second": 1000.0 * stream_chunk_count as f64
                / median_ms(&mut stream_decode_all_ms.clone()),
        },
        "notes": [
            "single-chunk decode is one decode_activation_chunk call; whole-set decode decodes all 64 chunks and drops the output",
            "assembly additionally verifies every chunk digest, concatenates the elements, and computes the stream root",
            "Huffman planes are not seekable, so a range read decodes exactly one page/chunk and slices it",
        ],
    });
    let text = serde_json::to_string_pretty(&receipt).expect("serialize receipt");
    fs::write(&output, format!("{text}\n")).expect("write receipt");
    println!("{text}");
}
