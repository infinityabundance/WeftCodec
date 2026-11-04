//! Deterministic 4 MiB bf16 WEIGHT_DELTA benchmark.
//!
//! Emits one JSON object on stdout. The receipt wrapper records the court,
//! revision, and limits that the codec itself cannot observe.
//!
//! Methodology: one 4 MiB bf16 base tensor and a target that differs in 1% of
//! its elements. Timings are the median of five runs after one warm-up. The
//! "independent" baseline is `zstd::bulk` over the target element bytes; it is
//! not a WeftCodec frame, so no frame overhead is included in that number.
//! Huffman plane sizes are computed by encoding each delta byte plane with the
//! same `huffman-canonical-v1` coder the mode uses.

use std::time::Instant;

use serde_json::json;

use weftcodec::dependency::MapResolver;
use weftcodec::hash::{hash_domain, DOMAIN_CHUNK};
use weftcodec::huffman;
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::tensor::{LogicalRecord, TensorDtype};
use weftcodec::weight_delta::{decode_with_base_frame, encode_weight_delta, WeightDtype};
use weftcodec::Frame;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unknown".to_owned())
}

fn main() {
    let limits = Limits::default();
    let element_count: usize = 2 * 1024 * 1024; // 4 MiB bf16
    let shape = vec![element_count as u64];

    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut base_bytes = Vec::with_capacity(element_count * 2);
    for _ in 0..element_count {
        let word = (xorshift(&mut state) & 0xFFFF) as u16;
        base_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut target_bytes = base_bytes.clone();
    let mut changed = 0usize;
    for index in (0..element_count).step_by(100) {
        target_bytes[index * 2] ^= 0x01;
        target_bytes[index * 2 + 1] ^= 0x20;
        changed += 1;
    }

    let base_frame = modes::encode_tensor(&base_bytes, TensorDtype::Bf16, &shape, &limits)
        .expect("base tensor frame");

    // One warm-up, then five measured runs.
    let warm = encode_weight_delta(
        &target_bytes,
        WeightDtype::Bf16,
        &shape,
        &base_frame,
        &limits,
    )
    .expect("warm-up encode");
    let mut encode_runs = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        let frame = encode_weight_delta(
            &target_bytes,
            WeightDtype::Bf16,
            &shape,
            &base_frame,
            &limits,
        )
        .expect("encode");
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(frame.len(), warm.len());
        encode_runs.push(elapsed);
    }

    let delta_frame = warm;
    let parsed = Frame::parse(&delta_frame, &limits).expect("parse delta");
    let mut decode_runs = Vec::with_capacity(5);
    let mut decoded = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        decoded = decode_with_base_frame(&parsed, &base_frame, &MapResolver::new(), &limits)
            .expect("decode");
        decode_runs.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    assert_eq!(decoded, target_bytes, "reconstruction must be byte-exact");

    let original_root =
        LogicalRecord::from_row_major(TensorDtype::Bf16, &shape, &target_bytes, &limits)
            .expect("logical record")
            .root_with_limits(&limits)
            .expect("logical root");
    let decoded_root = LogicalRecord::from_row_major(TensorDtype::Bf16, &shape, &decoded, &limits)
        .expect("decoded logical record")
        .root_with_limits(&limits)
        .expect("decoded logical root");

    // Delta byte planes, exactly as the mode splits them.
    let mut plane0 = Vec::with_capacity(element_count);
    let mut plane1 = Vec::with_capacity(element_count);
    for index in 0..element_count {
        let word = u16::from_le_bytes([base_bytes[index * 2], base_bytes[index * 2 + 1]]);
        let target = u16::from_le_bytes([target_bytes[index * 2], target_bytes[index * 2 + 1]]);
        let delta = target.wrapping_sub(word);
        plane0.push(delta as u8);
        plane1.push((delta >> 8) as u8);
    }
    let huffman0 = huffman::encode(&plane0).len();
    let huffman1 = huffman::encode(&plane1).len();

    let zstd3 = zstd::bulk::compress(&target_bytes, 3)
        .expect("zstd level 3")
        .len();
    let zstd19 = zstd::bulk::compress(&target_bytes, 19)
        .expect("zstd level 19")
        .len();

    let receipt = json!({
        "label": "synthetic, single host, Docker court; 4 MiB bf16 pair with 1% changed elements",
        "court": {
            "image": env("WEFT_COURT_IMAGE"),
            "image_id": env("WEFT_COURT_IMAGE_ID"),
            "kernel": env("WEFT_COURT_KERNEL"),
            "memory": env("WEFT_COURT_MEMORY"),
            "cpus": env("WEFT_COURT_CPUS"),
            "pids": env("WEFT_COURT_PIDS"),
            "docker_server": env("WEFT_DOCKER_SERVER"),
            "docker_client": env("WEFT_DOCKER_CLIENT"),
        },
        "revision": {
            "weftfs": env("WEFT_REPO_WEFTFS"),
            "weftcodec": env("WEFT_REPO_WEFTCODEC"),
            "weftcodec_go": env("WEFT_REPO_WEFTCODEC_GO"),
            "weftfs_go": env("WEFT_REPO_WEFTFS_GO"),
        },
        "limits": {
            "max_frame_bytes": limits.max_frame_bytes,
            "max_decoded_bytes": limits.max_decoded_bytes,
            "max_dependency_depth": limits.max_dependency_depth,
            "max_dependency_count": limits.max_dependency_count,
            "max_total_fetched_bytes": limits.max_total_fetched_bytes,
            "max_tensor_elements": limits.max_tensor_elements,
            "max_tensor_rank": limits.max_tensor_rank,
        },
        "workload": {
            "dtype": "bf16",
            "shape": shape,
            "element_count": element_count,
            "element_bytes": element_count * 2,
            "changed_elements": changed,
            "changed_fraction": changed as f64 / element_count as f64,
        },
        "bytes": {
            "base_tensor_frame": base_frame.len(),
            "delta_frame": delta_frame.len(),
            "independent_zstd_level3": zstd3,
            "independent_zstd_level19": zstd19,
            "delta_vs_zstd_level3": delta_frame.len() as f64 / zstd3 as f64,
        },
        "planes": {
            "raw": [element_count, element_count],
            "huffman": [huffman0, huffman1],
            "huffman_total": huffman0 + huffman1,
            "raw_total": element_count * 2,
        },
        "timings_ms": {
            "encode_runs": encode_runs,
            "encode_median": median(encode_runs),
            "decode_runs": decode_runs,
            "decode_median": median(decode_runs),
        },
        "exact": decoded == target_bytes,
        "logical_root_match": original_root == decoded_root,
        "logical_root": original_root.to_hex(),
        "target_content_hash": hash_domain(DOMAIN_CHUNK, &target_bytes).to_hex(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).expect("serialize receipt")
    );
}
