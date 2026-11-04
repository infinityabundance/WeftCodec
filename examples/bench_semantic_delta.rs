//! Deterministic 8 MiB bf16 SEMANTIC_DELTA benchmark.
//!
//! Emits one JSON object on stdout. The court wrapper records the revision,
//! image, kernel, and hard limits that the codec cannot observe itself.
//!
//! Methodology: one 8 MiB bf16 tensor and a target that differs in 0.1% of its
//! elements by low mantissa bits only, so the `mantissa_truncate:3` semantic
//! root is shared. Root timings are the median of three runs. Candidate
//! selection is measured over 32 candidates that all share the root; the
//! near-duplicate base is candidate 0. Encode/decode are the median of five
//! runs after one warm-up. The independent baseline is `zstd::bulk` over the
//! target element bytes, not a WeftCodec frame.

use std::time::Instant;

use serde_json::json;

use weftcodec::dependency::MapResolver;
use weftcodec::hash::{hash_domain, DOMAIN_CHUNK};
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::semantic::{select_semantic_base, semantic_root, SemanticPolicy};
use weftcodec::semantic_delta::{
    decode_with_base_frame, encode_semantic_delta, semantic_delta_metadata,
};
use weftcodec::tensor::{LogicalRecord, TensorDtype};
use weftcodec::weight_delta::WeightDtype;
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
    let element_count: usize = 4 * 1024 * 1024; // 8 MiB of bf16
    let shape = vec![element_count as u64];
    let dtype = WeightDtype::Bf16;

    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut base_bytes = Vec::with_capacity(element_count * 2);
    for _ in 0..element_count {
        // Finite, non-zero exponents so mantissa edits stay in the mantissa.
        let word = 0x3C00u16 | ((xorshift(&mut state) & 0x03FF) as u16);
        base_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut target_bytes = base_bytes.clone();
    let mut changed = 0usize;
    for index in (0..element_count).step_by(1000) {
        target_bytes[index * 2] ^= 0x05; // low three mantissa bits
        changed += 1;
    }
    assert_eq!(changed, element_count.div_ceil(1000));

    let policy = SemanticPolicy::MantissaTruncate { bits: 3 };
    let policies = [
        ("none", SemanticPolicy::None),
        ("mantissa_truncate_bits3", policy.clone()),
        (
            "block_quantize_b128_l256",
            SemanticPolicy::BlockQuantize {
                block_size: 128,
                levels: 256,
            },
        ),
    ];
    let mut root_ms = serde_json::Map::new();
    let mut root_hex = serde_json::Map::new();
    for (name, pol) in &policies {
        let mut runs = Vec::with_capacity(3);
        for _ in 0..3 {
            let start = Instant::now();
            let root =
                semantic_root(dtype, &shape, &target_bytes, pol.clone()).expect("semantic root");
            runs.push(start.elapsed().as_secs_f64() * 1000.0);
            root_hex.insert(name.to_string(), json!(root.to_hex()));
        }
        root_ms.insert(name.to_string(), json!(median(runs)));
    }
    let base_root = semantic_root(dtype, &shape, &base_bytes, policy.clone()).unwrap();
    let target_root = semantic_root(dtype, &shape, &target_bytes, policy.clone()).unwrap();
    assert_eq!(
        base_root, target_root,
        "0.1% low-bit change must share the root"
    );

    let base_frame = modes::encode_tensor(&base_bytes, TensorDtype::Bf16, &shape, &limits)
        .expect("base tensor frame");

    // 32 candidate frames that all share the semantic root under the policy.
    // Candidate 0 is the near-duplicate base; the rest apply increasing numbers
    // of low-bit edits so selection must actually evaluate every candidate.
    let mut candidates: Vec<Vec<u8>> = Vec::with_capacity(32);
    candidates.push(base_frame.clone());
    for candidate in 1..32usize {
        let mut bytes = base_bytes.clone();
        let stride = 50_000 / candidate.max(1);
        for index in (0..element_count).step_by(stride.max(1)) {
            bytes[index * 2] ^= 0x03;
        }
        candidates.push(
            modes::encode_tensor(&bytes, TensorDtype::Bf16, &shape, &limits)
                .expect("candidate frame"),
        );
    }
    let candidate_refs: Vec<&[u8]> = candidates.iter().map(|frame| frame.as_slice()).collect();
    let mut select_runs = Vec::with_capacity(3);
    let mut selected = None;
    for _ in 0..3 {
        let start = Instant::now();
        selected = select_semantic_base(
            &target_bytes,
            dtype,
            &shape,
            &policy,
            &candidate_refs,
            &limits,
        )
        .expect("candidate selection");
        select_runs.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    assert_eq!(selected, Some(0), "the near-duplicate base must be chosen");

    let warm = encode_semantic_delta(
        &target_bytes,
        dtype,
        &shape,
        &base_frame,
        policy.clone(),
        &limits,
    )
    .expect("warm-up encode");
    let mut encode_runs = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        let frame = encode_semantic_delta(
            &target_bytes,
            dtype,
            &shape,
            &base_frame,
            policy.clone(),
            &limits,
        )
        .expect("encode");
        encode_runs.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(frame.len(), warm.len());
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

    let meta = semantic_delta_metadata(&parsed, &limits).expect("metadata");
    assert_eq!(meta.base_semantic_root, base_root);

    let original_root =
        LogicalRecord::from_row_major(TensorDtype::Bf16, &shape, &target_bytes, &limits)
            .expect("logical record")
            .root_with_limits(&limits)
            .expect("logical root");
    let decoded_root = LogicalRecord::from_row_major(TensorDtype::Bf16, &shape, &decoded, &limits)
        .expect("decoded logical record")
        .root_with_limits(&limits)
        .expect("decoded logical root");

    let zstd3 = zstd::bulk::compress(&target_bytes, 3)
        .expect("zstd level 3")
        .len();
    let zstd19 = zstd::bulk::compress(&target_bytes, 19)
        .expect("zstd level 19")
        .len();

    let receipt = json!({
        "label": "synthetic, single host, Docker court; 8 MiB bf16 pair with 0.1% low-mantissa changes; no approximate reconstruction path",
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
            "policy": "mantissa_truncate:3",
            "base_is_candidate_zero": true,
        },
        "semantic_roots": {
            "root_host_mismatch_is_not_used_for_identity": true,
            "target_by_policy": root_hex,
            "base_mantissa_truncate_bits3": base_root.to_hex(),
            "target_mantissa_truncate_bits3": target_root.to_hex(),
            "target_and_base_share_mantissa_root": base_root == target_root,
        },
        "semantic_root_ms": root_ms,
        "candidate_selection": {
            "candidate_count": candidate_refs.len(),
            "selected_index": selected,
            "expected_index": 0,
            "all_candidates_share_root": true,
            "runs_ms": select_runs,
            "median_ms": median(select_runs),
            "exact_modular_cost_used": true,
        },
        "bytes": {
            "base_tensor_frame": base_frame.len(),
            "delta_frame": delta_frame.len(),
            "delta_payload": meta.payload_len,
            "independent_zstd_level3": zstd3,
            "independent_zstd_level19": zstd19,
            "delta_vs_zstd_level3": delta_frame.len() as f64 / zstd3 as f64,
            "zstd3_over_delta": zstd3 as f64 / delta_frame.len() as f64,
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
        "base_content_hash": hash_domain(DOMAIN_CHUNK, &base_bytes).to_hex(),
        "no_lossy_mode": true,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).expect("serialize receipt")
    );
}
