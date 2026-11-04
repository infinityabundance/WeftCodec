//! WEFT-TENSOR benchmark: logical root from row-major, 8 shards, and blocked
//! arrangements, plus a 1 MiB region read through row-major and sharded
//! physical layouts.
//!
//! Run inside the rust court (or any environment) with:
//!
//! ```sh
//! cargo run --release --example wefttensor_bench -- --receipt PATH
//! ```
//!
//! The tensor is synthetic and lives in process memory; this measures the
//! identity/address layer, not storage I/O.

use std::fs;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::json;

use weftcodec::limits::Limits;
use weftcodec::tensor::{self, DecodedShard, LayoutDescriptor, PhysicalTensor, Region};

const ELEMENT_BYTES: usize = 2; // bf16
const SHAPE: [u64; 3] = [128, 512, 512]; // 33,554,432 elements, 64 MiB
const SHARDS: u32 = 8;
const BLOCK_SHAPE: [u64; 3] = [16, 64, 64]; // 8 x 8 x 8 = 512 blocks
const SAMPLES: usize = 5;

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

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn time<T>(mut f: impl FnMut() -> T) -> (f64, Vec<f64>, T) {
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut result: Option<T> = None;
    for _ in 0..SAMPLES {
        let start = Instant::now();
        result = Some(f());
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    (
        median(&samples),
        samples,
        result.expect("at least one sample"),
    )
}

fn peak_rss_kib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let value = rest.trim().split_whitespace().next()?;
            return value.parse().ok();
        }
    }
    None
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (year + i64::from(month <= 2), month as u32, day as u32)
}

fn utc_stamp() -> (String, String) {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (seconds / 86_400) as i64;
    let secs_of_day = (seconds % 86_400) as u32;
    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;
    let compact = format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z");
    let iso = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z");
    (compact, iso)
}

fn env_or(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.to_owned())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut receipt_path: Option<String> = None;
    let mut receipt_dir: Option<String> = None;
    let mut label =
        String::from("synthetic, in-memory buffers (no I/O), single host; not a storage benchmark");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--receipt" => receipt_path = args.next(),
            "--receipt-dir" => receipt_dir = args.next(),
            "--label" => {
                if let Some(value) = args.next() {
                    label = value;
                }
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(64);
            }
        }
    }

    let limits = Limits::default();
    let dtype = tensor::dtype_from_name("bf16").expect("bf16 is a v1 dtype");
    let shape = SHAPE.to_vec();
    let element_count = tensor::validate_shape(&shape, &limits).expect("shape");
    let logical_bytes = element_count as usize * ELEMENT_BYTES;
    let values = pattern(logical_bytes, 0x5eed_2026);

    // (a) canonical row-major identity.
    let (row_major_ms, row_major_samples, row_root) =
        time(|| tensor::logical_root(dtype, &shape, &values, &limits).expect("row root"));

    // (b) eight shards on axis 0, then the shard-set verification path.
    let shard_payloads =
        tensor::split_shards(dtype, &shape, 0, SHARDS, &values, &limits).expect("split shards");
    let decoded: Vec<DecodedShard> = shard_payloads
        .iter()
        .enumerate()
        .map(|(index, payload)| DecodedShard {
            descriptor: LayoutDescriptor::sharded(dtype, &shape, 0, SHARDS, index as u32),
            bytes: payload.clone(),
        })
        .collect();
    let (shards_ms, shards_samples, shard_root) =
        time(|| tensor::shard_root(&decoded, &limits).expect("shard root"));

    // (c) blocked 16x64x64 (512 blocks).
    let block_shape = BLOCK_SHAPE.to_vec();
    let blocks = tensor::blocked_bytes(dtype, &shape, &block_shape, &values, &limits)
        .expect("blocked bytes");
    let block_count = blocks.len();
    let blocked = PhysicalTensor::blocked(dtype, &shape, &block_shape, blocks, &limits)
        .expect("blocked physical");
    let (blocked_ms, blocked_samples, blocked_root) =
        time(|| blocked.logical_root(&limits).expect("blocked root"));

    // Region read: 1 MiB = 2 x 512 x 512 bf16 elements.
    let region = Region::new(vec![64, 0, 0], vec![66, 512, 512]);
    let region_bytes = 2 * 512 * 512 * ELEMENT_BYTES;
    let row_physical =
        PhysicalTensor::row_major(dtype, &shape, values.clone(), &limits).expect("row physical");
    let sharded_physical = PhysicalTensor::sharded(dtype, &shape, 0, shard_payloads, &limits)
        .expect("sharded physical");
    let (row_region_ms, row_region_samples, row_read) =
        time(|| row_physical.read_region(&region).expect("row region"));
    let (sharded_region_ms, sharded_region_samples, sharded_read) = time(|| {
        sharded_physical
            .read_region(&region)
            .expect("sharded region")
    });
    assert_eq!(row_read.bytes.len(), region_bytes);
    assert_eq!(sharded_read.bytes, row_read.bytes);
    let shard_bytes = (element_count / u64::from(SHARDS)) as usize * ELEMENT_BYTES;

    let roots_equal = row_root == shard_root && row_root == blocked_root;

    let (stamp, iso) = utc_stamp();
    let receipt = match (receipt_path, receipt_dir) {
        (Some(path), _) => path,
        (None, Some(dir)) => format!("{}/wefttensor-{stamp}.json", dir.trim_end_matches('/')),
        (None, None) => format!("wefttensor-{stamp}.json"),
    };
    let output = json!({
        "schema": "weft.tensor-bench.v1",
        "kind": "tensor-bench",
        "label": label,
        "timestamp_utc": iso,
        "generated_utc": stamp,
        "package_version": env!("CARGO_PKG_VERSION"),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "revisions": {
            "WeftFS": env_or("WEFT_REPO_WEFTFS", "unknown"),
            "WeftFS-Go": env_or("WEFT_REPO_WEFTFS_GO", "unknown"),
            "WeftCodec": env_or("WEFT_REPO_WEFTCODEC", "unknown"),
            "WeftCodec-Go": env_or("WEFT_REPO_WEFTCODEC_GO", "unknown"),
        },
        "environment": {
            "court_image": env_or("WEFT_COURT_IMAGE", "unknown"),
            "court_image_id": env_or("WEFT_COURT_IMAGE_ID", "unknown"),
            "kernel": env_or("WEFT_COURT_KERNEL", "unknown"),
            "memory_limit": env_or("WEFT_COURT_MEMORY", "unknown"),
            "cpus": env_or("WEFT_COURT_CPUS", "unknown"),
            "pids_limit": env_or("WEFT_COURT_PIDS", "unknown"),
        },
        "tensor": {
            "dtype": dtype.as_str(),
            "shape": shape,
            "element_count": element_count,
            "logical_bytes": logical_bytes,
            "shard_axis": 0,
            "shards": SHARDS,
            "block_shape": block_shape,
            "blocks": block_count,
        },
        "measurements": {
            "logical_root_row_major": {
                "median_ms": row_major_ms,
                "samples_ms": row_major_samples,
                "bytes_hashed": logical_bytes,
                "logical_root": row_root.to_hex(),
            },
            "logical_root_8_shards": {
                "median_ms": shards_ms,
                "samples_ms": shards_samples,
                "bytes_read": logical_bytes,
                "shards_touched": SHARDS,
                "logical_root": shard_root.to_hex(),
            },
            "logical_root_blocked": {
                "median_ms": blocked_ms,
                "samples_ms": blocked_samples,
                "bytes_read": logical_bytes,
                "blocks_touched": block_count,
                "logical_root": blocked_root.to_hex(),
            },
            "region_1mib_row_major": {
                "median_ms": row_region_ms,
                "samples_ms": row_region_samples,
                "bytes_returned": row_read.bytes.len(),
                "units_touched": row_read.units_touched,
                "region": {"start": region.start, "end": region.end},
            },
            "region_1mib_sharded": {
                "median_ms": sharded_region_ms,
                "samples_ms": sharded_region_samples,
                "bytes_returned": sharded_read.bytes.len(),
                "units_touched": sharded_read.units_touched,
                "shards_touched": sharded_read.units_touched,
                "touched_unit_bytes": sharded_read.units_touched as usize * shard_bytes,
                "region": {"start": region.start, "end": region.end},
            },
        },
        "roots_equal": roots_equal,
        "peak_rss_kib": peak_rss_kib(),
    });
    let text = serde_json::to_string_pretty(&output).expect("serialize receipt");
    fs::write(&receipt, format!("{text}\n")).expect("write receipt");
    eprintln!("wrote {receipt}");
    if !roots_equal {
        std::process::exit(1);
    }
    println!(
        "row-major {:.1} ms, 8 shards {:.1} ms, blocked {:.1} ms, region row-major {:.3} ms ({} unit), sharded {:.3} ms ({} unit)",
        row_major_ms,
        shards_ms,
        blocked_ms,
        row_region_ms,
        row_read.units_touched,
        sharded_region_ms,
        sharded_read.units_touched
    );
}
