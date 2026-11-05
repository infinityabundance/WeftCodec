//! Real public weights: FLOAT_TENSOR (mode 9) vs zstd, and WEIGHT_DELTA
//! (mode 11) on consecutive checkpoints vs an independent fine-tune.
//!
//! This harness parses safetensors headers directly (8-byte little-endian
//! length followed by JSON; no transformers, no model framework) and feeds the
//! exact tensor bytes to the *shipped* codec entry points:
//!
//! * `modes::encode_float_tensor` / `modes::decode_float_tensor` (mode 9);
//! * `weight_delta::encode_weight_delta` / `weight_delta::decode_with_base_frame`
//!   (mode 11, base = the corresponding tensor's mode-9 frame).
//!
//! Independent baselines are produced by the real `zstd` CLI, one tensor at a
//! time (`zstd -19 -T1` and `zstd -3 -T1`), so the frame bytes are the CLI's
//! own output and not a library approximation. Every decode is verified
//! byte-exact against the original tensor bytes.
//!
//! Two phases:
//!
//! ```sh
//! real_float_tensor_bench tensors --file model.safetensors --label qwen-base \
//!     --scratch DIR --out tensors.json
//! real_float_tensor_bench delta --base-file older.safetensors \
//!     --target-file newer.safetensors --target-json tensors.json \
//!     --label pythia-step0-step1 --out delta.json
//! ```
//!
//! The harness writes only JSON; the court wrapper records the image, kernel,
//! limits, and corpus hashes.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{json, Value as Json};

use weftcodec::dependency::MapResolver;
use weftcodec::frame::Frame;
use weftcodec::limits::Limits;
use weftcodec::modes::{self, FloatDtype};
use weftcodec::weight_delta::{decode_with_base_frame, encode_weight_delta, WeightDtype};

const REPS: usize = 5;

fn main() {
    if let Err(err) = run() {
        eprintln!("real_float_tensor_bench: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let phase = args.first().cloned().unwrap_or_default();
    match phase.as_str() {
        "tensors" => tensors(&args[1..]),
        "delta" => delta(&args[1..]),
        other => Err(format!(
            "unknown phase `{other}` (expected `tensors` or `delta`)"
        )),
    }
}

fn value_of(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn required(args: &[String], flag: &str) -> Result<String, String> {
    value_of(args, flag).ok_or_else(|| format!("missing required {flag}"))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

// ---------------------------------------------------------------------------
// safetensors parsing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TensorEntry {
    name: String,
    /// Canonical WEFT-TENSOR text name (`bf16`, `fp16`, `fp32`).
    dtype: String,
    float_dtype: FloatDtype,
    weight_dtype: WeightDtype,
    shape: Vec<u64>,
    begin: u64,
    end: u64,
}

impl TensorEntry {
    fn byte_len(&self) -> u64 {
        self.end - self.begin
    }

    fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

struct Safetensors {
    path: PathBuf,
    file_len: u64,
    data_start: u64,
    tensors: Vec<TensorEntry>,
}

fn dtypes(text: &str) -> Result<(String, FloatDtype, WeightDtype), String> {
    match text {
        "BF16" => Ok(("bf16".to_owned(), FloatDtype::Bf16, WeightDtype::Bf16)),
        "F16" => Ok(("fp16".to_owned(), FloatDtype::F16, WeightDtype::Fp16)),
        "F32" => Ok(("fp32".to_owned(), FloatDtype::F32, WeightDtype::Fp32)),
        other => Err(format!(
            "safetensors dtype `{other}` has no FLOAT_TENSOR v1 mapping \
             (this harness measures bf16, fp16, and fp32)"
        )),
    }
}

fn element_bytes(dtype: FloatDtype) -> u64 {
    dtype.element_size()
}

fn parse_safetensors(path: &Path) -> Result<Safetensors, String> {
    let file_len = fs::metadata(path)
        .map_err(|err| format!("{}: {err}", path.display()))?
        .len();
    let mut file = File::open(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|err| format!("{}: {err}", path.display()))?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len > 64 * 1024 * 1024 || 8 + header_len > file_len {
        return Err(format!(
            "{}: implausible safetensors header length {header_len} (file {file_len})",
            path.display()
        ));
    }
    let mut header_json = vec![0u8; header_len as usize];
    file.read_exact(&mut header_json)
        .map_err(|err| format!("{}: {err}", path.display()))?;
    let parsed: Json = serde_json::from_slice(&header_json)
        .map_err(|err| format!("{}: header JSON: {err}", path.display()))?;
    let map = parsed
        .as_object()
        .ok_or_else(|| format!("{}: header is not a JSON object", path.display()))?;
    let data_start = 8 + header_len;
    let mut tensors = Vec::new();
    for (name, entry) in map {
        if name == "__metadata__" {
            continue;
        }
        let raw_dtype = entry
            .get("dtype")
            .and_then(Json::as_str)
            .ok_or_else(|| format!("{}/{name}: missing dtype", path.display()))?;
        let shape: Vec<u64> = entry
            .get("shape")
            .and_then(Json::as_array)
            .ok_or_else(|| format!("{}/{name}: missing shape", path.display()))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| format!("{}/{name}: non-integer shape", path.display()))
            })
            .collect::<Result<_, _>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Json::as_array)
            .ok_or_else(|| format!("{}/{name}: missing data_offsets", path.display()))?;
        if offsets.len() != 2 {
            return Err(format!(
                "{}/{name}: data_offsets must have 2 entries",
                path.display()
            ));
        }
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| format!("{}/{name}: bad data_offsets[0]", path.display()))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| format!("{}/{name}: bad data_offsets[1]", path.display()))?;
        let (dtype, float_dtype, weight_dtype) = dtypes(raw_dtype)?;
        let elements: u64 = shape.iter().product();
        let expected = elements
            .checked_mul(element_bytes(float_dtype))
            .ok_or_else(|| format!("{}/{name}: shape overflows byte length", path.display()))?;
        if end < begin || end - begin != expected {
            return Err(format!(
                "{}/{name}: data_offsets [{begin},{end}] do not match {elements} x {dtype}",
                path.display()
            ));
        }
        if data_start + end > file_len {
            return Err(format!(
                "{}/{name}: tensor extends past the file ({data_start}+{end} > {file_len})",
                path.display()
            ));
        }
        tensors.push(TensorEntry {
            name: name.clone(),
            dtype,
            float_dtype,
            weight_dtype,
            shape,
            begin,
            end,
        });
    }
    tensors.sort_by_key(|tensor| tensor.begin);
    let mut cursor = 0u64;
    for tensor in &tensors {
        if tensor.begin != cursor {
            return Err(format!(
                "{}: data section is not contiguous at {} (begin {}, expected {cursor})",
                path.display(),
                tensor.name,
                tensor.begin
            ));
        }
        cursor = tensor.end;
    }
    if data_start + cursor != file_len {
        return Err(format!(
            "{}: data section ends at {} but file is {file_len} bytes",
            path.display(),
            data_start + cursor
        ));
    }
    Ok(Safetensors {
        path: path.to_path_buf(),
        file_len,
        data_start,
        tensors,
    })
}

fn read_tensor_bytes(dataset: &Safetensors, tensor: &TensorEntry) -> Result<Vec<u8>, String> {
    let mut file =
        File::open(&dataset.path).map_err(|err| format!("{}: {err}", dataset.path.display()))?;
    file.seek(SeekFrom::Start(dataset.data_start + tensor.begin))
        .map_err(|err| format!("{}: {err}", dataset.path.display()))?;
    let mut bytes = vec![0u8; tensor.byte_len() as usize];
    file.read_exact(&mut bytes)
        .map_err(|err| format!("{}: {err}", dataset.path.display()))?;
    Ok(bytes)
}

fn file_blake3(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("{}: {err}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

// ---------------------------------------------------------------------------
// zstd CLI
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ZstdRun {
    bytes: u64,
    wall_ms: f64,
    stderr_tail: Vec<String>,
}

fn zstd_version() -> String {
    Command::new("zstd")
        .arg("--version")
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

/// Compress `input` with the real zstd CLI at `level` and return the frame
/// size. The caller supplies distinct paths so nothing is overwritten.
fn zstd_compress(
    level: u32,
    input: &Path,
    output: &Path,
    scratch: &Path,
) -> Result<ZstdRun, String> {
    if output.exists() {
        fs::remove_file(output).map_err(|err| format!("{}: {err}", output.display()))?;
    }
    let started = Instant::now();
    let result = Command::new("zstd")
        .arg(format!("-{level}"))
        .arg("-T1")
        .arg("--no-progress")
        .arg("-q")
        .arg("-f")
        .arg(input)
        .arg("-o")
        .arg(output)
        .output()
        .map_err(|err| format!("spawn zstd: {err}"))?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let _ = fs::remove_file(output);
        let _ = fs::remove_file(input);
        let _ = fs::remove_dir_all(scratch);
        return Err(format!(
            "zstd -{level} failed with {:?}: {}",
            result.status.code(),
            stderr.trim()
        ));
    }
    let bytes = fs::metadata(output)
        .map_err(|err| format!("{}: {err}", output.display()))?
        .len();
    let stderr_tail: Vec<String> = String::from_utf8_lossy(&result.stderr)
        .lines()
        .rev()
        .take(2)
        .map(|line| line.to_owned())
        .collect();
    Ok(ZstdRun {
        bytes,
        wall_ms,
        stderr_tail,
    })
}

// ---------------------------------------------------------------------------
// decode timing
// ---------------------------------------------------------------------------

fn timed_decode_frame(frame_bytes: &[u8], limits: &Limits) -> Result<(f64, Vec<u8>), String> {
    let mut samples = Vec::with_capacity(REPS);
    let mut decoded = Vec::new();
    for _ in 0..REPS {
        let started = Instant::now();
        let parsed = Frame::parse(frame_bytes, limits).map_err(|err| err.to_string())?;
        decoded = modes::decode_float_tensor(&parsed, limits).map_err(|err| err.to_string())?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    Ok((median(samples), decoded))
}

fn timed_decode_zstd(frame_bytes: &[u8], capacity: usize) -> Result<(f64, Vec<u8>), String> {
    let mut samples = Vec::with_capacity(REPS);
    let mut decoded = Vec::new();
    for _ in 0..REPS {
        let started = Instant::now();
        decoded = zstd::bulk::decompress(frame_bytes, capacity)
            .map_err(|err| format!("zstd decode: {err}"))?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    Ok((median(samples), decoded))
}

fn timed_decode_weight_delta(
    frame_bytes: &[u8],
    base_frame: &[u8],
    limits: &Limits,
) -> Result<(f64, Vec<u8>), String> {
    let resolver = MapResolver::new();
    let mut samples = Vec::with_capacity(REPS);
    let mut decoded = Vec::new();
    for _ in 0..REPS {
        let started = Instant::now();
        let parsed = Frame::parse(frame_bytes, limits).map_err(|err| err.to_string())?;
        decoded = decode_with_base_frame(&parsed, base_frame, &resolver, limits)
            .map_err(|err| err.to_string())?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    Ok((median(samples), decoded))
}

// ---------------------------------------------------------------------------
// tensors phase
// ---------------------------------------------------------------------------

fn tensors(args: &[String]) -> Result<(), String> {
    let file = PathBuf::from(required(args, "--file")?);
    let label = required(args, "--label")?;
    let scratch = PathBuf::from(required(args, "--scratch")?);
    let out = PathBuf::from(required(args, "--out")?);
    fs::create_dir_all(&scratch).map_err(|err| format!("{}: {err}", scratch.display()))?;
    fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))
        .map_err(|err| format!("{}: {err}", out.display()))?;

    let limits = Limits::default();
    let dataset = parse_safetensors(&file)?;
    let zstd_version = zstd_version();
    let payload_path = scratch.join("payload.bin");
    let level19_path = scratch.join("level19.zst");
    let level3_path = scratch.join("level3.zst");

    let mut rows = Vec::with_capacity(dataset.tensors.len());
    let mut totals = Totals::default();
    let mut ratios: Vec<(String, f64, u64, u64)> = Vec::new();
    let mut excess: Vec<(String, f64, u64, u64, f64)> = Vec::new();

    for tensor in &dataset.tensors {
        let bytes = read_tensor_bytes(&dataset, tensor)?;
        if bytes.is_empty() {
            return Err(format!("{}/{} is empty", file.display(), tensor.name));
        }

        let encode_started = Instant::now();
        let frame = modes::encode_float_tensor(&bytes, tensor.float_dtype, &tensor.shape, &limits)
            .map_err(|err| format!("mode 9 encode {}/{}: {err}", file.display(), tensor.name))?;
        let encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;

        let (decode_ms, decoded) = timed_decode_frame(&frame, &limits)?;
        if decoded != bytes {
            return Err(format!(
                "{}/{}: FLOAT_TENSOR round-trip is not byte-exact",
                file.display(),
                tensor.name
            ));
        }

        fs::write(&payload_path, &bytes)
            .map_err(|err| format!("{}: {err}", payload_path.display()))?;
        let z19 = zstd_compress(19, &payload_path, &level19_path, &scratch)?;
        let z3 = zstd_compress(3, &payload_path, &level3_path, &scratch)?;
        let compressed19 =
            fs::read(&level19_path).map_err(|err| format!("{}: {err}", level19_path.display()))?;
        let (zstd_decode_ms, zstd_decoded) = timed_decode_zstd(&compressed19, bytes.len())?;
        if zstd_decoded != bytes {
            return Err(format!(
                "{}/{}: zstd -19 round-trip is not byte-exact",
                file.display(),
                tensor.name
            ));
        }

        totals.raw_bytes += tensor.byte_len();
        totals.elements += tensor.elements();
        totals.mode9_bytes += frame.len() as u64;
        totals.zstd19_bytes += z19.bytes;
        totals.zstd3_bytes += z3.bytes;
        totals.mode9_encode_ms += encode_ms;
        totals.mode9_decode_ms_sum += decode_ms;
        totals.zstd19_decode_ms_sum += zstd_decode_ms;
        totals.zstd19_encode_ms += z19.wall_ms;
        totals.zstd3_encode_ms += z3.wall_ms;
        if frame.len() as u64 <= z19.bytes {
            totals.mode9_wins += 1;
        }
        if frame.len() as u64 <= bytes.len() as u64 {
            totals.mode9_not_larger_than_raw += 1;
        }
        ratios.push((
            tensor.name.clone(),
            ratio(frame.len() as u64, z19.bytes),
            frame.len() as u64,
            z19.bytes,
        ));
        if frame.len() as u64 > z19.bytes {
            excess.push((
                tensor.name.clone(),
                (frame.len() as u64 - z19.bytes) as f64,
                frame.len() as u64,
                z19.bytes,
                ratio(frame.len() as u64, z19.bytes),
            ));
        }

        rows.push(json!({
            "name": tensor.name,
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "elements": tensor.elements(),
            "bytes": tensor.byte_len(),
            "mode9_bytes": frame.len(),
            "mode9_ratio_of_raw": ratio(frame.len() as u64, tensor.byte_len()),
            "mode9_over_zstd19": ratio(frame.len() as u64, z19.bytes),
            "mode9_encode_ms": encode_ms,
            "mode9_decode_ms_median": decode_ms,
            "mode9_roundtrip_exact": true,
            "zstd19_bytes": z19.bytes,
            "zstd19_ratio_of_raw": ratio(z19.bytes, tensor.byte_len()),
            "zstd19_over_mode9": ratio(z19.bytes, frame.len() as u64),
            "zstd19_encode_ms": z19.wall_ms,
            "zstd19_decode_ms_median": zstd_decode_ms,
            "zstd19_roundtrip_exact": true,
            "zstd3_bytes": z3.bytes,
            "zstd3_ratio_of_raw": ratio(z3.bytes, tensor.byte_len()),
            "zstd3_encode_ms": z3.wall_ms,
            "zstd_cli_stderr_tail": z19.stderr_tail,
        }));

        let _ = fs::remove_file(&payload_path);
        let _ = fs::remove_file(&level19_path);
        let _ = fs::remove_file(&level3_path);
        drop(compressed19);
        drop(decoded);
        drop(bytes);
    }

    let mut ratio_values: Vec<f64> = ratios.iter().map(|(_, value, _, _)| *value).collect();
    let median_ratio = median(ratio_values.clone());
    ratio_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut by_ratio = ratios.clone();
    by_ratio.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let worst: Vec<Json> = by_ratio
        .iter()
        .take(5)
        .map(|(name, value, mode9, zstd)| {
            json!({"name": name, "mode9_over_zstd19": value, "mode9_bytes": mode9, "zstd19_bytes": zstd})
        })
        .collect();
    let best: Vec<Json> = by_ratio
        .iter()
        .rev()
        .take(5)
        .map(|(name, value, mode9, zstd)| {
            json!({"name": name, "mode9_over_zstd19": value, "mode9_bytes": mode9, "zstd19_bytes": zstd})
        })
        .collect();
    let mut by_excess = excess.clone();
    by_excess.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let worst_absolute: Vec<Json> = by_excess
        .iter()
        .take(5)
        .map(|(name, excess, mode9, zstd, value)| {
            json!({
                "name": name,
                "excess_bytes": excess,
                "mode9_bytes": mode9,
                "zstd19_bytes": zstd,
                "mode9_over_zstd19": value,
            })
        })
        .collect();

    let file_dtype = dataset
        .tensors
        .first()
        .map(|tensor| tensor.dtype.clone())
        .unwrap_or_default();
    let dtypes_seen: BTreeMap<String, u64> = {
        let mut map = BTreeMap::new();
        for tensor in &dataset.tensors {
            *map.entry(tensor.dtype.clone()).or_insert(0) += 1;
        }
        map
    };

    let doc = json!({
        "schema": "weftcodec.real.float_tensor.tensors.v1",
        "label": label,
        "file": file.display().to_string(),
        "file_bytes": dataset.file_len,
        "file_blake3": file_blake3(&file)?,
        "dtype": file_dtype,
        "dtypes_seen": dtypes_seen,
        "tensor_count": dataset.tensors.len(),
        "tensor_data_bytes": totals.raw_bytes,
        "elements": totals.elements,
        "zstd": {
            "binary": "zstd CLI",
            "version": zstd_version,
            "level19_args": "-19 -T1 --no-progress -q -f",
            "level3_args": "-3 -T1 --no-progress -q -f",
        },
        "limits": {
            "max_decoded_bytes": limits.max_decoded_bytes,
            "max_frame_bytes": limits.max_frame_bytes,
            "max_tensor_elements": limits.max_tensor_elements,
        },
        "decode_reps": REPS,
        "method": "per-tensor FLOAT_TENSOR mode 9 frame bytes vs per-tensor zstd CLI frame bytes; each decode median is over 5 reps; totals are sums over every tensor in the checkpoint",
        "totals": {
            "raw_bytes": totals.raw_bytes,
            "mode9_bytes": totals.mode9_bytes,
            "zstd19_bytes": totals.zstd19_bytes,
            "zstd3_bytes": totals.zstd3_bytes,
            "mode9_over_zstd19": ratio(totals.mode9_bytes, totals.zstd19_bytes),
            "zstd19_over_raw": ratio(totals.zstd19_bytes, totals.raw_bytes),
            "mode9_over_raw": ratio(totals.mode9_bytes, totals.raw_bytes),
            "zstd3_over_raw": ratio(totals.zstd3_bytes, totals.raw_bytes),
            "mode9_frames_not_larger_than_zstd19": totals.mode9_wins,
            "mode9_frames_not_larger_than_raw": totals.mode9_not_larger_than_raw,
            "median_tensor_mode9_over_zstd19": median_ratio,
            "mode9_encode_ms_sum": totals.mode9_encode_ms,
            "zstd19_encode_ms_sum": totals.zstd19_encode_ms,
            "zstd3_encode_ms_sum": totals.zstd3_encode_ms,
            "mode9_decode_ms_sum_of_medians": totals.mode9_decode_ms_sum,
            "zstd19_decode_ms_sum_of_medians": totals.zstd19_decode_ms_sum,
            "mode9_bytes_read": totals.mode9_bytes,
            "zstd19_bytes_read": totals.zstd19_bytes,
            "worst5_mode9_over_zstd19": worst,
            "worst5_mode9_absolute_excess_bytes": worst_absolute,
            "best5_mode9_over_zstd19": best,
        },
        "per_tensor": rows,
    });
    write_json(&out, &doc)
}

#[derive(Default)]
struct Totals {
    raw_bytes: u64,
    elements: u64,
    mode9_bytes: u64,
    zstd19_bytes: u64,
    zstd3_bytes: u64,
    mode9_encode_ms: f64,
    zstd19_encode_ms: f64,
    zstd3_encode_ms: f64,
    mode9_decode_ms_sum: f64,
    zstd19_decode_ms_sum: f64,
    mode9_wins: u64,
    mode9_not_larger_than_raw: u64,
}

// ---------------------------------------------------------------------------
// delta phase
// ---------------------------------------------------------------------------

fn delta(args: &[String]) -> Result<(), String> {
    let base_file = PathBuf::from(required(args, "--base-file")?);
    let target_file = PathBuf::from(required(args, "--target-file")?);
    let target_json = PathBuf::from(required(args, "--target-json")?);
    let label = required(args, "--label")?;
    let out = PathBuf::from(required(args, "--out")?);
    fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))
        .map_err(|err| format!("{}: {err}", out.display()))?;

    let limits = Limits::default();
    let base = parse_safetensors(&base_file)?;
    let target = parse_safetensors(&target_file)?;
    let target_rows = read_target_rows(&target_json)?;

    let base_names: BTreeMap<&str, &TensorEntry> =
        base.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let target_names: BTreeMap<&str, &TensorEntry> = target
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    if base_names.keys().ne(target_names.keys()) {
        return Err(format!(
            "{} and {} do not have identical tensor names",
            base_file.display(),
            target_file.display()
        ));
    }
    for (name, target_tensor) in &target_names {
        let base_tensor = base_names[name];
        if base_tensor.shape != target_tensor.shape || base_tensor.dtype != target_tensor.dtype {
            return Err(format!(
                "{name}: base {} {:?} vs target {} {:?}",
                base_tensor.dtype, base_tensor.shape, target_tensor.dtype, target_tensor.shape
            ));
        }
    }

    let mut rows = Vec::with_capacity(target.tensors.len());
    let mut total = DeltaTotals::default();
    let mut ratios: Vec<(String, f64)> = Vec::new();
    let mut excess: Vec<(String, f64, u64, u64, f64)> = Vec::new();
    let mut base_mode9_total = 0u64;

    for target_tensor in &target.tensors {
        let base_tensor = base_names[target_tensor.name.as_str()];
        let base_bytes = read_tensor_bytes(&base, base_tensor)?;
        let target_bytes = read_tensor_bytes(&target, target_tensor)?;

        let base_frame = modes::encode_float_tensor(
            &base_bytes,
            base_tensor.float_dtype,
            &base_tensor.shape,
            &limits,
        )
        .map_err(|err| format!("base mode 9 encode {}: {err}", base_tensor.name))?;
        base_mode9_total += base_frame.len() as u64;

        let encode_started = Instant::now();
        let frame = encode_weight_delta(
            &target_bytes,
            target_tensor.weight_dtype,
            &target_tensor.shape,
            &base_frame,
            &limits,
        )
        .map_err(|err| format!("mode 11 encode {}: {err}", target_tensor.name))?;
        let encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;

        let (decode_ms, decoded) = timed_decode_weight_delta(&frame, &base_frame, &limits)?;
        if decoded != target_bytes {
            return Err(format!(
                "{}: WEIGHT_DELTA round-trip is not byte-exact",
                target_tensor.name
            ));
        }

        let changed = changed_words(&base_bytes, &target_bytes, target_tensor.weight_dtype);
        let zstd19_bytes = target_rows
            .get(&target_tensor.name)
            .map(|row| row.zstd19_bytes)
            .ok_or_else(|| {
                format!(
                    "{}: missing zstd -19 baseline in {}",
                    target_tensor.name,
                    target_json.display()
                )
            })?;
        let zstd19_decode_ms = target_rows
            .get(&target_tensor.name)
            .map(|row| row.zstd19_decode_ms)
            .ok_or_else(|| {
                format!(
                    "{}: missing zstd -19 decode median in {}",
                    target_tensor.name,
                    target_json.display()
                )
            })?;

        total.raw_bytes += target_tensor.byte_len();
        total.elements += target_tensor.elements();
        total.delta_bytes += frame.len() as u64;
        total.zstd19_bytes += zstd19_bytes;
        total.delta_encode_ms += encode_ms;
        total.delta_decode_ms_sum += decode_ms;
        total.zstd19_decode_ms_sum += zstd19_decode_ms;
        total.changed_words += changed;
        if frame.len() as u64 <= zstd19_bytes {
            total.delta_wins += 1;
        }
        ratios.push((
            target_tensor.name.clone(),
            ratio(frame.len() as u64, zstd19_bytes),
        ));
        if frame.len() as u64 > zstd19_bytes {
            excess.push((
                target_tensor.name.clone(),
                (frame.len() as u64 - zstd19_bytes) as f64,
                frame.len() as u64,
                zstd19_bytes,
                ratio(frame.len() as u64, zstd19_bytes),
            ));
        }

        rows.push(json!({
            "name": target_tensor.name,
            "dtype": target_tensor.dtype,
            "shape": target_tensor.shape,
            "elements": target_tensor.elements(),
            "bytes": target_tensor.byte_len(),
            "changed_words": changed,
            "changed_fraction": ratio(changed, target_tensor.elements()),
            "base_mode9_frame_bytes": base_frame.len(),
            "delta_bytes": frame.len(),
            "delta_ratio_of_target_raw": ratio(frame.len() as u64, target_tensor.byte_len()),
            "delta_over_zstd19": ratio(frame.len() as u64, zstd19_bytes),
            "zstd19_bytes": zstd19_bytes,
            "delta_encode_ms": encode_ms,
            "delta_decode_ms_median": decode_ms,
            "zstd19_decode_ms_median": zstd19_decode_ms,
            "delta_roundtrip_exact": true,
            "bytes_read_to_restore": frame.len() as u64 + base_frame.len() as u64,
        }));

        drop(base_frame);
        drop(base_bytes);
        drop(target_bytes);
    }

    let ratio_values: Vec<f64> = ratios.iter().map(|(_, value)| *value).collect();
    let median_ratio = median(ratio_values);
    let mut by_ratio = ratios;
    by_ratio.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let worst: Vec<Json> = by_ratio
        .iter()
        .take(5)
        .map(|(name, value)| json!({"name": name, "delta_over_zstd19": value}))
        .collect();
    let best: Vec<Json> = by_ratio
        .iter()
        .rev()
        .take(5)
        .map(|(name, value)| json!({"name": name, "delta_over_zstd19": value}))
        .collect();
    let mut by_excess = excess;
    by_excess.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let worst_absolute: Vec<Json> = by_excess
        .iter()
        .take(5)
        .map(|(name, excess, delta, zstd, value)| {
            json!({
                "name": name,
                "excess_bytes": excess,
                "delta_bytes": delta,
                "zstd19_bytes": zstd,
                "delta_over_zstd19": value,
            })
        })
        .collect();

    let doc = json!({
        "schema": "weftcodec.real.weight_delta.pair.v1",
        "label": label,
        "base_file": base_file.display().to_string(),
        "base_file_bytes": base.file_len,
        "base_file_blake3": file_blake3(&base_file)?,
        "target_file": target_file.display().to_string(),
        "target_file_bytes": target.file_len,
        "target_file_blake3": file_blake3(&target_file)?,
        "tensor_count": target.tensors.len(),
        "dtype": target.tensors.first().map(|t| t.dtype.clone()).unwrap_or_default(),
        "method": "mode 11 WEIGHT_DELTA against the corresponding base tensor's mode 9 frame; zstd -19 bytes and decode medians per tensor come from the target checkpoint's tensors phase",
        "decode_reps": REPS,
        "totals": {
            "target_tensor_bytes": total.raw_bytes,
            "base_mode9_frame_bytes": base_mode9_total,
            "delta_bytes": total.delta_bytes,
            "zstd19_bytes": total.zstd19_bytes,
            "delta_over_zstd19": ratio(total.delta_bytes, total.zstd19_bytes),
            "delta_over_target_raw": ratio(total.delta_bytes, total.raw_bytes),
            "delta_frames_not_larger_than_zstd19": total.delta_wins,
            "median_tensor_delta_over_zstd19": median_ratio,
            "changed_words": total.changed_words,
            "elements": total.elements,
            "changed_fraction": ratio(total.changed_words, total.elements),
            "delta_encode_ms_sum": total.delta_encode_ms,
            "delta_decode_ms_sum_of_medians": total.delta_decode_ms_sum,
            "zstd19_decode_ms_sum_of_medians": total.zstd19_decode_ms_sum,
            "delta_bytes_read_to_restore": total.delta_bytes + base_mode9_total,
            "zstd19_bytes_read": total.zstd19_bytes,
            "worst5_delta_over_zstd19": worst,
            "worst5_delta_absolute_excess_bytes": worst_absolute,
            "best5_delta_over_zstd19": best,
        },
        "per_tensor": rows,
    });
    write_json(&out, &doc)
}

#[derive(Default)]
struct DeltaTotals {
    raw_bytes: u64,
    elements: u64,
    delta_bytes: u64,
    zstd19_bytes: u64,
    delta_encode_ms: f64,
    delta_decode_ms_sum: f64,
    zstd19_decode_ms_sum: f64,
    changed_words: u64,
    delta_wins: u64,
}

fn changed_words(base: &[u8], target: &[u8], dtype: WeightDtype) -> u64 {
    match dtype.word_bits() {
        16 => base
            .chunks_exact(2)
            .zip(target.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count() as u64,
        32 => base
            .chunks_exact(4)
            .zip(target.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count() as u64,
        _ => base
            .iter()
            .zip(target.iter())
            .filter(|(a, b)| a != b)
            .count() as u64,
    }
}

// ---------------------------------------------------------------------------
// target checkpoint rows
// ---------------------------------------------------------------------------

struct TargetRow {
    zstd19_bytes: u64,
    zstd19_decode_ms: f64,
}

fn read_target_rows(path: &Path) -> Result<BTreeMap<String, TargetRow>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let doc: Json =
        serde_json::from_str(&text).map_err(|err| format!("{}: {err}", path.display()))?;
    let rows = doc
        .get("per_tensor")
        .and_then(Json::as_array)
        .ok_or_else(|| format!("{}: no per_tensor array", path.display()))?;
    let mut map = BTreeMap::new();
    for row in rows {
        let name = row
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| format!("{}: tensor row without name", path.display()))?;
        let zstd19_bytes = row
            .get("zstd19_bytes")
            .and_then(Json::as_u64)
            .ok_or_else(|| format!("{}/{name}: no zstd19_bytes", path.display()))?;
        let zstd19_decode_ms = row
            .get("zstd19_decode_ms_median")
            .and_then(Json::as_f64)
            .ok_or_else(|| format!("{}/{}: no zstd19_decode_ms_median", path.display(), name))?;
        map.insert(
            name.to_owned(),
            TargetRow {
                zstd19_bytes,
                zstd19_decode_ms,
            },
        );
    }
    Ok(map)
}

fn write_json(path: &Path, doc: &Json) -> Result<(), String> {
    let text = serde_json::to_string_pretty(doc).map_err(|err| err.to_string())?;
    fs::write(path, text).map_err(|err| format!("{}: {err}", path.display()))
}
