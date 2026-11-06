//! Real public weights: mode 11 `WEIGHT_DELTA` vs mode 16 `SPARSE_DELTA` on
//! the same tensor pairs.
//!
//! This harness parses safetensors headers directly (8-byte little-endian
//! length followed by JSON; no transformers, no model framework) and feeds the
//! exact tensor bytes to the *shipped* codec entry points:
//!
//! * `weight_delta::encode_weight_delta` / `decode_with_base_frame` (mode 11);
//! * `sparse_delta::encode_sparse_delta` / `decode_with_base_frame` (mode 16).
//!
//! Both deltas use the same base: a mode-6 `TENSOR` frame over the base
//! tensor's exact bytes. Per-tensor `zstd -19` bytes and decode medians are
//! *read from an existing checkpoint receipt* (`--target-json`) instead of
//! being re-measured, so the delta numbers are compared against the already
//! recorded baseline rather than a fresh one.
//!
//! Usage:
//!
//! ```sh
//! real_sparse_delta_bench pair --base-file older.safetensors \
//!     --target-file newer.safetensors --target-json tensors.json \
//!     --label pythia-step0-step1 --kind consecutive-training-checkpoint \
//!     --out pair.json
//! ```

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value as Json};

use weftcodec::dependency::MapResolver;
use weftcodec::frame::Frame;
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::sparse_delta::{decode_with_base_frame as decode_sparse, encode_sparse_delta};
use weftcodec::weight_delta::{decode_with_base_frame as decode_weight, encode_weight_delta};
use weftcodec::WeightDtype;

const DECODE_REPS: usize = 5;

fn main() {
    if let Err(err) = run() {
        eprintln!("real_sparse_delta_bench: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let phase = args.first().cloned().unwrap_or_default();
    match phase.as_str() {
        "pair" => pair(&args[1..]),
        other => Err(format!("unknown phase `{other}` (expected `pair`)")),
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
// safetensors parsing (direct, no framework)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TensorEntry {
    name: String,
    /// Canonical WEFT-TENSOR text name (`bf16`, `fp16`, `fp32`).
    dtype: String,
    /// Mode-6 dtype name for the base frame.
    tensor_dtype: &'static str,
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

fn dtypes(text: &str) -> Result<(String, &'static str, WeightDtype, u64), String> {
    match text {
        "BF16" => Ok(("bf16".to_owned(), "bf16", WeightDtype::Bf16, 2)),
        "F16" => Ok(("fp16".to_owned(), "f16", WeightDtype::Fp16, 2)),
        "F32" => Ok(("fp32".to_owned(), "f32", WeightDtype::Fp32, 4)),
        other => Err(format!(
            "safetensors dtype `{other}` has no WEIGHT_DELTA/SPARSE_DELTA v1 mapping \
             (this harness measures bf16, fp16, and fp32)"
        )),
    }
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
        let (dtype, tensor_dtype, weight_dtype, element_bytes) = dtypes(raw_dtype)?;
        let elements: u64 = shape.iter().product();
        let expected = elements
            .checked_mul(element_bytes)
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
            tensor_dtype,
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
// per-mode helpers
// ---------------------------------------------------------------------------

fn run_count(frame_bytes: &[u8], limits: &Limits) -> Result<u32, String> {
    let parsed = Frame::parse(frame_bytes, limits).map_err(|err| err.to_string())?;
    let body = parsed.body;
    if body.len() < 4 {
        return Err("frame body is shorter than its metadata length".to_owned());
    }
    let meta_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let payload = &body[4 + meta_len..];
    if payload.len() < 4 {
        return Err("SPARSE_DELTA payload is shorter than its run_count".to_owned());
    }
    Ok(u32::from_le_bytes([
        payload[0], payload[1], payload[2], payload[3],
    ]))
}

fn timed_decode(
    frame_bytes: &[u8],
    base_frame: &[u8],
    limits: &Limits,
    sparse: bool,
) -> Result<(f64, Vec<u8>), String> {
    let resolver = MapResolver::new();
    let mut samples = Vec::with_capacity(DECODE_REPS);
    let mut decoded = Vec::new();
    for _ in 0..DECODE_REPS {
        let started = Instant::now();
        let parsed = Frame::parse(frame_bytes, limits).map_err(|err| err.to_string())?;
        decoded = if sparse {
            decode_sparse(&parsed, base_frame, &resolver, limits).map_err(|err| err.to_string())?
        } else {
            decode_weight(&parsed, base_frame, &resolver, limits).map_err(|err| err.to_string())?
        };
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    Ok((median(samples), decoded))
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
// baseline rows from an existing receipt
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

// ---------------------------------------------------------------------------
// pair phase
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Totals {
    target_bytes: u64,
    elements: u64,
    changed_words: u64,
    run_count: u64,
    mode11_bytes: u64,
    mode16_bytes: u64,
    zstd19_bytes: u64,
    mode11_encode_ms: f64,
    mode16_encode_ms: f64,
    mode11_decode_ms_sum: f64,
    mode16_decode_ms_sum: f64,
    zstd19_decode_ms_sum: f64,
    mode16_wins: u64,
    mode11_wins: u64,
    equal: u64,
}

fn pair(args: &[String]) -> Result<(), String> {
    let base_file = PathBuf::from(required(args, "--base-file")?);
    let target_file = PathBuf::from(required(args, "--target-file")?);
    let target_json = PathBuf::from(required(args, "--target-json")?);
    let label = required(args, "--label")?;
    let kind = value_of(args, "--kind").unwrap_or_else(|| "unspecified".to_owned());
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
    let mut total = Totals::default();
    let mut losses: Vec<(String, i64, u64, u64)> = Vec::new();
    for target_tensor in &target.tensors {
        let base_tensor = base_names[target_tensor.name.as_str()];
        let base_bytes = read_tensor_bytes(&base, base_tensor)?;
        let target_bytes = read_tensor_bytes(&target, target_tensor)?;
        let base_frame = modes::encode_tensor(
            &base_bytes,
            mode6_dtype(target_tensor.tensor_dtype)?,
            &target_tensor.shape,
            &limits,
        )
        .map_err(|err| format!("base mode-6 encode {}: {err}", target_tensor.name))?;

        let started = Instant::now();
        let weight_frame = encode_weight_delta(
            &target_bytes,
            target_tensor.weight_dtype,
            &target_tensor.shape,
            &base_frame,
            &limits,
        )
        .map_err(|err| format!("mode 11 encode {}: {err}", target_tensor.name))?;
        let weight_encode_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let sparse_frame = encode_sparse_delta(
            &target_bytes,
            target_tensor.weight_dtype,
            &target_tensor.shape,
            &base_frame,
            &limits,
        )
        .map_err(|err| format!("mode 16 encode {}: {err}", target_tensor.name))?;
        let sparse_encode_ms = started.elapsed().as_secs_f64() * 1000.0;

        let (weight_decode_ms, weight_decoded) =
            timed_decode(&weight_frame, &base_frame, &limits, false)?;
        let (sparse_decode_ms, sparse_decoded) =
            timed_decode(&sparse_frame, &base_frame, &limits, true)?;
        if weight_decoded != target_bytes {
            return Err(format!(
                "{}: mode 11 round-trip is not byte-exact",
                target_tensor.name
            ));
        }
        if sparse_decoded != target_bytes {
            return Err(format!(
                "{}: mode 16 round-trip is not byte-exact",
                target_tensor.name
            ));
        }
        if sparse_decoded != weight_decoded {
            return Err(format!(
                "{}: modes 11 and 16 reconstruct different bytes",
                target_tensor.name
            ));
        }
        let runs = run_count(&sparse_frame, &limits)
            .map_err(|err| format!("{}: {err}", target_tensor.name))?;

        let changed = changed_words(&base_bytes, &target_bytes, target_tensor.weight_dtype);
        let target_row = target_rows.get(&target_tensor.name).ok_or_else(|| {
            format!(
                "{}: missing zstd -19 baseline in {}",
                target_tensor.name,
                target_json.display()
            )
        })?;

        total.target_bytes += target_tensor.byte_len();
        total.elements += target_tensor.elements();
        total.changed_words += changed;
        total.run_count += u64::from(runs);
        total.mode11_bytes += weight_frame.len() as u64;
        total.mode16_bytes += sparse_frame.len() as u64;
        total.zstd19_bytes += target_row.zstd19_bytes;
        total.mode11_encode_ms += weight_encode_ms;
        total.mode16_encode_ms += sparse_encode_ms;
        total.mode11_decode_ms_sum += weight_decode_ms;
        total.mode16_decode_ms_sum += sparse_decode_ms;
        total.zstd19_decode_ms_sum += target_row.zstd19_decode_ms;
        match (sparse_frame.len() as u64).cmp(&(weight_frame.len() as u64)) {
            std::cmp::Ordering::Less => total.mode16_wins += 1,
            std::cmp::Ordering::Greater => total.mode11_wins += 1,
            std::cmp::Ordering::Equal => total.equal += 1,
        }
        if sparse_frame.len() as u64 > weight_frame.len() as u64 {
            losses.push((
                target_tensor.name.clone(),
                sparse_frame.len() as i64 - weight_frame.len() as i64,
                sparse_frame.len() as u64,
                weight_frame.len() as u64,
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
            "sparse_run_count": runs,
            "base_mode6_frame_bytes": base_frame.len(),
            "mode11_bytes": weight_frame.len(),
            "mode11_ratio_of_target_raw": ratio(weight_frame.len() as u64, target_tensor.byte_len()),
            "mode11_encode_ms": weight_encode_ms,
            "mode11_decode_ms_median": weight_decode_ms,
            "mode11_roundtrip_exact": true,
            "mode16_bytes": sparse_frame.len(),
            "mode16_ratio_of_target_raw": ratio(sparse_frame.len() as u64, target_tensor.byte_len()),
            "mode16_encode_ms": sparse_encode_ms,
            "mode16_decode_ms_median": sparse_decode_ms,
            "mode16_roundtrip_exact": true,
            "mode16_over_mode11": ratio(sparse_frame.len() as u64, weight_frame.len() as u64),
            "mode16_over_zstd19": ratio(sparse_frame.len() as u64, target_row.zstd19_bytes),
            "mode11_over_zstd19": ratio(weight_frame.len() as u64, target_row.zstd19_bytes),
            "zstd19_bytes": target_row.zstd19_bytes,
            "zstd19_decode_ms_median": target_row.zstd19_decode_ms,
            "bytes_read_to_restore_mode16": sparse_frame.len() as u64 + base_frame.len() as u64,
            "bytes_read_to_restore_mode11": weight_frame.len() as u64 + base_frame.len() as u64,
        }));

        drop(weight_frame);
        drop(sparse_frame);
        drop(weight_decoded);
        drop(sparse_decoded);
        drop(base_frame);
        drop(base_bytes);
        drop(target_bytes);
    }

    losses.sort_by_key(|(_, diff, _, _)| -diff);
    let losses_json: Vec<Json> = losses
        .iter()
        .take(10)
        .map(|(name, diff, sparse, weight)| {
            json!({
                "name": name,
                "mode16_excess_bytes": diff,
                "mode16_bytes": sparse,
                "mode11_bytes": weight,
            })
        })
        .collect();

    let doc = json!({
        "schema": "weftcodec.real.sparse_delta.pair.v1",
        "label": label,
        "kind": kind,
        "base_file": base_file.display().to_string(),
        "base_file_bytes": base.file_len,
        "base_file_blake3": file_blake3(&base_file)?,
        "target_file": target_file.display().to_string(),
        "target_file_bytes": target.file_len,
        "target_file_blake3": file_blake3(&target_file)?,
        "tensor_count": target.tensors.len(),
        "dtype": target.tensors.first().map(|t| t.dtype.clone()).unwrap_or_default(),
        "method": "per-tensor mode 11 WEIGHT_DELTA and mode 16 SPARSE_DELTA against the same mode-6 TENSOR base frame; zstd -19 bytes and decode medians are read from the target checkpoint's existing tensors-phase receipt, not re-measured here",
        "decode_reps": DECODE_REPS,
        "limits": {
            "max_decoded_bytes": limits.max_decoded_bytes,
            "max_frame_bytes": limits.max_frame_bytes,
            "max_sparse_runs": limits.max_sparse_runs,
        },
        "totals": {
            "target_tensor_bytes": total.target_bytes,
            "elements": total.elements,
            "changed_words": total.changed_words,
            "changed_fraction": ratio(total.changed_words, total.elements),
            "sparse_run_count": total.run_count,
            "mode11_bytes": total.mode11_bytes,
            "mode16_bytes": total.mode16_bytes,
            "mode16_over_mode11": ratio(total.mode16_bytes, total.mode11_bytes),
            "zstd19_bytes": total.zstd19_bytes,
            "mode11_over_zstd19": ratio(total.mode11_bytes, total.zstd19_bytes),
            "mode16_over_zstd19": ratio(total.mode16_bytes, total.zstd19_bytes),
            "tensors_mode16_smaller": total.mode16_wins,
            "tensors_mode11_smaller": total.mode11_wins,
            "tensors_equal": total.equal,
            "mode11_encode_ms_sum": total.mode11_encode_ms,
            "mode16_encode_ms_sum": total.mode16_encode_ms,
            "mode11_decode_ms_sum_of_medians": total.mode11_decode_ms_sum,
            "mode16_decode_ms_sum_of_medians": total.mode16_decode_ms_sum,
            "zstd19_decode_ms_sum_of_medians": total.zstd19_decode_ms_sum,
            "mode16_worst10_losses": losses_json,
        },
        "per_tensor": rows,
    });
    write_json(&out, &doc)
}

fn mode6_dtype(name: &str) -> Result<modes::TensorDtype, String> {
    match name {
        "bf16" => Ok(modes::TensorDtype::Bf16),
        "f16" => Ok(modes::TensorDtype::F16),
        "f32" => Ok(modes::TensorDtype::F32),
        other => Err(format!("no mode-6 dtype mapping for `{other}`")),
    }
}

fn write_json(path: &Path, doc: &Json) -> Result<(), String> {
    let text = serde_json::to_string_pretty(doc).map_err(|err| err.to_string())?;
    fs::write(path, text).map_err(|err| format!("{}: {err}", path.display()))
}
