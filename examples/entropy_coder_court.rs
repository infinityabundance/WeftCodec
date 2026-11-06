//! Experimental entropy-coder court.
//!
//! **This harness measures experimental coders. It writes no WeftCodec frame
//! and defines no wire mode.** Every plane input is extracted through the
//! shipped modes 9, 11, 12, and 15 encoders; the coders under test live in
//! `weftcodec::experimental::entropy`.
//!
//! Usage (inside the bench court, repository at `/work`):
//!
//! ```sh
//! cargo run --release --example entropy_coder_court -- \
//!   --base-qwen /work/.docker-cache/real-corpus/Qwen2.5-0.5B.safetensors \
//!   --instruct-qwen /work/.docker-cache/real-corpus/Qwen2.5-0.5B-Instruct.safetensors \
//!   --pythia-step0 /work/.docker-cache/real-corpus/pythia-70m-step0.safetensors \
//!   --pythia-step1 /work/.docker-cache/real-corpus/pythia-70m-step1.safetensors \
//!   --out /work/.docker-cache/entropy-court/rows.ndjson
//! ```
//!
//! The harness prints one JSON object per row and a final summary object. It
//! never writes outside `--out` and never mutates the store.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Value as Json};

use weftcodec::experimental::entropy::huffman::Huffman;
use weftcodec::experimental::entropy::planes::{
    float_planes, kv_page_planes, tiled_planes, weight_delta_planes,
};
use weftcodec::experimental::entropy::rans::Rans;
use weftcodec::experimental::entropy::raw::Raw;
use weftcodec::experimental::entropy::tunstall::{Dictionary, Tunstall};
use weftcodec::experimental::entropy::zstd::Zstd;
use weftcodec::experimental::entropy::Coder;
use weftcodec::kv_cache::KvDtype;
use weftcodec::limits::Limits;
use weftcodec::modes::{self, FloatDtype, TensorDtype};
use weftcodec::weight_delta::WeightDtype;

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Args {
    base_qwen: PathBuf,
    instruct_qwen: PathBuf,
    pythia_step0: PathBuf,
    pythia_step1: PathBuf,
    out: PathBuf,
    skip_family: bool,
    quick: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut base_qwen = None;
    let mut instruct_qwen = None;
    let mut pythia_step0 = None;
    let mut pythia_step1 = None;
    let mut out = None;
    let mut skip_family = false;
    let mut quick = false;
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--base-qwen" => base_qwen = Some(PathBuf::from(value()?)),
            "--instruct-qwen" => instruct_qwen = Some(PathBuf::from(value()?)),
            "--pythia-step0" => pythia_step0 = Some(PathBuf::from(value()?)),
            "--pythia-step1" => pythia_step1 = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--skip-family" => skip_family = true,
            "--quick" => quick = true,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(Args {
        base_qwen: base_qwen.ok_or("missing --base-qwen")?,
        instruct_qwen: instruct_qwen.ok_or("missing --instruct-qwen")?,
        pythia_step0: pythia_step0.ok_or("missing --pythia-step0")?,
        pythia_step1: pythia_step1.ok_or("missing --pythia-step1")?,
        out: out.ok_or("missing --out")?,
        skip_family,
        quick,
    })
}

// ---------------------------------------------------------------------------
// Minimal safetensors reader (direct, no framework)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TensorInfo {
    dtype_name: String,
    tensor_dtype: TensorDtype,
    float_dtype: Option<FloatDtype>,
    weight_dtype: WeightDtype,
    shape: Vec<u64>,
    offset: u64,
    len: u64,
}

impl TensorInfo {
    fn elements(&self) -> Result<u64, String> {
        self.shape
            .iter()
            .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| format!("element count overflows for shape {:?}", self.shape))
    }

    fn element_size(&self) -> u64 {
        self.tensor_dtype.element_size()
    }

    fn row_bytes(&self) -> Result<u64, String> {
        if self.shape.len() != 2 {
            return Err(format!("shape {:?} is not rank 2", self.shape));
        }
        self.shape[1]
            .checked_mul(self.element_size())
            .ok_or_else(|| "row byte count overflows".to_owned())
    }
}

struct Safetensors {
    path: PathBuf,
    file: File,
    file_len: u64,
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
    unsupported: BTreeSet<String>,
}

fn map_dtype(name: &str) -> Option<(TensorDtype, Option<FloatDtype>, WeightDtype)> {
    match name {
        "BF16" => Some((TensorDtype::Bf16, Some(FloatDtype::Bf16), WeightDtype::Bf16)),
        "F16" => Some((TensorDtype::F16, Some(FloatDtype::F16), WeightDtype::Fp16)),
        "F32" => Some((TensorDtype::F32, Some(FloatDtype::F32), WeightDtype::Fp32)),
        _ => None,
    }
}

impl Safetensors {
    fn open(path: PathBuf) -> Result<Self, String> {
        let file = File::open(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let file_len = file
            .metadata()
            .map_err(|error| format!("{}: {error}", path.display()))?
            .len();
        let mut length_bytes = [0u8; 8];
        file.read_exact_at(&mut length_bytes, 0)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let header_len = u64::from_le_bytes(length_bytes);
        if header_len > 256 << 20 {
            return Err(format!(
                "{}: header length {header_len} exceeds the 256 MiB sanity bound",
                path.display()
            ));
        }
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| "header length overflows".to_owned())?;
        if data_start > file_len {
            return Err(format!(
                "{}: header extends past the end of the file",
                path.display()
            ));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact_at(&mut header, 8)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let value: Json = serde_json::from_slice(&header)
            .map_err(|error| format!("{}: header JSON: {error}", path.display()))?;
        let object = value
            .as_object()
            .ok_or_else(|| format!("{}: header is not a JSON object", path.display()))?;
        let mut tensors = BTreeMap::new();
        let mut unsupported = BTreeSet::new();
        for (name, entry) in object {
            if name == "__metadata__" {
                continue;
            }
            let dtype_name = entry
                .get("dtype")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("{}: tensor {name} has no dtype", path.display()))?;
            let shape: Vec<u64> = entry
                .get("shape")
                .and_then(Json::as_array)
                .ok_or_else(|| format!("{}: tensor {name} has no shape", path.display()))?
                .iter()
                .map(|value| {
                    value.as_u64().ok_or_else(|| {
                        format!("{}: tensor {name} has a non-u64 dimension", path.display())
                    })
                })
                .collect::<Result<_, _>>()?;
            let offsets: Vec<u64> = entry
                .get("data_offsets")
                .and_then(Json::as_array)
                .ok_or_else(|| format!("{}: tensor {name} has no data_offsets", path.display()))?
                .iter()
                .map(|value| {
                    value.as_u64().ok_or_else(|| {
                        format!(
                            "{}: tensor {name} has a non-u64 data offset",
                            path.display()
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            if offsets.len() != 2 || offsets[0] > offsets[1] {
                return Err(format!(
                    "{}: tensor {name} has invalid data_offsets {offsets:?}",
                    path.display()
                ));
            }
            if data_start + offsets[1] > file_len {
                return Err(format!(
                    "{}: tensor {name} extends past the end of the file",
                    path.display()
                ));
            }
            let Some((tensor_dtype, float_dtype, weight_dtype)) = map_dtype(dtype_name) else {
                unsupported.insert(name.clone());
                continue;
            };
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype_name: dtype_name.to_owned(),
                    tensor_dtype,
                    float_dtype,
                    weight_dtype,
                    shape,
                    offset: offsets[0],
                    len: offsets[1] - offsets[0],
                },
            );
        }
        Ok(Safetensors {
            path,
            file,
            file_len,
            data_start,
            tensors,
            unsupported,
        })
    }

    fn get(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("{}: no float tensor named {name}", self.path.display()))
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        let info = self.get(name)?;
        let mut bytes = vec![0u8; info.len as usize];
        self.file
            .read_exact_at(&mut bytes, self.data_start + info.offset)
            .map_err(|error| format!("{}: {name}: {error}", self.path.display()))?;
        Ok(bytes)
    }

    fn float_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Input classes and workload shapes
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ShapeCase {
    shape: &'static str,
    planes: Vec<Vec<u8>>,
    rows: u64,
    columns: u64,
    elements: u64,
}

struct InputClass {
    id: &'static str,
    description: String,
    tensor: String,
    dtype: String,
    shape: Vec<u64>,
    anchor_row: u64,
    cases: Vec<ShapeCase>,
    family_members: usize,
    family_excluded: usize,
    family_histogram: Option<[u64; 256]>,
}

fn planes_bytes(planes: &[Vec<u8>]) -> u64 {
    planes.iter().map(|plane| plane.len() as u64).sum()
}

fn codec_error<T>(result: weftcodec::error::Result<T>) -> Result<T, String> {
    result.map_err(|error| format!("{error}"))
}

fn float_cases(
    reader: &Safetensors,
    name: &str,
    float_dtype: FloatDtype,
) -> Result<(Vec<ShapeCase>, Vec<u64>, String), String> {
    let info = reader.get(name)?;
    let Some(info_float) = info.float_dtype else {
        return Err(format!("{name} is not a float tensor"));
    };
    if info_float != float_dtype {
        return Err(format!(
            "{name} is {} not {}",
            info.dtype_name,
            match float_dtype {
                FloatDtype::Bf16 => "bf16",
                FloatDtype::F16 => "fp16",
                FloatDtype::F32 => "fp32",
                FloatDtype::Fp8E4m3 => "fp8e4m3",
                FloatDtype::Fp8E5m2 => "fp8e5m2",
            },
        ));
    }
    if info.shape.len() != 2 {
        return Err(format!("{name} has rank {} not 2", info.shape.len()));
    }
    let rows = info.shape[0];
    let columns = info.shape[1];
    let row_bytes = info.row_bytes()?;
    let bytes = reader.read(name)?;
    if bytes.len() as u64 != info.len {
        return Err(format!("{name}: short read"));
    }
    let mut cases = Vec::new();
    let whole = codec_error(float_planes(&bytes, float_dtype, info.elements()?))?;
    cases.push(ShapeCase {
        shape: "whole",
        planes: whole,
        rows,
        columns,
        elements: info.elements()?,
    });
    let region_rows = 100u64.min(rows);
    let region_len = (region_rows * row_bytes) as usize;
    let region = codec_error(float_planes(
        &bytes[..region_len],
        float_dtype,
        region_rows
            .checked_mul(columns)
            .ok_or_else(|| "region element count overflows".to_owned())?,
    ))?;
    cases.push(ShapeCase {
        shape: if region_rows == 100 {
            "rows_100"
        } else {
            "rows_all"
        },
        planes: region,
        rows: region_rows,
        columns,
        elements: region_rows * columns,
    });
    let one_len = row_bytes as usize;
    let one = codec_error(float_planes(&bytes[..one_len], float_dtype, columns))?;
    cases.push(ShapeCase {
        shape: "row_1",
        planes: one,
        rows: 1,
        columns,
        elements: columns,
    });
    Ok((cases, info.shape.clone(), info.dtype_name.clone()))
}

fn first_differing_row(
    base: &[u8],
    target: &[u8],
    row_bytes: usize,
    rows: u64,
) -> Result<u64, String> {
    if base.len() != target.len() {
        return Err("base and target lengths differ".to_owned());
    }
    let row_bytes = row_bytes.max(1);
    for row in 0..rows as usize {
        let start = row
            .checked_mul(row_bytes)
            .ok_or_else(|| "row offset overflows".to_owned())?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| "row end overflows".to_owned())?;
        if base[start..end] != target[start..end] {
            return Ok(row as u64);
        }
    }
    Ok(0)
}

fn delta_cases(
    base: &Safetensors,
    target: &Safetensors,
    name: &str,
    limits: &Limits,
) -> Result<(Vec<ShapeCase>, Vec<u64>, String, u64), String> {
    let base_info = base.get(name)?;
    let target_info = target.get(name)?;
    if base_info.shape != target_info.shape || base_info.tensor_dtype != target_info.tensor_dtype {
        return Err(format!("{name}: base and target descriptors differ"));
    }
    if base_info.shape.len() != 2 {
        return Err(format!("{name}: rank {} not 2", base_info.shape.len()));
    }
    let rows = base_info.shape[0];
    let columns = base_info.shape[1];
    let row_bytes = base_info.row_bytes()?;
    let base_bytes = base.read(name)?;
    let target_bytes = target.read(name)?;
    let anchor_row = first_differing_row(&base_bytes, &target_bytes, row_bytes as usize, rows)?;
    let mut cases = Vec::new();

    let whole_base_frame = codec_error(modes::encode_tensor(
        &base_bytes,
        base_info.tensor_dtype,
        &base_info.shape,
        limits,
    ))?;
    let whole = codec_error(weight_delta_planes(
        &target_bytes,
        &whole_base_frame,
        base_info.weight_dtype,
        &base_info.shape,
        limits,
    ))?;
    drop(whole_base_frame);
    cases.push(ShapeCase {
        shape: "whole",
        planes: whole,
        rows,
        columns,
        elements: base_info.elements()?,
    });

    let region_rows = 100u64.min(rows);
    let region_start = if rows > region_rows {
        anchor_row.min(rows - region_rows)
    } else {
        0
    };
    let region_shape = vec![region_rows, columns];
    let start_byte = (region_start * row_bytes) as usize;
    let end_byte = ((region_start + region_rows) * row_bytes) as usize;
    let region_base_frame = codec_error(modes::encode_tensor(
        &base_bytes[start_byte..end_byte],
        base_info.tensor_dtype,
        &region_shape,
        limits,
    ))?;
    let region = codec_error(weight_delta_planes(
        &target_bytes[start_byte..end_byte],
        &region_base_frame,
        base_info.weight_dtype,
        &region_shape,
        limits,
    ))?;
    drop(region_base_frame);
    cases.push(ShapeCase {
        shape: if region_rows == 100 {
            "rows_100"
        } else {
            "rows_all"
        },
        planes: region,
        rows: region_rows,
        columns,
        elements: region_rows * columns,
    });

    let one_shape = vec![1u64, columns];
    let one_start = (anchor_row * row_bytes) as usize;
    let one_end = one_start + row_bytes as usize;
    let one_base_frame = codec_error(modes::encode_tensor(
        &base_bytes[one_start..one_end],
        base_info.tensor_dtype,
        &one_shape,
        limits,
    ))?;
    let one = codec_error(weight_delta_planes(
        &target_bytes[one_start..one_end],
        &one_base_frame,
        base_info.weight_dtype,
        &one_shape,
        limits,
    ))?;
    drop(one_base_frame);
    cases.push(ShapeCase {
        shape: "row_1",
        planes: one,
        rows: 1,
        columns,
        elements: columns,
    });

    Ok((
        cases,
        base_info.shape.clone(),
        base_info.dtype_name.clone(),
        anchor_row,
    ))
}

fn largest_divisor(value: u64, maximum: u64) -> u64 {
    let mut candidate = maximum.min(value);
    while candidate > 1 && value % candidate != 0 {
        candidate -= 1;
    }
    candidate.max(1)
}

fn tiled_cases(
    reader: &Safetensors,
    name: &str,
    tile_shape: &[u64],
    limits: &Limits,
) -> Result<(Vec<ShapeCase>, Vec<u64>, String), String> {
    let info = reader.get(name)?;
    if info.shape.len() != 2 || tile_shape.len() != 2 {
        return Err(format!("{name}: tiling requires rank 2"));
    }
    if info.shape[0] % tile_shape[0] != 0 || info.shape[1] % tile_shape[1] != 0 {
        return Err(format!(
            "{name}: shape {:?} is not divisible by tile {:?}",
            info.shape, tile_shape
        ));
    }
    let rows = info.shape[0];
    let columns = info.shape[1];
    let row_bytes = info.row_bytes()?;
    let bytes = reader.read(name)?;
    let mut cases = Vec::new();
    let whole = codec_error(tiled_planes(
        &bytes,
        info.tensor_dtype,
        &info.shape,
        tile_shape,
        limits,
    ))?;
    cases.push(ShapeCase {
        shape: "whole",
        planes: whole,
        rows,
        columns,
        elements: info.elements()?,
    });

    let region_rows = 100u64.min(rows);
    let region_shape = vec![region_rows, columns];
    let region_tile_rows = largest_divisor(region_rows, tile_shape[0]);
    let region_tile_columns = largest_divisor(columns, tile_shape[1]);
    let region_tile = vec![region_tile_rows, region_tile_columns];
    let region_len = (region_rows * row_bytes) as usize;
    let region = codec_error(tiled_planes(
        &bytes[..region_len],
        info.tensor_dtype,
        &region_shape,
        &region_tile,
        limits,
    ))?;
    cases.push(ShapeCase {
        shape: if region_rows == 100 {
            "rows_100"
        } else {
            "rows_all"
        },
        planes: region,
        rows: region_rows,
        columns,
        elements: region_rows * columns,
    });

    let one_shape = vec![1u64, columns];
    let one_tile = vec![1u64, largest_divisor(columns, tile_shape[1])];
    let one_len = row_bytes as usize;
    let one = codec_error(tiled_planes(
        &bytes[..one_len],
        info.tensor_dtype,
        &one_shape,
        &one_tile,
        limits,
    ))?;
    cases.push(ShapeCase {
        shape: "row_1",
        planes: one,
        rows: 1,
        columns,
        elements: columns,
    });
    Ok((cases, info.shape.clone(), info.dtype_name.clone()))
}

fn kv_cases(
    reader: &Safetensors,
    name: &str,
    limits: &Limits,
) -> Result<(Vec<ShapeCase>, Vec<u64>, String), String> {
    let info = reader.get(name)?;
    if info.shape.len() != 2 {
        return Err(format!("{name}: rank {} not 2", info.shape.len()));
    }
    const HEADS: u32 = 16;
    const DIM: u32 = 64;
    const PAGE_ROWS: u64 = 512;
    let row_bytes = u64::from(HEADS) * u64::from(DIM) * 2;
    let page_bytes = PAGE_ROWS * row_bytes;
    if info.len < page_bytes {
        return Err(format!(
            "{name}: tensor is {} bytes, smaller than one {page_bytes}-byte KV page",
            info.len
        ));
    }
    let bytes = reader.read(name)?;
    let mut cases = Vec::new();
    let whole = codec_error(kv_page_planes(
        &bytes[..page_bytes as usize],
        PAGE_ROWS,
        HEADS,
        DIM,
        KvDtype::Bf16,
        limits,
    ))?;
    cases.push(ShapeCase {
        shape: "kv_page",
        planes: whole,
        rows: PAGE_ROWS,
        columns: u64::from(HEADS) * u64::from(DIM),
        elements: PAGE_ROWS * u64::from(HEADS) * u64::from(DIM),
    });
    let region_rows = 100u64;
    let region_bytes = region_rows * row_bytes;
    let region = codec_error(kv_page_planes(
        &bytes[..region_bytes as usize],
        region_rows,
        HEADS,
        DIM,
        KvDtype::Bf16,
        limits,
    ))?;
    cases.push(ShapeCase {
        shape: "rows_100",
        planes: region,
        rows: region_rows,
        columns: u64::from(HEADS) * u64::from(DIM),
        elements: region_rows * u64::from(HEADS) * u64::from(DIM),
    });
    let one = codec_error(kv_page_planes(
        &bytes[..row_bytes as usize],
        1,
        HEADS,
        DIM,
        KvDtype::Bf16,
        limits,
    ))?;
    cases.push(ShapeCase {
        shape: "row_1",
        planes: one,
        rows: 1,
        columns: u64::from(HEADS) * u64::from(DIM),
        elements: u64::from(HEADS) * u64::from(DIM),
    });
    Ok((
        cases,
        vec![PAGE_ROWS, u64::from(HEADS) * u64::from(DIM)],
        "bf16".to_owned(),
    ))
}

// ---------------------------------------------------------------------------
// Family histograms
// ---------------------------------------------------------------------------

fn accumulate(histogram: &mut [u64; 256], planes: &[Vec<u8>]) {
    for plane in planes {
        for &byte in plane {
            histogram[byte as usize] += 1;
        }
    }
}

fn float_family(reader: &Safetensors, limits: &Limits) -> Result<([u64; 256], usize), String> {
    let mut counts = [0u64; 256];
    let mut members = 0usize;
    for name in reader.float_names() {
        let info = reader.get(&name)?;
        let Some(float_dtype) = info.float_dtype else {
            continue;
        };
        let bytes = reader.read(&name)?;
        let planes = codec_error(float_planes(&bytes, float_dtype, info.elements()?))?;
        accumulate(&mut counts, &planes);
        members += 1;
        if members % 50 == 0 {
            eprintln!("  family float: {members} tensors");
        }
    }
    let _ = limits;
    Ok((counts, members))
}

fn delta_family(
    base: &Safetensors,
    target: &Safetensors,
    limits: &Limits,
) -> Result<([u64; 256], usize), String> {
    let mut counts = [0u64; 256];
    let mut members = 0usize;
    for name in base.float_names() {
        if !target.tensors.contains_key(&name) {
            continue;
        }
        let base_info = base.get(&name)?;
        let target_info = target.get(&name)?;
        if base_info.shape != target_info.shape
            || base_info.tensor_dtype != target_info.tensor_dtype
        {
            continue;
        }
        let base_bytes = base.read(&name)?;
        let target_bytes = target.read(&name)?;
        let base_frame = codec_error(modes::encode_tensor(
            &base_bytes,
            base_info.tensor_dtype,
            &base_info.shape,
            limits,
        ))?;
        let planes = codec_error(weight_delta_planes(
            &target_bytes,
            &base_frame,
            base_info.weight_dtype,
            &base_info.shape,
            limits,
        ))?;
        accumulate(&mut counts, &planes);
        members += 1;
        if members % 50 == 0 {
            eprintln!("  family delta: {members} tensors");
        }
    }
    Ok((counts, members))
}

fn tiled_family(
    reader: &Safetensors,
    tile_shape: &[u64],
    limits: &Limits,
) -> Result<([u64; 256], usize, usize), String> {
    let mut counts = [0u64; 256];
    let mut members = 0usize;
    let mut excluded = 0usize;
    for name in reader.float_names() {
        let info = reader.get(&name)?;
        if info.shape.len() != 2
            || info.shape[0] % tile_shape[0] != 0
            || info.shape[1] % tile_shape[1] != 0
        {
            excluded += 1;
            continue;
        }
        let bytes = reader.read(&name)?;
        let planes = codec_error(tiled_planes(
            &bytes,
            info.tensor_dtype,
            &info.shape,
            tile_shape,
            limits,
        ))?;
        accumulate(&mut counts, &planes);
        members += 1;
        if members % 50 == 0 {
            eprintln!("  family tiled: {members} tensors");
        }
    }
    Ok((counts, members, excluded))
}

fn kv_family(
    reader: &Safetensors,
    name: &str,
    limits: &Limits,
) -> Result<([u64; 256], usize), String> {
    const HEADS: u32 = 16;
    const DIM: u32 = 64;
    const PAGE_ROWS: u64 = 512;
    let row_bytes = u64::from(HEADS) * u64::from(DIM) * 2;
    let page_bytes = PAGE_ROWS * row_bytes;
    let bytes = reader.read(name)?;
    let mut counts = [0u64; 256];
    let pages = 8usize;
    if bytes.len() as u64 >= page_bytes * pages as u64 {
        for index in 0..pages {
            let start = index * page_bytes as usize;
            let end = start + page_bytes as usize;
            let planes = codec_error(kv_page_planes(
                &bytes[start..end],
                PAGE_ROWS,
                HEADS,
                DIM,
                KvDtype::Bf16,
                limits,
            ))?;
            accumulate(&mut counts, &planes);
        }
    }
    let members = (bytes.len() as u64 / page_bytes).min(pages as u64) as usize;
    Ok((counts, members))
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DictScope {
    PerPlane,
    PerTensor,
    Family,
}

impl DictScope {
    fn as_str(self) -> &'static str {
        match self {
            DictScope::PerPlane => "per-plane",
            DictScope::PerTensor => "per-tensor",
            DictScope::Family => "family-shared",
        }
    }
}

struct Measurement {
    encoded_bytes: u64,
    dictionary_bytes: u64,
    encoded_bytes_amortized: Option<f64>,
    encode_ms_median: f64,
    decode_ms_p50: f64,
    decode_ms_p99: f64,
    encode_reps: usize,
    decode_reps: usize,
    dictionary_build_ms: f64,
    peak_rss_kib: u64,
}

fn peak_rss_kib() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
        }
    }
    0
}

/// Reset the kernel's peak-RSS counter for this process before a row, so the
/// reported number is that row's high-water mark rather than a monotonic
/// process maximum. Returns false when the kernel does not support it
/// (`clear_refs` value 5).
fn reset_peak_rss() -> bool {
    use std::io::Write as _;
    match OpenOptions::new().write(true).open("/proc/self/clear_refs") {
        Ok(mut file) => file.write_all(b"5\n").is_ok(),
        Err(_) => false,
    }
}

fn median_ms(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

fn percentile_ms(samples: &mut [f64], percentile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((samples.len() as f64 - 1.0) * percentile).round() as usize;
    samples[index.min(samples.len() - 1)]
}

fn reps_for(bytes: u64, quick: bool) -> usize {
    let base = if bytes <= 64 * 1024 {
        31
    } else if bytes <= (1 << 20) {
        21
    } else if bytes <= (8 << 20) {
        9
    } else {
        3
    };
    if quick {
        base.min(3)
    } else {
        base
    }
}

fn measure_planes(
    coder: &dyn Coder,
    planes: &[Vec<u8>],
    encode_reps: usize,
    decode_reps: usize,
) -> Result<Measurement, String> {
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for plane in planes {
        let stream = coder.encode(plane);
        encoded.push(stream);
    }
    let dictionary_build_ms = 0.0;
    for (plane, stream) in planes.iter().zip(encoded.iter()) {
        let decoded = coder
            .decode(stream, plane.len())
            .map_err(|error| format!("{} roundtrip: {error}", coder.id()))?;
        if decoded != *plane {
            return Err(format!("{} roundtrip bytes differ", coder.id()));
        }
    }
    let encoded_bytes: u64 = encoded.iter().map(|stream| stream.len() as u64).sum();

    let mut encode_times = Vec::with_capacity(encode_reps);
    for _ in 0..encode_reps {
        let start = Instant::now();
        for plane in planes {
            let stream = coder.encode(plane);
            std::hint::black_box(&stream);
        }
        encode_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mut decode_times = Vec::with_capacity(decode_reps);
    for _ in 0..decode_reps {
        let start = Instant::now();
        for (plane, stream) in planes.iter().zip(encoded.iter()) {
            let decoded = coder
                .decode(stream, plane.len())
                .map_err(|error| format!("{} timed decode: {error}", coder.id()))?;
            std::hint::black_box(&decoded);
        }
        decode_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(Measurement {
        encoded_bytes,
        dictionary_bytes: 0,
        encoded_bytes_amortized: None,
        encode_ms_median: median_ms(&mut encode_times),
        decode_ms_p50: percentile_ms(&mut decode_times, 0.50),
        decode_ms_p99: percentile_ms(&mut decode_times, 0.99),
        encode_reps,
        decode_reps,
        dictionary_build_ms,
        peak_rss_kib: peak_rss_kib(),
    })
}

fn measure_tunstall(
    tunstall: &Tunstall,
    scope: DictScope,
    planes: &[Vec<u8>],
    family_histogram: Option<&[u64; 256]>,
    family_members: usize,
    encode_reps: usize,
    decode_reps: usize,
) -> Result<Measurement, String> {
    let build_start = Instant::now();
    let dictionaries: Vec<Dictionary> = match scope {
        DictScope::PerPlane => {
            let mut dictionaries = Vec::with_capacity(planes.len());
            for plane in planes {
                dictionaries.push(
                    tunstall
                        .build_dict_from_planes(&[plane.as_slice()])
                        .map_err(|error| format!("Tunstall per-plane dictionary: {error}"))?,
                );
            }
            dictionaries
        }
        DictScope::PerTensor => {
            let references: Vec<&[u8]> = planes.iter().map(|plane| plane.as_slice()).collect();
            vec![tunstall
                .build_dict_from_planes(&references)
                .map_err(|error| format!("Tunstall per-tensor dictionary: {error}"))?]
        }
        DictScope::Family => {
            let counts = family_histogram.ok_or_else(|| {
                "Tunstall family scope requested without a family histogram".to_owned()
            })?;
            vec![tunstall
                .build_dict(&*counts)
                .map_err(|error| format!("Tunstall family dictionary: {error}"))?]
        }
    };
    let dictionary_build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    let mut per_plane_dictionary_bytes = 0usize;
    for (index, plane) in planes.iter().enumerate() {
        let dictionary = match scope {
            DictScope::PerPlane => &dictionaries[index],
            _ => &dictionaries[0],
        };
        let stream = tunstall
            .encode_with_dict(plane, dictionary)
            .map_err(|error| format!("Tunstall encode: {error}"))?;
        encoded.push(stream);
    }
    for (index, (plane, stream)) in planes.iter().zip(encoded.iter()).enumerate() {
        let dictionary = match scope {
            DictScope::PerPlane => &dictionaries[index],
            _ => &dictionaries[0],
        };
        let decoded = tunstall
            .decode_with_dict(stream, plane.len(), dictionary)
            .map_err(|error| format!("Tunstall roundtrip: {error}"))?;
        if decoded != *plane {
            return Err("Tunstall roundtrip bytes differ".to_owned());
        }
    }
    let streams_bytes: u64 = encoded.iter().map(|stream| stream.len() as u64).sum();
    let dictionary_bytes: u64 = match scope {
        DictScope::PerPlane => {
            for dictionary in &dictionaries {
                per_plane_dictionary_bytes += dictionary.serialized_len();
            }
            per_plane_dictionary_bytes as u64
        }
        _ => dictionaries[0].serialized_len() as u64,
    };

    let mut encode_times = Vec::with_capacity(encode_reps);
    for _ in 0..encode_reps {
        let start = Instant::now();
        for (index, plane) in planes.iter().enumerate() {
            let dictionary = match scope {
                DictScope::PerPlane => &dictionaries[index],
                _ => &dictionaries[0],
            };
            let stream = tunstall
                .encode_with_dict(plane, dictionary)
                .map_err(|error| format!("Tunstall timed encode: {error}"))?;
            std::hint::black_box(&stream);
        }
        encode_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mut decode_times = Vec::with_capacity(decode_reps);
    for _ in 0..decode_reps {
        let start = Instant::now();
        for (index, (plane, stream)) in planes.iter().zip(encoded.iter()).enumerate() {
            let dictionary = match scope {
                DictScope::PerPlane => &dictionaries[index],
                _ => &dictionaries[0],
            };
            let decoded = tunstall
                .decode_with_dict(stream, plane.len(), dictionary)
                .map_err(|error| format!("Tunstall timed decode: {error}"))?;
            std::hint::black_box(&decoded);
        }
        decode_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let amortized = if scope == DictScope::Family && family_members > 0 {
        Some(streams_bytes as f64 + dictionary_bytes as f64 / family_members as f64)
    } else {
        None
    };
    Ok(Measurement {
        encoded_bytes: streams_bytes + dictionary_bytes,
        dictionary_bytes,
        encoded_bytes_amortized: amortized,
        encode_ms_median: median_ms(&mut encode_times),
        decode_ms_p50: percentile_ms(&mut decode_times, 0.50),
        decode_ms_p99: percentile_ms(&mut decode_times, 0.99),
        encode_reps,
        decode_reps,
        dictionary_build_ms,
        peak_rss_kib: peak_rss_kib(),
    })
}

// ---------------------------------------------------------------------------
// Row output
// ---------------------------------------------------------------------------

struct RowEmitter {
    file: BufWriter<File>,
    rows: Vec<Json>,
}

impl RowEmitter {
    fn new(path: &PathBuf) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("{}: {error}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(RowEmitter {
            file: BufWriter::new(file),
            rows: Vec::new(),
        })
    }

    fn emit(&mut self, row: Json) -> Result<(), String> {
        let line =
            serde_json::to_string(&row).map_err(|error| format!("row serialisation: {error}"))?;
        self.file
            .write_all(line.as_bytes())
            .and_then(|_| self.file.write_all(b"\n"))
            .map_err(|error| format!("row write: {error}"))?;
        println!("{line}");
        self.rows.push(row);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn push_row(
    emitter: &mut RowEmitter,
    class: &InputClass,
    case: &ShapeCase,
    coder: &str,
    tunstall_k: Option<u8>,
    scope: Option<DictScope>,
    measurement: &Measurement,
) -> Result<(), String> {
    let decoded_bytes = planes_bytes(&case.planes);
    let seconds_encode = measurement.encode_ms_median / 1000.0;
    let seconds_decode = measurement.decode_ms_p50 / 1000.0;
    let encode_mib_s = if seconds_encode > 0.0 {
        decoded_bytes as f64 / (1024.0 * 1024.0) / seconds_encode
    } else {
        0.0
    };
    let decode_mib_s = if seconds_decode > 0.0 {
        decoded_bytes as f64 / (1024.0 * 1024.0) / seconds_decode
    } else {
        0.0
    };
    let row = json!({
        "type": "row",
        "input_class": class.id,
        "description": class.description,
        "tensor": class.tensor,
        "dtype": class.dtype,
        "tensor_shape": class.shape,
        "anchor_row": class.anchor_row,
        "workload_shape": case.shape,
        "rows": case.rows,
        "columns": case.columns,
        "elements": case.elements,
        "plane_count": case.planes.len(),
        "plane_bytes": case.planes.iter().map(|plane| plane.len() as u64).collect::<Vec<_>>(),
        "decoded_bytes": decoded_bytes,
        "coder": coder,
        "tunstall_k": tunstall_k,
        "dictionary_scope": scope.map(DictScope::as_str),
        "dictionary_bytes": measurement.dictionary_bytes,
        "family_members": if scope == Some(DictScope::Family) { Some(class.family_members) } else { None },
        "encoded_bytes": measurement.encoded_bytes,
        "encoded_bytes_amortized": measurement.encoded_bytes_amortized,
        "encode_ms_median": measurement.encode_ms_median,
        "decode_ms_p50": measurement.decode_ms_p50,
        "decode_ms_p99": measurement.decode_ms_p99,
        "encode_mib_s": encode_mib_s,
        "decode_mib_s": decode_mib_s,
        "encode_reps": measurement.encode_reps,
        "decode_reps": measurement.decode_reps,
        "dictionary_build_ms": measurement.dictionary_build_ms,
        "process_peak_rss_kib": measurement.peak_rss_kib,
        "roundtrip_ok": true,
    });
    emitter.emit(row)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if let Err(error) = run() {
        eprintln!("entropy_coder_court: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let limits = Limits::default();
    eprintln!("== opening corpus ==");
    let qwen_base = Safetensors::open(args.base_qwen.clone())?;
    let qwen_instruct = Safetensors::open(args.instruct_qwen.clone())?;
    let pythia_step0 = Safetensors::open(args.pythia_step0.clone())?;
    let pythia_step1 = Safetensors::open(args.pythia_step1.clone())?;
    eprintln!(
        "qwen base: {} float tensors, {} unsupported",
        qwen_base.tensors.len(),
        qwen_base.unsupported.len()
    );
    eprintln!(
        "qwen instruct: {} float tensors",
        qwen_instruct.tensors.len()
    );
    eprintln!("pythia step0: {} float tensors", pythia_step0.tensors.len());
    eprintln!("pythia step1: {} float tensors", pythia_step1.tensors.len());
    if qwen_base.file_len == 0 || pythia_step0.file_len == 0 {
        return Err("empty corpus file".to_owned());
    }

    eprintln!("== building input classes ==");
    let qwen_tensor = "model.layers.0.mlp.down_proj.weight";
    let pythia_tensor = "gpt_neox.layers.0.mlp.dense_h_to_4h.weight";
    let sparse_tensor = "embed_out.weight";

    let (fp32_cases, fp32_shape, fp32_dtype) =
        float_cases(&pythia_step0, pythia_tensor, FloatDtype::F32)?;
    let (bf16_cases, bf16_shape, bf16_dtype) =
        float_cases(&qwen_base, qwen_tensor, FloatDtype::Bf16)?;
    let (dense_cases, dense_shape, dense_dtype, dense_anchor) =
        delta_cases(&qwen_base, &qwen_instruct, qwen_tensor, &limits)?;
    let (sparse_cases, sparse_shape, sparse_dtype, sparse_anchor) =
        delta_cases(&pythia_step0, &pythia_step1, sparse_tensor, &limits)?;
    let qwen_tile = vec![64u64, 128u64];
    let (tile_cases, tile_shape, tile_dtype) =
        tiled_cases(&qwen_base, qwen_tensor, &qwen_tile, &limits)?;
    let (kv_cases_vec, kv_shape, kv_dtype) = kv_cases(&qwen_base, qwen_tensor, &limits)?;

    let mut classes = Vec::new();
    classes.push(InputClass {
        id: "mode9_bf16",
        description: "mode-9 FLOAT_TENSOR exponent and sign+mantissa planes, BF16".to_owned(),
        tensor: qwen_tensor.to_owned(),
        dtype: bf16_dtype,
        shape: bf16_shape,
        anchor_row: 0,
        cases: bf16_cases,
        family_members: 0,
        family_excluded: 0,
        family_histogram: None,
    });
    classes.push(InputClass {
        id: "mode9_fp32",
        description: "mode-9 FLOAT_TENSOR exponent and sign+mantissa planes, FP32".to_owned(),
        tensor: pythia_tensor.to_owned(),
        dtype: fp32_dtype,
        shape: fp32_shape,
        anchor_row: 0,
        cases: fp32_cases,
        family_members: 0,
        family_excluded: 0,
        family_histogram: None,
    });
    classes.push(InputClass {
        id: "mode11_dense",
        description: "mode-11 WEIGHT_DELTA residual byte planes, dense fine-tune pair".to_owned(),
        tensor: qwen_tensor.to_owned(),
        dtype: dense_dtype,
        shape: dense_shape,
        anchor_row: dense_anchor,
        cases: dense_cases,
        family_members: 0,
        family_excluded: 0,
        family_histogram: None,
    });
    classes.push(InputClass {
        id: "mode11_sparse",
        description: "mode-11 WEIGHT_DELTA residual byte planes, sparse checkpoint pair".to_owned(),
        tensor: sparse_tensor.to_owned(),
        dtype: sparse_dtype,
        shape: sparse_shape,
        anchor_row: sparse_anchor,
        cases: sparse_cases,
        family_members: 0,
        family_excluded: 0,
        family_histogram: None,
    });
    classes.push(InputClass {
        id: "mode15_tiles",
        description: "mode-15 TENSOR_TILED byte planes, tile [64,128]".to_owned(),
        tensor: qwen_tensor.to_owned(),
        dtype: tile_dtype,
        shape: tile_shape,
        anchor_row: 0,
        cases: tile_cases,
        family_members: 0,
        family_excluded: 0,
        family_histogram: None,
    });
    classes.push(InputClass {
        id: "kv_page",
        description: "mode-12 KV_CACHE page byte planes, real tensor bytes".to_owned(),
        tensor: qwen_tensor.to_owned(),
        dtype: kv_dtype,
        shape: kv_shape,
        anchor_row: 0,
        cases: kv_cases_vec,
        family_members: 0,
        family_excluded: 0,
        family_histogram: None,
    });

    if !args.skip_family {
        eprintln!("== family histograms ==");
        for class in classes.iter_mut() {
            let started = Instant::now();
            let (histogram, members, excluded) = match class.id {
                "mode9_bf16" => {
                    let (counts, members) = float_family(&qwen_base, &limits)?;
                    (counts, members, 0)
                }
                "mode9_fp32" => {
                    let (counts, members) = float_family(&pythia_step0, &limits)?;
                    (counts, members, 0)
                }
                "mode11_dense" => {
                    let (counts, members) = delta_family(&qwen_base, &qwen_instruct, &limits)?;
                    (counts, members, 0)
                }
                "mode11_sparse" => {
                    let (counts, members) = delta_family(&pythia_step0, &pythia_step1, &limits)?;
                    (counts, members, 0)
                }
                "mode15_tiles" => {
                    let (counts, members, excluded) =
                        tiled_family(&qwen_base, &qwen_tile, &limits)?;
                    (counts, members, excluded)
                }
                "kv_page" => {
                    let (counts, members) = kv_family(&qwen_base, qwen_tensor, &limits)?;
                    (counts, members, 0)
                }
                other => return Err(format!("unknown class {other}")),
            };
            class.family_members = members;
            class.family_excluded = excluded;
            class.family_histogram = Some(histogram);
            eprintln!(
                "  {}: {members} family members ({excluded} excluded) in {:.1} ms",
                class.id,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    let mut emitter = RowEmitter::new(&args.out)?;
    let rss_reset_supported = reset_peak_rss();
    let build_peak_rss_kib = peak_rss_kib();
    let coders: Vec<Box<dyn Coder>> = vec![
        Box::new(Raw),
        Box::new(Zstd::new(3)),
        Box::new(Zstd::new(19)),
        Box::new(Huffman),
        Box::new(Rans),
    ];
    let tunstall_widths = [8u8, 10, 12];

    eprintln!("== measuring ==");
    for class in &classes {
        for case in &class.cases {
            let decoded = planes_bytes(&case.planes);
            let encode_reps = reps_for(decoded, args.quick);
            let decode_reps = encode_reps.max(if args.quick { 3 } else { 11 });
            eprintln!(
                "  {} / {}: {} planes, {} bytes, encode_reps={encode_reps}, decode_reps={decode_reps}",
                class.id,
                case.shape,
                case.planes.len(),
                decoded
            );
            for coder in &coders {
                let _ = reset_peak_rss();
                let measurement =
                    measure_planes(coder.as_ref(), &case.planes, encode_reps, decode_reps)?;
                push_row(
                    &mut emitter,
                    class,
                    case,
                    coder.id(),
                    None,
                    None,
                    &measurement,
                )?;
            }
            for k in tunstall_widths {
                let tunstall = Tunstall::new(k).map_err(|error| error.to_string())?;
                for scope in [DictScope::PerPlane, DictScope::PerTensor] {
                    let _ = reset_peak_rss();
                    let measurement = measure_tunstall(
                        &tunstall,
                        scope,
                        &case.planes,
                        class.family_histogram.as_ref(),
                        class.family_members,
                        encode_reps,
                        decode_reps,
                    )?;
                    push_row(
                        &mut emitter,
                        class,
                        case,
                        "tunstall",
                        Some(k),
                        Some(scope),
                        &measurement,
                    )?;
                }
                if !args.skip_family && class.family_histogram.is_some() {
                    let _ = reset_peak_rss();
                    let measurement = measure_tunstall(
                        &tunstall,
                        DictScope::Family,
                        &case.planes,
                        class.family_histogram.as_ref(),
                        class.family_members,
                        encode_reps,
                        decode_reps,
                    )?;
                    push_row(
                        &mut emitter,
                        class,
                        case,
                        "tunstall",
                        Some(k),
                        Some(DictScope::Family),
                        &measurement,
                    )?;
                }
            }
        }
    }

    // Summary: pooled throughput per coder configuration, plus the peak RSS.
    let mut pooled: BTreeMap<String, (u64, u64, f64, f64, f64)> = BTreeMap::new();
    for row in &emitter.rows {
        let key = match (
            row.get("coder").and_then(Json::as_str).unwrap_or(""),
            row.get("tunstall_k").and_then(Json::as_u64),
            row.get("dictionary_scope").and_then(Json::as_str),
        ) {
            ("tunstall", Some(k), Some(scope)) => format!("tunstall-{k}/{scope}"),
            (coder, _, _) => coder.to_owned(),
        };
        let decoded = row.get("decoded_bytes").and_then(Json::as_u64).unwrap_or(0);
        let encoded = row.get("encoded_bytes").and_then(Json::as_u64).unwrap_or(0);
        let encode_ms = row
            .get("encode_ms_median")
            .and_then(Json::as_f64)
            .unwrap_or(0.0);
        let decode_ms = row
            .get("decode_ms_p50")
            .and_then(Json::as_f64)
            .unwrap_or(0.0);
        let p99 = row
            .get("decode_ms_p99")
            .and_then(Json::as_f64)
            .unwrap_or(0.0);
        let entry = pooled.entry(key).or_insert((0, 0, 0.0, 0.0, 0.0));
        entry.0 += decoded;
        entry.1 += encoded;
        entry.2 += encode_ms;
        entry.3 += decode_ms;
        entry.4 = entry.4.max(p99);
    }
    let mut coders_summary = serde_json::Map::new();
    for (key, (decoded, encoded, encode_ms, decode_ms, max_p99)) in pooled {
        let encode_mib_s = if encode_ms > 0.0 {
            decoded as f64 / (1024.0 * 1024.0) / (encode_ms / 1000.0)
        } else {
            0.0
        };
        let decode_mib_s = if decode_ms > 0.0 {
            decoded as f64 / (1024.0 * 1024.0) / (decode_ms / 1000.0)
        } else {
            0.0
        };
        coders_summary.insert(
            key,
            json!({
                "decoded_bytes": decoded,
                "encoded_bytes": encoded,
                "encoded_ratio": if decoded > 0 { encoded as f64 / decoded as f64 } else { 0.0 },
                "pooled_encode_mib_s": encode_mib_s,
                "pooled_decode_mib_s": decode_mib_s,
                "max_p99_decode_ms": max_p99,
            }),
        );
    }
    let summary = json!({
        "type": "summary",
        "schema": "weft.entropy-coder.rows.v1",
        "rows": emitter.rows.len(),
        "all_roundtrip_ok": true,
        "build_peak_rss_kib": build_peak_rss_kib,
        "matrix_peak_rss_kib": emitter
            .rows
            .iter()
            .filter_map(|row| row.get("process_peak_rss_kib").and_then(Json::as_u64))
            .max()
            .unwrap_or(0),
        "process_peak_rss_kib": peak_rss_kib(),
        "peak_rss_method": if rss_reset_supported {
            "VmHWM reset immediately before each row; the six input classes' extracted planes stay resident, so small-region rows report a floor set by resident inputs"
        } else {
            "process high-water mark (reset unsupported)"
        },
        "experimental_only": true,
        "wire_modes_added": 0,
        "coders": coders_summary,
    });
    let line = serde_json::to_string(&summary).map_err(|error| format!("summary: {error}"))?;
    emitter
        .file
        .write_all(line.as_bytes())
        .and_then(|_| emitter.file.write_all(b"\n"))
        .and_then(|_| emitter.file.flush())
        .map_err(|error| format!("summary write: {error}"))?;
    println!("{line}");
    emitter
        .file
        .flush()
        .map_err(|error| format!("flush: {error}"))?;
    eprintln!("== done: {} rows ==", emitter.rows.len());
    Ok(())
}

// Keep `Read` in scope for the header read path even when the compiler can
// infer `read_exact_at` alone.
#[allow(dead_code)]
fn _read_trait_is_used(mut file: File) -> std::io::Result<()> {
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte)
}
