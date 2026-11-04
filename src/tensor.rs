//! Logical tensor identity, layout descriptors, and region addressing.
//!
//! Normative source: `spec/WEFT-TENSOR.md`.
//!
//! A *logical tensor* is `(dtype, shape)` plus element values in canonical
//! row-major order (last index fastest, little-endian elements). Its identity
//! deliberately ignores sharding, blocking, strides, and storage layout:
//!
//! ```text
//! values_hash       = H("weft/tensor/values/v1",  row-major element bytes)
//! LogicalTensorRoot = H("weft/tensor/logical/v1",
//!                       WCB({schema:"weft.tensor.logical.v1", dtype, shape,
//!                            element_count, values_hash}))
//! ```
//!
//! Layer separation (WEFT-TENSOR §1):
//!
//! * [`LogicalRecord`] — layout-independent identity of a tensor value.
//! * [`LayoutDescriptor`] — additive `weft.tensor.layout.v1` record that names
//!   `dtype`, `shape`, and how the bytes are physically arranged.
//! * [`PhysicalTensor`] — an in-memory physical arrangement (row-major,
//!   column-major, blocked, or a set of shards) with [`PhysicalTensor::assemble`]
//!   and [`PhysicalTensor::read_region`].
//! * [`encode_shard`] / [`decode_shard`] — the self-describing shard container
//!   `u32le(descriptor_len) || WCB(layout) || payload`, and [`shard_root`] to
//!   verify a complete shard set and return the logical root.
//!
//! This module is additive. It does not change mode-6 `TENSOR` frames, whose
//! `layout` field remains frozen to `row_major` (WEFT-AI-CODEC §2).

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::hash::{hash_domain, Hash32};
use crate::limits::Limits;
pub use crate::modes::TensorDtype;
use crate::wcb::{self, Value};

/// Canonical logical-tensor schema string.
pub const LOGICAL_SCHEMA: &str = "weft.tensor.logical.v1";
/// Canonical layout-descriptor schema string.
pub const LAYOUT_SCHEMA: &str = "weft.tensor.layout.v1";
/// Canonical shard-container schema string.
pub const SHARD_SCHEMA: &str = "weft.tensor.shard.v1";

/// Logical tensor identity domain.
pub const DOMAIN_TENSOR_LOGICAL: &str = "weft/tensor/logical/v1";
/// Canonical element-byte domain.
pub const DOMAIN_TENSOR_VALUES: &str = "weft/tensor/values/v1";
/// Shard-container payload domain.
pub const DOMAIN_TENSOR_SHARD_PAYLOAD: &str = "weft/tensor/shard-payload/v1";

// ---------------------------------------------------------------------------
// Shape and dtype validation
// ---------------------------------------------------------------------------

/// Parse a dtype name for the tensor-identity layer.
///
/// Unknown dtypes are `UnsupportedFeature` here (the value is structurally a
/// canonical record that names an element type this version does not know).
pub fn dtype_from_name(name: &str) -> Result<TensorDtype> {
    TensorDtype::from_name(name).map_err(|_| {
        Error::unsupported_feature(format!(
            "unknown tensor dtype `{name}` (v1 allows f16, bf16, f32, f64, i8, u8, i16, i32, i64)"
        ))
    })
}

/// Validate a logical shape and return its element count.
///
/// Rejects rank 0, zero dimensions, overflow, and counts or ranks above the
/// configured limits.
pub fn validate_shape(shape: &[u64], limits: &Limits) -> Result<u64> {
    if shape.is_empty() {
        return Err(Error::non_canonical(
            "tensor shape must have rank >= 1 (rank 0 is not a v1 logical tensor)",
        ));
    }
    if shape.len() as u64 > limits.max_tensor_rank {
        return Err(Error::limit_exceeded(format!(
            "tensor rank {} exceeds limit {}",
            shape.len(),
            limits.max_tensor_rank
        )));
    }
    let mut count: u64 = 1;
    for (axis, dim) in shape.iter().enumerate() {
        if *dim == 0 {
            return Err(Error::non_canonical(format!(
                "tensor shape dimension {axis} is zero"
            )));
        }
        count = count
            .checked_mul(*dim)
            .ok_or_else(|| Error::limit_exceeded("tensor shape product overflows u64"))?;
    }
    if count > limits.max_tensor_elements {
        return Err(Error::limit_exceeded(format!(
            "tensor element count {count} exceeds limit {}",
            limits.max_tensor_elements
        )));
    }
    Ok(count)
}

/// Element byte length for a count and dtype, with checked arithmetic.
pub fn payload_len(element_count: u64, dtype: TensorDtype) -> Result<u64> {
    element_count
        .checked_mul(dtype.element_size())
        .ok_or_else(|| Error::limit_exceeded("tensor payload byte count overflows u64"))
}

/// Check that `bytes` is exactly the payload for `shape` x `dtype`.
///
/// Returns the element count. A byte-length mismatch is `Malformed`; a payload
/// above `max_decoded_bytes` is `LimitExceeded` before any allocation.
pub fn check_payload(
    dtype: TensorDtype,
    shape: &[u64],
    bytes: &[u8],
    limits: &Limits,
) -> Result<u64> {
    let count = validate_shape(shape, limits)?;
    let expected = payload_len(count, dtype)?;
    crate::limits::check_len(expected, limits.max_decoded_bytes, "tensor payload")?;
    if expected != bytes.len() as u64 {
        return Err(Error::malformed(format!(
            "tensor payload is {} bytes but shape {:?} x {} requires {expected}",
            bytes.len(),
            shape,
            dtype.as_str()
        )));
    }
    Ok(count)
}

// ---------------------------------------------------------------------------
// Logical identity
// ---------------------------------------------------------------------------

/// `H("weft/tensor/values/v1", canonical row-major element bytes)`.
pub fn values_hash(row_major: &[u8]) -> Hash32 {
    hash_domain(DOMAIN_TENSOR_VALUES, row_major)
}

/// The canonical logical-tensor identity record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalRecord {
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub values_hash: Hash32,
}

impl LogicalRecord {
    /// Build the record from canonical row-major element bytes.
    pub fn from_row_major(
        dtype: TensorDtype,
        shape: &[u64],
        row_major: &[u8],
        limits: &Limits,
    ) -> Result<Self> {
        let element_count = check_payload(dtype, shape, row_major, limits)?;
        Ok(LogicalRecord {
            dtype,
            shape: shape.to_vec(),
            element_count,
            values_hash: values_hash(row_major),
        })
    }

    /// Strict validation of every record invariant.
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        let count = validate_shape(&self.shape, limits)?;
        if self.element_count != count {
            return Err(Error::non_canonical(format!(
                "tensor element_count {} does not match shape product {count}",
                self.element_count
            )));
        }
        // Checked multiplication; the record may not claim an impossible count.
        payload_len(count, self.dtype)?;
        Ok(())
    }

    /// Canonical WCB value.
    pub fn to_wcb_value(&self) -> Value {
        Value::map([
            ("schema", Value::Str(LOGICAL_SCHEMA.to_owned())),
            ("dtype", Value::Str(self.dtype.as_str().to_owned())),
            (
                "shape",
                Value::Array(self.shape.iter().map(|dim| Value::U64(*dim)).collect()),
            ),
            ("element_count", Value::U64(self.element_count)),
            (
                "values_hash",
                Value::Bytes(self.values_hash.as_bytes().to_vec()),
            ),
        ])
    }

    /// Canonical WCB bytes under explicit limits.
    pub fn canonical_bytes_with_limits(&self, limits: &Limits) -> Result<Vec<u8>> {
        self.validate(limits)?;
        wcb::encode(&self.to_wcb_value(), limits)
    }

    /// Canonical WCB bytes under default limits.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.canonical_bytes_with_limits(&Limits::default())
    }

    /// `H("weft/tensor/logical/v1", canonical bytes)`.
    pub fn root_with_limits(&self, limits: &Limits) -> Result<Hash32> {
        Ok(hash_domain(
            DOMAIN_TENSOR_LOGICAL,
            &self.canonical_bytes_with_limits(limits)?,
        ))
    }

    /// `H("weft/tensor/logical/v1", canonical bytes)` with default limits.
    pub fn root(&self) -> Result<Hash32> {
        self.root_with_limits(&Limits::default())
    }

    /// Decode canonical record bytes, rejecting every non-canonical form.
    pub fn from_canonical_bytes(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let value = wcb::decode(bytes, limits)?;
        let map = require_map(&value, "logical tensor record")?;
        check_allowed_keys(
            map,
            &["schema", "dtype", "shape", "element_count", "values_hash"],
            "logical tensor record",
        )?;
        let schema = require_text(map, "schema", "logical tensor record")?;
        if schema != LOGICAL_SCHEMA {
            return Err(Error::unknown_schema(format!(
                "unknown logical tensor schema `{schema}`"
            )));
        }
        let dtype = dtype_from_name(require_text(map, "dtype", "logical tensor record")?)?;
        let shape = require_u64_array(map, "shape", "logical tensor record")?;
        let element_count = require_u64(map, "element_count", "logical tensor record")?;
        let values_hash = require_hash(map, "values_hash", "logical tensor record")?;
        let record = LogicalRecord {
            dtype,
            shape,
            element_count,
            values_hash,
        };
        record.validate(limits)?;
        let reencoded = wcb::encode(&record.to_wcb_value(), limits)?;
        if reencoded != bytes {
            return Err(Error::non_canonical(
                "logical tensor bytes are not in canonical form",
            ));
        }
        Ok(record)
    }
}

/// `LogicalTensorRoot` computed directly from canonical row-major bytes.
pub fn logical_root(
    dtype: TensorDtype,
    shape: &[u64],
    row_major: &[u8],
    limits: &Limits,
) -> Result<Hash32> {
    LogicalRecord::from_row_major(dtype, shape, row_major, limits)?.root_with_limits(limits)
}

// ---------------------------------------------------------------------------
// Layout descriptor
// ---------------------------------------------------------------------------

/// Physical layout of a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorLayout {
    /// Last index fastest (canonical order).
    RowMajor,
    /// First index fastest (Fortran order).
    ColumnMajor,
    /// Tiled by `block_shape`; blocks are enumerated in row-major order over
    /// the block grid and each block is row-major internally.
    Blocked { block_shape: Vec<u64> },
    /// One slab along `axis`; `index` selects the slab in `[0, shards)`.
    Sharded { axis: u32, shards: u32, index: u32 },
}

impl TensorLayout {
    /// Canonical layout name.
    pub fn name(&self) -> &'static str {
        match self {
            TensorLayout::RowMajor => "row-major",
            TensorLayout::ColumnMajor => "column-major",
            TensorLayout::Blocked { .. } => "blocked",
            TensorLayout::Sharded { .. } => "sharded",
        }
    }
}

/// `weft.tensor.layout.v1`: a self-describing dtype/shape/layout record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutDescriptor {
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub layout: TensorLayout,
}

impl LayoutDescriptor {
    /// Row-major descriptor.
    pub fn row_major(dtype: TensorDtype, shape: &[u64]) -> Self {
        LayoutDescriptor {
            dtype,
            shape: shape.to_vec(),
            layout: TensorLayout::RowMajor,
        }
    }

    /// Column-major descriptor.
    pub fn column_major(dtype: TensorDtype, shape: &[u64]) -> Self {
        LayoutDescriptor {
            dtype,
            shape: shape.to_vec(),
            layout: TensorLayout::ColumnMajor,
        }
    }

    /// Blocked descriptor.
    pub fn blocked(dtype: TensorDtype, shape: &[u64], block_shape: &[u64]) -> Self {
        LayoutDescriptor {
            dtype,
            shape: shape.to_vec(),
            layout: TensorLayout::Blocked {
                block_shape: block_shape.to_vec(),
            },
        }
    }

    /// Sharded descriptor.
    pub fn sharded(dtype: TensorDtype, shape: &[u64], axis: u32, shards: u32, index: u32) -> Self {
        LayoutDescriptor {
            dtype,
            shape: shape.to_vec(),
            layout: TensorLayout::Sharded {
                axis,
                shards,
                index,
            },
        }
    }

    /// Strict validation of shape and layout invariants.
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        if self.shape.is_empty() {
            let reason = match self.layout {
                TensorLayout::ColumnMajor => {
                    "column-major cannot express a rank-0 shape (a scalar has no column order)"
                }
                _ => "tensor shape must have rank >= 1 (rank 0 is not a v1 tensor)",
            };
            return Err(Error::non_canonical(reason));
        }
        validate_shape(&self.shape, limits)?;
        match &self.layout {
            TensorLayout::RowMajor | TensorLayout::ColumnMajor => {}
            TensorLayout::Blocked { block_shape } => {
                if block_shape.len() != self.shape.len() {
                    return Err(Error::non_canonical(format!(
                        "blocked block_shape rank {} does not equal shape rank {}",
                        block_shape.len(),
                        self.shape.len()
                    )));
                }
                for (axis, block) in block_shape.iter().enumerate() {
                    if *block == 0 {
                        return Err(Error::non_canonical(format!(
                            "blocked block_shape dimension {axis} is zero"
                        )));
                    }
                }
            }
            TensorLayout::Sharded {
                axis,
                shards,
                index,
            } => {
                validate_shard_parameters(&self.shape, *axis, *shards)?;
                if *index >= *shards {
                    return Err(Error::non_canonical(format!(
                        "shard index {index} is not below shard count {shards}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Canonical WCB value.
    pub fn to_wcb_value(&self) -> Value {
        Value::Map(layout_map(self))
    }

    /// Canonical WCB bytes under explicit limits.
    pub fn canonical_bytes_with_limits(&self, limits: &Limits) -> Result<Vec<u8>> {
        self.validate(limits)?;
        wcb::encode(&self.to_wcb_value(), limits)
    }

    /// Canonical WCB bytes under default limits.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.canonical_bytes_with_limits(&Limits::default())
    }

    /// Short canonical hex rendering of the descriptor bytes.
    pub fn canonical_hex(&self) -> Result<String> {
        Ok(hex::encode(self.canonical_bytes()?))
    }

    /// Decode canonical descriptor bytes, rejecting every non-canonical form.
    pub fn from_canonical_bytes(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let value = wcb::decode(bytes, limits)?;
        let map = require_map(&value, "layout descriptor")?;
        check_allowed_keys(
            map,
            &[
                "schema",
                "dtype",
                "shape",
                "layout",
                "block_shape",
                "axis",
                "shards",
                "index",
            ],
            "layout descriptor",
        )?;
        let schema = require_text(map, "schema", "layout descriptor")?;
        if schema != LAYOUT_SCHEMA {
            return Err(Error::unknown_schema(format!(
                "unknown layout descriptor schema `{schema}`"
            )));
        }
        let descriptor = descriptor_from_map(map, limits)?;
        descriptor.validate(limits)?;
        let reencoded = wcb::encode(&descriptor.to_wcb_value(), limits)?;
        if reencoded != bytes {
            return Err(Error::non_canonical(
                "layout descriptor bytes are not in canonical form",
            ));
        }
        Ok(descriptor)
    }
}

fn layout_map(descriptor: &LayoutDescriptor) -> BTreeMap<String, Value> {
    let mut map = BTreeMap::new();
    map.insert("schema".to_owned(), Value::Str(LAYOUT_SCHEMA.to_owned()));
    map.insert(
        "dtype".to_owned(),
        Value::Str(descriptor.dtype.as_str().to_owned()),
    );
    map.insert(
        "shape".to_owned(),
        Value::Array(
            descriptor
                .shape
                .iter()
                .map(|dim| Value::U64(*dim))
                .collect(),
        ),
    );
    map.insert(
        "layout".to_owned(),
        Value::Str(descriptor.layout.name().to_owned()),
    );
    match &descriptor.layout {
        TensorLayout::RowMajor | TensorLayout::ColumnMajor => {}
        TensorLayout::Blocked { block_shape } => {
            map.insert(
                "block_shape".to_owned(),
                Value::Array(block_shape.iter().map(|dim| Value::U64(*dim)).collect()),
            );
        }
        TensorLayout::Sharded {
            axis,
            shards,
            index,
        } => {
            map.insert("axis".to_owned(), Value::U64(u64::from(*axis)));
            map.insert("shards".to_owned(), Value::U64(u64::from(*shards)));
            map.insert("index".to_owned(), Value::U64(u64::from(*index)));
        }
    }
    map
}

/// Build a descriptor from an already-schema-checked map.
fn descriptor_from_map(map: &BTreeMap<String, Value>, limits: &Limits) -> Result<LayoutDescriptor> {
    let dtype = dtype_from_name(require_text(map, "dtype", "layout descriptor")?)?;
    let shape = require_u64_array(map, "shape", "layout descriptor")?;
    let layout_name = require_text(map, "layout", "layout descriptor")?;
    let layout = match layout_name {
        "row-major" => {
            forbid_fields(
                map,
                &["block_shape", "axis", "shards", "index"],
                layout_name,
            )?;
            TensorLayout::RowMajor
        }
        "column-major" => {
            forbid_fields(
                map,
                &["block_shape", "axis", "shards", "index"],
                layout_name,
            )?;
            TensorLayout::ColumnMajor
        }
        "blocked" => {
            forbid_fields(map, &["axis", "shards", "index"], layout_name)?;
            TensorLayout::Blocked {
                block_shape: require_u64_array(map, "block_shape", "layout descriptor")?,
            }
        }
        "sharded" => {
            forbid_fields(map, &["block_shape"], layout_name)?;
            let axis = require_u32(map, "axis", "layout descriptor")?;
            let shards = require_u32(map, "shards", "layout descriptor")?;
            let index = require_u32(map, "index", "layout descriptor")?;
            TensorLayout::Sharded {
                axis,
                shards,
                index,
            }
        }
        other => {
            return Err(Error::unsupported_feature(format!(
            "unknown tensor layout `{other}` (v1 allows row-major, column-major, blocked, sharded)"
        )))
        }
    };
    let descriptor = LayoutDescriptor {
        dtype,
        shape,
        layout,
    };
    descriptor.validate(limits)?;
    Ok(descriptor)
}

/// Build the shard-container WCB map for a descriptor and payload metadata.
fn shard_map(
    descriptor: &LayoutDescriptor,
    payload_bytes: u64,
    payload_hash: Hash32,
) -> BTreeMap<String, Value> {
    let mut map = layout_map(descriptor);
    map.insert("schema".to_owned(), Value::Str(SHARD_SCHEMA.to_owned()));
    map.insert("payload_bytes".to_owned(), Value::U64(payload_bytes));
    map.insert(
        "payload_hash".to_owned(),
        Value::Bytes(payload_hash.as_bytes().to_vec()),
    );
    map
}

fn forbid_fields(map: &BTreeMap<String, Value>, fields: &[&str], layout: &str) -> Result<()> {
    for field in fields {
        if map.contains_key(*field) {
            return Err(Error::non_canonical(format!(
                "layout `{layout}` must not carry the `{field}` field"
            )));
        }
    }
    Ok(())
}

fn validate_shard_parameters(shape: &[u64], axis: u32, shards: u32) -> Result<()> {
    if shards == 0 {
        return Err(Error::non_canonical("shard count must be at least 1"));
    }
    let axis_usize = axis as usize;
    if axis_usize >= shape.len() {
        return Err(Error::non_canonical(format!(
            "shard axis {axis} is not below rank {}",
            shape.len()
        )));
    }
    if shape[axis_usize] % u64::from(shards) != 0 {
        return Err(Error::non_canonical(format!(
            "shape[{axis}] = {} is not divisible by shard count {shards}",
            shape[axis_usize]
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

/// Half-open hyper-rectangle in logical coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub start: Vec<u64>,
    pub end: Vec<u64>,
}

impl Region {
    /// Construct an unchecked region.
    pub fn new(start: Vec<u64>, end: Vec<u64>) -> Self {
        Region { start, end }
    }

    /// The full tensor region.
    pub fn full(shape: &[u64]) -> Self {
        Region {
            start: vec![0; shape.len()],
            end: shape.to_vec(),
        }
    }

    /// Per-axis extent.
    pub fn extent(&self) -> Vec<u64> {
        self.start
            .iter()
            .zip(self.end.iter())
            .map(|(start, end)| end.saturating_sub(*start))
            .collect()
    }

    /// True when any axis has zero extent.
    pub fn is_empty(&self) -> bool {
        self.start.iter().zip(self.end.iter()).any(|(s, e)| s >= e)
    }

    /// Validate rank and bounds against `shape`.
    pub fn validate(&self, shape: &[u64]) -> Result<()> {
        if self.start.len() != shape.len() || self.end.len() != shape.len() {
            return Err(Error::non_canonical(format!(
                "region rank ({}, {}) does not equal shape rank {}",
                self.start.len(),
                self.end.len(),
                shape.len()
            )));
        }
        for axis in 0..shape.len() {
            if self.start[axis] > self.end[axis] {
                return Err(Error::non_canonical(format!(
                    "region axis {axis} start {} is after end {}",
                    self.start[axis], self.end[axis]
                )));
            }
            if self.end[axis] > shape[axis] {
                return Err(Error::non_canonical(format!(
                    "region axis {axis} end {} exceeds shape extent {}",
                    self.end[axis], shape[axis]
                )));
            }
        }
        Ok(())
    }
}

/// Result of a region read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionRead {
    /// Shape of the returned region, in canonical row-major order.
    pub shape: Vec<u64>,
    /// Canonical row-major element bytes for the region.
    pub bytes: Vec<u8>,
    /// Physical units touched: shard slabs for [`PhysicalTensor::Sharded`],
    /// blocks for [`PhysicalTensor::Blocked`], and 1 for a non-empty
    /// single-buffer layout.
    pub units_touched: u64,
}

// ---------------------------------------------------------------------------
// Physical arrangements
// ---------------------------------------------------------------------------

/// An in-memory physical arrangement of one logical tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalTensor {
    RowMajor {
        dtype: TensorDtype,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    },
    ColumnMajor {
        dtype: TensorDtype,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    },
    Blocked {
        dtype: TensorDtype,
        shape: Vec<u64>,
        block_shape: Vec<u64>,
        /// Blocks in canonical block-grid order (row-major over the grid).
        blocks: Vec<Vec<u8>>,
    },
    Sharded {
        dtype: TensorDtype,
        shape: Vec<u64>,
        axis: u32,
        /// Shard payloads in index order.
        shards: Vec<Vec<u8>>,
    },
}

impl PhysicalTensor {
    /// Row-major arrangement from canonical bytes.
    pub fn row_major(
        dtype: TensorDtype,
        shape: &[u64],
        bytes: Vec<u8>,
        limits: &Limits,
    ) -> Result<Self> {
        check_payload(dtype, shape, &bytes, limits)?;
        Ok(PhysicalTensor::RowMajor {
            dtype,
            shape: shape.to_vec(),
            bytes,
        })
    }

    /// Column-major arrangement from column-major bytes.
    pub fn column_major(
        dtype: TensorDtype,
        shape: &[u64],
        bytes: Vec<u8>,
        limits: &Limits,
    ) -> Result<Self> {
        check_payload(dtype, shape, &bytes, limits)?;
        Ok(PhysicalTensor::ColumnMajor {
            dtype,
            shape: shape.to_vec(),
            bytes,
        })
    }

    /// Blocked arrangement from canonical block payloads.
    pub fn blocked(
        dtype: TensorDtype,
        shape: &[u64],
        block_shape: &[u64],
        blocks: Vec<Vec<u8>>,
        limits: &Limits,
    ) -> Result<Self> {
        let descriptor = LayoutDescriptor::blocked(dtype, shape, block_shape);
        descriptor.validate(limits)?;
        let element_count = validate_shape(shape, limits)?;
        crate::limits::check_len(
            payload_len(element_count, dtype)?,
            limits.max_decoded_bytes,
            "blocked tensor",
        )?;
        let extents = block_extents(shape, block_shape)?;
        if extents.len() != blocks.len() {
            return Err(Error::malformed(format!(
                "blocked tensor has {} block payloads but the grid needs {}",
                blocks.len(),
                extents.len()
            )));
        }
        for (index, (extent, payload)) in extents.iter().zip(blocks.iter()).enumerate() {
            let count = checked_product(extent)?;
            let expected = payload_len(count, dtype)?;
            if expected != payload.len() as u64 {
                return Err(Error::malformed(format!(
                    "block {index} is {} bytes but requires {expected}",
                    payload.len()
                )));
            }
        }
        Ok(PhysicalTensor::Blocked {
            dtype,
            shape: shape.to_vec(),
            block_shape: block_shape.to_vec(),
            blocks,
        })
    }

    /// Sharded arrangement from shard payloads in index order.
    pub fn sharded(
        dtype: TensorDtype,
        shape: &[u64],
        axis: u32,
        shards: Vec<Vec<u8>>,
        limits: &Limits,
    ) -> Result<Self> {
        validate_shape(shape, limits)?;
        let count = u32::try_from(shards.len())
            .map_err(|_| Error::limit_exceeded("shard count does not fit u32"))?;
        validate_shard_parameters(shape, axis, count)?;
        let shard_shape = shard_shape_for(shape, axis, count)?;
        let expected = payload_len(checked_product(&shard_shape)?, dtype)?;
        let total = expected
            .checked_mul(u64::from(count))
            .ok_or_else(|| Error::limit_exceeded("sharded tensor size overflows u64"))?;
        crate::limits::check_len(total, limits.max_decoded_bytes, "sharded tensor")?;
        for (index, payload) in shards.iter().enumerate() {
            if payload.len() as u64 != expected {
                return Err(Error::malformed(format!(
                    "shard {index} is {} bytes but requires {expected}",
                    payload.len()
                )));
            }
        }
        Ok(PhysicalTensor::Sharded {
            dtype,
            shape: shape.to_vec(),
            axis,
            shards,
        })
    }

    /// Element dtype.
    pub fn dtype(&self) -> TensorDtype {
        match self {
            PhysicalTensor::RowMajor { dtype, .. }
            | PhysicalTensor::ColumnMajor { dtype, .. }
            | PhysicalTensor::Blocked { dtype, .. }
            | PhysicalTensor::Sharded { dtype, .. } => *dtype,
        }
    }

    /// Logical shape.
    pub fn shape(&self) -> &[u64] {
        match self {
            PhysicalTensor::RowMajor { shape, .. }
            | PhysicalTensor::ColumnMajor { shape, .. }
            | PhysicalTensor::Blocked { shape, .. }
            | PhysicalTensor::Sharded { shape, .. } => shape,
        }
    }

    /// Reassemble canonical row-major logical bytes.
    pub fn assemble(&self) -> Result<Vec<u8>> {
        self.assemble_with_limits(&Limits::default())
    }

    /// Reassemble canonical row-major logical bytes under explicit limits.
    pub fn assemble_with_limits(&self, limits: &Limits) -> Result<Vec<u8>> {
        let esize = self.dtype().element_size() as usize;
        let count = checked_product(self.shape())?;
        let payload = payload_len(count, self.dtype())?;
        crate::limits::check_len(payload, limits.max_decoded_bytes, "assembled tensor")?;
        let total = usize::try_from(payload)
            .map_err(|_| Error::limit_exceeded("tensor payload does not fit usize"))?;
        match self {
            PhysicalTensor::RowMajor { bytes, .. } => Ok(bytes.clone()),
            PhysicalTensor::ColumnMajor { shape, bytes, .. } => {
                let mut out = vec![0u8; total];
                let mut position = 0usize;
                let stride_f = column_major_strides(shape)?;
                for_each_index(&vec![0; shape.len()], shape, |idx| {
                    let mut off: u64 = 0;
                    for k in 0..idx.len() {
                        off = off.saturating_add(idx[k].saturating_mul(stride_f[k]));
                    }
                    let source = off as usize * esize;
                    out[position..position + esize].copy_from_slice(&bytes[source..source + esize]);
                    position += esize;
                });
                Ok(out)
            }
            PhysicalTensor::Blocked {
                shape,
                block_shape,
                blocks,
                ..
            } => {
                let mut out = vec![0u8; total];
                let extents = block_extents(shape, block_shape)?;
                let grid = block_grid(shape, block_shape)?;
                for (block_index, extent) in extents.iter().enumerate() {
                    let coords = grid_coords(&grid, block_index);
                    let mut dst_start = vec![0u64; shape.len()];
                    for axis in 0..shape.len() {
                        dst_start[axis] = coords[axis].saturating_mul(block_shape[axis]);
                    }
                    copy_box_row_major_to_row_major(
                        &mut out,
                        shape,
                        &dst_start,
                        &blocks[block_index],
                        extent,
                        &vec![0u64; shape.len()],
                        extent,
                        esize,
                    )?;
                }
                Ok(out)
            }
            PhysicalTensor::Sharded {
                shape,
                axis,
                shards,
                ..
            } => {
                let mut out = vec![0u8; total];
                let shard_shape = shard_shape_for(shape, *axis, shards.len() as u32)?;
                let plane = shape[*axis as usize] / shards.len() as u64;
                for (index, payload) in shards.iter().enumerate() {
                    let mut dst_start = vec![0u64; shape.len()];
                    dst_start[*axis as usize] = index as u64 * plane;
                    copy_box_row_major_to_row_major(
                        &mut out,
                        shape,
                        &dst_start,
                        payload,
                        &shard_shape,
                        &vec![0u64; shape.len()],
                        &shard_shape,
                        esize,
                    )?;
                }
                Ok(out)
            }
        }
    }

    /// Canonical logical root of the assembled tensor.
    pub fn logical_root(&self, limits: &Limits) -> Result<Hash32> {
        let bytes = self.assemble_with_limits(limits)?;
        logical_root(self.dtype(), self.shape(), &bytes, limits)
    }

    /// Read a region, touching only the physical units it intersects.
    pub fn read_region(&self, region: &Region) -> Result<RegionRead> {
        self.read_region_with_limits(region, &Limits::default())
    }

    /// Read a region under explicit allocation limits.
    pub fn read_region_with_limits(&self, region: &Region, limits: &Limits) -> Result<RegionRead> {
        let shape = self.shape().to_vec();
        region.validate(&shape)?;
        let esize = self.dtype().element_size() as usize;
        let extent = region.extent();
        if region.is_empty() {
            return Ok(RegionRead {
                shape: extent,
                bytes: Vec::new(),
                units_touched: 0,
            });
        }
        let region_count = checked_product(&extent)?;
        let payload = payload_len(region_count, self.dtype())?;
        crate::limits::check_len(payload, limits.max_decoded_bytes, "region payload")?;
        let total = usize::try_from(payload)
            .map_err(|_| Error::limit_exceeded("region payload does not fit usize"))?;
        let mut out = vec![0u8; total];
        let zeros = vec![0u64; shape.len()];
        let units_touched = match self {
            PhysicalTensor::RowMajor { bytes, .. } => {
                copy_box_row_major_to_row_major(
                    &mut out,
                    &extent,
                    &zeros,
                    bytes,
                    &shape,
                    &region.start,
                    &extent,
                    esize,
                )?;
                1
            }
            PhysicalTensor::ColumnMajor { bytes, .. } => {
                copy_box_column_major_to_row_major(
                    &mut out,
                    &extent,
                    &zeros,
                    bytes,
                    &shape,
                    &region.start,
                    &extent,
                    esize,
                )?;
                1
            }
            PhysicalTensor::Blocked {
                block_shape,
                blocks,
                ..
            } => {
                let extents = block_extents(&shape, block_shape)?;
                let grid = block_grid(&shape, block_shape)?;
                let mut touched = 0u64;
                for (block_index, block_extent) in extents.iter().enumerate() {
                    let coords = grid_coords(&grid, block_index);
                    let mut block_start = vec![0u64; shape.len()];
                    for axis in 0..shape.len() {
                        block_start[axis] = coords[axis].saturating_mul(block_shape[axis]);
                    }
                    let mut inter_start = vec![0u64; shape.len()];
                    let mut inter_extent = vec![0u64; shape.len()];
                    let mut any = true;
                    for axis in 0..shape.len() {
                        let block_end = block_start[axis].saturating_add(block_extent[axis]);
                        let lo = region.start[axis].max(block_start[axis]);
                        let hi = region.end[axis].min(block_end);
                        if lo >= hi {
                            any = false;
                            break;
                        }
                        inter_start[axis] = lo;
                        inter_extent[axis] = hi - lo;
                    }
                    if !any {
                        continue;
                    }
                    touched += 1;
                    let mut src_start = vec![0u64; shape.len()];
                    let mut dst_start = vec![0u64; shape.len()];
                    for axis in 0..shape.len() {
                        src_start[axis] = inter_start[axis] - block_start[axis];
                        dst_start[axis] = inter_start[axis] - region.start[axis];
                    }
                    copy_box_row_major_to_row_major(
                        &mut out,
                        &extent,
                        &dst_start,
                        &blocks[block_index],
                        block_extent,
                        &src_start,
                        &inter_extent,
                        esize,
                    )?;
                }
                touched
            }
            PhysicalTensor::Sharded { axis, shards, .. } => {
                let axis_usize = *axis as usize;
                let plane = shape[axis_usize] / shards.len() as u64;
                let first = region.start[axis_usize] / plane;
                let last = (region.end[axis_usize] - 1) / plane;
                let shard_shape = shard_shape_for(&shape, *axis, shards.len() as u32)?;
                for shard_index in first..=last {
                    let payload = &shards[shard_index as usize];
                    let mut src_start = region.start.clone();
                    let mut dst_start = vec![0u64; shape.len()];
                    let mut inter_extent = extent.clone();
                    let slab_start = shard_index * plane;
                    let slab_end = slab_start + plane;
                    let lo = region.start[axis_usize].max(slab_start);
                    let hi = region.end[axis_usize].min(slab_end);
                    src_start[axis_usize] = lo - slab_start;
                    dst_start[axis_usize] = lo - region.start[axis_usize];
                    inter_extent[axis_usize] = hi - lo;
                    copy_box_row_major_to_row_major(
                        &mut out,
                        &extent,
                        &dst_start,
                        payload,
                        &shard_shape,
                        &src_start,
                        &inter_extent,
                        esize,
                    )?;
                }
                last - first + 1
            }
        };
        Ok(RegionRead {
            shape: extent,
            bytes: out,
            units_touched,
        })
    }
}

// ---------------------------------------------------------------------------
// Splitting a logical tensor into physical arrangements
// ---------------------------------------------------------------------------

/// Produce the column-major physical bytes of a canonical row-major tensor.
pub fn column_major_bytes(
    dtype: TensorDtype,
    shape: &[u64],
    row_major: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    check_payload(dtype, shape, row_major, limits)?;
    let esize = dtype.element_size() as usize;
    let mut out = vec![0u8; row_major.len()];
    let stride_f = column_major_strides(shape)?;
    let mut position = 0usize;
    for_each_index(&vec![0; shape.len()], shape, |idx| {
        let mut off: u64 = 0;
        for k in 0..idx.len() {
            off = off.saturating_add(idx[k].saturating_mul(stride_f[k]));
        }
        let target = off as usize * esize;
        out[target..target + esize].copy_from_slice(&row_major[position..position + esize]);
        position += esize;
    });
    Ok(out)
}

/// Block extents in canonical block-grid order.
pub fn block_extents(shape: &[u64], block_shape: &[u64]) -> Result<Vec<Vec<u64>>> {
    if shape.len() != block_shape.len() {
        return Err(Error::non_canonical(
            "block_shape rank must equal shape rank",
        ));
    }
    for block in block_shape {
        if *block == 0 {
            return Err(Error::non_canonical("block_shape dimensions must be >= 1"));
        }
    }
    let grid = block_grid(shape, block_shape)?;
    let grid_count = checked_product(&grid)?;
    let total = usize::try_from(grid_count)
        .map_err(|_| Error::limit_exceeded("block grid does not fit usize"))?;
    let mut extents = Vec::with_capacity(total);
    for block_index in 0..total {
        let coords = grid_coords(&grid, block_index);
        let mut extent = vec![0u64; shape.len()];
        for axis in 0..shape.len() {
            let start = coords[axis].saturating_mul(block_shape[axis]);
            let remaining = shape[axis].saturating_sub(start);
            extent[axis] = block_shape[axis].min(remaining);
        }
        extents.push(extent);
    }
    Ok(extents)
}

/// Split canonical row-major bytes into canonical block payloads.
pub fn blocked_bytes(
    dtype: TensorDtype,
    shape: &[u64],
    block_shape: &[u64],
    row_major: &[u8],
    limits: &Limits,
) -> Result<Vec<Vec<u8>>> {
    check_payload(dtype, shape, row_major, limits)?;
    let esize = dtype.element_size() as usize;
    let extents = block_extents(shape, block_shape)?;
    let grid = block_grid(shape, block_shape)?;
    let mut blocks = Vec::with_capacity(extents.len());
    for (block_index, extent) in extents.iter().enumerate() {
        let coords = grid_coords(&grid, block_index);
        let mut src_start = vec![0u64; shape.len()];
        for axis in 0..shape.len() {
            src_start[axis] = coords[axis].saturating_mul(block_shape[axis]);
        }
        let count = checked_product(extent)?;
        let size = usize::try_from(payload_len(count, dtype)?)
            .map_err(|_| Error::limit_exceeded("block payload does not fit usize"))?;
        let mut block = vec![0u8; size];
        copy_box_row_major_to_row_major(
            &mut block,
            extent,
            &vec![0u64; shape.len()],
            row_major,
            shape,
            &src_start,
            extent,
            esize,
        )?;
        blocks.push(block);
    }
    Ok(blocks)
}

/// Logical shape of shard `index` for a sharded layout.
pub fn shard_shape(shape: &[u64], axis: u32, shards: u32, index: u32) -> Result<Vec<u64>> {
    validate_shard_parameters(shape, axis, shards)?;
    if index >= shards {
        return Err(Error::non_canonical(format!(
            "shard index {index} is not below shard count {shards}"
        )));
    }
    shard_shape_for(shape, axis, shards)
}

fn shard_shape_for(shape: &[u64], axis: u32, shards: u32) -> Result<Vec<u64>> {
    validate_shard_parameters(shape, axis, shards)?;
    let mut shard = shape.to_vec();
    shard[axis as usize] /= u64::from(shards);
    Ok(shard)
}

/// Extract one shard's canonical row-major bytes.
pub fn shard_bytes(
    dtype: TensorDtype,
    shape: &[u64],
    axis: u32,
    shards: u32,
    index: u32,
    row_major: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    check_payload(dtype, shape, row_major, limits)?;
    let shard = shard_shape(shape, axis, shards, index)?;
    let esize = dtype.element_size() as usize;
    let plane = shape[axis as usize] / u64::from(shards);
    let mut src_start = vec![0u64; shape.len()];
    src_start[axis as usize] = u64::from(index) * plane;
    let count = checked_product(&shard)?;
    let size = usize::try_from(payload_len(count, dtype)?)
        .map_err(|_| Error::limit_exceeded("shard payload does not fit usize"))?;
    let mut out = vec![0u8; size];
    copy_box_row_major_to_row_major(
        &mut out,
        &shard,
        &vec![0u64; shape.len()],
        row_major,
        shape,
        &src_start,
        &shard,
        esize,
    )?;
    Ok(out)
}

/// Split canonical row-major bytes into all shards in index order.
pub fn split_shards(
    dtype: TensorDtype,
    shape: &[u64],
    axis: u32,
    shards: u32,
    row_major: &[u8],
    limits: &Limits,
) -> Result<Vec<Vec<u8>>> {
    check_payload(dtype, shape, row_major, limits)?;
    validate_shard_parameters(shape, axis, shards)?;
    (0..shards)
        .map(|index| shard_bytes(dtype, shape, axis, shards, index, row_major, limits))
        .collect()
}

/// Split a concatenated blocked container payload into canonical block payloads.
pub fn split_blocked_payload(
    dtype: TensorDtype,
    shape: &[u64],
    block_shape: &[u64],
    payload: &[u8],
    limits: &Limits,
) -> Result<Vec<Vec<u8>>> {
    check_payload(dtype, shape, payload, limits)?;
    let extents = block_extents(shape, block_shape)?;
    let mut blocks = Vec::with_capacity(extents.len());
    let mut position = 0usize;
    for extent in extents {
        let count = checked_product(&extent)?;
        let size = usize::try_from(payload_len(count, dtype)?)
            .map_err(|_| Error::limit_exceeded("block payload does not fit usize"))?;
        let end = position
            .checked_add(size)
            .ok_or_else(|| Error::limit_exceeded("block payload offset overflow"))?;
        if end > payload.len() {
            return Err(Error::malformed("blocked payload is truncated"));
        }
        blocks.push(payload[position..end].to_vec());
        position = end;
    }
    if position != payload.len() {
        return Err(Error::malformed("blocked payload has trailing bytes"));
    }
    Ok(blocks)
}

// ---------------------------------------------------------------------------
// Shard containers and reassembly
// ---------------------------------------------------------------------------

/// A decoded shard: its self-describing layout descriptor and payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedShard {
    pub descriptor: LayoutDescriptor,
    pub bytes: Vec<u8>,
}

/// Expected container payload length for a single descriptor.
pub fn descriptor_payload_len(descriptor: &LayoutDescriptor, limits: &Limits) -> Result<u64> {
    descriptor.validate(limits)?;
    let count = validate_shape(&descriptor.shape, limits)?;
    let elements = match &descriptor.layout {
        TensorLayout::RowMajor | TensorLayout::ColumnMajor | TensorLayout::Blocked { .. } => count,
        TensorLayout::Sharded { axis, shards, .. } => {
            checked_product(&shard_shape_for(&descriptor.shape, *axis, *shards)?)?
        }
    };
    payload_len(elements, descriptor.dtype)
}

/// `H("weft/tensor/shard-payload/v1", payload)`.
pub fn shard_payload_hash(payload: &[u8]) -> Hash32 {
    hash_domain(DOMAIN_TENSOR_SHARD_PAYLOAD, payload)
}

/// Encode a self-describing shard container.
///
/// Layout: `u32le(descriptor_len) || WCB(weft.tensor.shard.v1) || payload`.
pub fn encode_shard(
    descriptor: &LayoutDescriptor,
    payload: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    descriptor.validate(limits)?;
    let expected = descriptor_payload_len(descriptor, limits)?;
    if expected != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "shard payload is {} bytes but the descriptor requires {expected}",
            payload.len()
        )));
    }
    let descriptor_bytes = wcb::encode(
        &Value::Map(shard_map(descriptor, expected, shard_payload_hash(payload))),
        limits,
    )?;
    let length = u32::try_from(descriptor_bytes.len())
        .map_err(|_| Error::limit_exceeded("shard descriptor does not fit u32"))?;
    let mut out = Vec::with_capacity(4 + descriptor_bytes.len() + payload.len());
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&descriptor_bytes);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Decode a shard container, verifying descriptor, length, and payload hash.
pub fn decode_shard(bytes: &[u8], limits: &Limits) -> Result<DecodedShard> {
    if bytes.len() < 4 {
        return Err(Error::malformed(
            "shard container is shorter than its 4-byte descriptor length",
        ));
    }
    let length = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let end = 4usize
        .checked_add(length)
        .ok_or_else(|| Error::malformed("shard descriptor length overflow"))?;
    if end > bytes.len() {
        return Err(Error::malformed(
            "shard descriptor extends past the container",
        ));
    }
    let descriptor_bytes = &bytes[4..end];
    let payload = &bytes[end..];
    let value = wcb::decode(descriptor_bytes, limits)?;
    let map = require_map(&value, "shard descriptor")?;
    check_allowed_keys(
        map,
        &[
            "schema",
            "dtype",
            "shape",
            "layout",
            "block_shape",
            "axis",
            "shards",
            "index",
            "payload_bytes",
            "payload_hash",
        ],
        "shard descriptor",
    )?;
    let schema = require_text(map, "schema", "shard descriptor")?;
    if schema != SHARD_SCHEMA {
        return Err(Error::unknown_schema(format!(
            "unknown shard descriptor schema `{schema}`"
        )));
    }
    let descriptor = descriptor_from_map(map, limits)?;
    let declared_bytes = require_u64(map, "payload_bytes", "shard descriptor")?;
    let expected = descriptor_payload_len(&descriptor, limits)?;
    if declared_bytes != expected {
        return Err(Error::non_canonical(format!(
            "shard payload_bytes {declared_bytes} does not match layout length {expected}"
        )));
    }
    if declared_bytes != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "shard payload is {} bytes but the descriptor declares {declared_bytes}",
            payload.len()
        )));
    }
    let declared_hash = require_hash(map, "payload_hash", "shard descriptor")?;
    if declared_hash != shard_payload_hash(payload) {
        return Err(Error::corrupt("shard payload_hash mismatch"));
    }
    let reencoded = wcb::encode(
        &Value::Map(shard_map(&descriptor, declared_bytes, declared_hash)),
        limits,
    )?;
    if reencoded != descriptor_bytes {
        return Err(Error::non_canonical(
            "shard descriptor bytes are not in canonical form",
        ));
    }
    Ok(DecodedShard {
        descriptor,
        bytes: payload.to_vec(),
    })
}

/// Verify a complete shard set and return the logical tensor root.
///
/// Shard order is not significant: the `index` field is authoritative and
/// shards are reordered deterministically by it (both languages agree).
/// Duplicate, missing, out-of-range, and inconsistent shards are rejected.
pub fn shard_root(shards: &[DecodedShard], limits: &Limits) -> Result<Hash32> {
    if shards.is_empty() {
        return Err(Error::malformed("shard set is empty"));
    }
    let (axis, count) = match &shards[0].descriptor.layout {
        TensorLayout::Sharded { axis, shards, .. } => (*axis, *shards),
        other => {
            return Err(Error::malformed(format!(
                "shard root requires sharded descriptors, got `{}`",
                other.name()
            )))
        }
    };
    if count == 0 {
        return Err(Error::non_canonical("shard count must be at least 1"));
    }
    let n = count as usize;
    if shards.len() != n {
        return Err(Error::malformed(format!(
            "shard set has {} entries but the descriptor declares {count}",
            shards.len()
        )));
    }
    let dtype = shards[0].descriptor.dtype;
    let shape = shards[0].descriptor.shape.clone();
    let mut ordered: Vec<Option<&[u8]>> = vec![None; n];
    for shard in shards {
        let descriptor = &shard.descriptor;
        let (shard_axis, shard_count, index) = match descriptor.layout {
            TensorLayout::Sharded {
                axis,
                shards,
                index,
            } => (axis, shards, index),
            ref other => {
                return Err(Error::malformed(format!(
                    "shard root requires sharded descriptors, got `{}`",
                    other.name()
                )))
            }
        };
        if descriptor.dtype != dtype
            || descriptor.shape != shape
            || shard_axis != axis
            || shard_count != count
        {
            return Err(Error::non_canonical(
                "shard descriptors disagree on dtype, shape, axis, or shard count",
            ));
        }
        descriptor.validate(limits)?;
        let index_usize = index as usize;
        if index_usize >= n {
            return Err(Error::non_canonical(format!(
                "shard index {index} is not below shard count {count}"
            )));
        }
        if ordered[index_usize].is_some() {
            return Err(Error::non_canonical(format!(
                "duplicate shard index {index}"
            )));
        }
        let expected = payload_len(
            checked_product(&shard_shape_for(&shape, axis, count)?)?,
            dtype,
        )?;
        if shard.bytes.len() as u64 != expected {
            return Err(Error::malformed(format!(
                "shard {index} is {} bytes but requires {expected}",
                shard.bytes.len()
            )));
        }
        ordered[index_usize] = Some(&shard.bytes);
    }
    for (index, entry) in ordered.iter().enumerate() {
        if entry.is_none() {
            return Err(Error::non_canonical(format!("missing shard index {index}")));
        }
    }
    let payloads: Vec<Vec<u8>> = ordered
        .into_iter()
        .map(|entry| entry.expect("checked above").to_vec())
        .collect();
    let physical = PhysicalTensor::sharded(dtype, &shape, axis, payloads, limits)?;
    physical.logical_root(limits)
}

/// Reassemble a shard set to canonical row-major bytes.
pub fn assemble_shards(shards: &[DecodedShard], limits: &Limits) -> Result<Vec<u8>> {
    if shards.is_empty() {
        return Err(Error::malformed("shard set is empty"));
    }
    let (axis, count) = match &shards[0].descriptor.layout {
        TensorLayout::Sharded { axis, shards, .. } => (*axis, *shards),
        other => {
            return Err(Error::malformed(format!(
                "shard reassembly requires sharded descriptors, got `{}`",
                other.name()
            )))
        }
    };
    let n = count as usize;
    let mut ordered: Vec<Option<&[u8]>> = vec![None; n];
    for shard in shards {
        if let TensorLayout::Sharded { index, .. } = shard.descriptor.layout {
            let index = index as usize;
            if index >= n || ordered[index].is_some() {
                return Err(Error::non_canonical(
                    "shard indices are out of range or duplicated",
                ));
            }
            ordered[index] = Some(&shard.bytes);
        } else {
            return Err(Error::malformed(
                "shard reassembly requires shard descriptors",
            ));
        }
    }
    let payloads: Vec<Vec<u8>> = ordered
        .into_iter()
        .map(|entry| {
            entry
                .map(|bytes| bytes.to_vec())
                .ok_or_else(|| Error::non_canonical("shard set is missing an index"))
        })
        .collect::<Result<Vec<_>>>()?;
    PhysicalTensor::sharded(
        shards[0].descriptor.dtype,
        &shards[0].descriptor.shape,
        axis,
        payloads,
        limits,
    )?
    .assemble()
}

// ---------------------------------------------------------------------------
// Copy and index helpers
// ---------------------------------------------------------------------------

fn checked_product(values: &[u64]) -> Result<u64> {
    let mut product: u64 = 1;
    for value in values {
        product = product
            .checked_mul(*value)
            .ok_or_else(|| Error::limit_exceeded("tensor dimension product overflows u64"))?;
    }
    Ok(product)
}

fn row_major_strides(shape: &[u64]) -> Result<Vec<u64>> {
    let mut strides = vec![1u64; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1]
            .checked_mul(shape[axis + 1])
            .ok_or_else(|| Error::limit_exceeded("tensor stride overflows u64"))?;
    }
    Ok(strides)
}

fn column_major_strides(shape: &[u64]) -> Result<Vec<u64>> {
    let mut strides = vec![1u64; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1]
            .checked_mul(shape[axis - 1])
            .ok_or_else(|| Error::limit_exceeded("tensor stride overflows u64"))?;
    }
    Ok(strides)
}

/// Visit every multi-index in the box `[start, start+extent)` in row-major order.
///
/// A rank-0 box is a single point and calls `f` exactly once.
fn for_each_index<F: FnMut(&[u64])>(start: &[u64], ext: &[u64], mut f: F) {
    if ext.iter().any(|value| *value == 0) {
        return;
    }
    let rank = start.len();
    let mut index = start.to_vec();
    if index.is_empty() {
        f(&[]);
        return;
    }
    loop {
        f(&index);
        let mut axis = rank;
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < start[axis] + ext[axis] {
                break;
            }
            index[axis] = start[axis];
            if axis == 0 {
                return;
            }
        }
    }
}

/// Copy a box between two row-major buffers.
#[allow(clippy::too_many_arguments)]
fn copy_box_row_major_to_row_major(
    dst: &mut [u8],
    dst_shape: &[u64],
    dst_start: &[u64],
    src: &[u8],
    src_shape: &[u64],
    src_start: &[u64],
    ext: &[u64],
    esize: usize,
) -> Result<()> {
    let rank = ext.len();
    if rank == 0 || ext.iter().any(|value| *value == 0) {
        return Ok(());
    }
    let dst_strides = row_major_strides(dst_shape)?;
    let src_strides = row_major_strides(src_shape)?;
    let prefix = rank - 1;
    let run = usize::try_from(ext[prefix])
        .map_err(|_| Error::limit_exceeded("run length does not fit usize"))?
        * esize;
    let zeros = vec![0u64; prefix];
    let mut failure: Option<Error> = None;
    for_each_index(&zeros, &ext[..prefix], |local| {
        if failure.is_some() {
            return;
        }
        let mut dst_off: u64 = dst_start[prefix];
        let mut src_off: u64 = src_start[prefix];
        for axis in 0..prefix {
            let dst_index = dst_start[axis].saturating_add(local[axis]);
            let src_index = src_start[axis].saturating_add(local[axis]);
            dst_off = dst_off.saturating_add(dst_index.saturating_mul(dst_strides[axis]));
            src_off = src_off.saturating_add(src_index.saturating_mul(src_strides[axis]));
        }
        let dst_byte = dst_off as usize * esize;
        let src_byte = src_off as usize * esize;
        let dst_end = dst_byte.saturating_add(run);
        let src_end = src_byte.saturating_add(run);
        match (dst.get_mut(dst_byte..dst_end), src.get(src_byte..src_end)) {
            (Some(target), Some(source)) => target.copy_from_slice(source),
            _ => {
                failure = Some(Error::malformed(
                    "internal tensor copy exceeded its buffers",
                ))
            }
        }
    });
    match failure {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// Copy a box from a column-major buffer into a row-major buffer.
#[allow(clippy::too_many_arguments)]
fn copy_box_column_major_to_row_major(
    dst: &mut [u8],
    dst_shape: &[u64],
    dst_start: &[u64],
    src: &[u8],
    src_shape: &[u64],
    src_start: &[u64],
    ext: &[u64],
    esize: usize,
) -> Result<()> {
    let rank = ext.len();
    if rank == 0 || ext.iter().any(|value| *value == 0) {
        return Ok(());
    }
    let dst_strides = row_major_strides(dst_shape)?;
    let src_strides = column_major_strides(src_shape)?;
    let mut failure: Option<Error> = None;
    for_each_index(&dst_start, ext, |index| {
        if failure.is_some() {
            return;
        }
        let mut local = vec![0u64; rank];
        for axis in 0..rank {
            local[axis] = index[axis] - dst_start[axis];
        }
        let mut dst_off: u64 = 0;
        for axis in 0..rank {
            dst_off = dst_off.saturating_add(index[axis].saturating_mul(dst_strides[axis]));
        }
        let mut src_off: u64 = 0;
        for axis in 0..rank {
            let logical = src_start[axis].saturating_add(local[axis]);
            src_off = src_off.saturating_add(logical.saturating_mul(src_strides[axis]));
        }
        let dst_byte = dst_off as usize * esize;
        let src_byte = src_off as usize * esize;
        match (
            dst.get_mut(dst_byte..dst_byte.saturating_add(esize)),
            src.get(src_byte..src_byte.saturating_add(esize)),
        ) {
            (Some(target), Some(source)) => target.copy_from_slice(source),
            _ => {
                failure = Some(Error::malformed(
                    "internal tensor copy exceeded its buffers",
                ))
            }
        }
    });
    match failure {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

fn block_grid(shape: &[u64], block_shape: &[u64]) -> Result<Vec<u64>> {
    if shape.len() != block_shape.len() {
        return Err(Error::non_canonical(
            "block_shape rank must equal shape rank",
        ));
    }
    let mut grid = Vec::with_capacity(shape.len());
    for axis in 0..shape.len() {
        let block = block_shape[axis];
        if block == 0 {
            return Err(Error::non_canonical("block_shape dimensions must be >= 1"));
        }
        grid.push(shape[axis] / block + u64::from(shape[axis] % block != 0));
    }
    Ok(grid)
}

fn grid_coords(grid: &[u64], mut index: usize) -> Vec<u64> {
    let mut coords = vec![0u64; grid.len()];
    for axis in (0..grid.len()).rev() {
        let extent = grid[axis] as usize;
        coords[axis] = (index % extent) as u64;
        index /= extent;
    }
    coords
}

// ---------------------------------------------------------------------------
// WCB field accessors
// ---------------------------------------------------------------------------

fn require_map<'a>(value: &'a Value, what: &str) -> Result<&'a BTreeMap<String, Value>> {
    match value {
        Value::Map(map) => Ok(map),
        _ => Err(Error::non_canonical(format!("{what} must be a WCB map"))),
    }
}

fn check_allowed_keys(map: &BTreeMap<String, Value>, allowed: &[&str], what: &str) -> Result<()> {
    for key in map.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::non_canonical(format!(
                "{what} has unknown key `{key}`"
            )));
        }
    }
    Ok(())
}

fn require_text<'a>(map: &'a BTreeMap<String, Value>, key: &str, what: &str) -> Result<&'a str> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.as_str()),
        _ => Err(Error::non_canonical(format!("{what} `{key}` must be text"))),
    }
}

fn require_u64(map: &BTreeMap<String, Value>, key: &str, what: &str) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::non_canonical(format!("{what} `{key}` must be u64"))),
    }
}

fn require_u32(map: &BTreeMap<String, Value>, key: &str, what: &str) -> Result<u32> {
    let value = require_u64(map, key, what)?;
    u32::try_from(value)
        .map_err(|_| Error::non_canonical(format!("{what} `{key}` must fit u32, got {value}")))
}

fn require_u64_array(map: &BTreeMap<String, Value>, key: &str, what: &str) -> Result<Vec<u64>> {
    match map.get(key) {
        Some(Value::Array(items)) => {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(value) => values.push(*value),
                    _ => {
                        return Err(Error::non_canonical(format!(
                            "{what} `{key}` elements must be u64"
                        )))
                    }
                }
            }
            Ok(values)
        }
        _ => Err(Error::non_canonical(format!(
            "{what} `{key}` must be an array"
        ))),
    }
}

fn require_hash(map: &BTreeMap<String, Value>, key: &str, what: &str) -> Result<Hash32> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) => Hash32::from_slice(bytes)
            .map_err(|_| Error::non_canonical(format!("{what} `{key}` must be 32 bytes"))),
        _ => Err(Error::non_canonical(format!(
            "{what} `{key}` must be bytes"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    fn fixture() -> (TensorDtype, Vec<u64>, Vec<u8>) {
        // Rank-3 bf16, shape (4,3,2): axis 0 is divisible by 4 for sharding.
        let shape = vec![4, 3, 2];
        let bytes = pattern(4 * 3 * 2 * 2, 0x5eed);
        (TensorDtype::Bf16, shape, bytes)
    }

    fn slice_region(logical: &[u8], shape: &[u64], region: &Region) -> Vec<u8> {
        let esize = 2usize;
        let extent = region.extent();
        let count = checked_product(&extent).unwrap() as usize;
        let mut out = vec![0u8; count * esize];
        let zeros = vec![0u64; shape.len()];
        copy_box_row_major_to_row_major(
            &mut out,
            &extent,
            &zeros,
            logical,
            shape,
            &region.start,
            &extent,
            esize,
        )
        .unwrap();
        out
    }

    #[test]
    fn four_layouts_produce_one_logical_root() {
        let limits = Limits::default();
        let (dtype, shape, row_major) = fixture();
        let expected = logical_root(dtype, &shape, &row_major, &limits).unwrap();

        // (a) row-major
        let row = PhysicalTensor::row_major(dtype, &shape, row_major.clone(), &limits).unwrap();
        assert_eq!(row.logical_root(&limits).unwrap(), expected);

        // (b) column-major
        let column = column_major_bytes(dtype, &shape, &row_major, &limits).unwrap();
        let column_physical = PhysicalTensor::column_major(dtype, &shape, column, &limits).unwrap();
        assert_eq!(column_physical.assemble().unwrap(), row_major);
        assert_eq!(column_physical.logical_root(&limits).unwrap(), expected);

        // (c) blocked 2x2x2
        let block_shape = vec![2, 2, 2];
        let blocks = blocked_bytes(dtype, &shape, &block_shape, &row_major, &limits).unwrap();
        let blocked =
            PhysicalTensor::blocked(dtype, &shape, &block_shape, blocks, &limits).unwrap();
        assert_eq!(blocked.assemble().unwrap(), row_major);
        assert_eq!(blocked.logical_root(&limits).unwrap(), expected);

        // (d) sharded 4 ways on axis 0
        let shards = split_shards(dtype, &shape, 0, 4, &row_major, &limits).unwrap();
        let sharded = PhysicalTensor::sharded(dtype, &shape, 0, shards.clone(), &limits).unwrap();
        assert_eq!(sharded.assemble().unwrap(), row_major);
        assert_eq!(sharded.logical_root(&limits).unwrap(), expected);

        // shard_root from containers must agree.
        let decoded: Vec<DecodedShard> = shards
            .iter()
            .enumerate()
            .map(|(index, payload)| {
                let descriptor = LayoutDescriptor::sharded(dtype, &shape, 0, 4, index as u32);
                let container = encode_shard(&descriptor, payload, &limits).unwrap();
                decode_shard(&container, &limits).unwrap()
            })
            .collect();
        assert_eq!(shard_root(&decoded, &limits).unwrap(), expected);
    }

    #[test]
    fn region_reads_match_the_logical_slice() {
        let limits = Limits::default();
        let (dtype, shape, row_major) = fixture();
        let esize = dtype.element_size() as usize;

        let column = PhysicalTensor::column_major(
            dtype,
            &shape,
            column_major_bytes(dtype, &shape, &row_major, &limits).unwrap(),
            &limits,
        )
        .unwrap();
        let block_shape = vec![2, 2, 2];
        let blocked = PhysicalTensor::blocked(
            dtype,
            &shape,
            &block_shape,
            blocked_bytes(dtype, &shape, &block_shape, &row_major, &limits).unwrap(),
            &limits,
        )
        .unwrap();
        let sharded = PhysicalTensor::sharded(
            dtype,
            &shape,
            0,
            split_shards(dtype, &shape, 0, 4, &row_major, &limits).unwrap(),
            &limits,
        )
        .unwrap();
        let row = PhysicalTensor::row_major(dtype, &shape, row_major.clone(), &limits).unwrap();

        let regions = [
            Region::full(&shape),
            Region::new(vec![0, 0, 0], vec![1, 1, 1]), // single element
            Region::new(vec![3, 2, 1], vec![4, 3, 2]), // corner element
            Region::new(vec![0, 0, 1], vec![4, 3, 2]), // edge slab
            Region::new(vec![1, 0, 0], vec![3, 3, 2]), // interior
        ];
        for region in &regions {
            let expected = slice_region(&row_major, &shape, region);
            for physical in [&row, &column, &blocked, &sharded] {
                let read = physical.read_region(region).unwrap();
                assert_eq!(read.bytes, expected, "region {region:?}");
                assert_eq!(read.shape, region.extent());
            }
        }

        // A region inside one shard touches exactly one shard.
        let one = Region::new(vec![0, 0, 0], vec![1, 3, 2]);
        assert_eq!(sharded.read_region(&one).unwrap().units_touched, 1);
        let two = Region::new(vec![0, 0, 0], vec![2, 3, 2]);
        assert_eq!(sharded.read_region(&two).unwrap().units_touched, 2);
        let all = Region::full(&shape);
        assert_eq!(sharded.read_region(&all).unwrap().units_touched, 4);

        // Blocked regions count intersecting blocks.
        let single = Region::new(vec![0, 0, 0], vec![1, 1, 1]);
        assert_eq!(blocked.read_region(&single).unwrap().units_touched, 1);
        assert_eq!(blocked.read_region(&all).unwrap().units_touched, 4);
        assert!(blocked.read_region(&all).unwrap().bytes.len() == row_major.len());
        let _ = esize;
    }

    #[test]
    fn shuffled_shards_are_reordered_by_index() {
        let limits = Limits::default();
        let (dtype, shape, row_major) = fixture();
        let shards = split_shards(dtype, &shape, 0, 4, &row_major, &limits).unwrap();
        let mut decoded: Vec<DecodedShard> = shards
            .iter()
            .enumerate()
            .map(|(index, payload)| DecodedShard {
                descriptor: LayoutDescriptor::sharded(dtype, &shape, 0, 4, index as u32),
                bytes: payload.clone(),
            })
            .collect();
        decoded.reverse();
        let expected = logical_root(dtype, &shape, &row_major, &limits).unwrap();
        assert_eq!(shard_root(&decoded, &limits).unwrap(), expected);
    }

    #[test]
    fn duplicate_missing_and_inconsistent_shards_are_rejected() {
        let limits = Limits::default();
        let (dtype, shape, row_major) = fixture();
        let shards = split_shards(dtype, &shape, 0, 4, &row_major, &limits).unwrap();
        let make = |index: u32, payload: &[u8]| DecodedShard {
            descriptor: LayoutDescriptor::sharded(dtype, &shape, 0, 4, index),
            bytes: payload.to_vec(),
        };
        let good: Vec<DecodedShard> = shards
            .iter()
            .enumerate()
            .map(|(index, payload)| make(index as u32, payload))
            .collect();
        assert!(shard_root(&good, &limits).is_ok());

        let mut duplicate = good.clone();
        duplicate[3] = make(0, &shards[0]);
        assert_eq!(
            shard_root(&duplicate, &limits).unwrap_err().class(),
            crate::ErrorClass::NonCanonical
        );

        let missing = &good[..3];
        assert_eq!(
            shard_root(missing, &limits).unwrap_err().class(),
            crate::ErrorClass::Malformed
        );

        let mut wrong_shape = good.clone();
        wrong_shape[1].descriptor.shape = vec![2, 3, 4];
        assert!(shard_root(&wrong_shape, &limits).is_err());

        let mut wrong_len = good.clone();
        wrong_len[2].bytes.pop();
        assert!(shard_root(&wrong_len, &limits).is_err());
    }

    #[test]
    fn shape_and_layout_rejects() {
        let limits = Limits::default();
        let default = Limits::default();
        assert_eq!(
            validate_shape(&[], &default).unwrap_err().class(),
            crate::ErrorClass::NonCanonical
        );
        assert_eq!(
            validate_shape(&[2, 0, 3], &default).unwrap_err().class(),
            crate::ErrorClass::NonCanonical
        );
        assert_eq!(
            validate_shape(&[4_294_967_296, 4_294_967_296, 4_294_967_296], &default)
                .unwrap_err()
                .class(),
            crate::ErrorClass::LimitExceeded
        );
        assert_eq!(
            validate_shape(&[100_000, 100_000], &default)
                .unwrap_err()
                .class(),
            crate::ErrorClass::LimitExceeded
        );
        assert_eq!(
            validate_shape(&[1, 1, 1, 1, 1, 1, 1, 1, 1], &default)
                .unwrap_err()
                .class(),
            crate::ErrorClass::LimitExceeded
        );

        // Element byte length mismatch.
        assert_eq!(
            check_payload(TensorDtype::Bf16, &[2, 3], &[0u8; 11], &limits)
                .unwrap_err()
                .class(),
            crate::ErrorClass::Malformed
        );

        // Shard parameters.
        assert!(
            LayoutDescriptor::sharded(TensorDtype::Bf16, &[2, 3, 4], 0, 3, 0)
                .validate(&limits)
                .is_err()
        );
        assert!(
            LayoutDescriptor::sharded(TensorDtype::Bf16, &[4, 3, 2], 0, 4, 4)
                .validate(&limits)
                .is_err()
        );
        assert!(
            LayoutDescriptor::sharded(TensorDtype::Bf16, &[4, 3, 2], 3, 1, 0)
                .validate(&limits)
                .is_err()
        );
        assert!(
            LayoutDescriptor::sharded(TensorDtype::Bf16, &[4, 3, 2], 0, 0, 0)
                .validate(&limits)
                .is_err()
        );
        assert!(
            LayoutDescriptor::blocked(TensorDtype::Bf16, &[4, 3, 2], &[2, 2])
                .validate(&limits)
                .is_err()
        );
        assert!(
            LayoutDescriptor::blocked(TensorDtype::Bf16, &[4, 3, 2], &[2, 2, 0])
                .validate(&limits)
                .is_err()
        );
        assert!(LayoutDescriptor::column_major(TensorDtype::Bf16, &[])
            .validate(&limits)
            .is_err());
    }

    #[test]
    fn logical_record_round_trips() {
        let limits = Limits::default();
        let (dtype, shape, row_major) = fixture();
        let record = LogicalRecord::from_row_major(dtype, &shape, &row_major, &limits).unwrap();
        let bytes = record.canonical_bytes().unwrap();
        let decoded = LogicalRecord::from_canonical_bytes(&bytes, &limits).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(decoded.root().unwrap(), record.root().unwrap());

        let mut tampered = bytes.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xff;
        assert!(LogicalRecord::from_canonical_bytes(&tampered, &limits).is_err());
    }

    #[test]
    fn shard_container_round_trips_and_binds_payload() {
        let limits = Limits::default();
        let (dtype, shape, row_major) = fixture();
        let payload = shard_bytes(dtype, &shape, 0, 4, 2, &row_major, &limits).unwrap();
        let descriptor = LayoutDescriptor::sharded(dtype, &shape, 0, 4, 2);
        let container = encode_shard(&descriptor, &payload, &limits).unwrap();
        let decoded = decode_shard(&container, &limits).unwrap();
        assert_eq!(decoded.descriptor, descriptor);
        assert_eq!(decoded.bytes, payload);

        let mut corrupted = container.clone();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0x01;
        assert_eq!(
            decode_shard(&corrupted, &limits).unwrap_err().class(),
            crate::ErrorClass::Corrupt
        );

        let mut truncated = container;
        truncated.pop();
        assert!(decode_shard(&truncated, &limits).is_err());
    }
}
