//! Mode 20 `ENGRAM_SHARD` — incrementally updateable, region-addressable
//! row-group tables (WEFT-AI-CODEC §21, `spec/WEFT-ENGRAM.md`).
//!
//! A rank-2 table is stored as ordered row groups. Each group is split into
//! byte planes and each plane is coded with the frozen
//! `huffman-canonical-v1` coder, exactly as mode 15 codes a tile. A group can
//! be **stored inline** or emitted as a **reference** to an earlier artifact's
//! group root; the latter is what makes an incremental update write only the
//! changed groups.
//!
//! Decoding is exact: a full decode verifies every group root, every
//! supersession edge, the stored payload hash and the whole-table content hash.
//! A group or row-region read verifies only the roots of the groups it touches
//! and deliberately does **not** check the payload hash, frame trailer or
//! whole-table content hash (the same caveat as the mode-15 prefix reader).

use std::collections::{BTreeMap, BTreeSet};

use crate::dependency::Resolver;
use crate::error::{Error, Result};
use crate::frame::{
    encode_frame, Dependency, Frame, FrameParts, Mode, DEP_DOMAIN_PAYLOAD, DOMAIN_PAYLOAD,
    FORMAT_MAJOR, FORMAT_MINOR,
};
use crate::hash::{
    hash_domain, Hash32, DOMAIN_CHUNK, DOMAIN_ENGRAM, DOMAIN_ENGRAM_GROUP, DOMAIN_ENGRAM_PAYLOAD,
};
use crate::huffman;
use crate::limits::Limits;
use crate::tensor::{self, TensorDtype};
use crate::wcb::{self, Value};

/// Canonical engram descriptor schema.
pub const ENGRAM_SCHEMA: &str = "weft.ai.engram_shard.v1";
/// The only group plane coder implemented in v1.
pub const ENGRAM_GROUP_CODEC: &str = huffman::CODER_ID;
/// The only table layout (canonical row-major).
pub const ENGRAM_LAYOUT: &str = "row_major";
/// Hash-function registry: SHA-256 n-gram keys.
pub const ENGRAM_HASH_SHA256: &str = "ngram-sha256-v1";
/// Hash-function registry: BLAKE3 n-gram keys.
pub const ENGRAM_HASH_BLAKE3: &str = "ngram-blake3-v1";

/// A declared (and required-lossless) quantization block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramQuantization {
    pub scheme: String,
    pub lossless: bool,
}

/// Exact storage precision of a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramPrecision {
    /// Must equal the descriptor's `element_dtype` in v1.
    pub storage_dtype: TensorDtype,
    /// Optional declared transformation. Only `scheme = "none"` with
    /// `lossless = true` is accepted; anything else is `UnsupportedFeature`.
    pub quantization: Option<EngramQuantization>,
}

impl EngramPrecision {
    pub fn exact(dtype: TensorDtype) -> Self {
        EngramPrecision {
            storage_dtype: dtype,
            quantization: None,
        }
    }
}

/// A group emitted as a reference to an earlier artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngramReference {
    /// The group root the reference must resolve to.
    pub group_root: Hash32,
    /// Index into the frame dependency table naming the artifact.
    pub dependency: u64,
}

/// How one group's bytes are obtained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngramStorage {
    /// Stored in this frame's payload at `offset`, `compressed_len` bytes.
    Inline { offset: u64, compressed_len: u64 },
    /// Referenced from the frame dependency named by `dependency`.
    Reference { dependency: u64 },
}

/// One row group in a mode-20 descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramGroup {
    pub index: u64,
    pub row_start: u64,
    pub row_end: u64,
    pub storage: EngramStorage,
    pub group_root: Hash32,
    /// The group root this group replaces, or `None`.
    pub supersedes: Option<Hash32>,
}

/// A parsed mode-20 descriptor (without the payload bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramMeta {
    pub hash_function: String,
    pub vocab_size: u64,
    pub table_shape: Vec<u64>,
    pub dtype: TensorDtype,
    pub precision: EngramPrecision,
    pub row_group_edges: Vec<u64>,
    pub payload_len: u64,
    pub payload_hash: Hash32,
    pub groups: Vec<EngramGroup>,
}

impl EngramMeta {
    pub fn rows(&self) -> u64 {
        self.table_shape[0]
    }
    pub fn dim(&self) -> u64 {
        self.table_shape[1]
    }
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
    pub fn element_count(&self) -> u64 {
        self.rows().saturating_mul(self.dim())
    }
    pub fn decoded_len(&self) -> Result<u64> {
        self.element_count()
            .checked_mul(self.dtype.element_size())
            .ok_or_else(|| Error::limit_exceeded("engram decoded bytes overflow u64"))
    }
    /// `H("weft/engram/v1", WCB(descriptor))`.
    pub fn descriptor_root(&self, limits: &Limits) -> Result<Hash32> {
        Ok(hash_domain(
            DOMAIN_ENGRAM,
            &wcb::encode(&meta_value(self), limits)?,
        ))
    }
}

/// A parsed mode-20 frame: descriptor plus the stored payload slice.
#[derive(Clone, Debug)]
pub struct EngramDescriptor<'a> {
    pub meta: EngramMeta,
    pub payload: &'a [u8],
}

/// Input to [`encode_engram`] / [`encode_engram_with_dependencies`].
#[derive(Clone, Debug)]
pub struct EngramInput {
    pub hash_function: String,
    pub vocab_size: u64,
    pub table_shape: Vec<u64>,
    pub dtype: TensorDtype,
    pub precision: EngramPrecision,
    /// Strictly increasing row edges: first 0, last `rows`.
    pub row_group_edges: Vec<u64>,
    /// Groups to emit as references instead of storing: group index -> root
    /// and dependency index.
    pub references: BTreeMap<u64, EngramReference>,
    /// One entry per group; an empty vector means "all null".
    pub supersedes: Vec<Option<Hash32>>,
}

impl EngramInput {
    /// A fully inline table with no supersession edges.
    pub fn inline(
        dtype: TensorDtype,
        table_shape: Vec<u64>,
        row_group_edges: Vec<u64>,
        hash_function: impl Into<String>,
        vocab_size: u64,
    ) -> Self {
        EngramInput {
            hash_function: hash_function.into(),
            vocab_size,
            table_shape,
            dtype,
            precision: EngramPrecision::exact(dtype),
            row_group_edges,
            references: BTreeMap::new(),
            supersedes: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Small WCB / body helpers
// ---------------------------------------------------------------------------

fn ai_map<'a>(value: &'a Value, what: &str) -> Result<&'a BTreeMap<String, Value>> {
    match value {
        Value::Map(map) => Ok(map),
        _ => Err(Error::malformed(format!("{what} must be a WCB map"))),
    }
}

fn check_exact_keys(map: &BTreeMap<String, Value>, keys: &[&str], what: &str) -> Result<()> {
    for key in keys {
        if !map.contains_key(*key) {
            return Err(Error::malformed(format!("{what} is missing key `{key}`")));
        }
    }
    for key in map.keys() {
        if !keys.contains(&key.as_str()) {
            return Err(Error::malformed(format!("{what} has unknown key `{key}`")));
        }
    }
    Ok(())
}

fn meta_text<'a>(map: &'a BTreeMap<String, Value>, what: &str) -> Result<&'a str> {
    match map.get(what) {
        Some(Value::Str(text)) => Ok(text),
        _ => Err(Error::malformed(format!("`{what}` must be text"))),
    }
}

fn meta_u64(map: &BTreeMap<String, Value>, what: &str) -> Result<u64> {
    match map.get(what) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!("`{what}` must be u64"))),
    }
}

fn meta_bytes<'a>(map: &'a BTreeMap<String, Value>, what: &str) -> Result<&'a [u8]> {
    match map.get(what) {
        Some(Value::Bytes(bytes)) => Ok(bytes),
        _ => Err(Error::malformed(format!("`{what}` must be bytes"))),
    }
}

fn meta_u64_array(map: &BTreeMap<String, Value>, what: &str) -> Result<Vec<u64>> {
    match map.get(what) {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::U64(value) => out.push(*value),
                    _ => return Err(Error::malformed(format!("`{what}` elements must be u64"))),
                }
            }
            Ok(out)
        }
        _ => Err(Error::malformed(format!("`{what}` must be an array"))),
    }
}

fn hash32(bytes: &[u8], what: &str) -> Result<Hash32> {
    let hash = Hash32::from_slice(bytes)
        .map_err(|_| Error::malformed(format!("{what} must be 32 bytes")))?;
    if hash == Hash32::ZERO {
        return Err(Error::malformed(format!("{what} must not be all zero")));
    }
    Ok(hash)
}

fn split_body(body: &[u8]) -> Result<(&[u8], &[u8])> {
    if body.len() < 4 {
        return Err(Error::malformed(
            "engram body is shorter than its 4-byte descriptor length",
        ));
    }
    let len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let end = 4usize
        .checked_add(len)
        .ok_or_else(|| Error::limit_exceeded("engram descriptor length overflows usize"))?;
    if end > body.len() {
        return Err(Error::malformed(
            "engram descriptor extends past the frame body",
        ));
    }
    Ok((&body[4..end], &body[end..]))
}

fn build_body(meta: &Value, stored: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let descriptor = wcb::encode(meta, limits)?;
    let len = u32::try_from(descriptor.len())
        .map_err(|_| Error::limit_exceeded("engram descriptor does not fit u32"))?;
    let mut body = Vec::with_capacity(4 + descriptor.len() + stored.len());
    body.extend_from_slice(&len.to_le_bytes());
    body.extend_from_slice(&descriptor);
    body.extend_from_slice(stored);
    Ok(body)
}

// ---------------------------------------------------------------------------
// Group coder (byte planes, huffman-canonical-v1)
// ---------------------------------------------------------------------------

fn encode_group_planes(group: &[u8], element_size: usize) -> Vec<u8> {
    if element_size == 0 {
        return Vec::new();
    }
    let elements = group.len() / element_size;
    let mut out = Vec::with_capacity(group.len());
    for plane in 0..element_size {
        let mut plane_bytes = Vec::with_capacity(elements);
        let mut index = plane;
        while index < group.len() {
            plane_bytes.push(group[index]);
            index += element_size;
        }
        out.extend_from_slice(&huffman::encode(&plane_bytes));
    }
    out
}

fn decode_group_planes(stored: &[u8], element_size: usize, element_len: usize) -> Result<Vec<u8>> {
    if element_size == 0 || element_len % element_size != 0 {
        return Err(Error::malformed(
            "engram group length is not a whole number of elements",
        ));
    }
    let elements = element_len / element_size;
    let mut planes = Vec::with_capacity(element_size);
    let mut cursor = 0usize;
    for _ in 0..element_size {
        let remaining = stored
            .get(cursor..)
            .ok_or_else(|| Error::corrupt("engram group plane is truncated"))?;
        let (plane, consumed) = huffman::decode_prefix(remaining, elements)?;
        cursor = cursor
            .checked_add(consumed)
            .ok_or_else(|| Error::limit_exceeded("engram group plane offset overflows usize"))?;
        planes.push(plane);
    }
    if cursor != stored.len() {
        return Err(Error::corrupt(
            "engram group planes leave trailing bytes inside the group range",
        ));
    }
    let mut out = vec![0u8; element_len];
    for (plane_index, plane) in planes.iter().enumerate() {
        for (element, byte) in plane.iter().enumerate() {
            out[element * element_size + plane_index] = *byte;
        }
    }
    Ok(out)
}

/// `H("weft/engram/group/v1", WCB({group_index, row_start, row_end, dtype,
/// precision, values_hash}))`.
pub fn engram_group_root(
    dtype: TensorDtype,
    precision: &str,
    group_index: u64,
    row_start: u64,
    row_end: u64,
    group_bytes: &[u8],
    limits: &Limits,
) -> Result<Hash32> {
    let values_hash = hash_domain(tensor::DOMAIN_TENSOR_VALUES, group_bytes);
    let value = Value::map([
        ("group_index", Value::U64(group_index)),
        ("row_start", Value::U64(row_start)),
        ("row_end", Value::U64(row_end)),
        ("dtype", Value::Str(dtype.as_str().to_owned())),
        ("precision", Value::Str(precision.to_owned())),
        ("values_hash", Value::Bytes(values_hash.as_bytes().to_vec())),
    ]);
    Ok(hash_domain(
        DOMAIN_ENGRAM_GROUP,
        &wcb::encode(&value, limits)?,
    ))
}

// ---------------------------------------------------------------------------
// Validation and encoding
// ---------------------------------------------------------------------------

fn validate_hash_function(name: &str) -> Result<()> {
    match name {
        ENGRAM_HASH_SHA256 | ENGRAM_HASH_BLAKE3 => Ok(()),
        other => Err(Error::unsupported_mode(format!(
            "unknown engram hash_function `{other}` (v1 allows `{ENGRAM_HASH_SHA256}` or `{ENGRAM_HASH_BLAKE3}`)"
        ))),
    }
}

fn validate_precision(dtype: TensorDtype, precision: &EngramPrecision) -> Result<()> {
    if precision.storage_dtype != dtype {
        return Err(Error::unsupported_feature(format!(
            "engram storage_dtype `{}` differs from element_dtype `{}`; v1 has no storage transformation",
            precision.storage_dtype.as_str(),
            dtype.as_str()
        )));
    }
    if let Some(quantization) = &precision.quantization {
        if quantization.scheme != "none" {
            return Err(Error::unsupported_feature(format!(
                "engram quantization scheme `{}` is not implemented (v1 allows `none`)",
                quantization.scheme
            )));
        }
        if !quantization.lossless {
            return Err(Error::unsupported_feature(
                "engram quantization block declares lossless=false; a lossy table is rejected",
            ));
        }
    }
    Ok(())
}

fn validate_edges(shape: &[u64], edges: &[u64], limits: &Limits) -> Result<(u64, u64)> {
    if shape.len() != 2 {
        return Err(Error::malformed(
            "engram table_shape must have rank exactly 2",
        ));
    }
    let rows = shape[0];
    let dim = shape[1];
    if rows == 0 || dim == 0 {
        return Err(Error::malformed(
            "engram table_shape extents must be greater than zero",
        ));
    }
    if edges.len() < 2 {
        return Err(Error::malformed(
            "engram row_group_edges must contain at least two edges",
        ));
    }
    if edges[0] != 0 {
        return Err(Error::malformed("engram row_group_edges must start at 0"));
    }
    if edges[edges.len() - 1] != rows {
        return Err(Error::malformed(format!(
            "engram last edge {} must equal the row count {rows}",
            edges[edges.len() - 1]
        )));
    }
    let group_count = edges.len() - 1;
    if group_count as u64 > limits.max_structured_items {
        return Err(Error::limit_exceeded(format!(
            "engram group count {group_count} exceeds limit {}",
            limits.max_structured_items
        )));
    }
    for index in 1..edges.len() {
        if edges[index] <= edges[index - 1] {
            return Err(Error::malformed(format!(
                "engram row_group_edges are not strictly increasing at {index}: {} then {}",
                edges[index - 1],
                edges[index]
            )));
        }
        let group_rows = edges[index] - edges[index - 1];
        let elements = group_rows
            .checked_mul(dim)
            .ok_or_else(|| Error::limit_exceeded("engram group element count overflows u64"))?;
        if elements > limits.max_engram_group_elements {
            return Err(Error::limit_exceeded(format!(
                "engram group {} has {elements} elements, limit is {}",
                index - 1,
                limits.max_engram_group_elements
            )));
        }
    }
    Ok((rows, dim))
}

fn prepare_groups(
    table: &[u8],
    input: &EngramInput,
    limits: &Limits,
) -> Result<(EngramMeta, Vec<u8>)> {
    validate_hash_function(&input.hash_function)?;
    validate_precision(input.dtype, &input.precision)?;
    let (rows, dim) = validate_edges(&input.table_shape, &input.row_group_edges, limits)?;
    let element_count = tensor::validate_shape(&input.table_shape, limits)?;
    if input.vocab_size == 0 || input.vocab_size > rows {
        return Err(Error::malformed(format!(
            "engram vocab_size {} must be in 1..={rows}",
            input.vocab_size
        )));
    }
    let element_size = input.dtype.element_size();
    let expected_len = element_count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("engram decoded bytes overflow u64"))?;
    if table.len() as u64 != expected_len {
        return Err(Error::malformed(format!(
            "engram table is {} bytes but shape {:?} x {} requires {expected_len}",
            table.len(),
            input.table_shape,
            input.dtype.as_str()
        )));
    }
    crate::limits::check_len(table.len() as u64, limits.max_decoded_bytes, "engram table")?;
    let group_count = input.row_group_edges.len() - 1;
    let supersedes = if input.supersedes.is_empty() {
        vec![None; group_count]
    } else if input.supersedes.len() == group_count {
        input.supersedes.clone()
    } else {
        return Err(Error::malformed(format!(
            "engram supersedes has {} entries but there are {group_count} groups",
            input.supersedes.len()
        )));
    };
    for (index, reference) in &input.references {
        if *index >= group_count as u64 {
            return Err(Error::malformed(format!(
                "engram reference names group {index} but there are {group_count} groups"
            )));
        }
        if reference.group_root == Hash32::ZERO {
            return Err(Error::malformed(
                "engram reference group root must not be all zero",
            ));
        }
    }
    let element_size_usize = usize::try_from(element_size)
        .map_err(|_| Error::limit_exceeded("engram element size does not fit usize"))?;
    let row_bytes = dim
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("engram row byte length overflows u64"))?;
    let mut stored = Vec::with_capacity(table.len());
    let mut groups = Vec::with_capacity(group_count);
    for index in 0..group_count {
        let row_start = input.row_group_edges[index];
        let row_end = input.row_group_edges[index + 1];
        let byte_start = usize::try_from(
            row_start
                .checked_mul(row_bytes)
                .ok_or_else(|| Error::limit_exceeded("engram group start overflows u64"))?,
        )
        .map_err(|_| Error::limit_exceeded("engram group start does not fit usize"))?;
        let byte_end = usize::try_from(
            row_end
                .checked_mul(row_bytes)
                .ok_or_else(|| Error::limit_exceeded("engram group end overflows u64"))?,
        )
        .map_err(|_| Error::limit_exceeded("engram group end does not fit usize"))?;
        let group_bytes = table
            .get(byte_start..byte_end)
            .ok_or_else(|| Error::corrupt("engram group range is outside the table"))?;
        if let Some(reference) = input.references.get(&(index as u64)) {
            groups.push(EngramGroup {
                index: index as u64,
                row_start,
                row_end,
                storage: EngramStorage::Reference {
                    dependency: reference.dependency,
                },
                group_root: reference.group_root,
                supersedes: supersedes[index],
            });
            continue;
        }
        let encoded = encode_group_planes(group_bytes, element_size_usize);
        let root = engram_group_root(
            input.dtype,
            input.precision.storage_dtype.as_str(),
            index as u64,
            row_start,
            row_end,
            group_bytes,
            limits,
        )?;
        groups.push(EngramGroup {
            index: index as u64,
            row_start,
            row_end,
            storage: EngramStorage::Inline {
                offset: stored.len() as u64,
                compressed_len: encoded.len() as u64,
            },
            group_root: root,
            supersedes: supersedes[index],
        });
        stored.extend_from_slice(&encoded);
    }
    let payload_hash = hash_domain(DOMAIN_ENGRAM_PAYLOAD, &stored);
    let meta = EngramMeta {
        hash_function: input.hash_function.clone(),
        vocab_size: input.vocab_size,
        table_shape: input.table_shape.clone(),
        dtype: input.dtype,
        precision: input.precision.clone(),
        row_group_edges: input.row_group_edges.clone(),
        payload_len: stored.len() as u64,
        payload_hash,
        groups,
    };
    Ok((meta, stored))
}

/// Encode one fully inline mode-20 frame (no dependency table).
pub fn encode_engram(table: &[u8], input: &EngramInput, limits: &Limits) -> Result<Vec<u8>> {
    if !input.references.is_empty() {
        return Err(Error::malformed(
            "encode_engram cannot emit references without a dependency table; use encode_engram_with_dependencies",
        ));
    }
    let (meta, stored) = prepare_groups(table, input, limits)?;
    encode_frame(
        &FrameParts {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            domain: DOMAIN_PAYLOAD,
            mode: Mode::EngramShard,
            flags: 0,
            level: 0,
            decoded_len: table.len() as u64,
            content_hash: hash_domain(DOMAIN_CHUNK, table),
            dependencies: &[],
            chunks: &[],
            body: &build_body(&meta_value(&meta), &stored, limits)?,
        },
        limits,
    )
}

/// Encode a mode-20 frame that references groups in earlier artifacts.
///
/// `dependency_hashes[i]` is the content hash of the artifact named by
/// `EngramReference::dependency = i`. The caller must have verified that each
/// referenced root belongs to that artifact; every decoder re-verifies it.
pub fn encode_engram_with_dependencies(
    table: &[u8],
    input: &EngramInput,
    dependency_hashes: &[Hash32],
    limits: &Limits,
) -> Result<Vec<u8>> {
    if dependency_hashes.len() as u64 > limits.max_dependency_count {
        return Err(Error::limit_exceeded(format!(
            "engram dependency count {} exceeds limit {}",
            dependency_hashes.len(),
            limits.max_dependency_count
        )));
    }
    let (meta, stored) = prepare_groups(table, input, limits)?;
    for (index, reference) in &input.references {
        if reference.dependency as usize >= dependency_hashes.len() {
            return Err(Error::malformed(format!(
                "engram reference group {index} names dependency {} but only {} are supplied",
                reference.dependency,
                dependency_hashes.len()
            )));
        }
    }
    let dependencies: Vec<Dependency> = dependency_hashes
        .iter()
        .map(|hash| Dependency::new(*hash, table.len() as u64, DEP_DOMAIN_PAYLOAD, 20))
        .collect();
    encode_frame(
        &FrameParts {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            domain: DOMAIN_PAYLOAD,
            mode: Mode::EngramShard,
            flags: 0,
            level: 0,
            decoded_len: table.len() as u64,
            content_hash: hash_domain(DOMAIN_CHUNK, table),
            dependencies: &dependencies,
            chunks: &[],
            body: &build_body(&meta_value(&meta), &stored, limits)?,
        },
        limits,
    )
}

// ---------------------------------------------------------------------------
// Descriptor value and parsing
// ---------------------------------------------------------------------------

fn precision_value(precision: &EngramPrecision) -> Value {
    let mut map = BTreeMap::new();
    map.insert(
        "storage_dtype".to_owned(),
        Value::Str(precision.storage_dtype.as_str().to_owned()),
    );
    if let Some(quantization) = &precision.quantization {
        map.insert(
            "quantization".to_owned(),
            Value::map([
                ("scheme", Value::Str(quantization.scheme.clone())),
                ("lossless", Value::Bool(quantization.lossless)),
            ]),
        );
    }
    Value::Map(map)
}

fn group_value(group: &EngramGroup) -> Value {
    let (storage, offset, compressed_len, dependency) = match &group.storage {
        EngramStorage::Inline {
            offset,
            compressed_len,
        } => ("inline", *offset, *compressed_len, Value::Null),
        EngramStorage::Reference { dependency } => ("reference", 0, 0, Value::U64(*dependency)),
    };
    Value::map([
        ("index", Value::U64(group.index)),
        ("row_start", Value::U64(group.row_start)),
        ("row_end", Value::U64(group.row_end)),
        ("storage", Value::Str(storage.to_owned())),
        ("offset", Value::U64(offset)),
        ("compressed_len", Value::U64(compressed_len)),
        (
            "group_root",
            Value::Bytes(group.group_root.as_bytes().to_vec()),
        ),
        (
            "supersedes",
            match group.supersedes {
                Some(root) => Value::Bytes(root.as_bytes().to_vec()),
                None => Value::Null,
            },
        ),
        ("dependency", dependency),
    ])
}

fn meta_value(meta: &EngramMeta) -> Value {
    let edges = meta
        .row_group_edges
        .iter()
        .map(|edge| Value::U64(*edge))
        .collect();
    let shape = meta
        .table_shape
        .iter()
        .map(|dim| Value::U64(*dim))
        .collect();
    Value::map([
        ("schema", Value::Str(ENGRAM_SCHEMA.to_owned())),
        ("hash_function", Value::Str(meta.hash_function.clone())),
        ("vocab_size", Value::U64(meta.vocab_size)),
        ("table_shape", Value::Array(shape)),
        ("element_dtype", Value::Str(meta.dtype.as_str().to_owned())),
        ("precision", precision_value(&meta.precision)),
        ("row_group_edges", Value::Array(edges)),
        ("group_count", Value::U64(meta.groups.len() as u64)),
        (
            "groups",
            Value::Array(meta.groups.iter().map(group_value).collect()),
        ),
        ("payload_len", Value::U64(meta.payload_len)),
        (
            "payload_hash",
            Value::Bytes(meta.payload_hash.as_bytes().to_vec()),
        ),
    ])
}

/// Parse a mode-20 descriptor from its canonical WCB bytes.
pub fn parse_engram_descriptor(
    descriptor_bytes: &[u8],
    decoded_len: u64,
    payload_len: u64,
    _payload: &[u8],
    limits: &Limits,
) -> Result<EngramMeta> {
    let value = wcb::decode(descriptor_bytes, limits)?;
    let map = ai_map(&value, "engram descriptor")?;
    check_exact_keys(
        map,
        &[
            "schema",
            "hash_function",
            "vocab_size",
            "table_shape",
            "element_dtype",
            "precision",
            "row_group_edges",
            "group_count",
            "groups",
            "payload_len",
            "payload_hash",
        ],
        "engram descriptor",
    )?;
    let schema = meta_text(map, "schema")?;
    if schema != ENGRAM_SCHEMA {
        return Err(Error::malformed(format!(
            "unsupported engram schema `{schema}`"
        )));
    }
    let hash_function = meta_text(map, "hash_function")?.to_owned();
    validate_hash_function(&hash_function)?;
    let table_shape = meta_u64_array(map, "table_shape")?;
    let dtype = TensorDtype::from_name(meta_text(map, "element_dtype")?)?;
    let precision_map = match map.get("precision") {
        Some(Value::Map(precision)) => precision,
        _ => return Err(Error::malformed("engram `precision` must be a map")),
    };
    let storage_dtype_name = meta_text(precision_map, "storage_dtype")?;
    let storage_dtype = TensorDtype::from_name(storage_dtype_name).map_err(|_| {
        Error::unsupported_feature(format!(
            "unknown engram storage_dtype `{storage_dtype_name}`"
        ))
    })?;
    let quantization = match precision_map.get("quantization") {
        None | Some(Value::Null) => None,
        Some(Value::Map(block)) => {
            check_exact_keys(block, &["scheme", "lossless"], "engram quantization")?;
            let scheme = meta_text(block, "scheme")?.to_owned();
            let lossless = match block.get("lossless") {
                Some(Value::Bool(value)) => *value,
                _ => {
                    return Err(Error::malformed(
                        "engram quantization `lossless` must be a bool",
                    ))
                }
            };
            Some(EngramQuantization { scheme, lossless })
        }
        _ => {
            return Err(Error::malformed(
                "engram quantization must be a map or null",
            ))
        }
    };
    if !matches!(precision_map.len(), 1 | 2) {
        return Err(Error::malformed("engram precision has unknown keys"));
    }
    for key in precision_map.keys() {
        if key != "storage_dtype" && key != "quantization" {
            return Err(Error::malformed(format!(
                "engram precision has unknown key `{key}`"
            )));
        }
    }
    let precision = EngramPrecision {
        storage_dtype,
        quantization,
    };
    validate_precision(dtype, &precision)?;
    let row_group_edges = meta_u64_array(map, "row_group_edges")?;
    let (rows, _dim) = validate_edges(&table_shape, &row_group_edges, limits)?;
    let element_count = tensor::validate_shape(&table_shape, limits)?;
    let element_size = dtype.element_size();
    let expected_len = element_count
        .checked_mul(element_size)
        .ok_or_else(|| Error::limit_exceeded("engram decoded bytes overflow u64"))?;
    if decoded_len != expected_len {
        return Err(Error::corrupt(format!(
            "engram decoded_len is {decoded_len} but the descriptor requires {expected_len}"
        )));
    }
    let vocab_size = meta_u64(map, "vocab_size")?;
    if vocab_size == 0 || vocab_size > rows {
        return Err(Error::malformed(format!(
            "engram vocab_size {vocab_size} must be in 1..={rows}"
        )));
    }
    let group_count = meta_u64(map, "group_count")?;
    if group_count != (row_group_edges.len() - 1) as u64 {
        return Err(Error::malformed(format!(
            "engram group_count {group_count} does not match {} edges",
            row_group_edges.len() - 1
        )));
    }
    let declared_payload_len = meta_u64(map, "payload_len")?;
    if declared_payload_len != payload_len {
        return Err(Error::malformed(format!(
            "engram payload is {payload_len} bytes but the descriptor declares {declared_payload_len}"
        )));
    }
    let payload_hash = hash32(meta_bytes(map, "payload_hash")?, "engram payload_hash")?;
    let entries = match map.get("groups") {
        Some(Value::Array(items)) => items,
        _ => return Err(Error::malformed("engram `groups` must be an array")),
    };
    if entries.len() as u64 != group_count {
        return Err(Error::malformed(format!(
            "engram has {} group entries but group_count is {group_count}",
            entries.len()
        )));
    }
    let mut groups = Vec::with_capacity(entries.len());
    let mut next_offset: u64 = 0;
    for (position, item) in entries.iter().enumerate() {
        let entry = ai_map(item, "engram group")?;
        check_exact_keys(
            entry,
            &[
                "index",
                "row_start",
                "row_end",
                "storage",
                "offset",
                "compressed_len",
                "group_root",
                "supersedes",
                "dependency",
            ],
            "engram group",
        )?;
        let index = meta_u64(entry, "index")?;
        if index != position as u64 {
            return Err(Error::malformed(format!(
                "engram group entry {position} declares index {index}; entries must be in canonical order"
            )));
        }
        let row_start = meta_u64(entry, "row_start")?;
        let row_end = meta_u64(entry, "row_end")?;
        if row_start != row_group_edges[position] || row_end != row_group_edges[position + 1] {
            return Err(Error::malformed(format!(
                "engram group {position} rows [{row_start}, {row_end}) do not match edges [{}, {})",
                row_group_edges[position],
                row_group_edges[position + 1]
            )));
        }
        let storage_text = meta_text(entry, "storage")?;
        let offset = meta_u64(entry, "offset")?;
        let compressed_len = meta_u64(entry, "compressed_len")?;
        let dependency = match entry.get("dependency") {
            Some(Value::Null) => None,
            Some(Value::U64(value)) => Some(*value),
            _ => {
                return Err(Error::malformed(
                    "engram group `dependency` must be u64 or null",
                ))
            }
        };
        let storage = match storage_text {
            "inline" => {
                if dependency.is_some() {
                    return Err(Error::malformed(format!(
                        "engram inline group {position} must not name a dependency"
                    )));
                }
                if compressed_len == 0 {
                    return Err(Error::malformed(format!(
                        "engram inline group {position} has an empty stored range"
                    )));
                }
                if offset != next_offset {
                    return Err(Error::malformed(format!(
                        "engram group {position} offset {offset} is not the contiguous offset {next_offset}"
                    )));
                }
                next_offset = next_offset
                    .checked_add(compressed_len)
                    .ok_or_else(|| Error::limit_exceeded("engram group offset overflows u64"))?;
                EngramStorage::Inline {
                    offset,
                    compressed_len,
                }
            }
            "reference" => {
                if offset != 0 || compressed_len != 0 {
                    return Err(Error::malformed(format!(
                        "engram reference group {position} must have offset 0 and compressed_len 0"
                    )));
                }
                let dependency = dependency.ok_or_else(|| {
                    Error::malformed(format!(
                        "engram reference group {position} must name a dependency"
                    ))
                })?;
                EngramStorage::Reference { dependency }
            }
            other => {
                return Err(Error::malformed(format!(
                    "engram group {position} has unknown storage `{other}`"
                )))
            }
        };
        let group_root = hash32(meta_bytes(entry, "group_root")?, "engram group_root")?;
        let supersedes = match entry.get("supersedes") {
            Some(Value::Null) => None,
            Some(Value::Bytes(bytes)) => Some(hash32(bytes, "engram supersedes")?),
            _ => {
                return Err(Error::malformed(
                    "engram group `supersedes` must be bytes32 or null",
                ))
            }
        };
        groups.push(EngramGroup {
            index,
            row_start,
            row_end,
            storage,
            group_root,
            supersedes,
        });
    }
    if next_offset != declared_payload_len {
        return Err(Error::malformed(format!(
            "engram inline group ranges cover {next_offset} bytes but payload_len is {declared_payload_len}"
        )));
    }
    Ok(EngramMeta {
        hash_function,
        vocab_size,
        table_shape,
        dtype,
        precision,
        row_group_edges,
        payload_len: declared_payload_len,
        payload_hash,
        groups,
    })
}

/// Parse a mode-20 frame and return its descriptor with the stored payload.
pub fn parse_engram<'a>(frame: &'a Frame<'_>, limits: &Limits) -> Result<EngramDescriptor<'a>> {
    if frame.header.mode != Mode::EngramShard {
        return Err(Error::malformed(format!(
            "frame mode `{}` is not engram_shard",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor_bytes, payload) = split_body(frame.body)?;
    let meta = parse_engram_descriptor(
        descriptor_bytes,
        frame.header.decoded_len,
        payload.len() as u64,
        payload,
        limits,
    )?;
    for (index, dep) in frame.dependencies.iter().enumerate() {
        if dep.dep_domain != DEP_DOMAIN_PAYLOAD {
            return Err(Error::malformed(format!(
                "engram dependency {index} must use dep_domain 0 (payload)"
            )));
        }
        if dep.decoded_len != meta.decoded_len()? {
            return Err(Error::malformed(format!(
                "engram dependency {index} declares {} bytes but the table is {}",
                dep.decoded_len,
                meta.decoded_len()?
            )));
        }
    }
    for group in &meta.groups {
        if let EngramStorage::Reference { dependency } = group.storage {
            if dependency as usize >= frame.dependencies.len() {
                return Err(Error::malformed(format!(
                    "engram group {} references dependency {dependency} but the frame has {} dependencies",
                    group.index,
                    frame.dependencies.len()
                )));
            }
        }
    }
    if hash_domain(DOMAIN_ENGRAM_PAYLOAD, payload) != meta.payload_hash {
        return Err(Error::corrupt(
            "engram payload hash does not match the descriptor",
        ));
    }
    Ok(EngramDescriptor { meta, payload })
}

/// Return a mode-20 frame's descriptor without its payload.
pub fn engram_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<EngramMeta> {
    Ok(parse_engram(frame, limits)?.meta)
}

// ---------------------------------------------------------------------------
// Group decoding and reads
// ---------------------------------------------------------------------------

fn group_byte_len(meta: &EngramMeta, group: &EngramGroup) -> Result<u64> {
    let rows = group.row_end - group.row_start;
    rows.checked_mul(meta.dim())
        .and_then(|elements| elements.checked_mul(meta.dtype.element_size()))
        .ok_or_else(|| Error::limit_exceeded("engram group byte length overflows u64"))
}

fn decode_inline_group(
    meta: &EngramMeta,
    group: &EngramGroup,
    payload: &[u8],
    offset: u64,
    compressed_len: u64,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let start = usize::try_from(offset)
        .map_err(|_| Error::limit_exceeded("engram group offset does not fit usize"))?;
    let end = start
        .checked_add(
            usize::try_from(compressed_len)
                .map_err(|_| Error::limit_exceeded("engram group length does not fit usize"))?,
        )
        .ok_or_else(|| Error::limit_exceeded("engram group range overflows usize"))?;
    let stored = payload
        .get(start..end)
        .ok_or_else(|| Error::corrupt("engram group range is outside the payload"))?;
    let element_len = group_byte_len(meta, group)?;
    let bytes = decode_group_planes(
        stored,
        usize::try_from(meta.dtype.element_size())
            .map_err(|_| Error::limit_exceeded("engram element size does not fit usize"))?,
        usize::try_from(element_len)
            .map_err(|_| Error::limit_exceeded("engram group length does not fit usize"))?,
    )?;
    let root = engram_group_root(
        meta.dtype,
        meta.precision.storage_dtype.as_str(),
        group.index,
        group.row_start,
        group.row_end,
        &bytes,
        limits,
    )?;
    if root != group.group_root {
        return Err(Error::corrupt(format!(
            "engram group {} root {root} does not match the descriptor's {}",
            group.index, group.group_root
        )));
    }
    Ok(bytes)
}

struct FoundGroup {
    meta: EngramMeta,
    group: EngramGroup,
    bytes: Vec<u8>,
}

/// A reference names the *same* group, so its row range and type must match.
fn compatible_reference(
    meta_a: &EngramMeta,
    group_a: &EngramGroup,
    meta_b: &EngramMeta,
    group_b: &EngramGroup,
) -> bool {
    group_a.row_start == group_b.row_start
        && group_a.row_end == group_b.row_end
        && meta_a.dtype == meta_b.dtype
        && meta_a.precision.storage_dtype == meta_b.precision.storage_dtype
}

/// Supersession may re-group rows, so only the element type and storage
/// precision must agree. A different dtype/precision is an incompatible
/// replacement and is `Corrupt`.
fn compatible_supersession(meta_a: &EngramMeta, meta_b: &EngramMeta) -> bool {
    meta_a.dtype == meta_b.dtype && meta_a.precision.storage_dtype == meta_b.precision.storage_dtype
}

/// Find a group root in a frame (or its dependencies) and decode it.
fn find_group_in_frame(
    frame: &Frame<'_>,
    root: Hash32,
    resolver: &(impl Resolver + ?Sized),
    limits: &Limits,
    visiting: &mut BTreeSet<Hash32>,
    depth: u64,
) -> Result<Option<FoundGroup>> {
    if depth == 0 {
        return Ok(None);
    }
    let descriptor = parse_engram(frame, limits)?;
    if let Some(group) = descriptor
        .meta
        .groups
        .iter()
        .find(|candidate| candidate.group_root == root)
        .cloned()
    {
        let bytes = match group.storage {
            EngramStorage::Inline {
                offset,
                compressed_len,
            } => decode_inline_group(
                &descriptor.meta,
                &group,
                descriptor.payload,
                offset,
                compressed_len,
                limits,
            )?,
            EngramStorage::Reference { dependency } => {
                let dep = frame.dependencies.get(dependency as usize).ok_or_else(|| {
                    Error::corrupt("engram reference dependency index is out of range")
                })?;
                if !visiting.insert(dep.hash) {
                    return Err(Error::dependency_cycle(
                        "engram dependency cycle while resolving a referenced group",
                    ));
                }
                let nested = resolver.resolve_frame(&dep.hash)?;
                let found = match nested {
                    Some(bytes) => {
                        let nested_frame = Frame::parse(&bytes, limits)?;
                        find_group_in_frame(
                            &nested_frame,
                            root,
                            resolver,
                            limits,
                            visiting,
                            depth - 1,
                        )?
                    }
                    None => None,
                };
                visiting.remove(&dep.hash);
                found
                    .ok_or_else(|| {
                        Error::dependency_missing(format!(
                            "engram referenced group {root} is absent from dependency {}",
                            dep.hash
                        ))
                    })?
                    .bytes
            }
        };
        return Ok(Some(FoundGroup {
            meta: descriptor.meta,
            group,
            bytes,
        }));
    }
    for dep in &frame.dependencies {
        if !visiting.insert(dep.hash) {
            return Err(Error::dependency_cycle(
                "engram dependency cycle while resolving a group",
            ));
        }
        let found = match resolver.resolve_frame(&dep.hash)? {
            Some(bytes) => {
                let nested = Frame::parse(&bytes, limits)?;
                find_group_in_frame(&nested, root, resolver, limits, visiting, depth - 1)?
            }
            None => None,
        };
        visiting.remove(&dep.hash);
        if found.is_some() {
            return Ok(found);
        }
    }
    Ok(None)
}

/// Read explicit group indices. Returns the concatenated canonical group bytes
/// in ascending index order and the touched group indices.
///
/// Only the touched groups are decoded and verified. The payload hash, frame
/// trailer and whole-table content hash are **not** checked on this path.
pub fn read_engram_groups<R: Resolver + ?Sized>(
    frame: &Frame<'_>,
    descriptor: &EngramDescriptor<'_>,
    indices: &[u32],
    resolver: &R,
    limits: &Limits,
) -> Result<(Vec<u8>, Vec<u32>)> {
    let mut unique: BTreeSet<u32> = BTreeSet::new();
    for index in indices {
        if *index as usize >= descriptor.meta.groups.len() {
            return Err(Error::malformed(format!(
                "engram group index {index} is outside the descriptor"
            )));
        }
        unique.insert(*index);
    }
    let mut out = Vec::new();
    for index in &unique {
        let group = &descriptor.meta.groups[*index as usize];
        let bytes = match group.storage {
            EngramStorage::Inline {
                offset,
                compressed_len,
            } => decode_inline_group(
                &descriptor.meta,
                group,
                descriptor.payload,
                offset,
                compressed_len,
                limits,
            )?,
            EngramStorage::Reference { .. } => {
                let found = find_group_in_frame(
                    frame,
                    group.group_root,
                    resolver,
                    limits,
                    &mut BTreeSet::new(),
                    limits.max_dependency_depth,
                )?
                .ok_or_else(|| {
                    Error::dependency_missing(format!(
                        "engram referenced group {} ({}) is absent",
                        group.index, group.group_root
                    ))
                })?;
                if !compatible_reference(&descriptor.meta, group, &found.meta, &found.group) {
                    return Err(Error::corrupt(format!(
                        "engram referenced group {} is incompatible with its source group",
                        group.index
                    )));
                }
                found.bytes
            }
        };
        out.extend_from_slice(&bytes);
    }
    Ok((out, unique.into_iter().collect()))
}

/// Group indices whose row intervals intersect `[row_start, row_end)`.
pub fn engram_region_groups(meta: &EngramMeta, row_start: u64, row_end: u64) -> Result<Vec<u32>> {
    if row_start > row_end {
        return Err(Error::non_canonical(format!(
            "engram row region start {row_start} is after end {row_end}"
        )));
    }
    if row_end > meta.rows() {
        return Err(Error::non_canonical(format!(
            "engram row region end {row_end} exceeds {} rows",
            meta.rows()
        )));
    }
    let mut out = Vec::new();
    if row_start == row_end {
        return Ok(out);
    }
    for group in &meta.groups {
        if group.row_start < row_end && row_start < group.row_end {
            out.push(
                u32::try_from(group.index)
                    .map_err(|_| Error::limit_exceeded("engram group index does not fit u32"))?,
            );
        }
    }
    Ok(out)
}

/// Read a row region, decoding only the groups whose row intervals intersect it.
/// Returns the region's canonical row-major bytes and the touched group indices.
///
/// The payload hash, frame trailer and whole-table content hash are **not**
/// checked on this path; each touched group's root is.
pub fn read_engram_region<R: Resolver + ?Sized>(
    frame: &Frame<'_>,
    descriptor: &EngramDescriptor<'_>,
    row_start: u64,
    row_end: u64,
    resolver: &R,
    limits: &Limits,
) -> Result<(Vec<u8>, Vec<u32>)> {
    let meta = &descriptor.meta;
    let indices = engram_region_groups(meta, row_start, row_end)?;
    if indices.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let (bytes, touched) = read_engram_groups(frame, descriptor, &indices, resolver, limits)?;
    let word = meta.dtype.element_size();
    let row_bytes = usize::try_from(
        meta.dim()
            .checked_mul(word)
            .ok_or_else(|| Error::limit_exceeded("engram row byte length overflows u64"))?,
    )
    .map_err(|_| Error::limit_exceeded("engram row byte length does not fit usize"))?;
    let first = &meta.groups[indices[0] as usize];
    let lead = (row_start - first.row_start) as usize;
    let last = &meta.groups[*indices.last().unwrap() as usize];
    let tail_rows = (last.row_end - row_end) as usize;
    let lead_bytes = lead
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::limit_exceeded("engram region lead overflows usize"))?;
    let tail_bytes = tail_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::limit_exceeded("engram region tail overflows usize"))?;
    let end = bytes
        .len()
        .checked_sub(tail_bytes)
        .ok_or_else(|| Error::corrupt("engram region trim exceeds the decoded group bytes"))?;
    if lead_bytes > end {
        return Err(Error::corrupt(
            "engram region lead exceeds the decoded group bytes",
        ));
    }
    Ok((bytes[lead_bytes..end].to_vec(), touched))
}

// ---------------------------------------------------------------------------
// Full decode
// ---------------------------------------------------------------------------

fn verify_supersession<R: Resolver + ?Sized>(
    meta: &EngramMeta,
    frame: &Frame<'_>,
    resolver: &R,
    limits: &Limits,
) -> Result<()> {
    for group in &meta.groups {
        let mut seen = BTreeSet::new();
        seen.insert(group.group_root);
        let mut current = group.supersedes;
        while let Some(root) = current {
            if !seen.insert(root) {
                return Err(Error::corrupt(format!(
                    "engram supersession cycle at group {}",
                    group.index
                )));
            }
            match meta
                .groups
                .iter()
                .find(|candidate| candidate.group_root == root)
            {
                Some(parent) => {
                    if !compatible_supersession(meta, meta) {
                        return Err(Error::corrupt(format!(
                            "engram supersession for group {} names an incompatible in-frame group",
                            group.index
                        )));
                    }
                    current = parent.supersedes;
                }
                None => {
                    let found = find_group_in_frame(
                        frame,
                        root,
                        resolver,
                        limits,
                        &mut BTreeSet::new(),
                        limits.max_dependency_depth,
                    )?;
                    let found = match found {
                        Some(found) => found,
                        None => {
                            return Err(Error::corrupt(format!(
                                "engram supersession root {root} does not resolve to a group root"
                            )))
                        }
                    };
                    if !compatible_supersession(meta, &found.meta) {
                        return Err(Error::corrupt(format!(
                            "engram supersession for group {} names an incompatible group",
                            group.index
                        )));
                    }
                    break;
                }
            }
        }
    }
    Ok(())
}

/// Decode a complete mode-20 frame with a dependency resolver.
///
/// Every group root, every supersession edge, the payload hash, the exact
/// length and the whole-table content hash are verified.
pub fn decode_engram<R: Resolver + ?Sized>(
    frame: &Frame<'_>,
    resolver: &R,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let descriptor = parse_engram(frame, limits)?;
    let meta = &descriptor.meta;
    let total = usize::try_from(meta.decoded_len()?)
        .map_err(|_| Error::limit_exceeded("engram decoded bytes do not fit usize"))?;
    let mut out = vec![0u8; total];
    let row_bytes = usize::try_from(
        meta.dim()
            .checked_mul(meta.dtype.element_size())
            .ok_or_else(|| Error::limit_exceeded("engram row byte length overflows u64"))?,
    )
    .map_err(|_| Error::limit_exceeded("engram row byte length does not fit usize"))?;
    for group in &meta.groups {
        let bytes = match group.storage {
            EngramStorage::Inline {
                offset,
                compressed_len,
            } => decode_inline_group(
                meta,
                group,
                descriptor.payload,
                offset,
                compressed_len,
                limits,
            )?,
            EngramStorage::Reference { .. } => {
                let found = find_group_in_frame(
                    frame,
                    group.group_root,
                    resolver,
                    limits,
                    &mut BTreeSet::new(),
                    limits.max_dependency_depth,
                )?
                .ok_or_else(|| {
                    Error::dependency_missing(format!(
                        "engram referenced group {} ({}) is absent",
                        group.index, group.group_root
                    ))
                })?;
                if !compatible_reference(meta, group, &found.meta, &found.group) {
                    return Err(Error::corrupt(format!(
                        "engram referenced group {} is incompatible with its source group",
                        group.index
                    )));
                }
                found.bytes
            }
        };
        let start = usize::try_from(
            group
                .row_start
                .checked_mul(row_bytes as u64)
                .ok_or_else(|| Error::limit_exceeded("engram group row offset overflows u64"))?,
        )
        .map_err(|_| Error::limit_exceeded("engram group row offset does not fit usize"))?;
        let end = start
            .checked_add(bytes.len())
            .ok_or_else(|| Error::limit_exceeded("engram group destination overflows usize"))?;
        if end > out.len() {
            return Err(Error::corrupt(
                "engram group destination is outside the assembled table",
            ));
        }
        out[start..end].copy_from_slice(&bytes);
    }
    verify_supersession(meta, frame, resolver, limits)?;
    if hash_domain(DOMAIN_CHUNK, &out) != frame.header.content_hash {
        return Err(Error::corrupt("engram whole-table content hash mismatch"));
    }
    Ok(out)
}

/// Decode with no external bytes available. Any referenced group (or external
/// supersession edge) fails closed with `DependencyMissing`.
pub fn decode_engram_without_dependencies(frame: &Frame<'_>, limits: &Limits) -> Result<Vec<u8>> {
    struct Empty;
    impl Resolver for Empty {
        fn resolve(&self, hash: &Hash32, _decoded_len: u64, _dep_domain: u8) -> Result<Vec<u8>> {
            Err(Error::dependency_missing(format!(
                "no dependency resolver for {hash}"
            )))
        }
    }
    decode_engram(frame, &Empty, limits)
}

/// Decode a mode-20 frame through the generic mode dispatch.
pub fn decode_frame(
    frame: &Frame<'_>,
    resolver: &dyn Resolver,
    limits: &Limits,
) -> Result<Vec<u8>> {
    decode_engram(frame, resolver, limits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dependency::MapResolver;

    fn table(rows: u64, dim: u64, seed: u8) -> Vec<u8> {
        let mut state = u64::from(seed);
        (0..rows * dim * 2)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xff) as u8
            })
            .collect()
    }

    #[test]
    fn round_trip_and_region() {
        let limits = Limits::default();
        let bytes = table(8, 4, 3);
        let meta = EngramInput::inline(
            TensorDtype::Bf16,
            vec![8, 4],
            vec![0, 3, 8],
            ENGRAM_HASH_BLAKE3,
            8,
        );
        let frame_bytes = encode_engram(&bytes, &meta, &limits).unwrap();
        let parsed = Frame::parse(&frame_bytes, &limits).unwrap();
        let decoded = decode_engram_without_dependencies(&parsed, &limits).unwrap();
        assert_eq!(decoded, bytes);
        let descriptor = parse_engram(&parsed, &limits).unwrap();
        assert_eq!(descriptor.meta.groups.len(), 2);
        let (region, touched) =
            read_engram_region(&parsed, &descriptor, 2, 5, &MapResolver::new(), &limits).unwrap();
        assert_eq!(touched, vec![0, 1]);
        let row = 4 * 2;
        assert_eq!(region, bytes[2 * row..5 * row]);
    }

    #[test]
    fn reference_round_trip() {
        let limits = Limits::default();
        let bytes = table(8, 4, 5);
        let first = EngramInput::inline(
            TensorDtype::Bf16,
            vec![8, 4],
            vec![0, 4, 8],
            ENGRAM_HASH_BLAKE3,
            8,
        );
        let first_frame = encode_engram(&bytes, &first, &limits).unwrap();
        let first_parsed = Frame::parse(&first_frame, &limits).unwrap();
        let first_meta = engram_metadata(&first_parsed, &limits).unwrap();
        let mut updated = bytes.clone();
        updated[0] ^= 0xff;
        let mut input = EngramInput::inline(
            TensorDtype::Bf16,
            vec![8, 4],
            vec![0, 4, 8],
            ENGRAM_HASH_BLAKE3,
            8,
        );
        input.references.insert(
            1,
            EngramReference {
                group_root: first_meta.groups[1].group_root,
                dependency: 0,
            },
        );
        let second = encode_engram_with_dependencies(
            &updated,
            &input,
            &[first_parsed.header.content_hash],
            &limits,
        )
        .unwrap();
        let second_parsed = Frame::parse(&second, &limits).unwrap();
        let mut resolver = MapResolver::new();
        resolver.insert(first_parsed.header.content_hash, first_frame.clone());
        let decoded = decode_engram(&second_parsed, &resolver, &limits).unwrap();
        assert_eq!(decoded, updated);
        let descriptor = parse_engram(&second_parsed, &limits).unwrap();
        assert!(matches!(
            descriptor.meta.groups[1].storage,
            EngramStorage::Reference { .. }
        ));
        assert!(descriptor.meta.payload_len < first_meta.payload_len);
    }
}
