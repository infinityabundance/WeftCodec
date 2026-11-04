# WEFT-TENSOR v1 — logical tensor identity and region addressing

Status: **implemented** in both codecs (Rust `WeftCodec::tensor`, Go
`WeftCodec-Go/tensor`) and replayed by committed conformance vectors.

This document defines a **logical tensor identity independent of physical
layout**, plus enough layout and region vocabulary to prove that a region is
assembled from only the physical units it intersects. It is the prerequisite
for topology-independent checkpoint restore.

## 1. Scope and constraints

1. A logical tensor is `(dtype, shape)` plus element values in **canonical
   row-major order**. Its identity MUST NOT depend on sharding, blocking,
   strides, storage layout, file name, or host.
2. The layer is **additive**. It defines new schema strings and records. It
   does **not** change mode-6 `TENSOR` frames (WEFT-AI-CODEC §2), which remain
   frozen to `layout: "row_major"`, and it does not change any committed
   `weftcodec-format` vector.
3. Canonical encoders only. Every record is WCB (WEFTCODEC-FORMAT §8); the
   decoder rejects non-canonical bytes.
4. All arithmetic is checked. Every allocation is bounded by
   `max_tensor_elements`, `max_tensor_rank`, `max_decoded_bytes`, and the WCB
   limits.
5. Hash mismatch and descriptor mismatch fail closed.

## 2. Logical tensor

`dtype` is one of the v1 element types:

```
f16  bf16  f32  f64  i8  u8  i16  i32  i64
```

with element sizes `2, 2, 4, 8, 1, 1, 2, 4, 8` bytes respectively. Elements are
fixed-width little-endian bytes.

### 2.1 Additive quantized logical dtypes (WEFT-AI-CODEC §11)

Mode 10 (`TENSOR_SHARDED`) adds two logical dtype names, `q8_0` and `q4_0`.
They are used only with the `weft.ai.tensor_sharded.v1` descriptor; they are
not mode-6 element types and they do not change the dtype table above or any
committed vector. For these dtypes `element_count` is the number of quantized
values, and `values_hash` is

```
H("weft/tensor/values/v1", W || B)
```

where `W` is the whole canonical row-major sequence of quantized values
each sign-extended to one `u8` byte (`q4_0` values are in `-8..=7`; `q8_0`
values are `i8`), and `B` is the concatenation, in block order, of
`u32le(scale_f32_bits) || i64le(zero_point)`. The logical record and
`LogicalTensorRoot` construction are otherwise identical to §3. This additive
type set exists so a sharded quantized tensor has the same layout-independent
identity as a full-precision tensor; it does not define dequantization.

`shape` is an array of `u64` extents. It MUST satisfy:

- rank `1 <= len(shape) <= max_tensor_rank`;
- every extent is non-zero;
- the product of extents does not overflow `u64`;
- the product is `<= max_tensor_elements`.

The **canonical row-major** element order is the C order: the last index varies
fastest. The payload length MUST equal

```
element_count * dtype_size
```

exactly. An element payload of the wrong length is `Malformed`.

Rank 0 is not a v1 logical tensor. It is rejected for every layout; for
`column-major` the rejection is called out explicitly because a scalar has no
column order (§4.3).

## 3. Logical identity

Define

```
values_hash       = H("weft/tensor/values/v1", canonical row-major element bytes)
LogicalTensorRoot = H("weft/tensor/logical/v1", WCB(logical record))
```

where `H(domain, data) = BLAKE3-256(u32le(len(domain)) || domain || data)`
(WEFTCODEC-FORMAT §1.1) and the **logical record** is the WCB map

```
{
  "schema":        "weft.tensor.logical.v1",   text
  "dtype":         <canonical dtype name>,     text
  "shape":         [u64, ...],                 array
  "element_count": <shape product>,            u64
  "values_hash":   [u8; 32],                   bytes
}
```

Validation order and classes (normative):

| Check | Class |
|---|---|
| unknown top-level key | `NonCanonical` |
| `schema` is not `weft.tensor.logical.v1` | `UnknownSchema` |
| unknown `dtype` | `UnsupportedFeature` |
| `shape` not an array of `u64` | `NonCanonical` |
| rank 0 | `NonCanonical` |
| rank above `max_tensor_rank` | `LimitExceeded` |
| zero extent | `NonCanonical` |
| shape product overflow | `LimitExceeded` |
| product above `max_tensor_elements` | `LimitExceeded` |
| `element_count` differs from the shape product | `NonCanonical` |
| `values_hash` is not exactly 32 bytes | `NonCanonical` |
| record does not re-encode to the same bytes | `NonCanonical` |

`element_count * dtype_size` is also checked for overflow (`LimitExceeded`).
The direct encoder additionally requires the supplied row-major payload length
to equal `element_count * dtype_size` (`Malformed`).

The four-layout claim is normative: two artifacts that describe the same
element values with different physical layouts produce the **same**
`LogicalTensorRoot`.

## 4. Layout descriptor

A layout descriptor is the WCB map `weft.tensor.layout.v1`:

```
{
  "schema": "weft.tensor.layout.v1",  text
  "dtype":  <canonical dtype name>,   text
  "shape":  [u64, ...],               array
  "layout": "row-major" | "column-major" | "blocked" | "sharded",
  ... layout-specific fields ...
}
```

`dtype` and `shape` are repeated so a shard is self-describing. They MUST equal
the logical tensor's. The allowed layout-specific fields are:

| `layout` | Additional fields |
|---|---|
| `row-major` | none |
| `column-major` | none |
| `blocked` | `block_shape`: array of `u64`, rank equal to `shape` |
| `sharded` | `axis`: u32, `shards`: u32, `index`: u32 |

Any field belonging to another layout is `NonCanonical`. An unknown layout name
is `UnsupportedFeature`. An unknown schema is `UnknownSchema`. u32 fields are
carried as WCB `u64` and MUST fit `u32`.

### 4.1 Validation

`shape` and `dtype` are validated exactly as in §2. Then:

- `blocked`: `len(block_shape) == len(shape)` and every block extent is `>= 1`,
  otherwise `NonCanonical`.
- `sharded`: `shards >= 1`, `axis < rank`, `index < shards`, and
  `shape[axis] % shards == 0`, otherwise `NonCanonical`.
- `column-major` with a rank-0 shape: `NonCanonical` ("column-major cannot
  express a rank-0 shape").

### 4.2 Physical orders

- **row-major**: linear index
  `((i0 * s1 + i1) * s2 + i2) ...`; last index fastest.
- **column-major**: linear index `i0 + s0 * (i1 + s1 * (i2 + ...))`; first
  index fastest.
- **blocked**: let `g[k] = ceil(shape[k] / block_shape[k])`. Blocks are
  enumerated in row-major order over the block grid. Block `(q0, ...)` covers
  logical `[q[k] * block_shape[k], min((q[k]+1) * block_shape[k], shape[k]))`
  on each axis, and its payload is that block's elements in row-major order.
  Edge blocks are smaller; no padding is stored. The physical buffer is the
  concatenation of block payloads in grid order.
- **sharded** on `axis = a` with `n` shards and index `j`: the shard shape is
  `shape` with `shape[a]` replaced by `shape[a] / n`; the shard covers
  `[j * shape[a]/n, (j+1) * shape[a]/n)` on `a` and the full extent on every
  other axis. Its payload is that sub-tensor in row-major order.

### 4.3 Reassembly

Reassembling all shards in `index` order, or all blocks in grid order, MUST
reproduce the canonical row-major element bytes exactly, and therefore the same
`LogicalTensorRoot`. This equality is a required test in both languages.

## 5. Region addressing

A **region** is a half-open hyper-rectangle

```
Region { start: [u64; rank], end: [u64; rank] }
```

with `0 <= start[k] <= end[k] <= shape[k]` for every axis. A region with equal
start and end on any axis is empty. An invalid rank or bound is `NonCanonical`.

A region read returns the region's elements in **canonical row-major order**
with shape `end - start`. It MUST be assembled from only the physical units the
region intersects:

- `row-major` and `column-major`: one unit (the tensor buffer); zero for an
  empty region;
- `blocked`: exactly the blocks whose logical boxes intersect the region;
- `sharded`: exactly the shard slabs whose range on `axis` intersects the
  requested range on `axis`; zero for an empty region, and in particular a
  region entirely inside one shard touches **exactly one** shard.

The API reports the number of units touched. This count, not a timing, is the
proof that region addressing does not fan out across the whole topology.

## 6. Shard container

A stored shard is a self-describing container:

```
u32le(descriptor_len) || WCB(shard descriptor) || payload
```

The shard descriptor is the layout record with `schema` replaced by
`"weft.tensor.shard.v1"` and two payload fields added:

```
{
  "schema":        "weft.tensor.shard.v1",
  "dtype":         ..., "shape": ..., "layout": ...,
  <layout-specific fields>,
  "payload_bytes": <derived length>,            u64
  "payload_hash":  H("weft/tensor/shard-payload/v1", payload)   bytes32
}
```

`payload_bytes` MUST equal the length derived from the descriptor
(`row-major`, `column-major`, and `blocked` containers carry the whole tensor
payload; a `sharded` container carries exactly its shard). A mismatch is
`NonCanonical`; a payload length different from `payload_bytes` is `Malformed`;
a hash mismatch is `Corrupt`; a non-canonical descriptor is `NonCanonical`.

This container is deliberately **not** a mode-6 `TENSOR` frame. Mode 6 is
frozen to `row_major` and its reject vectors must not change. The container uses
the same additive `u32le || WCB || payload` framing as AI bodies.

### 6.1 Shard-set verification

`shard_root(shards)` accepts a set of decoded shards and returns the logical
root. It MUST:

1. require every descriptor to be `sharded` (`Malformed` otherwise);
2. require all descriptors to agree on `dtype`, `shape`, `axis`, and shard
   count (`NonCanonical` otherwise);
3. require exactly `shards` entries, with indices forming a permutation of
   `0..shards` (`Malformed` for a wrong count, `NonCanonical` for a duplicate,
   missing, or out-of-range index);
4. require every shard payload length to equal its sub-shape payload length
   (`Malformed`);
5. reassemble in index order and return `LogicalTensorRoot`.

**Ordering rule:** shard order in the input sequence is *not* significant. The
descriptor's `index` is authoritative, so a shuffled input is accepted and
reordered deterministically by `index`. Both languages implement this rule
identically and the same conformance vectors are replayed by both.

## 7. Limits and allocation bounds

The layer uses the existing limits:

| Limit | Use |
|---|---|
| `max_tensor_rank` | shape rank and `block_shape` rank |
| `max_tensor_elements` | checked shape product |
| `max_decoded_bytes` | element payloads, assembled tensors, region buffers |
| `max_structured_depth` / `max_structured_items` / `max_string_bytes` | WCB records |

A payload or buffer above `max_decoded_bytes` is `LimitExceeded` **before**
allocation. Checked multiplication is used for every product, stride, offset,
and byte count.

## 8. API surface (normative shape)

Both implementations expose, with language-appropriate naming:

- `logical_root(dtype, shape, row_major_bytes) -> Hash32`
- `LogicalRecord` encode / decode / root
- `LayoutDescriptor` encode / decode / validate
- `shard_root(decoded_shards) -> Hash32`
- `read_region(shape, layout, region) -> bytes` (over an in-memory physical
  arrangement, returning the region bytes and the number of units touched)

Rust: `WeftCodec/src/tensor.rs`. Go: `WeftCodec-Go/tensor`.

CLI verbs (maintainer tools, not wire formats):

```
tensor root       --input FILE --dtype D --shape S [--layout L] [--block-shape B]
tensor root       --shards-dir DIR --dtype D --shape S --axis A --shards N
tensor split      --input FILE --dtype D --shape S --axis A --shards N --out-dir DIR
tensor reassemble --shards-dir DIR --dtype D --shape S --axis A --shards N
                  [--output FILE] [--expect-root HEX]
```

`split` writes `shard-<index>.bin` containers. `reassemble` verifies every
container and the shard set, and fails closed unless the reassembled logical
root matches (and matches `--expect-root` when given).

## 9. Conformance vectors

`conformance/v1/tensor/` (byte-identical in both codec repositories):

- `001`–`003`: valid logical records with `values_hex`, `values_hash`,
  `canonical_hex`, and `logical_root`;
- `101`–`110`: logical rejects (rank 0, zero extent, product overflow,
  element-count mismatch, unknown schema, unknown dtype, short hash, unknown
  key, rank limit, element limit);
- `201`–`204`: valid layouts (row-major, column-major, blocked, sharded);
- `301`–`310`: layout rejects (unknown layout, non-divisible shards, index,
  zero shards, block rank, zero block, rank-0 column-major, extra field, axis,
  shape overflow);
- `401`–`402`: shard sets with per-shard payloads, canonical layout
  descriptors, byte-identical containers, and the expected logical root.

The Rust generator is
`cargo run -- conformance --dir conformance/v1 --tensor-only`; the Go suite
replays the files unchanged.

## 10. Non-goals and stated limitations

- This layer does not read or write a filesystem, object store, or network
  transport. FUSE/store integration is separate work.
- It does not claim a storage-cost win. The benchmark measures identity and
  region addressing only; all buffers are synthetic and in memory.
- It does not change mode-6 `TENSOR` or any frozen frame vector.
- It does not define cross-vendor dtype equivalence, lossy layouts, padding,
  strides, or partially resident tensors.
- `blocked` requires full block extents to be derivable from `shape` and
  `block_shape`; arbitrary irregular block lists are not representable.
- The Go representation-manifest v2 codec mode is unrelated and unchanged; this
  layer is present in both codecs independently.
