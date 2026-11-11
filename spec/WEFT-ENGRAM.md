# WEFT-ENGRAM v1.0.0 — incrementally updateable, region-addressable row-group tables

Status: **implemented** in Rust `weftcodec` and Go `weftcodec-go` (codec);
store integration is implemented in the Rust store (writer, region reads,
incremental update) and in the Go store for reads, verification, and region
reads — Go has no mode-20 writer. Mode value **20** in `WEFT-AI-CODEC.md`.

## 1. The workload this mode addresses

Large n-gram / embedding-lookup tables (the `Engram`-class workload) are
~10^2 GB, are read by a sparse set of row groups, and are updated a few groups
at a time. Stored as one monolithic array they must either be mapped whole or
gathered through the page cache; an update rewrites the whole object.

Mode 20 `ENGRAM_SHARD` stores the table as an ordered list of **row groups**.
Each group is independently encoded, independently rooted and independently
decodable. A read fetches only the groups it touches; an incremental update
writes only the changed groups and references the unchanged groups by their
existing roots.

This document is normative for the wire format. It is not a claim about cache
behaviour, throughput, or that supersession reduces bytes without the measured
delta in `bench/RESULTS-engram.md`.

## 2. Frame shape

```text
domain        = 0 (payload)
mode          = 20
flags         = 0
dep_count     = 0
chunk_count   = 0
decoded_len   = rows * dim * element_size
content_hash  = H("weft/chunk/v1", whole canonical row-major table bytes)
body          = u32le(descriptor_len) || WCB(descriptor) || stored_payload
```

`stored_payload` is the concatenation of the stored (non-reference) groups in
canonical group order. `descriptor.payload_len` is exactly its byte length.
The frame carries no dependency table and no chunk table.

## 3. Descriptor

Schema `weft.ai.engram_shard.v1`. Exactly these keys, no others:

| Key | Type | Rule |
|---|---|---|
| `schema` | text | exactly `weft.ai.engram_shard.v1` |
| `hash_function` | text | one of the registry below; an unknown id is `UnsupportedMode` |
| `vocab_size` | u64 | `> 0` and `<= table_shape[0]` |
| `table_shape` | `[u64; 2]` | `[rows, dim]`; rank exactly 2; both extents `> 0`; product within `max_tensor_elements` |
| `element_dtype` | text | a `WEFT-TENSOR` element dtype |
| `precision` | map | exactly `storage_dtype` and optional `quantization` |
| `row_group_edges` | `[u64]` | length `group_count + 1`; strictly increasing; first `0`; last `rows` |
| `group_count` | u64 | exactly `row_group_edges.len() - 1` |
| `groups` | array | exactly `group_count` entries in canonical order |
| `payload_len` | u64 | exact stored payload length |
| `payload_hash` | bytes32 | `H("weft/engram/payload/v1", stored_payload)` |

### 3.1 `hash_function` registry

| Value | Meaning |
|---|---|
| `ngram-sha256-v1` | the workload derives n-gram keys with SHA-256 |
| `ngram-blake3-v1` | the workload derives n-gram keys with BLAKE3 |

The registry is **descriptive**: the codec never hashes n-gram keys, and
`hash_function` does not change a group root or the payload. It is part of the
descriptor identity so two tables built for different keying schemes cannot be
silently interchanged. Any other value is `UnsupportedMode`.

### 3.2 `precision`

Exactly these keys:

| Key | Type | Rule |
|---|---|---|
| `storage_dtype` | text | the exact storage element dtype. In v1 it MUST equal `element_dtype`; a difference is `UnsupportedFeature` (no storage-dtype transformation is implemented). |
| `quantization` | map, optional | when present, exactly `scheme` (text) and `lossless` (bool) |

Implemented quantization schemes: `none` (with `lossless = true`). Any other
scheme, or `lossless = false`, is `UnsupportedFeature`: a declared lossy
transformation is explicitly rejected rather than silently accepted. The group
root's `precision` field is `storage_dtype`; a lossless `quantization` block
therefore does not change group identities but does change the descriptor root.

### 3.3 Group entry

Each `groups` element has exactly these keys:

| Key | Type | Rule |
|---|---|---|
| `index` | u64 | canonical group index; equals its array position |
| `row_start` | u64 | equals `row_group_edges[index]` |
| `row_end` | u64 | equals `row_group_edges[index + 1]` |
| `storage` | text | `"inline"` or `"reference"` |
| `offset` | u64 | inline: byte offset inside `stored_payload`; reference: `0` |
| `compressed_len` | u64 | inline: stored bytes, `> 0`; reference: `0` |
| `group_root` | bytes32 | see §4; never all-zero |
| `supersedes` | null or bytes32 | the group root this group replaces, or null |

Inline groups must have contiguous offsets: the first inline `offset` is `0`,
each next inline `offset` is the previous inline `offset + compressed_len`, and
the last inline end equals `payload_len`. References consume no payload bytes.

`row_start`/`row_end` are row indices in the table, half-open. Group `i` holds
rows `[edges[i], edges[i+1])` and therefore `(edges[i+1] - edges[i]) * dim`
elements. Every group's element count must not exceed
`Limits::max_engram_group_elements` (default `2^26`); the group count must not
exceed `max_structured_items`.

## 4. Group identity and payload

A group's decoded bytes are its canonical row-major element bytes
(`group_rows * dim * element_size`). It is split into `element_size` byte
planes: plane `p` is byte `p` of every element in row-major order. Each plane is
coded independently with `huffman-canonical-v1` (its own 256-byte table and
bitstream). A stored group is the concatenation of its planes in plane order.
The decoder consumes exactly the plane's own table and stream and rejects
trailing bytes inside the declared group range.

This is the same exact per-plane coder mode 15 `TENSOR_TILED` uses (§16 of
`WEFT-AI-CODEC.md`). It is reused because it is already frozen, independently
decodable, lossless for every element byte pattern, and has a strict decoder
that rejects truncation and trailing bytes; no new entropy coder is introduced.

```text
values_hash = H("weft/tensor/values/v1", group canonical row-major bytes)
group_root  = H("weft/engram/group/v1",
                WCB({group_index, row_start, row_end,
                     dtype, precision, values_hash}))
```

`WCB` keys are exactly `group_index` (u64), `row_start` (u64), `row_end`
(u64), `dtype` (text, `element_dtype`), `precision` (text,
`precision.storage_dtype`) and `values_hash` (bytes32).

The group root binds the group's index, its exact row range, the element dtype,
the storage precision and the decoded values. A group fetched from the wrong
artifact, placed at a different index or decoded under a different dtype can
therefore never validate.

## 5. Identity

```text
descriptor_root = H("weft/engram/v1", WCB(descriptor))
payload_hash    = H("weft/engram/payload/v1", stored_payload)
```

`descriptor_root` is derived, not stored (a field holding its own hash would be
circular); the decoder recomputes it and returns it with the decoded table.
`payload_hash` is stored in the descriptor and compared to the bytes actually
read.

## 6. Decode paths

### 6.1 Full decode

Full decode verifies, in order:

1. every inline group's root (by decoding its planes and recomputing the root),
2. every reference group's root after the resolver returns its exact bytes,
3. every supersession digest resolves (see §7),
4. the assembled table's `decoded_len` and frame `content_hash`.

### 6.2 Group and row-region reads

A group read takes a set of group indices. A row-region read takes a half-open
`[row_start, row_end)` range and computes exactly the groups whose row interval
intersects it. Either path:

1. reads the frame prefix (header plus descriptor),
2. reads only the stored ranges of the groups it touches (and asks the
   resolver only for referenced groups it touches),
3. decodes only those groups, verifying each `group_root`,
4. returns the touched group indices and the canonical bytes.

Untouched groups are never read or decoded. A range reader that does not
materialise the whole frame **cannot** verify `payload_hash`, the frame trailer
or the whole-table `content_hash`; the per-group roots are the exactness
guarantee on that path. The implementation states this explicitly in its
region-read documentation and in `bench/RESULTS-engram.md`. Full decode
continues to verify all three.

## 7. Supersession and reference encoding

`supersedes` records that a group replaces an earlier group. It is a group-root
digest, not a table root, so it names exact rows, dtype, precision and values.

Rules:

* A non-null `supersedes` digest MUST resolve to a group root that the decoder
  can see. "Sees" means either a `group_root` of this descriptor, or a group
  supplied by the caller-supplied resolver.
* The supersession graph over the group roots visible to the decoder MUST be
  acyclic. A cycle is `Corrupt`.
* A digest that resolves to nothing is `Corrupt` on full decode. A reference
  group whose bytes are absent is `DependencyMissing`.

`storage: "reference"` lets an incremental update emit a group without storing
its bytes. The reference is the group root alone; the resolver returns the exact
group bytes and the decoder verifies the root (including index, rows, dtype,
precision and values). A reference to a group with different edges, dtype or
precision therefore fails closed.

A store that references an earlier artifact SHOULD record that earlier artifact
as a representation dependency, so reachability-based reclamation retains it.
The store integration in `src/tensor_store.rs` does this; the raw codec does not
require a frame dependency table.

## 8. Incremental update

`update_engram_groups(object, changed)` (store API) reads the current table,
replaces the rows of the listed groups with caller-supplied bytes, and publishes
a new object version whose mode-20 frame:

* stores the changed groups inline (their new roots),
* emits every unchanged group as `storage:"reference"` with its existing root,
* sets `supersedes` on a changed group to the root of the group it replaced
  (when the previous version is known),
* carries the same `table_shape`, `element_dtype`, `precision` and edges.

The previous version is not rewritten. The new frame's payload contains only
the changed groups; the new descriptor and frame are the only other new bytes.
The new whole-table `content_hash` and the store's `logical_root` reflect the
updated table. Unchanged groups retain their group roots.

## 9. Validation order (normative for error classes)

1. Frame header/tables (`Malformed`, `UnsupportedVersion`, `UnsupportedFeature`);
   mode must be 20.
2. Body prefix shorter than 4 bytes or descriptor past the body (`Malformed`).
3. WCB decode of the descriptor (`Malformed`, `NonCanonical`).
4. Exact key set, value types, `schema` (`Malformed`).
5. `hash_function` registry (`UnsupportedMode`).
6. `table_shape` rank/extents/product, `element_dtype`, `vocab_size`
   (`Malformed`, `LimitExceeded`).
7. `precision`: key set, `storage_dtype` support and equality, quantization
   scheme and `lossless` (`Malformed`, `UnsupportedFeature`).
8. `row_group_edges` strictly increasing / first zero / last `rows`;
   `group_count`; per-group element bound (`Malformed`, `LimitExceeded`).
9. `groups` count, canonical `index`, `row_start`/`row_end`, `storage` spelling,
   offset/compressed-length rules, contiguous inline ranges, `payload_len`
   (`Malformed`).
10. `group_root` and `supersedes` lengths; all-zero roots (`Malformed`).
11. `payload_hash` over the stored payload (`Corrupt`).
12. `decoded_len == rows * dim * element_size` (`Corrupt`).
13. Per-group plane decode and `group_root` (`Corrupt`, `NonCanonical`).
14. Reference resolution; missing reference (`DependencyMissing`), wrong bytes
    (`Corrupt`).
15. Supersession resolution and acyclicity (`Corrupt`).
16. Whole-table `content_hash` (`Corrupt`).

## 10. Limits

`max_tensor_elements`, `max_tensor_rank`, `max_decoded_bytes`,
`max_structured_items` (group count), `max_string_bytes` (descriptor) and
`max_frame_bytes` apply. The new `max_engram_group_elements` bounds one
group's element count before allocation. A prefix reader bounds the descriptor
read by `max_string_bytes`.

## 11. Conformance vectors

`conformance/v1/engram/` (byte-identical in both codec repositories):

Valid:

* `001_engram_bf16_one_group` — bf16, one group covering the whole table.
* `002_engram_fp16_many_groups` — fp16, four groups, uneven row counts.
* `003_engram_fp32_basic` — fp32, two groups.
* `004_engram_empty_supersession` — explicit null supersession for every group.
* `005_engram_chain_two` — group 1 supersedes group 0's root (an in-frame,
  acyclic chain of length two).
* `006_engram_region_subset` — a row-region read that touches a strict subset.

Rejects:

* `101_edges_not_increasing`
* `102_last_edge_not_rows`
* `103_group_count_mismatch`
* `104_group_root_mismatch`
* `105_payload_truncated`
* `106_payload_hash_mismatch`
* `107_supersession_cycle`
* `108_supersession_missing`
* `109_supersession_incompatible`
* `110_unknown_hash_function`
* `111_unknown_precision`
* `112_reserved_mode_21`
* `113_non_canonical_descriptor`
* `114_reference_missing`

The reserved-mode reject moved from mode 20 to mode 21;
`017_unknown_mode_reserved20` is retained with `superseded_by`.

## 12. Non-claims

* No claim about cache behaviour, page-cache residency, TLB behaviour, or
  access-pattern privacy.
* No claim that supersession reduces bytes without the measured delta in
  `bench/RESULTS-engram.md`.
* No claim that a group is the optimal update granularity for any workload; the
  caller chooses the edges.
* No encryption, confidentiality, or authorization semantics. The store's
  capability layer is separate and unchanged.
* No claim about a distributed or multi-writer engram table; the store path is
  the existing single-writer object model plus MW2 transactions where enabled.
