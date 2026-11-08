# WEFT-MODEL v1.0.0 — identity, state, transactions, durability

**Status:** Normative for the v1 single-host store. Some safety properties are
model-checked or tested; claims are labelled. Where a property is not proven,
this document says so.

---

## 1. Entities

| Entity | Definition | Stability |
|---|---|---|
| `NamespaceId` | Text name scoped to one store; v1 has `default` | configured |
| `ObjectId` | 32 bytes; `H("weft/object/v1", seed)` or explicitly supplied | immutable, independent of path/content |
| `ContentId` | `H("weft/chunk/v1", exact bytes)` for a whole file | immutable |
| `ManifestId` | `H("weft/manifest/v1", WCB(manifest))` | immutable |
| `VersionId` | `H("weft/version/v1", WCB(version record))` | immutable |
| `RelationId` | `H("weft/relation/v1", WCB(assertion))` | immutable assertion |
| `Context` | A selector AST (`weft.selector.v1`) | expression identity |
| `SelectorId` | `H("weft/selector/v1", WCB(selector))` | expression identity |
| `SnapshotId` | The commit root hash of an immutable snapshot | immutable |
| `ResolvedRef` | `(namespace, snapshot, object, version, content)` | pinned |
| `Principal` | `(uid, gid, supplementary groups)` from the kernel | per-operation |

A **hard link** creates a second directory entry (`EntryId`) targeting the same
`ObjectId`. **Copy** creates a new `ObjectId` (except an explicit
`copy --reflink` which creates a new object whose content root is shared; the
identity remains distinct). A **rename** changes an entry's parent/name and
never its target `ObjectId`.

The namespace root directory has `ObjectId = 32 zero bytes` and is created in
generation 1.

---

## 2. On-disk layout (portable; Rust and Go MUST agree)

```
FORMAT                         text: "weftfs-store 1\n"
LOCK                           empty; flock(LOCK_EX) for writers
superblock                     8192 bytes = two 4096-byte slots
commits/<generation>.commit    immutable commit records
objects/<ab>/<hash>            immutable raw chunks (64 KiB boundaries in v1)
manifests/<ab>/<hash>          canonical WCB manifests (may be zstd-framed)
refs/snapshots/<name>          pointer file -> commit hash (text hex)
refs/tags/<name>               pointer file -> object version (text hex)
tmp/                           staging for atomic writes; cleaned on open
```

`ab` is the first two lowercase hex characters of the hash. Files are written
with `O_CREAT|O_EXCL` under `tmp/`, `fsync`ed, renamed into place, and the
containing directory `fsync`ed. Existing object files are verified, never
overwritten.

### 2.1 Superblock slots

Each 4096-byte slot:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | magic `WFSB` |
| 4 | 2 | format major = 1 |
| 6 | 2 | format minor = 0 |
| 8 | 8 | generation (u64le) |
| 16 | 32 | commit root hash |
| 48 | 8 | commit record length |
| 56 | 8 | commit timestamp (i64le ns since epoch) |
| 64 | 4 | CRC32C over bytes `[0,64)` |
| 68 | 4028 | zero |

Publishing a commit:
1. write and fsync all new object/manifest files;
2. write `commits/<gen>.commit` (header `WFC1`, u64le length, WCB commit, u32le
   CRC32C of the WCB bytes), fsync file;
3. fsync `commits/`;
4. write the superblock slot `gen % 2`, fsync `superblock`;
5. fsync the store root directory.

A commit is visible only after step 5. Recovery selects the highest generation
whose slot CRC is valid, whose commit file exists, whose record verifies, and
whose referenced object files all exist. Otherwise it falls back to the
previous valid generation and reports the loss boundary. It never "repairs" by
inventing data.

Recovery MAY use a **bounded** replay window: start at the selected head
generation, read `commits/<generation>.commit` downward, and stop at the newest
commit whose `checkpoint` verifies (declared `generation` equal to the commit's
generation, manifest hash and length valid). In that mode the implementation
need not enumerate commit files below the checkpoint, because the checkpoint
state is content-addressed, hash-verified, and pinned to that generation. The
recovered state is therefore exact for the head.

Bounded replay MUST be reported: an implementation that stops at a checkpoint
MUST surface a recovery note naming the checkpoint generation and stating that
older commit files were not enumerated. Any inconsistency found while walking
the window (missing file, generation/name disagreement, hash or parent
mismatch, truncated record, unverifiable or mislabelled checkpoint) MUST fall
back to the full enumeration path, which is authoritative for its rejected set
and notes. `verify --full` MUST enumerate every commit file and report gaps in
older history that bounded open did not observe.

### 2.2 Commit record

Canonical WCB map:

```
{
  "schema": "weftfs.commit.v1",
  "generation": u64,
  "parent": bytes32 | null,
  "timestamp_ns": timestamp,
  "ops": [ op, ... ],
  "checkpoint": bytes32 | null
}
```

A commit MAY instead carry schema `weftfs.commit.v2`, which is the v1 key set
plus one optional field:

```
{
  "schema": "weftfs.commit.v2",
  "generation": u64,
  "parent": bytes32 | null,
  "timestamp_ns": timestamp,
  "ops": [ op, ... ],
  "checkpoint": bytes32 | null,
  "index_attestation": map | null
}
```

`index_attestation` records the shape of the advisory selector index the commit
published, so an index segment can be trusted only when the commit says what it
must contain:

```
{
  "object_count": u64,
  "relation_count": u64,
  "entry_count": u64,
  "state_root": bytes32,
  "candidates_root": bytes32
}
```

`state_root` is `H("weft/index/state/v1", WCB)` over the six v1 fields (schema
`weftfs.commit.v1`, generation, parent, timestamp_ns, ops, checkpoint) with no
attestation, so a v1 and a v2 commit with the same fields share it.
`candidates_root` is `H("weft/index/candidates/v1", sections)` over the WFX2
candidate section bytes. The index segment layout is unchanged (internal,
advisory, rebuildable).

**Migration / compatibility.** `weftfs.commit.v1` remains fully valid and is
still read and written (a store may be configured to write only v1, which is
how the frozen conformance fixture is reproduced). A v1 commit carries **no**
attestation: any index segment named by it is rejected on load and rebuilt —
never trusted. Both implementations here read and write v2. Adding the field
changes the commit identity (`H("weft/store/commit/v1", WCB)`, domain
unchanged), so a v2 commit is a different hash than the same v1 commit; old
stores keep their v1 chain and continue to replay. A reader that checks exact
map keys MUST be updated before it can read a v2 store.

`checkpoint` is the manifest hash of a full state snapshot (§2.4) used as a
replay base. Without it, replay starts at generation 1. Checkpoints are written
every 256 commits (configurable) and by `weftfs checkpoint`. A checkpoint named
by a commit MUST describe the state **after** applying that commit's ops and
MUST declare `generation` equal to the commit's generation. A mismatch is
`Corrupt`, and replay ignores the checkpoint and falls back to an earlier base.
When replay reaches a commit whose checkpoint verifies, it loads that state and
**skips** that commit's own ops (they are already included), then continues with
the next generation.

Ops (maps with `"op"` text):

| `op` | Fields |
|---|---|
| `put_object` | `id:bytes32`, `kind:text`, `mode:u64`, `uid:u64`, `gid:u64`, `mtime_ns:timestamp`, `version:bytes32`, `content:bytes32`, `size:u64`, `attrs:map` |
| `delete_object` | `id:bytes32` |
| `put_entry` | `id:bytes32`, `parent:bytes32`, `name:bytes`, `target:bytes32` |
| `del_entry` | `id:bytes32` |
| `put_relation` | `id:bytes32`, `subject:bytes32`, `predicate:text`, `kind:text` (`object`|`value`), `object:bytes32|scalar`, `provenance:map|null`, `attrs:map` |
| `del_relation` | `id:bytes32` |
| `set_snapshot` | `name:text`, `commit:bytes32` |
| `set_tip` | `object:bytes32`, `version:bytes32` |
| `set_quota_usage` | `reset:bool`, `usage:[{kind:text, uid:u64?, gid:u64?, tenant:text?, used_bytes:u64, used_objects:u64}, ...]` |

Ops apply in array order. A commit whose ops cannot be applied (missing parent,
duplicate entry name, dangling target) is invalid; it is not partially applied.

`set_quota_usage` is an **accounting assertion**, not namespace state. It
MUST be the last op in its commit and MUST describe the cumulative admitted
usage after every other op in that commit. A reset (`reset: true`) MUST be the
only op in its commit and MUST declare an empty `usage` array; it establishes a
new accounting baseline. The `usage` array is sorted by principal key
(`p:%020u:%020u` before `t:<name>`) with no duplicates. The commit schema stays
`weftfs.commit.v1`; older commits contain no usage op and replay unchanged.
Replay recomputes the admission totals from the object ops and rejects an
assertion that disagrees as `Corrupt`. A commit that carries no assertion is
valid: replay derives usage from its ops.

### 2.3 File content manifests

Files are split into 64 KiB chunks in v1 (content-defined chunking is an
experimental profile, not v1 persistence). The manifest is:

```
{
  "schema": "weft.manifest.v1",
  "size": u64,
  "chunk_size": u64,
  "chunks": [ {"hash":bytes32,"len":u64}, ... ],
  "content_hash": bytes32
}
```

Manifest identity is the manifest hash; whole-content identity is
`H("weft/chunk/v1", plaintext)` and MUST be verified after assembly. Empty files
have zero chunks and content hash of the empty byte string.

### 2.3.1 Representation manifests (experimental, opt-in)

A store MAY persist a file through a **representation manifest**, schema
`"weft.manifest.v2"` (one spelling, no alias), which has the v1 fields plus a
canonical WCB `representation` map:

```
{
  "kind": "raw" | "codec",
  "frame": bytes32 | null,        # WFT1 frame blob for codec
  "dependencies": [bytes32, ...], # base content roots (manifest ids)
  "strategy": "raw" | "zstd" | "delta" | "reference" | "zstd-patch",
  "decoded_size": u64            # MUST equal size
}
```

Rules:

- A v1 manifest MUST NOT contain `representation`; a v2 manifest MUST.
  `kind:"raw"` is a v2 manifest whose `chunks` are the physical layout.
  `kind:"codec"` stores bytes only inside the named frame and has no chunks.
- `strategy` MUST equal the decoded frame's payload mode; a mismatch is
  `Corrupt`. The frame is validated with the codec's own limits.
- `zstd-patch` is an additive strategy: the frame is mode 4 `DICTIONARY`, the
  single dependency is the whole base as a raw-content dictionary, and the
  encoder MUST set a zstd window that covers that base (the
  `zstd --patch-from` analogue). No existing byte changes: v1 manifests, and
  v2 manifests using the four original strategies, are unaffected.
- `dependencies` are content roots (manifest ids). Decoding a dependent MUST
  fail closed if a dependency is missing, unreadable, or cyclic; dependency
  depth MUST be bounded (default 8). No approximate or partial reconstruction
  is permitted. A `zstd-patch` decoder MUST bound its decompression window by
  the already-loaded dependency, never by a frame-declared size.
- `content_hash` remains the frozen whole-content digest
  `H("weft/chunk/v1", plaintext)` and MUST verify over the decoded bytes.
- Retention MUST follow dependencies: a GC that keeps a dependent MUST keep
  its transitive dependency closure, or refuse the deletion.
- Writers MUST select this representation explicitly. Absent that selection,
  the v1 chunked manifest is written and remains byte-identical.
- v2 is **experimental**; an implementation that does not understand it MUST
  reject the store/manifest (`UnsupportedVersion`) rather than guess.

#### Tensor representations (additive within v2)

A v2 manifest MAY use `kind:"tensor"` to name a WeftTensor frame as the
physical representation. `kind:"raw"` and `kind:"codec"` are unchanged; the
tensor key set is exactly:

```
{
  "kind": "tensor",
  "frame": bytes32,               # WFT1 frame blob id, never null
  "tensor_mode": 10|11|12|13|14,  # u32 carried as WCB u64
  "dtype": text,                  # WEFT-TENSOR dtype, or q8_0/q4_0
  "shape": [u64, ...],            # logical extents
  "logical_root": bytes32,        # WEFT-TENSOR LogicalTensorRoot
  "layout": map,                  # canonical weft.tensor.layout.v1
  "dependencies": [bytes32, ...], # base content roots for 11 and 14
  "decoded_size": u64,            # logical element bytes; MUST equal size
  "strategy": text                # mode name (below)
}
```

The `strategy` text is fixed by `tensor_mode`: `10 = "tensor-sharded"`,
`11 = "weight-delta"`, `12 = "kv-cache"`, `13 = "activation"`,
`14 = "semantic-delta"`. A mismatch is `Malformed`; a frame whose actual mode
disagrees with `tensor_mode` is `Corrupt` on read. No other mode is legal, and
no planner bookkeeping field may be added to this map.

Rules:

- `frame` MUST be a manifest-domain blob; the reader verifies its blob id and
the frame's own checksum, mode, and descriptor.
- `dtype` and `shape` MUST agree with the frame descriptor. The logical bytes
are the canonical row-major element bytes implied by `(dtype, shape)`; for
`q8_0` the bytes are one sign-extended byte per logical element and the root is
the additive quantized root of `WEFT-AI-CODEC.md` §11.5. `decoded_size` MUST
equal the manifest `size` and the decoded logical byte length.
- `layout` MUST be the canonical `weft.tensor.layout.v1` map from
`WEFT-TENSOR.md` §4, with `dtype`/`shape` equal to the manifest fields. The
store canonicalises the caller's physical arrangement to row-major before
writing, so the same tensor written row-major, column-major, blocked, or as a
single shard has the same `logical_root` (`WEFT-TENSOR.md` §3).
- `dependencies` follows the frozen v2 content-root rule. Modes 11 and 14 have
exactly one dependency: the base representation's content root. The base frame
is reconstructed deterministically as a self-contained mode-6 `TENSOR` frame
over the base's canonical element bytes; its content hash MUST equal the delta
descriptor's `base`. No extra base-frame blob is stored. Modes 10, 12, and 13
have no dependencies. Missing, unreadable, cyclic, or mismatched dependencies
fail closed with `Corrupt`; depth is bounded by the codec dependency bound.
- `content_hash` remains the frozen whole-content digest over the decoded
logical bytes and MUST verify on a full read. `verify --full` decodes every
tensor representation, recomputes `logical_root`, and checks dtype/shape,
frames, and dependencies.
- Retention follows dependencies, exactly as for `kind:"codec"`: a live tensor
dependent keeps its base's transitive closure and frame blobs.
- **B1 choice:** one object per tensor with exactly one frame. A multi-shard
tensor is therefore not a single bulk object; a `sharded` write with
`shards > 1` is rejected with `InvalidArgument` rather than silently
truncated. Modes 12 and 13 are readable but are not produced by the B1 writer
(they require workload metadata outside the tensor contract).
- A tensor written through `kind:"raw"` or `kind:"codec"` (the planner may
choose either) records the same identity in reserved object attributes:
`weft.tensor.dtype`, `weft.tensor.shape`, `weft.tensor.logical_root`,
`weft.tensor.layout` (canonical WCB bytes), and `weft.tensor.strategy`. These
are ordinary additive attributes, not manifest fields.
- `q8_0` is only stored through mode 10 because its additive identity binds the
mode-10 scale plane; `q4_0` store support is not implemented in B1.
- **The planner is not part of the format.** The cost function
`physical_bytes + lambda_ns * estimated_decode_ns` and the chosen/runner-up
prices are exposed by `weft plan --json` and `Store::plan_tensor` only. The
manifest records exactly the chosen representation, never a cost table.

### 2.3.2 Merkle content identity (v3, opt-in)

A writer MAY persist a raw chunk table under schema `"weft.manifest.v3"`
(one spelling, no alias) so that a small range write costs O(changed chunks +
tree hashing) instead of O(file size):

```
{
  "schema": "weft.manifest.v3",
  "size": u64,
  "chunk_size": u64,
  "chunks": [ {"hash":bytes32,"len":u64}, ... ],
  "content_root": bytes32,
  "content_identity": "merkle-v1",
  "representation": map            # optional, kind:"raw" only
}
```

The key set is exactly the fields above: `content_root` replaces `content_hash`,
`content_identity` is required and MUST be `"merkle-v1"`, and
`representation` may be omitted. A v3 manifest MUST NOT carry a frame-backed
(codec or tensor) representation; that is `Corrupt`.

Construction (all hashes are the domain-separated construction of §2.3.1):

```
chunk_hash_i = H("weft/chunk/v1", chunk_bytes_i)      # unchanged chunk domain
leaf_i       = H("weft/merkle/leaf/v1", u64le(chunk_len_i) || chunk_hash_i)
parent(l,r)  = H("weft/merkle/node/v1", l || r)
empty_root   = H("weft/merkle/empty/v1", empty)
```

The tree is built over chunk order. At every level, nodes are paired left to
right; an odd trailing node is **promoted unchanged** to the next level (it is
never duplicated and never hashed with itself). With one chunk the root is that
chunk's leaf. An empty file has zero chunks and root `empty_root`.

`content_root` MUST equal the fold of the manifest's own chunk table; a
mismatch is `Corrupt` on parse. Because `chunk_size` and each chunk's `len` are
part of the leaf preimage, the root is deliberately **chunk-size dependent**:
writing the same bytes with a different `chunk_size` produces a different root.
The chunk size is recorded in the manifest, so the identity is unambiguous.

Rules:

- A file's `ContentId` for a v3 manifest is `content_root`. The manifest id
  remains `H("weft/manifest/v1", WCB(manifest))`.
- A range write re-hashes and re-stores only chunks whose bytes change; every
  other chunk reference is reused verbatim and its blob is not read. The root
  is folded from the manifest's stored chunk hashes plus the changed leaves; no
  whole-file read is performed for hashing.
- A full read verifies every chunk blob and recomputes the root from the chunk
table. A range read verifies the touched blobs and their declared hashes.
`verify_region(content_root, offset, len)` verifies the touched blobs, then
recomputes the root from all chunk hashes (O(#chunks) small hashes) and compares
it with the declared root; no per-level sibling table is stored. A failure
names the chunk index.
- A v3 file MUST NOT be rewritten as v1/v2 implicitly. Conversion is an
explicit operation; absent one, every rewrite of a v3 file keeps v3. Reading
dispatches by schema and an unknown schema is `UnsupportedVersion` (fail
closed).
- The default remains the frozen v1 whole-file identity. A writer selects v3
only through the `content_identity = merkle` option / `--content-identity
merkle` flag; otherwise v1 (or v2) bytes are unchanged.
- v3 is byte-identical across implementations: the same bytes and `chunk_size`
MUST produce the same root in Rust and Go.

### 2.3.3 Immutable overlay manifests (v4, experimental)

A writer MAY represent a file as an immutable **base** manifest plus an ordered
list of immutable **extents**, schema `"weft.manifest.v4"` (one spelling, no
alias). The key set is exact:

```
{
  "schema": "weft.manifest.v4",
  "size": u64,
  "chunk_size": u64,
  "content_root": bytes32,
  "content_identity": "merkle-v1",
  "representation": map
}
```

A v4 manifest carries **no top-level chunk table**. The `representation` map is
`kind:"overlay"` with exactly these fields:

```
{
  "kind": "overlay",
  "base": bytes32,               # previous manifest id
  "base_content_root": bytes32,
  "base_size": u64,
  "depth": u64,                  # chained overlay manifests, this one included
  "overlay_bytes": u64,          # total logical bytes carried by the chain
  "decoded_size": u64,           # MUST equal size
  "extents": [ extent, ... ],
  "strategy": "overlay"
}

extent = {
  "offset": u64,                 # chunk-grid aligned
  "length": u64,                 # positive
  "chunks": [ {"hash":bytes32,"len":u64}, ... ]
}
```

Rules:

- **Base binding.** `base` MUST be the manifest id of an immutable base, and
  `base_content_root` / `base_size` MUST equal that base manifest's
  `content_root` / `size`. The base MUST itself be a Merkle chunk table: a
  `weft.manifest.v3` raw manifest or another v4 overlay. A frame-backed v2
  representation, a v1/v2 whole-file manifest, a missing base, or a mismatched
  base is `Corrupt` on read. A base chain MUST be acyclic.
- **Depth.** `depth` MUST equal one for a v3 base and `base.depth + 1` for a v4
  base. Readers MUST reject a chain deeper than 256 manifests with
  `LimitExceeded`; a writer MUST compact rather than exceed that bound. A
  writer MAY enforce a stricter configured bound (the Rust store defaults to 64)
  and MUST refuse with `LimitExceeded` naming compaction rather than publish a
  chain that exceeds it. A writer MUST also bound the total logical bytes
  carried by the chain (default 1 GiB in the Rust store) and refuse past that
  bound rather than let read amplification grow without a limit.
- **Extent grid.** Extents in one manifest MUST be sorted by ascending
  `offset`, MUST NOT overlap, and MAY be adjacent (`end == next.offset`).
  `offset` MUST be a multiple of `chunk_size`; `length` MUST be positive and
  MUST end at EOF or on a chunk boundary. Each listed chunk MUST have the full
  `chunk_size` except the final chunk of the file. The chunk count MUST equal
  `ceil(length / chunk_size)` and the chunk lengths MUST sum to `length`. A
  manifest lists between 1 and 64 extents. Non-canonical ordering or overlap
  is `Malformed`; an extent past EOF, a zero-length chunk, a chunk-count or
  length-sum mismatch, or a chunk-length disagreement is `Corrupt`.
- **Accounting.** `overlay_bytes` MUST equal the base chain's `overlay_bytes`
  (zero for a v3 base) plus the sum of this manifest's extent lengths. A value
  smaller than either component is `Corrupt`.
- **Derived table.** The canonical chunk table starts from the base's canonical
  table and applies this manifest's extents in order. An extent replaces the
  chunk references at its grid positions and may append past the base's end;
  a hole between the base end and an extent is `Corrupt`. The resulting table
  MUST sum to `size` and MUST have every chunk full except the final one.
- **Logical identity.** `content_root` MUST equal the Merkle root of the derived
  canonical chunk table. This is exactly the root a canonical single-table
  `weft.manifest.v3` over the same logical bytes at the same `chunk_size`
  produces. A mismatch is `Corrupt`. Overlap between layers follows
  newest-wins: the later (higher) layer's extent replaces the earlier
  representation at those chunks.
- **Reads.** Resolving an overlay reads manifest blobs only; it MUST NOT fetch
  a content chunk merely to reconstruct the table. A range read MUST fetch only
  the chunks intersecting the requested range, verify each chunk blob against
  its declared hash, and, for a whole-file read, recompute the Merkle root.
- **Retention.** While any object, snapshot, checkpoint, or pin reaches a v4
  manifest, GC MUST retain that manifest blob, its base chain, and every chunk
  named by every retained layer. A chunk or base manifest reachable only
  through an unreferenced overlay is collectable.
- **Additive.** v1, v2, and v3 encodings and decodings are unchanged. A v4
  manifest is written only when overlay representation is selected explicitly;
  absent that selection the existing writers emit their existing bytes. v4 is
  experimental: an implementation that does not understand it MUST reject it
  with `UnsupportedVersion` rather than guess.

### 2.3.1 Derived identities

```
version_record = {"schema":"weftfs.version.v1", "object":bytes32,
                  "content":bytes32, "size":u64, "mtime_ns":timestamp,
                  "parent":bytes32|null}
VersionId      = H("weft/version/v1", WCB(version_record))

ObjectId       = H("weft/object/v1", seed)       # seed = 32 random bytes by
                                                  # default; tests may supply
EntryId        = H("weft/entry/v1", WCB({"parent":bytes32,"name":bytes}))
RelationId     = H("weft/relation/v1", WCB({"subject":bytes32,
                  "predicate":text,"kind":text,"object":bytes32|scalar,
                  "provenance":map|null,"attrs":map}))
SnapshotId     = commit root hash of the snapshot commit
```

An entry identity is the (parent,name) pair: renaming removes and inserts an
entry, while the target ObjectId is unchanged. Two entries with the same
parent/name cannot coexist. A relation identity excludes any edge-local
metadata that is not part of the assertion; the fields above are all part of
it in v1.

### 2.4 Checkpoint state manifest

`commit.checkpoint` names a WCB state manifest stored content-addressed under
`manifests/`. Its bytes are:

```
{
  "schema": "weftfs.state.v1",
  "generation": u64,
  "objects":  [ object_record, ... ],
  "entries":  [ entry_record, ... ],
  "relations":[ relation_record, ... ],
  "snapshots": { text_name: bytes32_commit, ... },
  "tips":      { bytes32_object: bytes32_version, ... }
}
```

`object_record`:

```
{"id":bytes32, "kind":text, "mode":u64, "uid":u64, "gid":u64,
 "mtime_ns":timestamp, "version":bytes32, "content":bytes32, "size":u64,
 "attrs":map}
```

`entry_record`:

```
{"id":bytes32, "parent":bytes32, "name":bytes, "target":bytes32}
```

`relation_record`:

```
{"id":bytes32, "subject":bytes32, "predicate":text, "kind":text,
 "object":bytes32|scalar, "provenance":map|null, "attrs":map}
```

A symlink's target is stored in `attrs["weft.symlink.target"]` as `bytes`.
A checkpoint MUST describe exactly the state after applying the commit that
names it; a mismatch is `Corrupt` and replay falls back to an earlier base.
The state manifest hash is `H("weft/manifest/v1", WCB bytes)`.

### 2.5 Reserved relation values, saved contexts, and xattrs

Reserved predicate `tagged_as` is sugar for a tag: target is a text value.

Reserved predicate `saved_context` stores a named selector as a normal
relation, so it participates in the same transactions and snapshots:

```
subject   = root ObjectId (32 zero bytes)
predicate = "saved_context"
kind      = "value"
target    = text name (unique among saved contexts)
attrs     = {"selector_hex": text}, where selector_hex is lowercase hex of
            the canonical selector bytes from WEFT-SELECTORS.md
```

A `saved_context` relation is visible only to principals allowed to read the
root object. Two relations with the same name but different selectors are a
validation error; replacement is delete+add in one commit.

Extended attributes are stored in the object's `attrs` map with the key
`xattr.<name>` and a `bytes` value. `listxattr` exposes the `<name>` suffixes;
`getxattr`/`setxattr`/`removexattr` operate on those entries. Names are UTF-8
without NUL and without `/`; values are bounded by the store's attribute
limit. Reserved `weft.*` keys are not exposed through xattr calls.

### 2.6 Pointer files

`refs/snapshots/<name>` and `refs/tags/<name>` contain lowercase hex of a
32-byte hash followed by a single `\n`. They are **caches**, not authority:
a reader MUST validate the named commit by replay. Names are UTF-8 without
`/` or NUL; a name collision is an error, not a silent overwrite.

---

## 3. Operations and state transitions

- `create(parent, name, bytes)`: allocate ObjectId, write chunks, put_object,
  put_entry, set_tip, commit. Implementations SHOULD also provide a streaming
  form that reads the source in bounded memory and commits once.
- `write(path, offset, bytes)`: read the current manifest, materialise only the
  intersecting chunks, store new chunks for changed regions, reuse unchanged
  chunk references verbatim, create a new VersionId, update content root,
  commit. Old versions remain readable through snapshots. The defensive file
  bound is 1 TiB; the effective bound for a chunked file is also the manifest
  decode limit (`max_array_items` chunks), which at the default 64 KiB chunk
  size is far below 1 TiB. Implementations MUST fail with `LimitExceeded`
  rather than truncate.
- `truncate`, `unlink` (delete entry; object remains while referenced by a
  snapshot or open handle), `rename` (delete + insert entry atomically in one
  commit), `link`, `mkdir`, `rmdir`, `symlink`, `setattr`, `xattr`.
- `relation add/delete`: one commit, idempotent by RelationId.
- `snapshot create(name)`: commit a `set_snapshot` op pointing at the parent
  commit; returns the resulting SnapshotId.
- `resolve(selector, snapshot, principal)`: read-only; evaluates against the
  pinned commit state.

**Isolation:** v1 is single-writer (flock). A reader pins one superblock
generation; every referenced object is immutable; therefore readers see exactly
the state of that commit, never a partial one. This is snapshot isolation with
no write conflicts because writes serialize. Multi-writer concurrency is out of
scope and MUST be refused by the lock.

**Open handles:** a handle pins `(snapshot, object, version)`. Rename, unlink,
and later commits do not change what that handle reads. An unlinked but open
file remains readable/writable in-process and is reclaimed after release; after
a crash it is gone (POSIX-permitted).

---

## 4. Durability contract

| Operation | Survives process crash | Survives kernel crash | Survives power loss |
|---|---|---|---|
| `commit` returns | yes (all fsyncs completed) | yes, given host fs honors fsync | yes, given host fs honors fsync and stable media |
| `fsync(file)` | current version durable | yes | stated host guarantee |
| `fsyncdir(dir)` | directory entries durable | yes | stated host guarantee |
| `write` without fsync | no guarantee | no guarantee | no guarantee |
| `unlink` commit | yes | yes | yes |
| snapshot create | yes | yes | yes |

The store does not claim stronger guarantees than the underlying filesystem's
`fsync` semantics. Power-loss claims require the separate hardware/VM court;
software crash tests are labelled as process-fault proxies, not physical power
loss.

Replay is idempotent: a generation is applied at most once, selected by the
superblock. Torn commit files fail CRC or hash and are ignored; a torn
superblock slot fails CRC and the other slot is used.

### 4.1 Storage backend contract

The store addresses durable artifacts through typed keys and performs all
physical I/O through a storage backend. The on-disk logical layout in §2 and the
commit protocol above are unchanged; a backend supplies the publication and
`fsync` primitives. The Rust reference implementation defines:

```
trait StorageBackend {
    fn read(&self, key, offset, len) -> Result<Vec<u8>>;
    fn read_all(&self, key) -> Result<Vec<u8>>;
    fn write_at(&mut self, key, offset, bytes) -> Result<()>;   // superblock slots
    fn put_atomic(&mut self, key, bytes) -> Result<()>;         // all-or-nothing
    fn create_zeroed(&mut self, key, len) -> Result<()>;
    fn delete(&mut self, key) -> Result<()>;
    fn list_prefix(&self, prefix) -> Result<Vec<Key>>;
    fn sync(&mut self, key) -> Result<()>;
    fn sync_dir(&mut self, key) -> Result<()>;
    // plus len, exists, modified, layout, writer lock, staging cleanup
}
```

A backend declares `Capabilities { atomic_publish, durable_fsync,
  true_range_reads, max_object_bytes, consistency, publication }`.

Required guarantees:

- `atomic_publish` — a `put_atomic` write becomes visible all-or-nothing under
  its final key. Every content blob, manifest, commit record, index segment,
  and snapshot pointer is published this way after its bytes are durable.
- `durable_fsync` — `sync`, `put_atomic`, and `write_at` return only after the
  data is stable. The commit protocol depends on this for the commit file and
  the superblock slot.
- `true_range_reads` — `read(offset, len)` reads only the requested range. If
  false, a backend may still be used, but range reads materialise whole
  objects; this is a performance property, not a durability guarantee, and the
  store never claims otherwise.
- `max_object_bytes` — a hard per-object bound. Blobs and pages above it are
  `LimitExceeded`, never silently split or truncated.

Publication primitive. `LocalBackend` is the normative POSIX namespace:
temporary file, `fsync`, `rename(2)`, parent `fsync`. `ObjectBackend` maps keys
to an immutable object namespace (`objects/<2-hex>/<name>`) and publishes with
temporary file, `fsync`, `link(2)`, unlink of the temporary name, parent
`fsync`. The final name appears only after the inode is complete, so it does not
rely on POSIX rename replacement. If `link(2)` is unavailable, the backend
falls back to `rename(2)` and states that in `Capabilities::publication` and in
its diagnostics; it never claims a link-published run.

Commit protocol mapping. The sequence in §4 maps one-to-one onto backend calls:

1. content and checkpoint blobs: `put_atomic` per blob;
2. commit record: `put_atomic` of `commits/<generation>.commit`, then its
   directory fsync;
3. superblock: `write_at` the inactive 4096-byte slot, then `sync`;
4. store root: `sync_dir`.

A crash between any two steps leaves either the previous whole generation or
the new whole generation. `put_atomic` supplies the property `create_new` plus
rename supplies on the local backend: an existing key with different bytes is
an error, and identical bytes are a no-op.

Fail-closed capability enforcement. The store consults capabilities rather than
assuming them:

- a writable open or initialization requires `durable_fsync` and refuses with
  `UnsupportedVersion` if it is absent;
- every publication requires `atomic_publish` and refuses rather than
  degrading to a non-atomic write;
- a read-only open may use a backend that declares weaker write guarantees,
  because it performs no publication.

Backend selection is explicit (`Store::init_with_backend`,
`Store::open_with_backend`, `Options`-equivalent CLI `--backend local|object|s3`).
The default is `LocalBackend`, so every existing fixture and handoff is
unchanged.

### 4.1.1 Conditional-write head publication (S3-compatible backends)

An object-store backend has no POSIX `rename(2)` and no in-place range write.
`S3Backend` maps the contract as follows:

- **Keys.** The logical layout of §2 is used verbatim: `FORMAT`, `superblock`,
  `LOCK`, `commits/<gen>.commit`, `objects/<ab>/<hash>`,
  `manifests/<ab>/<hash>`, `indexes/<ab>/<commit>.idx`,
  `refs/snapshots/<name>`, `refs/tags/<name>`, `quota`, `quota.migrated`.
  Consequently a local store and an S3 store built from identical operations
  publish identical artifact bytes under identical names.
- **`put_atomic`.** A single `PUT` with `If-None-Match: *`. `200` publishes;
  `412` means the key already exists. The writer then reads the existing bytes:
  identical bytes are an idempotent success, different bytes are `Exists`.
  A backend that accepts the second `PUT` has ignored the precondition and is
  not usable for this store.
- **`write_at` (superblock).** Read the whole `superblock` object and its ETag,
  patch the inactive 4096-byte slot, then `PUT` the whole object with
  `If-Match: <ETag>`. A `412` is a lost race: re-read and retry a bounded number
  of times, then fail `Busy`. A successful write is visible all-or-nothing under
  the key; the previous generation remains readable because the other slot is
  unchanged. The 8192-byte object shape and both slot CRCs are exactly those of
  §2.1.
- **Writer lock.** A `LOCK` object created with `If-None-Match: *`, carrying an
  owner and an expiry. A second live writer fails `Busy`. An expired lease is
  replaced with `If-Match`, so a crashed writer does not brick the store. The
  lock is advisory and provider-side; it is not a POSIX file lock and it is not
  a multi-host coordination protocol.
- **Probe.** A writable open performs a temporary `If-None-Match: *` create, a
  duplicate-create attempt, and a stale-`If-Match` attempt. If either
  precondition is ignored, the store refuses writable mode with
  `UnsupportedVersion`. It never degrades to an unconditional overwrite.
- **Durability semantics.** The backend declares `durable_fsync` only in the
  sense that the provider acknowledged the write. There is no local `fsync`.
  This is a provider-level durability assumption, not a media or power-loss
  guarantee; the MinIO court backs its data with a tmpfs and makes no such
  claim.

A backend that cannot provide both conditional forms must not be used for this
store. The refusal is the required behaviour, not a limitation to work around.

Network and distributed filesystems. A POSIX-namespace target (for example 3FS)
is addressed through `BackendKind::ThreeFs { config }`, which has two
transports over the **same** normative layout:

- `posix` — ordinary positional `pread`/`pwrite`, `fsync`, and `rename` through
the target's mount. This is the correctness baseline and it is what carries all
control-state I/O in both modes.
- `usrbio` — the hf3fs user-space ring API (`libhf3fs_api_shared.so`) for the
immutable data plane only: one shared region, a read ring and a write ring,
batched `hf3fs_prep_io`/`hf3fs_submit_ios`/`hf3fs_wait_for_ios`. The library is
loaded with `dlopen`, never linked, and a missing library or missing symbol is
`UnsupportedVersion` — never a silent fall back to POSIX.

The buffer boundary is explicit. `BlobPlane::read_into` and
`StorageBackend::read_into` fill a caller-owned buffer (one positional read; no
intermediate `Vec`), and `read_ranges`/`read_ranges_into` express a batch. For
`usrbio`, a destination that lies inside the shared region is passed to
`hf3fs_prep_io` directly; any other destination is staged through the region
and copied once. A request whose `[ptr, ptr+len)` is outside the region, an
unregistered descriptor, or a batch larger than the ring or region bound is
refused before submission.

What is exercised, and what is not: the `posix` transport is tested for
byte-identical state against the local backend, and the `usrbio` transport is
tested against a conformance implementation of the published ABI over real
POSIX I/O (`bench/threefs/hf3fs_shim.c`), including struct layout, registration
lifetime, batched submission counts, partial completion, short reads, and error
paths. **No 3FS deployment is run by this repository**, and that conformance
library reports `hf3fs_is_hf3fs = 0`, which the transport records in
diagnostics and receipts. No 3FS throughput, latency, RDMA, durability, or
multi-node consistency claim follows.

A caller that configures any other POSIX target must declare its real
capabilities with `BackendKind::Posix { capabilities }`. If the target cannot
honour `fsync` or atomic publication, the store refuses writable mode; before
any production use its consistency and durability claims must be measured, not
inferred from the API shape.

---

## 5. Garbage collection and retention

Reachability roots: every published snapshot commit, every object tip, every
open handle pin, every checkpoint manifest, and every codec dependency closure
of exported artifacts while retained.

- GC runs only under the exclusive writer lock, never concurrently with a
  commit.
- Phase 1 marks reachable hashes and persists a mark record.
- Phase 2 deletes only unmarked object/manifest files older than a grace period
  (default 1 hour) and never deletes a file referenced by any valid commit.
- A mark-and-sweep fallback exists for interrupted GC and treats the mark record
  as advisory, recomputing from roots.
- Context relation cycles are handled with a visited set.
- Codec dependency closure is acyclic by format; a cycle there is an error, not
  a graph property to preserve.

---

## 6. Authorization (v1)

A `Principal` is `(uid, gid, groups)`. Access checks:

1. Namespace root must be readable/traversable.
2. An object is visible if its POSIX mode permits the requested access for the
   principal, and, if `attrs["weft.acl.read"]` exists, it is an array of `u64`
   uid values and the principal's uid is listed (uid 0 is not special in the
   authorization predicate; mount-level root behavior is a FUSE policy).
3. A relation is visible only if its subject and target are visible.
4. Selector resolution filters the universe **before** matching; result counts
   and evidence contain only authorized objects.
5. Denial is indistinguishable from absence where the threat model requires
   non-disclosure.
6. Deduplication MUST NOT be used to infer content across namespaces or
   principals in the same store; v1 authorization is store-local and the threat
   model spells out this limitation.

v1 does not implement per-edge ACLs or tenancy isolation beyond the above. This
is stated in `security/THREAT_MODEL.md`; no stronger claim is made.

---

## 7. Invariants

- **I1 Identity:** rename, retag, relocation, and content update never change
  an ObjectId.
- **I2 Integrity:** materialization verifies every chunk hash, manifest hash,
  and whole-content hash; mismatch fails closed.
- **I3 Cardinality:** selectors return 0..n; singular APIs reject ambiguity.
- **I4 Determinism:** selector + snapshot + schema + authorization yields the
  same ordered result in Rust and Go.
- **I5 Atomicity:** no reader observes a commit prefix; no op partially applies.
- **I6 Reference integrity:** no reachable snapshot/handle loses a dependency.
- **I7 Isolation:** counts/evidence do not disclose unauthorized objects.
- **I8 Graph kinds:** context relations may cycle; codec dependencies must be a
  DAG with bounded depth and work.

Evidence status: I1–I6 are exercised by integration and crash tests; I7 by
authorization tests; I8 by codec cycle tests. I5 and replay idempotence are
also checked by bounded state exploration in `verification/` where feasible.
No property in this section is claimed model-checked until the corresponding
artifact exists and is named in `CLAIM_LEDGER`.

---

## 8. Recovery contract

On open (writer):
1. `flock` the LOCK file exclusively; non-blocking attempts return `Busy`.
2. Validate `FORMAT`.
3. Read both superblock slots; validate magic, version, CRC, and generation.
4. Verify the selected commit record's CRC and hash; verify referenced object
   files exist and are readable.
5. If the highest generation is invalid, fall back one generation at a time,
   recording the rejected generations; never delete them automatically.
6. Rebuild in-memory indexes by replaying commits from the nearest valid
   checkpoint.
7. Remove stale `tmp/` files only when they are unreferenced and older than the
   grace period.

Recovery is read-only until `verify --repair` explicitly requests a destructive
action; repair defaults to quarantine, not deletion.

---

## 9. Context Roots, causal identity, and capabilities (additive)

This section is additive to the frozen v1 model above. It introduces no change
to any existing v1 byte, identity, or commit record. The normative definitions
live in `spec/WEFT-CONTEXT-ROOT.md`, `spec/WEFT-CAUSAL.md`, and
`spec/WEFT-CAPABILITIES.md`.

### 9.1 New entities

| Entity | Definition | Stability |
|---|---|---|
| `ContextRootId` | `H("weft/context-root/v1", WCB(context-root record))` | immutable selection identity |
| `CausalClosureId` | `H("weft/causal/v1", WCB(causal record))` | immutable ancestry identity |
| `CapabilityHash` | `H("weft/capability/v1", WCB(capability record))` | immutable grant identity |

A Context Root binds a canonical selector, its `SelectorId`, a pinned snapshot
commit, a resolver/policy version, an ordered causal closure, an optional
capability hash, an optional budget, and the v1 ordering policy
`selector_then_object_id`. It denotes a selection, never a result set and never
the first match.

### 9.2 Version-record causal fields (opt-in only)

`WEFT-CAUSAL.md` §2 defines two identity modes. `bytes_only` is the **v1
default and is byte-identical to §2.3.1 above**; its version record carries no
additional fields. `causal_exact` adds exactly two fields to the version
record:

```
"identity_mode": "causal_exact",
"causal_hash":   bytes32
```

and derives `VersionId = H("weft/version/v1", WCB(record))` over the extended
record. `ContentId`, `ManifestId`, and every existing frozen vector are
unchanged. An implementation that does not implement `causal_exact` MUST reject
it with `UnsupportedVersion`; it MUST NOT ignore the fields.

### 9.3 Reserved attributes

`weft.causal.parents`, `weft.causal.closure`, `weft.causal.hash`, and
`weft.identity.mode` are reserved `weft.*` attribute keys and are not exposed
through xattr calls (§2.5). Their encodings and validation are defined in
`WEFT-CAUSAL.md` §3.

### 9.4 Invariants (additive)

- **I9 Root pinning:** a `ContextRootId` always resolves at exactly its named
  snapshot; it never follows a tag or the live head.
- **I10 Root cardinality:** a Context Root exposes the authorized cardinality of
  its selection and never collapses n>1 to one result.
- **I11 Identity separation:** content identity does not imply selection
  identity, and selection identity does not imply content existence.
- **I12 Causal graphs may cycle; codec dependency graphs must not.**
- **I13 Capability narrowing:** a capability context can only narrow, never
  widen, the authorization already enforced by §6.

`I9`/`I10` are implemented by the codec-level canonicalization and specified for
resolution in F2/F3; `I11` is tested by F1 vectors; `I12` is normative here and
in both code docs; `I13` is specified for F2. No property in this section is
claimed model-checked until the corresponding artifact is named in
`CLAIM_LEDGER`.
