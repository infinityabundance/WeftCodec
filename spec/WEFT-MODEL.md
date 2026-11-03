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

Ops apply in array order. A commit whose ops cannot be applied (missing parent,
duplicate entry name, dangling target) is invalid; it is not partially applied.

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
  put_entry, set_tip, commit.
- `write(path, offset, bytes)`: read current content, apply bounded splice in
  memory (v1 bound: 256 MiB per file for random writes; larger writes must be
  streaming append or truncate), create a new VersionId, update content root,
  commit. Old versions remain readable through snapshots.
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
