# WEFT-BUCK2 v1.0.0 — action keys and a store-backed build cache

**Status:** Normative for `weft-buck2 1.0` (the parent adapter contract).
**Depends on:** `WEFT-MODEL.md` (ContentId, ObjectId, relations, GC roots),
`WEFT-SELECTORS.md` (WeftQL), `WEFT-CONTEXT-ROOT.md`, `WEFTCODEC-FORMAT.md`
(BLAKE3 domain separation, WCB), `WEFT-PROVENANCE.md` (computation records).
**Source of truth:** `spec/WEFT-BUCK2.md` in `infinityabundance/WeftFS`.

This document defines how a Buck2-style action cache is expressed on WeftFS. It
is an *adapter contract*: a local content-addressed store plus canonical
action keys. It is **not** a Buck2 plugin ABI, not a Buck2 worker, and not a
REAPI gRPC server. No part of this document claims that Buck2 was executed.

The design keeps two identities separate and explicit:

| Identity | Definition | Purpose |
|---|---|---|
| `ActionKey` | `H("weft/buck2/action/v1", WCB(action))` | names one action declaration |
| `ContentId` | `H("weft/chunk/v1", exact bytes)` | names one immutable blob |

`ActionKey` never names a result set, a path, or a snapshot. `ContentId` never
names an action. A build hit is a *lookup* of the action record; it is not a
property of the selector.

---

## 1. Action declaration (`weft.buck2.action.v1`)

The canonical WCB value is a map with exactly these keys:

| Key | Type | Required | Rule |
|---|---|---|---|
| `schema` | text | yes | exactly `weft.buck2.action.v1` |
| `action_type` | text | yes | 1..=256 bytes, valid UTF-8 |
| `toolchain` | text | yes | 1..=256 bytes, valid UTF-8 |
| `inputs` | array of text | yes | sorted ascending content-root hex, duplicates preserved |
| `params` | map text→text | yes | may be empty; keys sorted by WCB encoding |

Input **paths are not key material**. Two actions that consume the same bytes
under the same toolchain share a key even if the producer spelled the paths
differently. The action record keeps the paths for evidence and materialization.

The key is:

```
ActionKey = H("weft/buck2/action/v1", WCB(action))
```

`H` is the frozen domain-separated construction from `WEFTCODEC-FORMAT.md` §1:
`BLAKE3(LE32(len(domain)) || domain || data)`.

### 1.1 Input content roots

Each declared input carries a `ContentId`, the 32-byte whole-file content root
`H("weft/chunk/v1", bytes)`, rendered as 64 lowercase hex characters. Inputs
are sorted by `(content_id, path)`; the key hashes only the sorted
`content_id` sequence. A caller may compute the content root from a file without
touching the store, so key derivation is a pure function of the declaration.

---

## 2. Action record (`weft.buck2.action-record.v1`)

The record is a canonical WCB map stored as an ordinary WeftFS file at
`/cas/records/<ActionKey>`. It is content-addressed by its own
`ContentId`; its `ObjectId` is deterministic:

```
RecordSeed   = "weft.buck2.record:" || ActionKey
RecordObjectId = H("weft/object/v1", RecordSeed)
```

| Key | Type | Rule |
|---|---|---|
| `schema` | text | exactly `weft.buck2.action-record.v1` |
| `action_key` | text | 64 lowercase hex; must equal the path name |
| `action_type` | text | same as the declaration |
| `toolchain` | text | same as the declaration |
| `inputs` | array of map | each `{path, content_id, bytes}`; sorted by `(content_id, path)` |
| `params` | map text→text | same as the declaration |
| `output_path` | text | clean relative path, 1..=4096 bytes |
| `output_content_id` | text | 64 lowercase hex |
| `output_object_id` | text | 64 lowercase hex |
| `output_bytes` | u64 | 0..=1 GiB in v1 |

Decoders MUST reject unknown or missing fields, non-canonical WCB, unsorted
inputs, and any re-encoding that differs from the stored bytes. A record whose
`action_key` does not re-derive from its own declaration is `Corrupt`.

---

## 3. Store layout

```
/cas/records/<action-key>                     action record file
/cas/outputs/<action-key>/<output-path>       output blob file, one per action
```

Every blob is an ordinary WeftFS file object. There is no private blob store:
the output file's `ContentId` is the CAS blob address, and ordinary WeftFS GC,
verification, snapshots, Context Roots, and FUSE mounts apply.

The output `ObjectId` is deterministic:

```
OutputSeed     = "weft.buck2.output:" || ActionKey
OutputObjectId = H("weft/object/v1", OutputSeed)
```

Directory `ObjectId`s are deterministic from their absolute store path
(`"weft.buck2.dir:" || path`). Determinism is a convenience for cross-language
tests and evidence; it is not a security boundary.

---

## 4. Relations and attributes

Publishing an action writes both edges:

| Subject | Predicate | Target | Meaning |
|---|---|---|---|
| action record object | `cas_action` | output object | the record produced the output |
| output object | `cas_output_of` | action record object | reverse edge for one-hop selectors |

The reverse edge exists because WeftQL `relation(pred, obj:<id>)` returns the
edge **subject**. `cas_action` alone would select the record, not the output.

The output object also carries these non-reserved attributes:

| Attribute | Value |
|---|---|
| `cas.action_key` | lowercase `ActionKey` |
| `cas.action_record` | record `ObjectId` hex |
| `cas.content_id` | output `ContentId` hex |
| `cas.output_path` | declared output path |

---

## 5. `by:action=<key>` as a Context Root

The canonical selector is:

```
select relation(cas_output_of, obj:<record-object-id>)
```

It resolves to exactly the output object, because the output is the subject of
the reverse edge. A Context Root named `by:action=<ActionKey>` pins this
selector to a named snapshot:

```
weftfs context create --name by:action=<key> \
  --selector 'select relation(cas_output_of, obj:<record-object-id>)' \
  --snapshot <snapshot>
```

`--require-one` semantics are real: resolution either returns exactly one
object or fails (`NotFound` / `Ambiguous`). The implementation never picks one
silently.

A Context Root pins a commit. In v1 the store's GC roots are the live state,
named snapshots, and checkpoints; a Context Root whose pinned commit is not a
named snapshot is not itself a GC root. Retained contexts MUST therefore pin a
retained snapshot when GC is expected to preserve their output.

---

## 6. Determinism and conflict

`store(action, blob)`:

1. Derive `ActionKey` and `ContentId(blob)` without mutating the store.
2. If `/cas/records/<ActionKey>` exists:
   - same `output_content_id` → idempotent hit; verify and return the existing
     record and output;
   - different bytes → fail closed with `Conflict` **before any mutation**.
3. Otherwise publish the output file, the record file, and both relations.

The conflict rule is a correctness rule, not a policy: a cache that returns
different bytes for the same action key is not a cache.

The multi-file publish is not one atomic store transaction in v1. A process
crash between the output file and the record file can leave an unreferenced
output object. That object is ordinary garbage; it is collected by the store's
GC and never changes the meaning of an existing key.

---

## 7. Cross-language contract

A store written by one implementation MUST be readable by the other:

- canonical WCB bytes for the action record are identical;
- `ActionKey`, `ContentId`, `ObjectId`, relation ids, and Context Root ids are
  identical;
- both implementations publish the same paths and edges.

The adapter's court builds a store through the Go implementation, verifies it
with the Rust CLI (`verify --full`, `cat`, `stat`, `context resolve`), then
builds a second store entirely through the Rust CLI and reads it with the Go
implementation (`fetch`, `verify`, `query`).

---

## 8. CLI contract (adapter wrapper)

The adapter is a command-line wrapper. A build system may invoke it; the
adapter does not load into the build system.

### 8.1 Verbs

```
buck2cas init    --root DIR [--gc-grace-secs N] [--checkpoint-interval N]
buck2cas key     --key-file FILE [--json]
buck2cas store   --root DIR --key-file FILE --blob FILE [--output-path REL] [--json]
buck2cas fetch   --root DIR --key HEX64 --out FILE [--json]
buck2cas query   --root DIR --by action=HEX64 [--json]
buck2cas verify  --root DIR (--key HEX64 [--blob FILE] | --all) [--json]
buck2cas context --root DIR --key HEX64 [--name NAME] [--snapshot NAME] [--replace] [--json]
buck2cas resolve --root DIR --name NAME [--json]
buck2cas delete  --root DIR --key HEX64
buck2cas gc      --root DIR [--apply] [--grace-secs N] [--max-deletes N] [--json]
```

### 8.2 Key file

`--key-file` is a JSON object:

```json
{
  "action_type": "compile_object",
  "toolchain": "rust-1.85.0",
  "inputs": [
    {"path": "src/input.txt", "file": "out/input.txt"}
  ],
  "params": {"opt": "2"},
  "output_path": "out/object.bin"
}
```

An input may instead supply `content_id` and `bytes` directly. When `file` is
present the adapter computes the WeftFS `ContentId`; a supplied `content_id`
that disagrees with the file fails closed.

### 8.3 Environment and encoding

| Variable | Meaning |
|---|---|
| `WEFTFS_RUST_BIN` | Rust oracle used by tests/demos; never needed at runtime |
| `NO_COLOR` | accepted and ignored (output is plain text) |

Keys are 64 lowercase hex characters. Paths are clean relative
slash-separated paths. JSON output is stable field-named data; text output is
for humans.

### 8.4 Exit codes

| Code | Meaning |
|---:|---|
| 0 | success |
| 1 | general failure |
| 2 | usage error |
| 3 | cache miss |
| 4 | verification mismatch or corrupt store data |
| 5 | same key, different bytes (`Conflict`) |

---

## 9. Limits (v1)

| Bound | Value |
|---|---|
| inputs per action | 4,096 |
| params per action | 256 (16 KiB of key+value bytes per param) |
| action type / toolchain text | 256 bytes each |
| output path | 4,096 bytes; 255 bytes per segment |
| output blob | 1 GiB |
| record bytes | 1 MiB |

These are adapter bounds layered on the store's own WCB and file bounds.

---

## 10. Explicit non-goals

- No Buck2 plugin ABI, worker protocol, or scheduler integration.
- No REAPI gRPC, remote execution, or bytestream service.
- No claim that Buck2 itself was run. The court records whether a `buck2`
  binary was present; a present binary is probed only with `--version`.
- No performance claim beyond the measured receipt, which is labelled
  single-host and synthetic.
- No multi-output action in v1. One action publishes one output blob; the
  record and relation shape extend additively if that changes.
