# WEFT-TRAINING-READY v1.0.0 — training-ready materialization is a refusal

Status: **implemented** (P7, 2026-11-05). Code: `src/training_ready.rs`,
CLI `weft materialize --training-ready` and `weft training-ready`, tests
`tests/training_ready.rs`. This document is normative for v1.

"Training-ready" is not a promise about a result set. It is a set of
preconditions that a Context Root must satisfy before any output is written.
When any precondition fails, the operation returns a typed error and creates no
output directory and no partial file.

## 1. Terms

| Term | Meaning |
|---|---|
| pinned snapshot | the `snapshot` commit recorded in the Context Root |
| result set | the exact ordered objects the pinned selector resolves to |
| closure | the Context Root's declared causal identities plus every identity reachable from the objects that carry them |
| tensor object | an object whose `kind:"tensor"` representation or `weft.tensor.*` attributes declare a logical tensor identity |
| training-ready set | the result set, union the causal closure, union every checkpoint shard object named by a checkpoint manifest in that set |

## 2. Preconditions

All checks run against the pinned snapshot and read bytes through the store's
ordinary verified read path. No check trusts a stored hash.

### 2.1 Exact resolution

The Context Root must recompute to its recorded `ContextRootId`, its named
snapshot must replay, its resolver must be the v1 resolver, and:
`matched_count == results.len()` and `truncated == false`. A selector with a
`limit` that removes matches is **not** training-ready (`Corrupt`).

### 2.2 Content verification

Every object in the training-ready set must exist at the pinned snapshot, be
readable by the principal, and its full decoded bytes must verify:

* v1 / v2 `kind:"raw"`: every chunk blob hash and the whole-content hash;
* v3 `weft.manifest.v3`: every chunk blob hash and the Merkle content root;
* v2 `kind:"codec"` / `kind:"tensor"`: the representation decodes exactly and
  the decoded length equals the manifest size;
* symlinks: the target bytes equal the recorded whole-content digest.

A mismatch is `Corrupt`. An unreadable object is `Unauthorized`. A missing
object is `NotFound` or `DependencyMismatch` (see 2.3).

### 2.3 Causal completeness

Every identity in the Context Root's declared causal closure must resolve to an
object present at the pinned snapshot, by ObjectId, VersionId, or ContentId.
For every such object, each recorded causal parent and closure identity must
also resolve, transitively. The walk uses a visited set, so cycles are legal
and must terminate. A missing identity is `DependencyMismatch` and the message
names the missing identity in lowercase hex. An identity that matches objects
with different content roots is `Ambiguous`. The walk is bounded by the
codec's `max_causal_entries`; exceeding it is `LimitExceeded`.

### 2.4 Capability authorization

When an explicit capability or an attached verified envelope is supplied, the
principal must be authorized for every object in the training-ready set and
every relation predicate in the selector:

* namespace-wide grants authorize every readable object;
* root grants authorize only objects in the authorized result set of a granted
  Context Root, resolved under that capability;
* the selector's relation/tag predicates must all be in the predicate
  allowlist.

A denial is `Unauthorized` with a non-disclosing message. An explicit
capability bound to the Context Root must match the recorded `capability_hash`
(enforced by resolution).

### 2.5 Tensor verification (`--require-tensors`)

Every tensor object in the set must re-verify its declared
`LogicalTensorRoot`/layout against its decoded bytes. These identities are
recomputed, never compared to themselves:

* `kind:"tensor"` mode 10: the mode-10 root is checked while decoding;
* `kind:"tensor"` modes 11/14: the base dependency is resolved and the delta
  root is checked while decoding;
* raw/zstd tensor objects: the `weft.tensor.logical_root` attribute is
  recomputed from the decoded bytes and dtype/shape;
* every `weft.checkpoint.v1` manifest in the set is verified in full: every
  shard is decoded, every per-shard logical root is recomputed, the tensor is
  reassembled, and the assembled content root and `LogicalTensorRoot` are
  recomputed. Every shard object named by the manifest is added to the set.

A mismatch is `Corrupt`. A manifest that names a shard object which does not
resolve to the recorded ObjectId is `Corrupt`.

### 2.6 Cost envelope

Before reading any materialized object payload and before creating anything,
the implementation computes (checkpoint manifest metadata may be read first to
expand the shard set; those reads are themselves hash-verified):

```
estimated_bytes   = sum of decoded sizes of the training-ready set
estimated_objects = |training-ready set|
estimated_decode_ms
    = ceil(estimated_bytes * 1000 / 33_554_432)
    + 1 ms per tensor object
```

If any of `--max-bytes`, `--max-objects`, or `--max-decode-ms` is below the
estimate, the operation fails with `BudgetExceeded` before writing. After the
verification pass the *actual* bytes, object count, and measured verification
wall time are checked against the same envelopes; failure is again
`BudgetExceeded` and nothing is written. Actual costs are recorded.

## 3. Command surface

```
weft materialize --root <name-or-uri> --training-ready --out DIR
    [--max-bytes N] [--max-objects N] [--max-decode-ms N]
    [--capability FILE] [--envelope FILE] [--revocations FILE]
    [--require-tensors] [--uid U] [--gid G] [--group G ...]

weft training-ready --root <name-or-uri> --check
    [same limits, capability, envelope, and --require-tensors]

weft training-ready --verify-out DIR
```

`--out` must not exist; an existing path is `Exists`, never overwritten.
`--check` writes nothing and prints a structured report. `--verify-out`
verifies an already materialized directory without a store.

## 4. Output format

Canonical WCB bytes only. No JSON is written into the output directory.

```
<out>/<store path>              verified decoded bytes, one file per object
<out>/manifest.wcb              weft.training_ready.manifest.v1
<out>/training-ready.wcb        weft.training_ready.v1
```

`manifest.wcb` (canonical WCB map):

```
schema              text = "weft.training_ready.manifest.v1"
context_root_id     bytes32
context_root_name   text
uri                 text
snapshot_id         bytes32
selector_id         bytes32
capability_hash     bytes32|null
declared_closure    [bytes32]
preconditions       map {resolution_exact, content_verified, causal_complete,
                         authorized, tensors_verified, budget_enforced: bool}
cost                map {max_bytes, max_objects, max_decode_ms: u64|null;
                         estimated_bytes, estimated_objects,
                         estimated_decode_ms, actual_bytes, actual_objects,
                         actual_decode_ms: u64}
files               [ file ]
```

`file`:

```
path             text                store path, leading "/"
kind             text                file | dir | symlink
object_id        bytes32
version_id       bytes32
content_root     bytes32             ObjectRecord::content
content_hash     bytes32             H("weft/chunk/v1", decoded bytes)
content_identity text                whole-file | merkle | directory | symlink
size             u64                 decoded bytes
causal_closure   [bytes32]
origin           text                result | causal | checkpoint
tensor           map|null {
  dtype, shape, logical_root, source (object|checkpoint), layout, strategy,
  checkpoint: text|null, tensor_name: text|null,
  offline_verifiable: bool
}
checkpoint       map|null {
  name, manifest_root, complete, source_sha256: bytes32|null, tensors
}
```

`training-ready.wcb` records the same header, preconditions, and cost, plus
`files` (count), `checkpoint_tensors_verified`, and
`tensor_objects_verified`.

The output directory is assembled in a sibling temporary directory and renamed
into place. Any failure removes the temporary directory. The final directory
therefore appears atomically and is either complete or absent.

## 5. Verification of an output directory

`weft training-ready --verify-out DIR` decodes `manifest.wcb` and:

1. rejects unsafe or non-relative paths;
2. checks every file's decoded length and `content_hash`;
3. recomputes every tensor file's `LogicalTensorRoot` when
   `offline_verifiable` is true;
4. for every checkpoint manifest, decodes it, reads its sibling shard files,
   checks each shard content root and logical root, reassembles each tensor,
   and recomputes the assembled content root and logical root;
5. reports file/byte/tensor/checkpoint counts.

It never contacts the store.

## 6. Failure classes

| Condition | Class |
|---|---|
| Context Root identity, snapshot, resolver, or exactness failure | `Corrupt` |
| byte hash, Merkle root, decode, length, tensor root, checkpoint root | `Corrupt` |
| principal lacks POSIX/ACL read access | `Unauthorized` |
| capability/predicate denial | `Unauthorized` |
| declared causal identity absent | `DependencyMismatch` |
| causal identity matches different content roots | `Ambiguous` |
| cost envelope exceeded (estimate or actual) | `BudgetExceeded` |
| output path already exists | `Exists` |
| causal closure exceeds `max_causal_entries` | `LimitExceeded` |
| bad arguments / unsafe output path | `InvalidArgument` / `InvalidName` |

## 7. Non-goals

* v1 does not claim that the materialized bytes are sufficient to train a
  model, only that every declared storage precondition holds.
* There is no partial, best-effort, or "warn and continue" mode.
* There is no implicit repair: a missing causal parent, a renamed checkpoint
  shard, or a tampered blob is a refusal.
* The cost model is a declared deterministic estimate plus a measured
  verification wall time; it is not a hardware-independent performance claim.
* `verify --verify-out` verifies the output directory's canonical manifest and
  bytes; it does not re-open the source store.
