# WEFT-TRAINING-ARTIFACT v1.0.0 — a composable training artifact

**Status:** Implemented (P01, 2026-11-15) in Rust `weftfs::training_artifact` and
Go `WeftFS-Go/store` (`training_artifact.go`). Normative for
`weft.training-artifact.v1` and `weft.training-grant.v1`.
**Depends on:** `WEFT-MODEL.md` (object/version identity),
`WEFT-TENSOR.md` (logical tensor identity), `WEFT-CONTEXT-ROOT.md`,
`WEFT-CAUSAL.md`, `WEFT-PROVENANCE.md`, `WEFT-TRAINING-READY.md`,
`WEFT-RETENTION.md`.
**Source of truth:** `spec/WEFT-TRAINING-ARTIFACT.md` in
`infinityabundance/WeftFS`.

A **Training Artifact** is the canonical composition of what the records above
already establish: exact logical tensor values, the checkpoint that produced
them, the Context Root that selects the state, the causal ancestry, and the
provenance receipts. It adds no new byte-level identity: every member identity
is reused verbatim.

## 1. The three identity classes stay separate

This is the governing rule and it is not negotiable.

| Class | Example | Derived from | Mutable |
|---|---|---|---|
| **Exact logical root** | `LogicalTensorRoot`, whole-tensor content root | decoded tensor bytes, dtype, shape | no |
| **Contextual artifact root** | `TrainingArtifactRoot` | immutable content identities + declarations | no |
| **Mutable authority** | grants, retention bindings, quarantine | local policy state | yes |

`TrainingArtifactRoot` is derived **only** from the record in §3. It MUST NOT
include a grant, a retention class or lifetime, a quarantine decision, a store
path, a snapshot, a principal, a wall clock, or a mutable name binding. Those
are separate records:

* **Grant** — `weft.training-grant.v1` (§6), keyed by `TrainingArtifactRoot`.
* **Retention** — the existing `weft.retention.v1` binding (`WEFT-RETENTION.md`),
  keyed by a descriptor root, never by the artifact record.
* **Quarantine** — the existing `weft.quarantine.v1` record (`WEFT-TAINT.md`),
  keyed by an object identity.

Editing, adding or revoking any of them MUST NOT change `TrainingArtifactRoot`.
Conversely, a change to any member content identity or declared field MUST
change it.

The snapshot at which a record was observed is reported by resolution and is
never part of the artifact identity. Two stores that hold the same immutable
records therefore compute the same `TrainingArtifactRoot`; the RFC-0001 warning
that Context Root identities are snapshot-dependent does not apply to this
record. A consumer MUST still resolve the record at a pinned snapshot.

## 2. Terms

| Term | Meaning |
|---|---|
| artifact record | the canonical WCB map in §3 |
| `TrainingArtifactRoot` | `H("weft/training-artifact/v1", WCB(record))` |
| member tensor | one entry of `tensors` |
| member shard | one entry of a tensor's `shards` |
| declared lineage | `causal.parents` and `causal.closure` |
| declared policy | the `policy` map in §3.3 |
| grant epoch | the monotonic counter of the mutable grant record (§6) |

## 3. Record schema `weft.training-artifact.v1`

A canonical WCB map with exactly these keys. Unknown or missing keys are
`NonCanonical`; every array is in canonical order and is never sorted by a
reader.

| Key | Type | Required | Rule |
|---|---|---|---|
| `schema` | text | yes | exactly `weft.training-artifact.v1` |
| `name` | text | yes | 1..=256 bytes, ASCII printable, no NUL |
| `context_root` | bytes(32) \| null | yes | a ContextRootId, or null |
| `checkpoint` | map \| null | yes | §3.2 |
| `tensors` | array of map | yes | §3.1, sorted by `name`, unique |
| `causal` | map \| null | yes | `{parents, closure}`, `WEFT-CAUSAL.md` §1 |
| `provenance` | array of bytes(32) | yes | strictly ascending, unique, nonzero |
| `policy` | map | yes | §3.3 |

`checkpoint` and `tensors` MUST NOT both be empty: an artifact with no member
tensor is not a training artifact (`Malformed`).

### 3.1 Member tensor

Exact keys:

```
name           text        unique in the artifact, sorted ascending
dtype          text        a WEFT-TENSOR dtype name
shape          [u64]       non-empty, every dimension nonzero
logical_root   bytes(32)   LogicalTensorRoot of the whole tensor
content_root   bytes(32)   H("weft/chunk/v1", canonical row-major bytes)
element_bytes  u64         decoded size of the whole tensor
source_offset  u64         the source container offset, or 0
shards         [map]       at least one, ordered by `index`, unique
```

`element_bytes` MUST equal `product(shape) * element_size(dtype)` and the sum of
`shards[*].byte_len`. Shard ordering is semantic: a reader MUST NOT sort it.

### 3.2 Member shard

Exact keys:

```
axis         u32     shard axis
shards       u32     total shard count on that axis (>= 1)
index        u32     this shard's index (0..shards)
object       bytes(32) ObjectId of the shard object in the named store
content_root bytes(32) H("weft/chunk/v1", shard bytes)
logical_root bytes(32) LogicalTensorRoot of the shard sub-tensor
dtype        text    the shard's logical dtype
shape        [u64]   the shard's shape
byte_len     u64     the shard's decoded byte length
```

All shards of one tensor MUST agree on `axis` and `shards`, indices MUST be
`0..shards` exactly once each, `shards * shape[axis] == tensor.shape[axis]`
(no silent padding), and the other dimensions MUST equal the tensor's.

`checkpoint` is an optional provenance anchor:

```
name            text
manifest_object bytes(32)   ObjectId of the stored checkpoint manifest file
manifest_content bytes(32)  its whole-content digest
manifest_root   bytes(32)   H("weft/checkpoint/manifest/v1", manifest bytes)
complete        bool
source_sha256   bytes(32) | null
tensor_count    u64
element_bytes   u64
```

### 3.3 Declared policy

A canonical map with exactly these keys. Policy is an immutable declaration of
what the artifact *claims*; it is not an authorization decision and it never
grants access.

| Key | Type | Rule |
|---|---|---|
| `lineage` | text | `none`, `declared` or `complete` |
| `allowed_dtypes` | array of text | sorted, unique; empty means any |
| `minimum_tensors` | u64 | `>= 1`; `tensors.len()` must be at least this |
| `checkpoint_complete` | bool | when true, a `checkpoint` MUST be present and `complete` |

`lineage = "none"`: no lineage claim; `causal` MAY be null.
`lineage = "declared"`: `causal` MUST be present and every member **shard
object** MUST carry a stored `provenance` relation whose receipt id is one of
the artifact's `provenance` entries. A relation on the shard object (not a
content-root comparison) is used deliberately: it is a snapshot-scoped store
fact, and it keeps the provenance module's own output identity (which is the
object's recorded content identity) out of `TrainingArtifactRoot`.
`lineage = "complete"`: `declared`, and every entry of `causal.closure` MUST
resolve at the pinned snapshot.

## 4. Validation (normative)

Validation takes a pinned snapshot `S` and an artifact root `R`. It MUST
recompute every identity; it MUST NOT trust a stored hash.

1. **Record** — read `/training/artifacts/<R-hex>` at `S`; decode strictly;
   recompute `H("weft/training-artifact/v1", bytes)` and require it equals `R`.
   A mismatch is `Corrupt`; non-canonical bytes are `NonCanonical`.
2. **Checkpoint anchor** — when `checkpoint` is present: read
   `manifest_object` at `S`; require its decoded bytes hash to
   `manifest_content`; require `H("weft/checkpoint/manifest/v1", bytes)` equals
   `manifest_root`; decode `weft.checkpoint.v1` strictly; require `name`,
   `complete`, `source_sha256`, `tensor_count`, `element_bytes` and the full
   tensor/shard set (names, dtypes, shapes, roots, object ids, byte lengths) to
   equal the artifact's declaration. Any difference is `Corrupt`.
3. **Exact tensor values** — for every member tensor and every member shard:
   resolve `object` at `S`; decode it through the ordinary verified read path;
   require `H("weft/chunk/v1", bytes) == content_root` and the shard's
   `LogicalTensorRoot` recomputed from `dtype`/`shape`/bytes equals
   `logical_root`. Reassemble the tensor in shard-index order and require the
   assembled `content_root` and whole-tensor `logical_root`. A mutated value is
   `Corrupt` and the message names the tensor and shard.
4. **Declared lineage** — verify the `causal` record's canonical arrays
   (`WEFT-CAUSAL.md` §1.1). For `lineage = "complete"`, every closure entry
   MUST resolve at `S`; under a capability, an entry that does not resolve is
   `DependencyMismatch` with the same non-disclosing message whether it is
   absent or unauthorized. The closure query itself is defined in §5.
5. **Declared policy** — recompute: `tensors.len() >= minimum_tensors`; every
   dtype is in `allowed_dtypes` when non-empty; `checkpoint_complete` holds;
   the `lineage` requirement holds. A contradicted claim is `Corrupt` and the
   message names the policy key.
6. **Independent classes** — read the grant records and the retention binding
   when asked, and report them. Their presence, content or epoch MUST NOT alter
   steps 1–5 or `R`.

### 4.1 Refusal classes

| Condition | Class |
|---|---|
| record bytes not canonical, unknown/missing key, bad type, unsorted array | `NonCanonical` |
| `TrainingArtifactRoot` mismatch, tensor/shard byte or root mismatch, manifest disagreement, contradicted policy | `Corrupt` |
| a required causal ancestor does not resolve (absent or unauthorized) | `DependencyMismatch` |
| an ancestor exists but is quarantined | `Quarantined` (explicit mode) / `NotFound` (non-disclosing mode) |
| a bound (`max_depth`, `max_nodes`, tensor/shard count) exceeded | `LimitExceeded` |
| the artifact record is absent at `S` | `NotFound` |
| the pinned snapshot itself is absent | `SnapshotMissing` |

No reader may repair. There is no "warn and continue" mode.

## 5. Causal closure modes

The closure is computed over the *artifact graph*: nodes are artifact
identities (a VersionId, a CausalClosureId, a ContentId, or a
TrainingArtifactRoot), and a node's declared `parents` are its outgoing edges.
The record's `closure` array is a declared transitive set, not an edge list.

Two modes, both bounded by `max_depth` (edge hops from any root, default 64)
and `max_nodes` (default 65,536), both deterministic:

* **complete** — every ancestor declared by the artifact and by every member
  object reachable from it. Start set: the artifact's `causal.parents`, every
  member object's stored causal parents, and every provenance record's inputs
  and outputs. The result is the union of the transitive parent sets.
* **minimal** — only what verifying *this* artifact needs: the artifact's own
  `causal.parents`, the transitive parent edges reachable from them, the
  provenance record ids the artifact names, and the representation dependency
  closure of its member tensors. Member objects' unrelated causal ancestors are
  excluded. Minimal is a subset of complete for the same request.

Ordering is ascending by identity bytes. Depth is the length of the shortest
parent path from any root; it is reported, never used to order. The result
always reports `total_closure_size` (authorized size **before** any budget
filter), `returned_closure_size`, and `truncated`.

A missing or unresolvable parent is always a refusal; an implementation MUST
NOT return a closure that silently omits it. The message names the identity the
caller already supplied. It MUST NOT distinguish an absent ancestor from an
unauthorized one.

## 6. Mutable grant record `weft.training-grant.v1`

Stored under `/training/grants/<artifact-root-hex>/` as a grant record plus a
monotonic `epoch` file. Exact keys:

```
schema          text            "weft.training-grant.v1"
artifact_root   bytes(32)       the TrainingArtifactRoot this grant names
principal       map {uid,gid}
actions         [text]          sorted unique from train|evaluate|materialize|replicate
policy_epoch    u64
issued_at_ns    i64
expires_at_ns   i64 | null
supersedes      bytes(32) | null
signature       bytes(64) | null
```

`GrantId = H("weft/training-grant/v1", WCB(record))`. Every grant mutation
increments `epoch`, which is a text decimal integer in the same directory. The
epoch is local policy state: it is not content-addressed and it never enters
`TrainingArtifactRoot`. A reader that needs a grant MUST check the current
epoch and expiry; an expired grant authorizes nothing.

## 7. Storage

```
training/artifacts/<root-hex>          canonical artifact record bytes
training/grants/<root-hex>/epoch       decimal counter
training/grants/<root-hex>/<grant-id>  canonical grant record bytes
```

Artifact records and grant records are ordinary store files, so they are
committed, snapshotted, verified and garbage-collected by the existing rules.
The artifact record file is a GC root while it is reachable from the head or a
snapshot; its member objects are retained by the ordinary dependency rules
(`WEFT-MODEL.md` §5).

## 8. Limits

| Limit | Default |
|---|---:|
| `max_artifact_tensors` | 1,000,000 |
| `max_artifact_shards` | 16,777,216 |
| `max_name_bytes` | 256 |
| `max_depth` (closure) | 64 |
| `max_nodes` (closure) | 65,536 |

## 9. Non-goals in v1

* Not a training guarantee. It validates storage preconditions and declared
  lineage, not that a model converges or that a resumed step is bitwise equal.
* Not an authorization grant. A grant is separate, mutable and epoch-checked.
* Not a cross-store equivalence proof. Equal roots mean equal immutable member
  identities and declarations; they do not prove the stores kept identical
  commit histories (no snapshot is in the root).
* Not a signature. The root is an unsigned content address; authenticity of
  the writer remains the producer-attestation and store-authority surfaces, and
  is outside this record.
* No scheduling, transport, cache policy or placement decision.
