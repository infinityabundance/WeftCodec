# WEFT-RETENTION v1.0.0 — retention bindings for cached artifacts

**Status:** Normative for `weft.retention.v1` and for class-aware GC ordering.
Implemented in Rust (`weftfs::retention`, `Store::set_retention`,
`Store::retention_of`, `Store::retention_history`, `Store::gc_plan`,
`Store::retention_gc`) and in Go (`WeftFS-Go/store`). The court that compares
retention policies is `weft.retention-court.v1`.

**Depends on:** `WEFTCODEC-FORMAT.md` (WCB, `H(domain, data)`, limits),
`WEFT-KV-PROVENANCE.md` (`descriptor_root`, the identity of a mode-19 KV page),
`WEFT-MODEL.md` (manifests and content roots), `WEFT-MULTIWRITER.md` (live
prepared transactions as GC roots).

---

## 1. The defect this document closes

A storage layer sees bytes and references. It does not see *utility*. A KV page
from a sliding-window cache is useless ninety seconds after it is written; a
long-context block cache written in the same minute is useful for a day. The
bytes can be identical in shape and size, and usually differ not at all in any
property the storage layer can observe.

Production systems therefore do one of two bad things:

1. they persist both classes under one coarse TTL, so the short-lived class
   occupies capacity long after it can help, or
2. they discard the short-lived class immediately, because a coarse TTL cannot
   be shortened without also discarding the long-lived class.

Both are consequences of the same missing fact: the artifact does not carry a
statement of how long it is expected to be useful.

This document adds that statement as a first-class, inspectable record — and
nothing else. It does not add a cache, an eviction engine, a scheduler, or a
wall-clock promise. It adds a contract that a GC planner can read.

---

## 2. The two layers, and why retention must not be identity

This is the load-bearing constraint of the design.

An artifact already has an **identity**: for a mode-19 KV page it is

```text
descriptor_root = H("weft/kv/prov/v1", WCB(descriptor))
```

which covers the payload hash and every provenance dimension
(`WEFT-KV-PROVENANCE.md` §2–§3). For a file it is the manifest `ContentId`; for
a tensor it is the `LogicalTensorRoot`. Identity is immutable and is the key
for deduplication, content addressing, references, and every hash check in the
program.

Utility is **not** identity. The same exact bytes, produced by the same run
with the same provenance, are the same artifact whether the producer expects to
reuse them for ninety seconds or for a day. If retention were folded into
identity, then changing a retention policy would:

* fork deduplication (two copies of the same bytes),
* invalidate every existing reference and content root,
* turn a policy change into a data migration, and
* make the "same artifact" question unanswerable without reading a policy.

Therefore retention is a **second layer**:

| Layer | Mutability | Answers | Input to identity? |
|---|---|---|---|
| identity | immutable | *what is this artifact?* | yes |
| retention binding | mutable, versioned | *how long is it expected to be useful?* | **no** |

A retention binding references the artifact's `descriptor_root`. No field of a
binding is an input to any descriptor, content root, `ContentId`, `ObjectId`,
manifest, selector identity, or codec frame. Publishing a new binding writes
one metadata record and **zero artifact bytes**. Two artifacts with identical
bytes and different retention policies have identical content roots and
different retention ids.

The converse is equally important: a binding never vouches for bytes. A binding
whose `descriptor_root` does not exist, or whose stored artifact no longer
hashes to that root, is a stale binding. A stale binding is reported and is not
a read authorization.

### 2.1 What the binding is not allowed to do

* It does not make an artifact reachable. Retention is **not** a GC root
  (§5.3).
* It does not encrypt, sign, or authorize the artifact. An optional signature
  (§3.3) authenticates the *binding record*, not the page and not the writer's
  intent.
* It does not override reachability. GC may reclaim an expired binding's bytes
  only when the artifact is already unreachable (§5.3).

---

## 3. The binding record

Schema: `weft.retention.v1`. Exactly these keys, no others.

| Key | Type | Rule |
|---|---|---|
| `schema` | text | exactly `weft.retention.v1` |
| `descriptor_root` | bytes | exactly 32 bytes, not all zero; the artifact identity |
| `class` | text | exactly `volatile`, `session`, `ephemeral`, or `durable` (§4) |
| `declared_lifetime_ms` | u64 | `1 ..= MAX_DECLARED_LIFETIME_MS` for the first three classes; exactly `0` for `durable`; an **admission hint only** (§5.2) |
| `producer` | text | non-empty, at most 256 UTF-8 bytes: the producing run |
| `policy_epoch` | u64 | monotonic policy generation; a rebind with a lower epoch is `Conflict` |
| `issued_at` | timestamp | nanoseconds since the Unix epoch; when the binding became current |
| `supersedes` | bytes or null | exactly 32 bytes for a rebind (the previous `RetentionId`); `null` for a first binding |
| `signature` | map or null | optional; §3.3 |

`MAX_DECLARED_LIFETIME_MS` is `3_155_760_000_000` (100 years). Values above it
are `LimitExceeded`; a non-durable binding with `0`, or a durable binding with
a non-zero value, is `Malformed`.

There is no ninth field and no free-form extension in v1. A new field is a new
schema (`weft.retention.v2`), never an extra key.

### 3.1 Identity

```text
RetentionId = H("weft/retention/v1", WCB(binding))
```

`RetentionId` is the identity of the *binding*, not of the artifact. It is
derived, never stored inside the record. When a signature is present it is part
of the hashed bytes, so re-signing the same policy with a different key yields
a different binding id; that is correct, because the signature is a property of
the binding, not of the artifact.

`H("weft/retention/v1", …)` is computed only after the record is fully valid, so
a caller never sees an id for a record that cannot decode.

### 3.2 Rebinding

`supersedes` creates an append-only chain per `descriptor_root`. A rebind:

* publishes a new binding blob and a new current pointer,
* sets `supersedes` to the previous current `RetentionId`,
* never rewrites, deletes, or mutates the previous binding,
* requires `policy_epoch` greater than or equal to the previous binding's
  epoch; a lower epoch is `Conflict` (a stale policy writer), an equal epoch is
  allowed for a corrected record,
* requires `supersedes` to equal the current `RetentionId`; a stale
  `supersedes` is `Conflict` (compare-and-swap semantics).

The first binding for a descriptor requires `supersedes = null`; rebinding a
descriptor that already has a current binding with `supersedes = null` is
`Conflict`.

The chain is bounded at `MAX_RETENTION_HISTORY = 4096`. Rebinding beyond it is
`LimitExceeded` naming the bound; history is never silently truncated.

### 3.3 Optional signature

`signature`, when present, is exactly `{algorithm: text, public_key: bytes,
signature: bytes}` with:

* `algorithm` exactly `ed25519`;
* `public_key` exactly 32 bytes;
* `signature` exactly 64 bytes, over `WCB(record without the signature key)`.

A present signature MUST verify; a malformed or non-verifying signature is
`Corrupt`. When the store is configured with one or more trust keys, the
embedded `public_key` MUST be one of them; otherwise `Unauthorized`. When no
trust key is configured, a valid signature proves only that the holder of the
embedded key signed the record — it is not an authorization claim (§10).

---

## 4. Classes

The four classes are **semantic**, not clock-based. They describe the intended
reuse horizon of the producing workload; they are not deadlines and they are
not derived from any observable property of the bytes.

| Class | Meaning | Intended horizon |
|---|---|---|
| `volatile` | recomputed in seconds-to-minutes; a sliding-window cache, a speculative decode | seconds to minutes |
| `session` | useful for the duration of one interactive or batch session | hours |
| `ephemeral` | useful across a bounded multi-hour or multi-day job; a long-context block cache | hours to days |
| `durable` | expected to be useful indefinitely; requires an explicit pin or snapshot root | indefinite |

Why these four and no more:

* They are the distinctions a *producer* can actually make. A producer knows
  whether it will ask again in ninety seconds, in this session, or tomorrow.
* They are coarse enough that a misclassification is bounded by the lifetime
  hint, not by an arbitrary taxonomy.
* They are ordered by expected utility: `volatile` < `session` < `ephemeral` <
  `durable`. GC ordering uses that order (§5.4).
* `durable` is the only class with a structural requirement, because
  "indefinite" is not a retention promise the storage layer may make on its
  own. A durable binding without a pin or snapshot root is refused (§5.1).

### 4.1 Inference is forbidden

Implementations and operators MUST NOT infer a class from:

* object size, frame size, or decoded size;
* the artifact's age, mtime, commit generation, or access count;
* the presence or absence of a previous binding;
* the file extension, path, tenant, model, dtype, or codec mode;
* the declared lifetime of a *different* artifact.

Class comes from the binding record and from nowhere else. The court
deliberately mixes sizes across the volatile and ephemeral populations so that
a size-based heuristic would be visibly wrong (`bench/RESULTS-retention.md`).

---

## 5. Rules

### 5.1 A binding never changes identity

Publishing, rebinding, or deleting a binding changes no artifact byte, content
root, `descriptor_root`, `ObjectId`, `ContentId`, manifest, codec frame, or
dedup key. The only bytes written are the binding record and the current
pointer (§6). A store that has never seen a binding and a store that has a
binding for every artifact produce byte-identical artifacts.

`durable` requires an explicit pin or snapshot root at bind time:

* an in-memory object pin on the artifact, or
* an object with the artifact's `descriptor_root` present in a named snapshot
  state, or
* an object with that `descriptor_root` present in a checkpoint state manifest.

A mutable namespace entry (for example a live KV prefix record) is **not**
sufficient: it can be deleted at any moment, which is the opposite of
"indefinite". A durable binding without one of the three roots above is
`InvalidArgument`, before any byte is written.

### 5.2 Lifetime is an admission hint

`declared_lifetime_ms` tells the GC planner how long the producer expects the
artifact to remain useful. It is:

* an **admission** input: a live binding may cause the planner to *retain*
  unreachable bytes it would otherwise reclaim;
* an **ordering** input: among expired bindings, the class and expiry decide
  which bytes are reclaimed first;
* **not** a guarantee: no implementation promises that bytes survive the
  declared lifetime, and none promises that they are deleted at its end;
* **not** a safety input: it never causes reachable bytes to be reclaimed.

An expired binding remains a valid binding. The artifact is simply no longer
protected by the hint.

### 5.3 Reachability is computed first, and class never touches it

GC computes the reachable set exactly as reachability-only GC does:

> every committed root (the current state), every snapshot commit state, every
> checkpoint state manifest (including rejected or unpublished generations),
> every in-memory object and content pin, every provenance receipt and its
> declared inputs and outputs, and every live prepared transaction's pins.

A class-aware GC pass MAY reclaim an expired binding's bytes **only** when the
artifact is unreachable from every one of those roots.

Class affects **ordering**, never reachability safety. Concretely:

* A blob in the reachable set is never a candidate, regardless of class or
  lifetime.
* A class-aware pass deletes a *subset* of what a reachability-only pass would
  delete; it may retain more (live bindings) but never less.
* Retaining unreachable bytes because a binding is live is a space decision,
  not a safety decision, and it can always be overridden by an explicit
  reachability-only GC.

**Safety invariant (normative):**

> For every class-aware GC pass, the set of reclaimed blobs is a subset of the
> set of blobs that the same pass's reachability computation marks unreachable.
> Equivalently: class-aware GC never reclaims a blob that reachability-based GC
> would keep.

The invariant is tested by `class_aware_gc_never_reclaims_a_reachable_blob`
(Rust) and `TestClassAwareGCNeverReclaimsReachable` (Go). Both build a store
with snapshot-, pin-, provenance-, checkpoint-, and transaction-retained
artifacts of every class, run a class-aware pass with a zero budget and an
expired binding for every artifact, and assert that every retained blob is
still present and every reclaimed blob was unreachable.

### 5.4 Ordering

Reclamation order is total and deterministic. Ascending rank:

| Rank | Population | Why |
|---:|---|---|
| 0 | `unbound` (no binding) | no declared utility at all; nothing to protect |
| 1 | `volatile` | cheapest to recompute; shortest declared utility |
| 2 | `session` | |
| 3 | `ephemeral` | |
| 4 | `durable` | longest declared utility; normally pinned and therefore unreachable only after an explicit unpin |

Within one rank: earliest expiry first (`issued_at + declared_lifetime_ms`),
then `descriptor_root` ascending, then blob key ascending. A blob shared by
several bindings takes the **most conservative** rank and the **latest** expiry
of its owners, so a blob shared by a `volatile` and an `ephemeral` artifact is
ordered as `ephemeral` and is retained if either binding is live.

A blob with no owning binding is `unbound` and is reclaimable as soon as it is
unreachable and past the ordinary grace gate.

Under a byte budget the planner takes candidates in this order until the budget
is reached. It always takes the first candidate even when that candidate alone
exceeds the budget, so a pass always makes progress; the overshoot is reported.

### 5.5 Honest non-guarantees

* A lifetime is a declaration by a producer. Nothing verifies that the producer
  told the truth, and a dishonest or mistaken producer can waste capacity.
* Class affects ordering. It does not change what is reachable, what is
  durable, or what a reader may read.
* Deleting a binding removes the hint, not the artifact. The artifact then
  behaves as `unbound` for ordering.

---

## 6. Store layout (normative)

Retention records are ordinary committed state, so they replicate, snapshot,
verify, and recover with the rest of the store. They live under one reserved
directory:

```text
/retention/bindings/<binding_id_hex>     immutable canonical binding record (file)
/retention/current/<descriptor_hex>      fixed 104-byte current pointer (file)
```

The pointer is exactly:

```text
magic      "RPT1"       4 bytes
version    u8 = 1       1 byte
reserved   zero         3 bytes
descriptor_root        32 bytes
binding_id             32 bytes
artifact               32 bytes   # manifest ContentId of the artifact frame
```

The `artifact` field is the physical anchor that lets a GC planner map a
binding to blobs without replaying history. It is metadata: it is not part of
`RetentionId` and not part of artifact identity. `retention set` derives it from
the live object that carries the artifact and rejects a pointer whose anchor
does not resolve to a present manifest.

The pointer is overwritten in place with a fixed-length range write, so a
reader observes either the old complete pointer or the new complete pointer.
Binding files are immutable and content-addressed by name; re-publishing an
existing binding is idempotent (`Exists` is treated as success only when the
existing bytes hash to the same `RetentionId`).

Binding history is therefore durable by construction: binding files are
reachable from the current head and are never rewritten. `retention_history`
follows `supersedes` from the current pointer and is bounded by
`MAX_RETENTION_HISTORY`.

---

## 7. Validation order (normative for error classes)

Decoding a binding:

1. WCB decode: non-minimal varints, unsorted or duplicate map keys, non-text
   map keys, trailing bytes, invalid UTF-8, unknown tags → `NonCanonical`;
   truncation → `Malformed`.
2. Not a map, unknown key, missing key, wrong value type → `Malformed`;
   `schema` differs → `Malformed`.
3. `descriptor_root` is not exactly 32 bytes or is all zero → `Malformed`.
4. `class` is not one of the four names → `Malformed`.
5. `declared_lifetime_ms` violates §3 → `Malformed`; above
   `MAX_DECLARED_LIFETIME_MS` → `LimitExceeded`.
6. `producer` is empty or exceeds 256 bytes → `Malformed` (the byte bound is
   checked before hashing).
7. `supersedes` is neither exactly 32 bytes nor null → `Malformed`.
8. `signature` present but malformed → `Malformed`; present and non-verifying →
   `Corrupt`; embedded key not in the configured trust set → `Unauthorized`.
9. The recomputed id differs from the address that named the record →
   `Corrupt` (checked by the store, not by the pure decoder).

`H("weft/retention/v1", …)` is computed only after steps 1–8 pass.

---

## 8. GC plan report

`gc_plan` is pure: it computes reachability, binds blobs to classes, orders
candidates, and returns a plan without deleting. `gc_apply` executes a plan.
Both report:

* `now_ns`, `budget_bytes` (optional), `grace_secs`;
* reachable manifest and chunk counts;
* per class (`unbound`, `volatile`, `session`, `ephemeral`, `durable`):
  retained blob count and bytes, reclaimable blob count and bytes;
* every **retained-but-expired** artifact with a reason string. A reachable
  artifact is reported as `reachable from <root>`, where `<root>` is a stable
  label (`head:<commit>`, `snapshot:<commit>`, `checkpoint:<commit>`,
  `transaction:<id>`, `pin:<content>`, `provenance:<id>`). A non-expired
  artifact is reported as `retention live until <timestamp>`;
* every reclaimed blob in order, with its class, owning descriptor root, size,
  and whether the artifact was expired;
* `over_budget_bytes` when live bindings keep the retained footprint above the
  budget.

The reason is evidence, not decoration: a reviewer must be able to see *why*
each expired artifact was kept, without rerunning the pass.

---

## 9. CLI

```text
weftfs retention set   --descriptor-root HEX (--class CLASS --lifetime-ms N | --class durable) \
                       --producer TEXT [--policy-epoch N] [--issued-at-ns N] [--object PATH]
weftfs retention show    (--descriptor-root HEX | --object PATH)
weftfs retention history (--descriptor-root HEX | --object PATH) [--limit N]
weftfs retention plan    [--budget-bytes N] [--now-ns N] [--json]
weftfs retention gc      --apply [--budget-bytes N] [--now-ns N]
```

`retention set` with `--class durable` requires §5.1. `retention plan` is a dry
run and never deletes. `retention gc` is a dry run without `--apply`.

Go exposes the same record format, pointer layout, and GC ordering through
`store.NewRetentionBinding`, `Store.SetRetention`, `Store.RetentionOf`,
`Store.RetentionHistory`, `Store.GCPlan`, and `Store.RetentionGC`. Because the
Go store has no mode-19 KV writer, its `RetentionRequest` supplies the artifact
anchor (`Artifact`) explicitly instead of resolving it from a live object's
`weft.kv.provenance_root` attribute. The bytes written and the ordering rules
are identical; the cross-language court reads a Rust-written binding in Go and
a Go-written binding in Rust, byte-for-byte (`scripts/retention_cross.sh`).

---

## 10. Non-claims

* **No wall-clock guarantee.** `declared_lifetime_ms` is a hint. No byte is
  promised to survive it and no byte is promised to be deleted at its end.
* **No clock-skew defence.** `issued_at`, expiry, and ordering use the store's
  clock. A backward-jumping clock can delay reclaim; a forward-jumping clock
  can advance it. Safety still holds, because reachability is computed from
  committed state and is not clock-dependent.
* **No truthfulness claim.** Nothing verifies that a producer's class or
  lifetime matches the workload. The court *measures* the cost of a wrong
  declaration; it does not prevent one.
* **No signing authority.** An embedded Ed25519 key authenticates a binding
  against that key only. Without a configured trust set it is not an
  authorization or identity claim.
* **No size or age inference.** See §4.1.
* **No cache implementation.** This document defines a contract and a GC
  ordering. It does not define an eviction engine, a serving integration, an
  admission controller, or a scheduler.
* **No fleet or media claim.** The court is one host, a local store, a
  synthetic trace, and a simulated clock.
