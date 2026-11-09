# WEFT-MULTIWRITER — writer capability levels and the multi-writer contract

Status: **normative for the MW2 surfaces implemented in subphase R4.** The
single-committer surfaces documented here are already implemented and frozen;
the optimistic-committer surfaces are implemented in Rust and Go for the
`local` and `object` commit planes and in Rust for `s3`/`gcs`/`azure`. Anything
not listed as implemented in §12 is specified design, not a claim.

Version: `weft.multiwriter/v1`.

This document is the authority for:

* which writer capability level a store is at,
* what a caller may conclude from a successful publication, an ambiguous
  timeout, or a typed conflict,
* which concurrent histories are safe, retried, branched, rejected, or merged,
* the durability boundary of an acknowledgement.

It does not specify a consensus protocol. MW2 is **not** a distributed
consensus protocol and **not** a fleet protocol. See §11.

---

## 1. Capability levels

A store is at the **highest level for which it has an executed, tested
protocol**. A level is not a product tier and is never implied by a lower
level.

### MW0 — concurrent immutable blob writes

**Implemented.** Immutable blobs are content-addressed, so any number of
producers may publish them concurrently with create-once semantics. A blob
worker (`Store::open_blob_worker`, `WeftFS-Go` `OpenBlobWorker`) is read-only
for control state and may only `put_if_absent` immutable bytes.

MW0 is the prepare half of MW1/MW2. It is not a safe mode for logical state on
its own: publish a blob and no reader can find it until a commit names it.

### MW1 — many producers, one committer

**Implemented; this is the default and the level of every pre-R4 surface.**

N producers publish immutable fragments independently. Exactly one committer
holds the exclusive writer proof (POSIX `LOCK` lock or the fenced object-store
lease of `docs/FENCING.md`), re-verifies fragments, and publishes one commit
against one observed head. The fenced lease decides *which* single writer may
publish; it says nothing about how two writers would agree.

The store must be opened with `--writer-mode single` (the default) for every
operation not explicitly listed as MW2 in §12. A second writer is refused
`Busy`.

### MW2 — independent committers

**Implemented for the surfaces in §12; opt-in with `--writer-mode optimistic`.**

Two or more *independent processes* may prepare transactions and publish
against one shared store. Publication is a conditional head update
(`compare_exchange_head`) against the head observed by the transaction. A lost
race is either retried against the new head after conflict validation, or
rejected with a typed `Conflict` naming the class. The store does not run a
leader, a quorum, or a lock service.

MW2 guarantees linearizable publication **of the authoritative head sequence**
(§4.1), not serializability of arbitrary multi-object reads, and not
snapshot isolation across an entire workload. What is guaranteed per operation
is stated in §4.

### MW3 — distributed scalable publication

**Not implemented. Not claimed.**

MW3 would require conflict domains, bounded cross-node contention, a
coordinated multi-writer GC, and fleet evidence. Nothing in this document or
in the R4 courts supports it. Every R4 court runs on one host against one
local namespace or one MinIO/Azurite endpoint.

---

## 2. Terms

* **Authoritative state**: the two-slot superblock head and the commit chain it
  selects (`commits/<generation>.commit`). Blobs are only reachable through a
  committed state.
* **Commit**: one canonical `weftfs.commit.v2` record with a parent, a
  generation, an op list, and optional checkpoint manifest and index
  attestation.
* **Transaction**: a bounded, identified set of ops plus preconditions,
  published as exactly one commit.
* **Prepared transaction**: the immutable blobs and the op set are durable, but
  no commit names them yet.
* **Visible**: a reader that selects the commit as its head can decode it.
* **Durable**: the backend has acknowledged the commit record and the head
  publication under its documented durability contract.
* **Committer**: a process that may call the publication primitive.
* **Winner**: the transaction whose conditional head publication succeeds.
* **Loser**: a transaction whose conditional head publication is refused
  because the head changed.

---

## 3. Backend contract required by MW2

MW2 needs exactly one primitive beyond MW1:

1. **Atomic conditional head publication**: `compare_exchange_head(expected,
   next)` succeeds iff the plane still reports `expected` as the head
   (compared by generation and commit id), and otherwise fails `Busy`. The
   8 KiB superblock image is preserved byte for byte except for the slot
   `next.generation % 2`.
2. **Create-once commit records**: `commits/<generation>.commit` is immutable
   and create-once. Identical bytes under the same generation are idempotent;
   different bytes are an error.
3. **Create-once immutable blobs** (MW0).
4. **A durable acknowledgement** for both the record and the head.

It does **not** need multi-object atomicity. A transaction spanning several
objects is safe because the commit record is the single publication point: the
ops inside one commit become visible together even though the commit file and
the head are two physical writes (§5).

Primitive per backend:

| backend | conditional head | publication helper | concurrency proof |
|---|---|---|---|
| `local`, `object` | read–verify–write under a short publication lock on `LOCK.publish` (`fcntl` OFD + compatibility `flock`), then one 4 KiB slot write | store acquires the publication lock for the critical section only | the publication lock serialises only the head critical section; prepare and encode run outside it |
| `s3` | `GET` + `PUT` with `If-Match: <ETag>` on `superblock` | native conditional write | the object store's conditional PUT |
| `gcs` | metadata+media read, then upload with `ifGenerationMatch=<observed>` | native conditional write | the object store's generation precondition |
| `azure` | `Get Blob` + `Put Blob` with `If-Match: <ETag>` | native conditional write | the object store's conditional PUT |

`LOCK.publish` is **not** the single-writer `LOCK`. It is created lazily only
by an optimistic store, is never read for correctness, and carries no bytes
that a reader interprets. A single-writer store and an optimistic store cannot
share a namespace: the `writers` policy record (§10) makes the mode explicit
and a mode mismatch is refused `UnsupportedVersion`.

A backend whose conditional update is unproven is refused. `CommitPlane::probe`
already establishes that a stale compare-and-swap is rejected; MW2 reuses that
probe unchanged.

---

## 4. Consistency contract

### 4.1 Publication

**Guarantee P1 — linearizable authoritative head.** The sequence of published
heads is a total order. If transaction A publishes commit `cA` and transaction
B publishes commit `cB`, then every reader observes some order consistent with
the physical conditional writes: either `cA` precedes `cB` (B observed `cA`
as its parent or observed a later head) or `cB` precedes `cA`. No two commits
occupy the same generation, no head regresses, and a commit selected as head
has every ancestor present.

**Guarantee P2 — atomic visibility of one commit.** Every op in one commit
becomes visible together. A reader that selects commit `g` never sees a strict
subset of `g`'s ops: the ops are replayed from the commit record, not from
individual object writes.

**Guarantee P3 — no lost acknowledged transaction.** If a publish call returns
`Ok(commit)`, the commit is durable at its backend's acknowledged durability
boundary and is an ancestor of every later head. It is never replaced,
overwritten, or skipped by a concurrent winner. A later head may make it
non-current, but not non-existent.

**Guarantee P4 — snapshot-consistent readers.** A read handle selects one head
and replays that commit chain. Reopening may select a newer head. A handle
never mixes objects from two heads within one API call. A handle's own writes
are durable before the call returns; visibility to *other* handles is
eventual in the sense that they must select the new head (reopen or explicit
refresh), not that the data is unsafe.

### 4.2 Per-operation scope

The table states the contract of the op *inside one commit*. Multi-op
transactions inherit P2. There is **no** guarantee of cross-commit
serializability, and no guarantee of a global order among reads.

| operation | MW2 outcome | visibility |
|---|---|---|
| immutable blob publish | always safe (content-addressed, create-once) | not visible until a commit names it |
| object create (fresh `ObjectId`) | safe; a duplicate identical creation is idempotent | with its commit |
| object replace (existing id) | requires an observed `version`; otherwise `Conflict(ObjectChanged)` | with its commit |
| directory entry create (parent, name) | safe if the name is still unused; otherwise `Conflict(NameExists)` unless the existing entry has the same target (idempotent) | with its commit |
| directory entry remove | requires the observed entry id; a changed entry is `Conflict(EntryChanged)` | with its commit |
| rename | source entry remove + target entry create + object update; any changed precondition is a typed conflict | atomic as one commit |
| relation create | deterministic `RelationId`; identical duplicate is idempotent; a different assertion under the same id is `Conflict(RelationConflict)` | with its commit |
| relation delete | requires the observed relation id | with its commit |
| snapshot / context-root publish | name-keyed; absent → create, same commit → idempotent, different commit → `Conflict(SnapshotConflict)`; **no branch is created automatically** | with its commit |
| tip update | requires the observed tip | with its commit |
| quota predicate | evaluated inside the transaction against the rebased state; a concurrent winner can only make the predicate *more* likely to fail, never bypass it | the commit that charges is the commit that is visible |
| capability grant/revoke | a capability record and its relation are ordinary ops; an authorization precondition is re-evaluated on every rebase attempt | with its commit |
| retention / GC | GC marks from every committed head and from every live `prepared` transaction record; it never reclaims a reachable blob | GC is a separate exclusive administration pass; it does not publish logical state |

**Explicitly not guaranteed:** serializable execution of arbitrary read/write
mixes; read-your-neighbour's-write without selecting the new head; a global
order between two independent transactions; atomicity across two commits; any
multi-host order.

### 4.3 Multi-object transactions

A commit with N ops publishes all N or none. If the publication primitive is
lost, the loser's commit record may remain as an unreferenced file; no reader
can reach it, no head names it, and GC may reclaim any blob only it referenced.
The loser is therefore **not** partially visible.

---

## 5. Transaction lifecycle

```
                 begin
                   │
                   ▼
     ┌──────── prepared ────────┐
     │  (blobs durable, txs/<id> │
     │   record = prepared)      │
     └───────┬───────────┬───────┘
             │ publish ok │ publish lost / crash
             ▼            ▼
          committed    outcome_unknown
             │            │  (txs/<id> still prepared)
             │            │  lookup by tx id + op digest
             ▼            ▼
          visible      aborted (typed Conflict)
             │
             ▼
          durable
```

| state | observable | what a client may conclude |
|---|---|---|
| `prepared` | only through the `txs/<id>` control record and the immutable blobs; not a head, never through the namespace | the transaction has not published; a retry is safe |
| `committed` | yes: it is the head or an ancestor of the head | the commit exists and is durable at the backend boundary; reading the new head sees it |
| `visible` | yes: any reader that selects this head or a descendant | all ops of the commit are present |
| `durable` | yes | the backend acknowledged the record and the head under its documented contract |
| `aborted` | the typed error return | nothing was published by this attempt; the namespace is unchanged by it |
| `outcome_unknown` | the `txs/<id>` record says `prepared` and the call timed out / crashed | **must not** be read as either success or failure. The caller resolves it with the outcome lookup (§6) before retrying. |

`committed` implies `visible` at the publishing handle and at any handle that
re-selects the head. `durable` is the backend's acknowledgement, not a media
claim; see §9.

### 5.1 Allowed and forbidden transitions

* `prepared → committed`: exactly one conditional head publication.
* `prepared → aborted`: a typed conflict or a validation failure. No head
  write occurred.
* `prepared → outcome_unknown`: the publication call failed ambiguously
  (transport error, process death after the record, timeout). The store must
  not report success or failure.
* `outcome_unknown → committed`: outcome lookup finds the transaction's ops in
  a commit that is an ancestor of the current head.
* `outcome_unknown → aborted`: outcome lookup establishes that no commit
  contains the transaction's op digest within the bounded window. The caller
  may then retry as a *new* attempt; the transaction id is retained so the
  record is repaired.
* `committed → anything else`: forbidden. A commit is immutable.

---

## 6. Idempotency and ambiguous timeouts

Every transaction has a caller-supplied 128-bit **transaction id**, rendered
as 32 lowercase hex characters. The id is echoed in the `txs/<id>` control
record. It is not part of the commit format (the commit format is frozen), so
outcome discovery uses a deterministic **op digest**:

```
tx_digest = BLAKE3("weft/tx/v1", WCB({
    id: tx_id, parent_generation: u64, ops: [canonical op WCB, ...]
}))
```

The `txs/<id>` record is canonical WCB `weft.tx.v1`:

```
schema = "weft.tx.v1"
id = <32 hex>
state = "prepared" | "committed"
digest = <64 hex>
parent_generation = u64
commit = <64 hex or "-">
updated_ns = i64
pins = [<64 hex manifest id>, ...]      # sorted, unique
```

`pins` names the manifest ids the prepared transaction still needs. GC marks
them while the record is live and within the grace window, which is what stops
a concurrent or immediately following sweep from reclaiming a blob the eventual
commit names.

Protocol:

1. Before publishing, the committer writes `txs/<id>` with state `prepared`,
   the digest, and the observed parent generation. The write is create-once;
   an existing record with a different digest is `Conflict(TransactionIdReused)`.
2. After a successful publication the committer replaces the record with
   state `committed` and the commit id. This write is advisory for
   *discovery*, never for correctness; a failure to write it is a returned
   diagnostic, not a publication failure.
3. A retry with the same transaction id **must** call `transaction_outcome`
   first. If `committed`, the store verifies the named commit is the head or an
   ancestor within a bounded window and returns it. If `prepared`, the store
   scans commits from the current head back to the recorded parent generation
   (bounded by `max_outcome_window`, default 256) and compares canonical op
   bytes. A match is the outcome; the record is repaired to `committed`.
4. If no match is found, the outcome is `aborted` for that attempt. Reusing the
   id with the same digest is allowed and is a new attempt under the same
   logical transaction. Reusing the id with a different digest is
   `Conflict(TransactionIdReused)` and publishes nothing.

An ambiguous timeout therefore never requires the caller to guess. The two
legal client behaviours are: call `transaction_outcome`, or retry with the same
id (which performs the lookup itself).

---

## 7. Conflict model

Conflict validation happens **after** the transaction has prepared its blobs
and **before** each publication attempt, against the head the attempt will
use. The transaction records the state it observed when it was built (the
*p0* pre-image). Every candidate head is checked against p0.

| class | detection | outcome | rule that makes it safe |
|---|---|---|---|
| `NameExists` | a directory entry for `(parent, name)` now targets a different object | reject (`Conflict`) | a name is a unique key; two different targets cannot both be visible under one name |
| `NameIdempotent` | the entry now targets the same object | merge (no-op) | the transactions are observably identical for this op |
| `ObjectChanged` | `PutObject` for an existing id whose current `version` differs from the observed version | reject (`Conflict`) | a version is an immutable identity of the object content; replacing it without having observed it would be last-writer-wins |
| `ObjectAbsent` | `PutObject` replacing an object that has disappeared | reject (`Conflict`) | the update's base is gone; it cannot be replayed |
| `EntryChanged` | `DelEntry` or rename whose entry id no longer names the observed target | reject (`Conflict`) | deleting or moving an entry you did not observe is unsafe |
| `RelationConflict` | the same `RelationId` now names a different assertion | reject (`Conflict`) | the id is derived from the assertion; a different assertion is a different relation with a colliding id, which is corruption |
| `SnapshotConflict` | a snapshot/context-root name now names a different commit | reject (`Conflict`); **no automatic branch** | a named pointer is single-valued; branching must be explicit (`docs/branching.md`) |
| `TipChanged` | the tip changed since observation | reject (`Conflict`) | tips are observed versions |
| `QuotaExceeded` | the rebased state would exceed a budget | reject (`BudgetExceeded`) | the predicate is re-evaluated with the winner's state; a concurrent commit cannot spend the same bytes twice |
| `CapabilityDenied` | an authorization precondition fails on the rebased state | reject (`Unauthorized`) | the precondition is re-checked on every attempt, after the winner's grant/revoke |
| `ChildRace` | a directory entry op would violate the current parent/child shape (for example the parent was removed) | reject (`Conflict` or `NotFound`) | the op is validated by `State::apply_ops` on the candidate state |
| `GenerationCollision` | the commit file for the chosen generation already exists with different bytes | retry against the new head | commit files are create-once and generation-keyed; a collision proves another writer won |
| `ParentAdvanced` | the observed head is no longer the head | retry (bounded) | the transaction is revalidated against the new state before re-publishing |

**No last-writer-wins rule exists anywhere.** A `PutObject` without an observed
version is only legal for a fresh object id. A `SetSnapshot`/`SetTip` without
an observed value is only legal for a create. An entity that changed under a
transaction is either an explicit idempotent merge or a typed conflict.

Retries are bounded: `max_attempts` (default 8) with exponential backoff
starting at 200 µs and capped at 25 ms, deterministic (no randomised jitter),
and no retry is attempted when the conflict table says reject. Exhausting the
budget returns `Busy` with `outcome_unknown`; the caller resolves it by
transaction id.

---

## 8. Authorization, quota, provenance inside the transaction

These checks are **not** pre-publication advisory checks:

* **Authorization.** A transaction records the principal and the capability
  facts it relied on. On every retry the store re-evaluates the recorded
  precondition against the rebased state. A revoke that wins the race makes the
  transaction `Unauthorized`; it cannot publish.
* **Quota.** Admission is computed against the candidate state and the winner's
  committed usage. The usage assertion is part of the same commit, so a reader
  never observes data charged to a different generation.
* **Provenance.** A `weft.provenance.v1` record published by the transaction is
  an ordinary op in the same commit; it is visible exactly when its outputs
  are.
* **Payload verification.** Every manifest and checkpoint a commit names was
  fully written and verified before the commit record is published. A loser's
  payload is unreferenced, never partially visible, and GC may reclaim it.

---

## 9. Durability boundary

An `Ok` from publication means:

1. every blob and checkpoint manifest named by the commit was published
   create-once and acknowledged by the blob plane;
2. the commit record was published create-once and acknowledged;
3. the conditional head publication was acknowledged;
4. the backend's documented durability contract has been met for (2) and (3).

It does **not** mean physical media has survived a power cut. On local POSIX
the backend uses `fsync` and rename/link; on S3/GCS/Azure it means the provider
acknowledged the PUT. No R4 court is a physical power-loss court; the crash
courts are process-fault proxies, and the one device-model court is labelled as
such.

An ambiguous return (transport error, timeout, `outcome_unknown`) makes **no**
durability claim in either direction. Use §6.

---

## 10. Policy records and mode identity

Optimistic publication is opt-in. The first optimistic writable open creates a
control record:

```
weft.writers.v1
mode = "optimistic"
min_reader = "weftfs 1.2.0"
created_ns = i64
```

Rules:

* A `single` writable open refuses a store whose `writers` record says
  `optimistic` (`UnsupportedVersion`), and an `optimistic` open refuses a
  `writers` record that says `single`. An absent record means `single` and is
  created by the first optimistic open.
* A read-only open is never refused and never creates the record.
* The record is advisory metadata for mode identity; it is not a source of
  truth for the head.
* A store that has ever run optimistic publication cannot be safely opened by a
  pre-R4 binary as a writer; the record is the detection mechanism. The
  `min_reader` field exists so a future format can refuse older readers too.

---

## 11. GC, retention, and in-flight transactions

GC marks from:

1. the current head and every named snapshot;
2. every checkpoint generation reachable from those roots;
3. every capability, context-root, provenance, and retention root recorded in
   committed state;
4. every **live prepared transaction** recorded under `txs/<id>`, bounded by
   `gc_transaction_grace_secs` (default 300 s) and a hard bound on the number
   of live records.

A prepared transaction pins only the blobs its ops name, not the whole store.
GC is an **exclusive administration pass**. In R4 it does **not** run
concurrently with an optimistic publisher: a caller that wants to run GC on an
optimistic store must quiesce its committers first. The `pins` field is what
makes the interleaving *prepared → GC → publish* exact; true concurrent
GC/publication is not claimed in R4. The `local`/`object` GC pass does not take
the publication lock, because it also republishes logical state for orphan
removal; making that path transaction-aware is future work. A prepared
transaction older than the grace is treated as abandoned: GC may reclaim its
unreferenced blobs and the record is deleted. This is safe because an abandoned
prepared transaction has no commit naming it.

GC never deletes a blob named by a committed head or by a live (unexpired)
prepared transaction. A test must demonstrate that a GC pass racing an
in-flight prepared transaction leaves the eventual publication readable and
exact.

---

## 12. What R4 implements, honestly

| surface | Rust | Go | evidence |
|---|---|---|---|
| MW0 blob workers | yes, frozen | yes | `tests/parallel_checkpoint.rs`, `store/group_test.go` |
| MW1 single committer | yes, default | yes, default | all pre-R4 suites |
| MW2 explicit transaction API on `local`/`object` | yes | yes | `tests/multiwriter.rs`, `store/mw2_test.go` |
| MW2 on `s3` (MinIO, Ceph RGW) | yes | no (R4) | `examples/bench_multiwriter.rs` MinIO receipt |
| MW2 on `gcs`/`azure` | code path only, no concurrent court | no | not claimed |
| MW2 for FUSE file mutations | **not implemented** | **not implemented** | FUSE remains MW1 |
| MW2 for checkpoint publish | **not implemented as a generic path**; a tensor/shard set can be expressed as a transaction | no | not claimed |
| mixed Rust/Go concurrent writers | yes | yes | `scripts/multiwriter_court.sh` |
| model-checked multi-committer invariants | bounded TLA+ | n/a | `verification/tla/WeftMultiWriter.tla` |
| linearizability proof | **no** | **no** | finite history check only |
| MW3 (distributed scalable) | **no** | **no** | not claimed |

The default remains `single` after R4. The R4 evidence supports optimistic
publication for the explicit transaction API on the planes above; it does not
support making every file operation optimistic, and it does not support a
claim that optimistic mode is faster for a single writer. A deployment that
wants MW2 must opt in and must route its logical publishes through the
transaction API.

---

## 13. Limits and non-goals

* No consensus, no quorum, no leader election, no multi-node ordering.
* No serializable isolation and no distributed snapshot isolation.
* No automatic branching or merge policy for named pointers: a conflict is
  returned to the caller.
* No last-writer-wins and no blind overwrite, anywhere.
* No physical power-loss claim.
* No claim that a finite history check proves linearizability.
* No claim that optimistic mode is faster; the benchmark reports both the
  throughput and the conflict cost, including all losses.
* The transaction API is bounded: op count, total bytes, retry budget, outcome
  window, and in-flight prepared records are all limited and fail closed.

## 14. References

* `docs/PLANES.md` — blob/commit plane contracts.
* `docs/FENCING.md` — the MW1 fenced lease; **not** an MW2 protocol.
* `docs/AI-STORAGE-AUGMENTATION.md` §4 — capability levels.
* `docs/MULTIWRITER-ARCHITECTURE.md` — R4 design and backend tradeoffs.
* `bench/CONTRACTS-multiwriter.md` — frozen benchmark contract.
* `bench/RESULTS-multiwriter.md` — R4 measurements and every loss.
* `verification/tla/WeftMultiWriter.tla` — bounded model-checked invariants.
