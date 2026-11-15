# WEFT-TAINT — taint tracing and quarantine

Status: **implemented in the Rust store (T075)**. Normative for
`weft.taint-trace.v1` and `weft.quarantine.v1`. A Go implementation does not
exist; a Go store MUST reject both schemas rather than guess. All frozen bytes,
modes and roots are unchanged: both records are new, additive, and
content-addressed.

## 0. What this is not

**Quarantining is not unlearning.** A quarantine revokes visibility and refuses
to serve an artifact. It cannot remove information that has already been learned
into model weights, and it does not inspect, edit, retrain or repair a model.
The deliverable is **traceability plus refusal to serve**. Any claim that this
constitutes unlearning would be false.

## 1. Taint trace

A trace starts from one or more **tainted content roots** — the invariant
identity of a datum, not a path, name or object id — and walks the provenance
graph forward.

```
TaintTrace {
  roots: [bytes32],                   # sorted, deduplicated
  tainted: [ { content, depth, via, record, operation, role, objects } ],
  artifacts: [ { object, content, verdict, witness, gaps } ],
  gaps: [text],                       # global gaps
  iterations: u64,
  records_scanned: u64
}
```

`via` is `"root"` or `"provenance"`. `objects` is every live object whose
content is that root, ascending. The forward walk visits only provenance
records whose declared inputs intersect the tainted set and whose own evidence
is consistent (see §3). It terminates when no new content root is tainted; a
walk that would exceed the caller's depth bound or the node bound is
`LimitExceeded` — **a provenance walk is never truncated**.

### 1.1 Verdicts

Each artifact the caller names receives exactly one verdict:

| Verdict | Meaning |
|---|---|
| `tainted` | A forward path from a tainted root exists. `witness` names the content roots and provenance records on it. |
| `clear` | The artifact has its own provenance receipt, every record on its backward closure has consistent evidence, and no node is tainted. |
| `unproven` | The graph is incomplete on this closure. `gaps` names exactly what is missing. |

`unproven` is never presented as `clear`. The trace's own status is `closed`
when no gap exists and no artifact is `unproven`, and `unproven` otherwise.

A **leaf with no receipt is a source, not a gap**: provenance records name the
inputs they consumed, so a source's identity is known even when nothing recorded
how it came to exist. A gap is:
* the artifact itself has no provenance receipt (its derivation is unknown);
* a receipt on the closure names an input `object` hint that no longer carries
  that content, or that is gone (the chain is broken);
* a `provenance` relation cannot be interpreted (a global gap: nothing
  downstream of it is trustworthy);
* a declared tainted root resolves to no live object (global).

Paths, names, modes and timestamps are **not** identity and never appear in the
walk. A rewrite that keeps the bytes is the same content root and is therefore
tainted by the same trace.

## 2. Quarantine record

```
weft.quarantine.v1 {
  schema: "weft.quarantine.v1",
  kind: "quarantine" | "release",
  content: bytes32,                 # the governed content root
  mode: "non-disclosing" | "explicit",
  reason: text,                     # 1..8192 bytes
  trace: bytes32 | null,            # root of the weft.taint-trace.v1 decision input
  trace_status: "closed" | "unproven",
  policy_epoch: u64,
  issued_at: timestamp,
  supersedes: bytes32 | null,
  signature: { algorithm: "ed25519", public_key, signature } | null
}
```

* `QuarantineId = H("weft/quarantine/v1", WCB(record))`, over the record
  **including** `signature`. The signature covers the canonical bytes without
  the `signature` key, using the same Ed25519 construction as
  `weft.retention.v1`.
* `trace` is required for a `quarantine` decision when one exists, and refused
  for a `release`.
* **Authorization.** A decision MUST be signed, and the signer MUST be in the
  authority set the caller supplies. An empty authority set refuses both
  quarantining and lifting: there is no implicit authority. A quarantine whose
  `trace_status` is `unproven` is permitted — the record states that the graph
  was incomplete rather than pretending otherwise.
* **Append-only and epoch-fenced.** A second decision for the same content must
  name a strictly higher `policy_epoch` and must set `supersedes` to the
  current record. A release must name an epoch above the active quarantine's.
  Rebinding never changes identity, bytes, content root or dedup key.

## 3. Storage

* Records are ordinary namespace files under `/quarantine/records/<hex>`.
* The current decision per content root is one fixed **72-byte** pointer file
  under `/quarantine/current/<content-hex>`, replaced (not appended) in place:
  `WQP1 || 4 zero bytes || content root || record id`.
* Both live in the namespace, so they are ordinary reachable state: snapshots,
  checkpoints and pins retain them, and GC collects them with the tree.
* A pointer that does not parse, or names a content root or record that
  disagrees with its key, is `Corrupt` — a tampered pointer is never read as a
  release.

## 4. Enforcement

The gate is evaluated on the object's **content root**, so every object sharing
that content is refused, including a copy under a new name and a rewrite that
produced the same bytes.

| Entry point | Enforced |
|---|---|
| `Store::read_object`, `read_object_at`, `read_object_authorized` | yes |
| `Store::read_range`, `read_range_authorized` | yes |
| `Store::read_content`, `read_content_authorized` | yes |
| `Store::load_checkpoint_manifest`, `restore_checkpoint*`, `restore_named_tensor*` | yes (manifest object and every shard content root) |
| FUSE reads, `weftfs cat`, `weftfs range` | via the object/range entry points above |

The refusal is the policy's:

* `non-disclosing` returns `NotFound: no such object`, **byte-identical** to an
  absent object, so the refusal discloses nothing about existence;
* `explicit` returns `Quarantined: <reason>`.

Quarantine records and pointers are read through an exempt path so the gate can
never make its own state unreachable.

A store handle observes the quarantine state of the snapshot it holds. Another
process's decision becomes visible after `refresh_to_head` or a reopen, exactly
like every other namespace mutation.

## 5. Bounds

| Field | Bound |
|---|---|
| tainted roots | 1,024 |
| artifacts | 4,096 |
| tainted nodes | 65,536 |
| trace depth | 64 |
| reason | 8,192 bytes |
| pointer | exactly 72 bytes |

Every violation is `LimitExceeded` or `Malformed`. There is no partial trace.

## 6. What a verdict does not mean

* `clear` is **relative to the recorded graph**. An actor that omitted a
  receipt makes an artifact `unproven`, not `clear`; an actor that records a
  false receipt makes it `clear` dishonestly. The mechanism detects missing
  evidence, not lying evidence.
* `tainted` does not say how much a model was influenced. It says a path
  exists.
* A quarantine is not a cryptographic erasure and does not revoke bytes from a
  backend that still holds them. It refuses to serve them through this store.
