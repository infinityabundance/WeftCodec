# WEFT-CAUSAL v1.0.0 — causal closure and causal identity

**Status:** Normative for `weft-causal 1.0`. Implemented in Rust
`weftcodec::context` and Go `WeftCodec-Go/context` as of F1.
**Depends on:** `WEFT-MODEL.md` §2.3.1 (derived identities),
`WEFT-CONTEXT-ROOT.md`, `WEFTCODEC-FORMAT.md` §8 (WCB).
**Source of truth:** `spec/WEFT-CAUSAL.md` in `infinityabundance/WeftFS`.

Weft distinguishes **content identity** from **causal identity**. The bytes of
an artifact say what it is; its causal closure says what it was derived from
under which ancestry. This document defines the closure record, its canonical
hash, the two v1 identity modes, and the storage/authorization rules.

Causal *graphs* are allowed to cycle. Codec *reconstruction dependency* graphs
remain acyclic; the two graphs are different and MUST NOT be conflated.

---

## 1. Causal closure record `weft.causal.v1`

```
{
  "schema":   "weft.causal.v1",
  "parents":  [ bytes(32), ... ],   # explicit parent artifact identities
  "closure":  [ bytes(32), ... ]    # transitive ancestor closure
}
```

Both arrays are required (an empty array is present and encodes as
`0x07 0x00`). WCB map-key order is the wire order; array order is semantic and
MUST be canonical (below). Unknown keys are `NonCanonical`.

### 1.1 Canonical array rules

- Entries are 32-byte hashes (artifact or `ContextRootId` identities).
- Entries are **strictly ascending** by unsigned byte order; duplicates are
  rejected.
- The all-zero hash is rejected.
- `closure` MUST contain every element of `parents` (closure is a superset).
- `closure` MUST NOT be assumed to contain the artifact's own identity. An
  artifact that participates in a cycle may legitimately list itself as an
  ancestor; this is recorded explicitly and is not inferred.
- Length is bounded by `max_causal_entries` (default 4,096 for the record,
  configurable downward).

A record whose arrays violate any rule is `NonCanonical`. Implementations MUST
NOT sort-and-accept a record; sorting is a *construction* convenience only
(§1.3).

### 1.2 Identity

```
causal_bytes       = WCB(causal record)
CausalClosureId    = H("weft/causal/v1", causal_bytes)
```

`CausalClosureId` is a content address for the ancestry statement. It is not a
signature, not an authorization token, and not proof that any parent still
exists or is authorized. It identifies a claim, whose truth is checked at
resolution.

### 1.3 Construction

`Closure.from_parents(p1..pn)`:

1. validate every parent is nonzero 32 bytes;
2. sort ascending and remove duplicates;
3. set `closure = parents` when no further graph is available;
4. if a graph is available, `closure` is the union of all transitive ancestors
   reachable from any parent, plus the parents, sorted/unique. The artifact's
   own identity is excluded from this computed set (but may be present in a
   received record for cyclic artifacts).

The CLI `causal id --parents a,b` constructs the explicit-parent-only record
(step 3). A resolver that has the full graph MUST compute the full closure
(step 4); it MUST NOT claim the explicit-parents form is the transitive closure
when it has the graph.

---

## 2. Identity modes

v1 defines exactly two identity modes.

| Mode | Meaning |
|---|---|
| `bytes_only` | VersionId depends on content/version bytes only. **Default in v1.** |
| `causal_exact` | VersionId additionally includes the causal closure hash. |

The mode name is the text stored in `attrs["weft.identity.mode"]`; absence means
`bytes_only`.

### 2.1 `bytes_only`

The version record and `VersionId` are exactly as frozen in `WEFT-MODEL.md`
§2.3.1. The causal fields are storage metadata; they do not change the
`VersionId`. This preserves every existing v1 identity byte-for-byte.

### 2.2 `causal_exact`

The version record is the v1 record **plus** two fields:

```
{
  "schema":        "weftfs.version.v1",
  "object":        bytes(32),
  "content":       bytes(32),
  "size":          u64,
  "mtime_ns":      timestamp,
  "parent":        bytes(32) | null,
  "identity_mode": "causal_exact",
  "causal_hash":   bytes(32)
}
VersionId = H("weft/version/v1", WCB(record))
```

Rules:

- `identity_mode` MUST be exactly `causal_exact`.
- `causal_hash` MUST equal `H("weft/causal/v1", WCB(closure record))` for the
  artifact's recorded closure.
- Omitting either field is `NonCanonical` in this mode.
- `bytes_only` records MUST omit both fields; a record carrying
  `identity_mode: "bytes_only"` is `NonCanonical` (canonical form is the frozen
  v1 record). If a received record carries `identity_mode: "causal_exact"`
  without `causal_hash`, or vice versa, it is `NonCanonical`.
- `ContentId` is unchanged. Two artifacts with identical bytes and different
  closures have the same `ContentId` and different `VersionId` and
  `CausalClosureId` in `causal_exact` mode. In `bytes_only` mode they are the
  same version identity; the difference remains visible in the recorded
  closure, and consumers MUST NOT assume closure equality from `VersionId`
  equality.

Unknown mode names are `UnsupportedVersion`. A future mode requires a new
specification and new vectors.

---

## 3. Storage

### 3.1 Reserved attributes

Object/version records MAY carry these reserved keys in their `attrs` map. They
are **not** exposed through POSIX `xattr` calls (`WEFT-MODEL.md` §2.5).

| Attribute key | WCB type | Rule |
|---|---|---|
| `weft.causal.parents` | array of bytes(32) | strictly ascending, unique, nonzero |
| `weft.causal.closure` | array of bytes(32) | strictly ascending, unique, nonzero, superset of parents |
| `weft.causal.hash` | bytes(32) | equals the hash of the closure record |
| `weft.identity.mode` | text | `bytes_only` (default/omitted) or `causal_exact` |

When `weft.causal.parents` is present, `weft.causal.closure` and
`weft.causal.hash` MUST be present and consistent. A missing/extra/mismatched
causal attribute is `Corrupt`, not a silent repair. When `weft.identity.mode`
is `causal_exact`, `weft.causal.hash` MUST be present.

### 3.2 Version record

The optional causal field in the version record is defined in §2.2. It is
`causal_hash` plus `identity_mode`; there is no separate "parents" field in the
version record. The closure record itself is stored content-addressed and
referenced by `causal_hash`; this keeps version records fixed-size and makes
closure sharing exact.

### 3.3 Transactions and retention

Causal closure records obey the same commit, snapshot, and GC rules as other
immutable records:

- a closure record is published atomically (write + fsync + rename + directory
  fsync) before any version record that references it;
- a snapshot that references a version record keeps its closure reachable;
- GC treats every non-`bytes_only` version record's `causal_hash` as a root,
  and every recorded parent as a root while the referencing version is
  retained. The dependency closure of exported artifacts is retained as
  `WEFT-MODEL.md` §5 already requires.

### 3.4 Cycle policy (normative, with justification)

**Causal graphs MAY contain cycles.** Examples are legitimate: two agents whose
memories ground each other; a document and its errata; mutually referencing
iterative checkpoint notes that were later reconciled.

Consequences:

- Closure computation MUST use a visited set and MUST terminate. Each node is
  enumerated at most once; work is bounded by the graph's reachable node count
  and by `max_causal_entries` after canonicalization.
- A cycle is not an error. It is an ancestry fact. An artifact on a cycle may
  list itself in `closure`; readers MUST interpret that as "cyclic ancestor",
  not as corruption.
- A cycle MUST NOT be used to imply mutual authorization or mutual derivation
  correctness. Each edge is checked independently.
- **Codec reconstruction dependencies are a different graph and MUST remain
  acyclic** (`WEFTCODEC-FORMAT.md` §5). A `DependencyCycle` there is an error.
  The causal graph's cycles never authorize a cyclic reconstruction; a decoder
  that would need one fails closed.

Why not forbid cycles? Because real provenance is often cyclic and truthfully
recording a cycle is safer than forcing a lie (dropping an edge) or an arbitrary
break. Determinism is preserved by the visited-set rule and canonical sorting.

---

## 4. Minimal sufficient closure queries

A **minimal sufficient closure** for a target set `T`, relative to a principal
and a materialization state, is the smallest set `C` such that:

1. `C` is exactly the union of the transitive ancestor sets of `T` that are
   needed to verify causal identity and reproduce the requested artifacts;
2. every member of `C` is authorized for the principal;
3. no member of `C` is already present in the requester's verified local
   materialization when the requester asks for "missing only";
4. `C` is sorted and unique and has no extra members.

The query MUST expose:

- `total_closure_size` — authorized closure size after filtering;
- `returned_closure_size` — entries actually returned;
- `truncated` — true if a budget prevented returning the full authorized set;
- `roots` — the requested target identities.

Rules:

- Authorization filtering happens **before** counting. A count, evidence list,
  error message, or timing hint MUST NOT reveal an unauthorized ancestor.
- If non-disclosure applies, a denied ancestor is indistinguishable from a
  non-existent one at every observable output.
- "Minimal sufficient" never means "omit an ancestor to save bytes". It means
  "omit ancestors that are not required for this request and this verified local
  state". Omitting a required ancestor is a correctness failure, not an
  optimization.
- A closure query result is a *claim about the graph*, not an authorization
  grant. Resolution still checks each artifact.

### 4.1 Evidence and leakage

Closure evidence may name an edge `(child, parent, predicate)` only when the
principal is authorized to see both endpoints and the edge itself. Counts are
over the authorized subgraph. Error detail MUST NOT distinguish "denied" from
"absent" when the threat model requires non-disclosure. v1 makes no timing
side-channel claim; see `security/THREAT_MODEL.md`.

---

## 5. Errors and limits

| Class | Cause |
|---|---|
| `NonCanonical` | unsorted/duplicate/zero entries, closure missing a parent, wrong types/keys, mode/field mismatch |
| `Malformed` | truncated/non-WCB bytes |
| `UnknownSchema` | schema string mismatch |
| `UnsupportedVersion` | unknown identity mode |
| `LimitExceeded` | closure entry count or string bound exceeded |
| `Corrupt` | stored closure hash or attribute inconsistency |
| `DependencyCycle` | only for **codec reconstruction** graphs, never for causal graphs |
| `Unauthorized` | closure or evidence access denied |
| `BudgetExceeded` | closure query budget exceeded |

`max_causal_entries` defaults to 4,096 (configurable downward). A closure that
exceeds it is `LimitExceeded` **before** evaluation, even if the graph would
otherwise be computable.

---

## 6. Non-goals in v1

- No cryptographic proof of ancestry (no signatures, no Merkle inclusion proofs
  beyond the hash binding). A closure is an assertion; its integrity is the hash,
  its truth is checked against stored records.
- No global causality across stores or organizations.
- No automatic rebase/merge of closures.
- No causal garbage collection beyond the reachability rules above.
- No claim that `bytes_only` artifacts with different closures are the same
  artifact in the causal sense. The default mode is a compatibility choice, not
  a semantic claim.
