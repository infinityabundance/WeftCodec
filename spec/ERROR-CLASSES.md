# Error classes and the RFC §13 reconciliation

Status: **normative for the shipped registry** listed in §2; the rows in §3
marked *proposed* are **not** normative, **not frozen**, and **not implemented**
unless the row says otherwise. This document performs the reconciliation RFC
§13 asked for ("these strings are **proposed** and must be reconciled against
the existing class registry before freeze").

The class string is the interoperable contract. Human-readable messages, their
wording, and their detail are **not** a contract and may change in a patch
release (`WEFT-COMPATIBILITY.md` §3). Conformance vectors record the class,
never the wording.

## 1. Two registries, one vocabulary

There are two shipped registries because there are two crates:

| registry | source of truth | normative for |
|---|---|---|
| codec | `WeftCodec/src/error.rs` (`ErrorClass`) and `WeftCodec-Go/werrors` | frame and selector decode/encode results, every conformance vector |
| store | `src/error.rs` (`Class`) and `WeftFS-Go/werr` (`Kind`) | store operations, CLI output, `verify` problem classes |

Both spell the shared classes identically. A store implementation maps a codec
error onto its own registry without changing the class (`FromCodec` in both Go
and Rust). A new class must be added to both registries in the same change if
both layers can emit it.

## 2. Shipped store classes

`R` = emitted by the Rust store, `G` = emitted by the Go store. Where `G` is
blank the class is absent from `WeftFS-Go/werr` and that language has no code
path that returns it (an external reader must not assume the class exists
there). The first twelve rows are the codec classes as re-used by the store.

| class | meaning | R | G |
|---|---|:-:|:-:|
| `Malformed` | structural byte/format error | ✓ | ✓ |
| `Corrupt` | checksum, hash, length, or chain mismatch | ✓ | ✓ |
| `UnsupportedVersion` | format major/minor, backend contract, or writer mode not supported | ✓ | ✓ |
| `UnsupportedMode` | (codec only) a payload mode this reader does not implement | — | — |
| `UnsupportedFeature` | (codec only) unknown critical flag/feature | — | — |
| `LimitExceeded` | a configured resource bound would be exceeded | ✓ | ✓ |
| `DependencyMissing` | (codec only) a dependency cannot be found | — | — |
| `DependencyDenied` | (codec only) authorization refuses a dependency | — | — |
| `DependencyCycle` | (codec only) the reconstruction graph is cyclic | — | — |
| `NonCanonical` | canonical bytes are not canonical | ✓ | ✓ |
| `Io` | transport/read/write failure, never reported as damage | ✓ | ✓ |
| `DependencyMismatch` | a declared causal dependency is missing, unreachable, or mismatched | ✓ | |
| `Parse` | selector grammar error | ✓ | ✓ |
| `UnknownSchema` | schema string is not the expected one | ✓ | ✓ |
| `Busy` | the writer lock or a conditional head CAS refused this writer | ✓ | ✓ |
| `NotFound` | authorized absence of an object, entry, snapshot, relation, or commit | ✓ | ✓ |
| `Ambiguous` | a singular API matched more than one object | ✓ | ✓ |
| `Unauthorized` | the principal may not read/traverse; nil-returning for absent and denied where nondisclosure is required | ✓ | ✓ |
| `NamespaceMismatch` | a selector named a namespace other than `default` | ✓ | ✓ |
| `ReadOnly` | a mutation on a read-only handle | ✓ | ✓ |
| `Exists` | the name or store already exists | ✓ | ✓ |
| `NotDir` | a path component was expected to be a directory | ✓ | ✓ |
| `IsDir` | an operation is invalid for a directory | ✓ | ✓ |
| `NotEmpty` | the directory is not empty | ✓ | ✓ |
| `InvalidName` | empty, over-long, NUL, `/`, `.`, `..` | ✓ | ✓ |
| `InvalidArgument` | a non-name argument is invalid | ✓ | ✓ |
| `UnsupportedRepresentation` | the operation is not expressible for this physical representation (e.g. shrink over an immutable overlay) | ✓ | |
| `UnmeasuredCost` | the planner was asked to price an unmeasured class; it refuses rather than substituting a neighbour | ✓ | |
| `BudgetExceeded` | a configured write budget would be exceeded | ✓ | ✓ |
| `Conflict` | a union merge found one path with different content, or a transaction precondition failed | ✓ | ✓ |
| `CrossDomain` | an atomicity across MW3 conflict domains that the protocol does not provide | ✓ | |
| `Cancelled` | the caller cancelled while waiting for shared work | ✓ | |
| `SnapshotMissing` | a pinned snapshot named by a Context Root or artifact request does not exist (distinct from an authorized absence inside a snapshot) | ✓ | |
| `CapabilityRevoked` | a capability, or its revocation id, is revoked at the observed revocation epoch; the message is fixed and non-disclosing | ✓ | |
| `StaleEpoch` | the client's revocation observation is older than the store's epoch or its freshness window; refresh, never fall back to the head | ✓ | |

Notes:

* `NotFound` and `Unauthorized` are deliberately interchangeable at some
  boundaries: a revocation refusal must be indistinguishable from absence
  (`docs/REVOCATION-EPOCH.md`). The class that *is* returned is still stable for
  the caller that is entitled to know.
* `Corrupt` is never used for a transient transport failure; `Io` is. This was
  a real defect fixed in R13 and is enforced by the distributed court.
* `verify` reports `VerifyProblem.class` strings from the same registry, one
  per finding, rather than failing.

## 3. RFC §13 reconciliation

| RFC §13 string | status | mapping | why |
|---|---|---|---|
| `UnsupportedProfile` | **proposed; mapped onto shipped classes** | `UnsupportedVersion` for a version/feature/backend-profile gate; `UnsupportedRepresentation` for a representation gate | "Profile" is not a first-class object in the shipped store. Introducing a string would split one refusal into two indistinguishable ones for clients. No code returns it. |
| `Unproven` | **proposed; not mapped (genuinely new)** | nearest shipped: `DependencyMismatch` (a declared dependency cannot be verified) or `Unauthorized` (a capability proof is absent). Not used. | "Cannot prove" is not "mismatch". Mapping it onto `Corrupt`/`DependencyMismatch` would lose the distinction RFC §13 requires between an unavailable dependency, a missing capability proof, and a mismatched numerical contract. Keep it proposed until a capability-proof surface exists. |
| `Incompatible` | **proposed; mapped onto shipped classes** | `UnsupportedVersion` when versions/profiles disagree; `DependencyMismatch` when declared inputs (dtype, shape, root) disagree with what resolved | The shipped store already distinguishes these two; a third string would be a synonym. |
| `SnapshotMissing` | **shipped** | `Class::SnapshotMissing` (Rust); Go reports the string only in `verify` problems | Implemented with the artifact API (T072): a missing pinned snapshot is never silently replaced by the head. |
| `CapabilityRevoked` | **shipped** | `Class::CapabilityRevoked` (Rust, T078) | Implemented with revocation epochs; fixed, non-disclosing message. |
| `TransformationRequired` | **proposed; not mapped on purpose** | none. `UnsupportedRepresentation` is the *opposite* statement ("not expressible") and must not be reused. | The store today refuses a shrink/collapse/insert over an immutable overlay with `UnsupportedRepresentation`; that is a refusal, not "a rewrite would make this work". No code returns `TransformationRequired`. |
| `MaterializationIncomplete` | **proposed; partially expressed** | a missing declared dependency → `DependencyMismatch`; a budget interruption → `BudgetExceeded`; an ENOSPC/transport failure → `Io` | There is no partial-result surface to attach the class to: `weft materialize --training-ready` fails closed and publishes nothing. Keep it proposed. |

Also relevant, though not proposed by RFC §13:

* `StaleEpoch` (T078) is the class RFC §13's "stale policy epochs" threat
  needs; it is shipped and distinguishable from `CapabilityRevoked`.
* RFC §13's cursor-binding requirement ("cursors MUST bind to the exact
  snapshot, selector, principal and rule version") is implemented as
  **snapshot-scoped calls** in the artifact API (`docs/ARTIFACT-API.md`), not as
  an opaque cursor type. There is no cursor object to mis-bind, and a request
  that names no snapshot gets a provisional result that never silently follows
  the head.

## 4. Rules

1. **Do not add a class to make a message nicer.** A distinct class is a
   compatibility surface: clients branch on it.
2. **Adding a class is additive** (minor): old clients see an unknown string
   and must treat it as an error, never as success. Removing or renaming a
   class is major.
3. **A conformance vector's recorded class never changes.** A correction adds a
   vector and marks the old one with `superseded_by`.
4. **Nondisclosure wins over precision** where a denial would be an oracle. The
   class may be the same for two different internal causes; that is the point.
5. **Proposed classes are documentation, not behaviour.** Nothing in this
   repository returns a string from §3 marked *proposed*. A future
   implementation that returns one must freeze it here, in the codec and store
   registries, and in a conformance vector in the same change.
