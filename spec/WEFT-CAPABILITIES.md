# WEFT-CAPABILITIES v1.0.0 — capability contexts, budgets, and enforcement

**Status:** Normative design for `weft-capabilities 1.0`. The record type and
hash are implemented in F1 (`weftcodec::context` / `WeftCodec-Go/context`);
resolver/store **enforcement** is specified here and arrives in F2. Claims are
labelled accordingly.
**Depends on:** `WEFT-CONTEXT-ROOT.md`, `WEFT-CAUSAL.md`, `WEFT-MODEL.md` §6.
**Source of truth:** `spec/WEFT-CAPABILITIES.md` in `infinityabundance/WeftFS`.

A capability context is the explicit, non-ambient authority under which a
Context Root is resolved and materialized. It answers: *who* may see *which
namespaces, roots, and relation predicates*, *within which budget*, *until
when*, and *what happens when it is revoked*. It does not replace store-local
POSIX/ACL checks; it narrows them.

**v1 does not claim cryptographic unforgeability.** A capability hash is a
content identifier, not a signature or MAC. It becomes a bearer credential only
when the store keeps it out of unauthorized hands. This is stated plainly here
and in `security/THREAT_MODEL.md`.

---

## 1. Capability context record `weft.capability.v1`

```
{
  "schema":      "weft.capability.v1",
  "principal":   text,                  # non-empty, 1..=256 bytes
  "namespaces":  [ text, ... ],         # sorted, unique; may be empty
  "roots":       [ bytes(32), ... ],    # sorted, unique, nonzero; may be empty
  "predicates":  [ text, ... ],         # optional; sorted, unique
  "budget":      { ... },               # optional; same schema as Context Root §4
  "expires_at":  timestamp              # optional; UTC ns since Unix epoch
}
```

Rules:

- `namespaces` and `roots` MUST NOT both be empty. A capability that grants
  neither grants nothing and is `NonCanonical`; it is not an implicit "all".
- An empty `namespaces` list means "no namespace grant". It never means "all
  namespaces".
- An empty `roots` list means "no root grant".
- `predicates` is an allowlist of **relation predicate names** that may be
  traversed. If the field is absent, relation traversal is **not** granted. If
  it is present and empty, relation traversal is also not granted. There is no
  wildcard in v1.
- `budget` uses the identical schema and validation as
  `WEFT-CONTEXT-ROOT.md` §4. Where both a Context Root budget and a capability
  budget exist, the **effective** bound is the component-wise minimum.
- `expires_at` is inclusive: a capability is valid at exactly `expires_at` and
  invalid one nanosecond later. Clock skew is an operational concern; v1 uses
  the store's clock for enforcement and records it.
- No unknown keys, no implicit defaults for the grant lists.
- Array lengths are bounded by `max_capability_entries` (default 4,096).

### 1.1 Identity

```
capability_bytes = WCB(capability record)
CapabilityHash   = H("weft/capability/v1", capability_bytes)
```

`ContextRoot.capability_hash` binds a root to a capability scope. The binding is
inclusion in the root identity, not a signature: a resolver MUST look up the
capability record by hash (or receive it from the caller) and validate that its
hash matches; it MUST NOT treat a bare hash as proof of possession.

Two capabilities with the same hash denote the same grant. Any change
(principal, list, budget, expiry) changes the hash. Revocation is keyed by hash.

---

## 2. No ambient authority

In capability mode:

1. Every operation receives an explicit capability context. There is no
   "default allow" and no implicit union with the process uid/gid.
2. The absence of a capability is denial, not fallback to POSIX mode.
3. The capability can only **narrow** what the store already authorizes. It
   never grants more than the underlying principal's POSIX/ACL rights.
4. A capability does not authorize a namespace or root that contains objects
   the principal cannot read. Both checks apply; the stricter result wins.
5. Selector predicates that reference ungranted namespaces, roots, or relation
   predicates fail with `Unauthorized`/`NamespaceDenied`/`RootDenied`/
   `PredicateDenied`. They do not silently match nothing in a way that could be
   mistaken for a complete empty result — unless non-disclosure explicitly
   requires indistinguishability, in which case the response is the
   non-disclosing form with an explicit flag.

The store's own authorization (`WEFT-MODEL.md` §6) remains in force. Capability
contexts are layered above it; they are not a replacement for kernel/FUSE
access control.

---

## 3. Revocation

- Revocation is recorded by `CapabilityHash`, with a revocation timestamp and
  the store generation that observed it.
- Revocation is monotonic in v1: once revoked, a hash stays revoked. There is
  no un-revoke operation. A new grant has a new hash.
- A revoked capability fails at every enforcement point with
  `CapabilityRevoked`, even if its lists, budget, and expiry are still valid.
- Revocation takes effect at the enforcement point's pinned snapshot. A
  resolution already in progress MAY finish under the snapshot it pinned, but it
  MUST report the revocation boundary (the generation at which revocation was
  observed) if it completes after revocation was published. A materialization
  that begins after the revoking commit MUST fail.
- Revocation records are themselves immutable and GC-rooted while referenced by
  any retained audit record.
- v1 makes **no distributed revocation claim**. Revocation is store-local and
  consistent within a single writer's commit order. Cross-store revocation is a
  non-goal.

---

## 4. Enforcement points

Enforcement is mandatory at each of these boundaries. A missing check is a
vulnerability, not an optimization.

| # | Point | Required checks |
|---|---|---|
| 1 | **Context Root construction** | If `capability_hash` is present, the resolver must be able to resolve the capability record; hash mismatch is `NonCanonical`/`Corrupt`. The record's namespace/roots/predicates are not trusted until validated. |
| 2 | **Selector resolution** | Namespace grant, root grant, predicate grant, budget; authorized universe computed before evaluation, counting, and evidence. |
| 3 | **Relation traversal** | Every traversed edge's predicate must be granted; both endpoints must be visible; closure queries filter before counting (`WEFT-CAUSAL.md` §4). |
| 4 | **Materialization** | Each object/version/content must be within the granted root's resolved set and readable by the principal; budget `max_bytes`/`max_depth` enforced; hash verification is mandatory and independent of capability checks. |
| 5 | **Exact-id access** | `ContentId`/`ObjectId`/`VersionId`/`ContextRootId` access is checked against grants. Possession of an ID is not authority. A `ContentId` reached only through an ungranted root is denied. |
| 6 | **FUSE views** | The mount's capability context is applied to `lookup`, `readdir`, `open`, `read`, `readlink`, `getxattr`, `statfs` counts, and metadata. No file handle outlives revocation silently: operations after the revocation boundary fail with `CapabilityRevoked` (already-open handles may be torn down; the contract is stated in `docs/fuse-status.md`). |
| 7 | **Closure/evidence queries** | Filter before counting; no unauthorized names, counts, sizes, or existence hints. |

### 4.1 Budgets

Budgets are ceilings, not targets. Enforcement points decrement a work/budget
account and fail with `BudgetExceeded` **before** exceeding a bound. A truncated
result MUST carry `truncated = true` plus the bound that stopped it. A truncated
result MUST NOT be presented as a complete set, and a singular accessor MUST NOT
return one item from a truncated ambiguous set.

`max_work` is implementation-defined but MUST be deterministic for a given
resolver version. Implementations MUST document their work unit; v1 uses
"authorized objects inspected plus relation edges traversed plus bytes hashed".

---

## 5. Error taxonomy

| Class | Cause |
|---|---|
| `Unauthorized` | generic denial; non-disclosing when required |
| `NamespaceDenied` | namespace not in the capability grant |
| `RootDenied` | Context Root / object root not in the grant |
| `PredicateDenied` | relation predicate not in the allowlist |
| `CapabilityExpired` | `expires_at` passed |
| `CapabilityRevoked` | hash revoked |
| `CapabilityMissing` | needed capability record not found |
| `BudgetExceeded` | a stated budget would be exceeded |
| `Ambiguous` | singular access on a set with authorized count ≠ 1 |
| `NotFound` | authorized absence (must be indistinguishable from denial where non-disclosure applies) |
| `SnapshotMissing` / `Corrupt` | pinned snapshot unavailable or failing integrity |
| `MaterializationMismatch` | content hash mismatch |

The class is the interoperable contract; messages are human text. When
non-disclosure applies, `NamespaceDenied`, `RootDenied`, `PredicateDenied`, and
`NotFound` MUST be collapsed into the same externally visible response; the
store MAY log the precise class privately.

---

## 6. Non-disclosure requirements

1. Result counts, evidence arrays, closure sizes, error classes (where
   non-disclosure applies), and truncation flags MUST be computed over the
   authorized set only.
2. Denial MUST NOT reveal the existence, size, type, mtime, or hash of an
   unauthorized object.
3. A capability hash MUST NOT be usable to enumerate namespaces, roots, or
   predicates. Probing a hash returns at most "valid/revoked/expired" if the
   caller is already authorized to learn that.
4. Deduplication MUST NOT be used to infer content across capability contexts.
   This is the same limitation stated in `WEFT-MODEL.md` §6 and is restated
   because capability contexts make it more visible.
5. Cache keys and logs MUST include the capability hash when the cached value
   depends on the authorized view. A cache hit from another capability context
   is a disclosure bug.
6. v1 makes no timing side-channel claim. Constant-time authorization is not
   claimed; the threat model states this.

---

## 7. What is not claimed in v1

- **Not cryptographic authorization.** No signatures, MACs, key derivation, or
  unforgeable tokens. Hashes are identifiers.
- **Not distributed.** No cross-store capability transfer, no federated
  revocation, no global ordering.
- **Not delegation.** A principal cannot mint a narrower capability for another
  principal under a verifiable chain in v1. That would require a signature
  scheme and is out of scope.
- **Not per-object grants.** Grants are namespace/root/predicate scoped. A root
  grant covers the authorized results of that root; it is not a per-inode ACL.
- **Not an OS sandbox.** FUSE/kernel permissions still apply. Capabilities add
  restrictions; they do not replace the kernel's.
- **Not constant-time.** See §6.6.
- **Not implemented end-to-end yet.** F1 implements the record, strict
  canonicalization, identity, and CLI. Enforcement points 2–7 are **specified,
  not implemented** until F2/F3. No test claims otherwise.

---

## 8. Compatibility

- `weft-capabilities 1.0` freezes the schema string, field set, grant-list
  semantics, budget schema, and hash domain.
- New optional fields require a minor version and a default old readers
  implement; otherwise a major version.
- New error classes may be added; existing classes never change meaning.
- Capability records are append-only vectors under `conformance/v1/` when
  implemented; absent vectors mean "not yet implemented", not "optional".
