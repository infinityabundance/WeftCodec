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
"authorized objects inspected plus visible relation edges traversed plus bytes
hashed".

Work accounting MUST be computed over the **authorized universe only**:
objects the principal may read, and relation edges whose subject and (for
object targets) target the principal may read. Unauthorized objects and edges
MUST NOT contribute to the count, whether or not the selector mentions
relations. Therefore, two stores with identical authorized views and different
unauthorized contents MUST report the same work units and the same budget
outcome for the same selector and principal.

A budget denial MUST be non-disclosing: it MUST NOT include the computed work
count, the configured bound, or any object/relation identifier. v1 emits the
fixed messages `resolver work budget exceeded` and `traversal depth exceeds
max_depth`. The precise value MAY be logged privately. Authorization (namespace,
root, predicate, and namespace-root readability) MUST be decided before work
accounting so that a budget failure cannot distinguish an unauthorized store
from an empty one.

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
   unauthorized object, nor the number of unauthorized objects or relations
   (directly, through work accounting, or through budget failures).
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

- **The bare `weft.capability.v1` hash is not cryptographic authorization.**
  It is an identifier. Authenticated grants are an *additive* layer in §9:
  an Ed25519-signed `weft.capability.envelope.v1`. A bare hash is never
  accepted as a credential.
- **Not distributed.** No cross-store capability transfer, no federated
  revocation, no global ordering. §9 revocation sets are caller-supplied.
- **Not delegation.** §9 defines a signed grant but no delegation chain or
  certificate graph. A grantor signs grants directly; transitive authority is
  out of scope.
- **Not per-object grants.** Grants are namespace/root/predicate scoped. A root
  grant covers the authorized results of that root; it is not a per-inode ACL.
- **Not an OS sandbox.** FUSE/kernel permissions still apply. Capabilities add
  restrictions; they do not replace the kernel's.
- **Not constant-time.** See §6.6.
- **Not per-object visibility in the kernel.** FUSE applies the mount's
  capability context and `default_permissions`; v1 does not claim a
  per-inode capability protocol.
- **Not deployment-tested at scale.** Enforcement points 2–7 are implemented
  and exercised by `tests/context.rs`, `tests/capability_nondisclosure.rs`,
  `tests/capability_envelope.rs`, `tests/capability_isolation.rs`, and the FUSE
  context-mount court. The adversarial two-store non-disclosure court covers
  authorized-universe work accounting; it is not a proof against timing
  channels (§6.6).

---

## 8. Compatibility

- `weft-capabilities 1.0` freezes the schema string, field set, grant-list
  semantics, budget schema, and hash domain.
- New optional fields require a minor version and a default old readers
  implement; otherwise a major version.
- New error classes may be added; existing classes never change meaning.
- Capability records are append-only vectors under `conformance/v1/` when
  implemented; absent vectors mean "not yet implemented", not "optional".

---

## 9. Authenticated grants (`weft.capability.envelope.v1`)

This section is **additive**. It does not change §1–§8 and does not retroactively
make a `CapabilityHash` a credential. A capability envelope is a signed public
record; only the grantor's private key is secret.

### 9.1 Canonical envelope

A canonical WCB map with exactly these keys (every key is always present):

```
{
  "schema":        "weft.capability.envelope.v1",
  "grantor":       bytes(32),          # Ed25519 public key, nonzero
  "grantee":       bytes(32),          # public key or stable principal hash, nonzero
  "tenant":        text,               # 1..=256 bytes
  "roots":         [ bytes(32), ... ], # sorted, unique, nonzero; may be empty
  "namespaces":    [ text, ... ],      # sorted, unique; may be empty
  "predicates":    [ text, ... ],      # sorted, unique; may be empty
  "expires_at_ns": timestamp | null,   # inclusive expiry, UTC ns
  "budget":        map | null,         # WEFT-CONTEXT-ROOT §4 fields
  "revocation_id": bytes(32),          # nonzero
  "issued_at_ns":  timestamp,          # UTC ns
  "nonce":         bytes(32)           # nonzero; distinguishes re-issues
}
```

Rules:

- No unknown keys, no missing keys. Decoding rejects both.
- The grant lists reuse §1 validation: sorted, unique, bounded by
  `max_capability_entries`; `namespaces` and `roots` MUST NOT both be empty.
- `budget`, when present, is the exact §1 budget map and is validated the same
  way. `null` means "no additional ceiling from this envelope".
- `expires_at_ns` is inclusive: valid at exactly the timestamp and invalid one
  nanosecond later. If present it MUST NOT precede `issued_at_ns`.
- `grantor`, `grantee`, `revocation_id`, and `nonce` MUST NOT be all-zero.
- Canonical bytes are unique. A decoder MUST re-encode and compare; any
  difference is `NonCanonical`.

Identities:

```
EnvelopeId = H("weft/capability/envelope/v1", WCB(envelope))
sign_input = H("weft/capability/sign/v1",     WCB(envelope))
signature  = Ed25519.sign(grantor_private_key, sign_input)     # 64 bytes
Blob       = WCB({ "envelope": bytes(WCB(envelope)), "signature": bytes(64) })
BlobId     = H("weft/capability/blob/v1",   WCB(Blob))
```

`sign_input` is a 32-byte digest, and Ed25519 signs that digest as its message.
The signature is deterministic (RFC 8032), so the same envelope bytes and key
produce byte-identical blobs in every implementation.

**Secrets never enter the envelope or any content id.** The envelope, its
signature, and both identities are public. Only the grantor's seed/private key
is secret; it is never stored in the envelope, a blob, a relation, or a log.

### 9.2 Offline verification

`verify_envelope(blob, revocation_set, now_ns[, expected_grantee])` performs, in
this normative order:

1. **Canonical form** — decode the outer blob and inner envelope, re-encode, and
   compare bytes. Failures are `Malformed`, `NonCanonical`, or `UnknownSchema`.
2. **Signature** — parse `grantor` as an Ed25519 public key and verify the
   signature over `sign_input`. Failure is `BadSignature`. Implementations MUST
   use a strict verification path (reject small-order/non-canonical keys).
3. **Grantor/grantee binding** — if the caller supplies an expected grantee,
   the envelope's `grantee` MUST equal it. Failure is `WrongGrantee`.
4. **Expiry** — fail with `Expired` when `now_ns > expires_at_ns`.
5. **Revocation** — fail with `Revoked` when `revocation_id` is in the
   caller-supplied set.

`Malformed`, `NonCanonical`, `UnknownSchema`, `BadSignature`, `WrongGrantee`,
`Expired`, and `Revoked` are the interoperable rejection names. Verification
performs **no I/O and no network access**; it needs only the blob, the caller's
revocation set, the clock, and optionally the expected grantee.

### 9.3 Storage attachment

The store writes the canonical blob to the content-addressed manifest namespace
under `BlobId`, then publishes one relation:

```
predicate:  capability_envelope
subject:    grantee (bytes32)
target:     value text = lowercase hex EnvelopeId
attrs:      capability.envelope_blob = lowercase hex BlobId
            capability.envelope_tenant = tenant label (discovery only)
```

Publishing the same envelope twice is idempotent. A stored envelope is retained
by GC exactly while its attachment relation is retained. `verify --full`
checks the blob's content address, canonical form, and signature, and that the
recomputed `EnvelopeId` equals the relation target. It does **not** reject on
expiry or revocation, because those are caller policy at use time.

### 9.4 Revocation semantics

- Revocation is keyed by `revocation_id`, not by `EnvelopeId`. Revocation is
  therefore independent of re-signing the same grant for a different tenant or
  time.
- The revocation set is **explicit and caller-supplied**. There is no network
  lookup, no global revocation service, and no implicit trust of the store.
  Trusted coordination of that set remains external and is the caller's
  responsibility.
- A revoked envelope fails closed at every use: it cannot be revived by reusing
  its hash or BlobId. Only a newly signed envelope with a different
  `revocation_id` (and therefore a different `EnvelopeId`) can grant again.
- Expiry is monotonic for fixed bytes; the same blob is rejected forever after
  `expires_at_ns`.
- Replacing an attachment is a new envelope, never an un-revoke. The store does
  not provide an un-revoke operation.

### 9.5 Enforcement

A verified envelope is attached to the `Principal` (not to a bare hash). At
every enforcement point the envelope grant is applied **in addition to** POSIX
permissions and in addition to any explicit or Context-Root-bound capability:

- every grant that is present must independently allow the namespace, root, and
  relation predicates of the operation;
- budgets take the component-wise minimum across all grants;
- expiry takes the earliest bound;
- the authorized universe is still computed from POSIX/ACL readability first.

An envelope cannot widen `weft.capability.v1`, POSIX bits, or an ACL. Expired,
revoked, wrong-grantee, or non-overlapping grants are denied with the existing
non-disclosing classes (`Unauthorized` at the resolver, `NotFound` at exact-id).
The precise rejection is available to the holder of the envelope through the
offline `capability verify` command; it is not exposed through data queries.

### 9.6 Cross-tenant isolation and deduplication

Tenants share one store. Isolation is the combination of POSIX/ACL visibility
and tenant-scoped envelopes. Consequences:

- Resolution, exact-id access, and enumeration are computed over the
  authorized universe only; another tenant's objects contribute neither
  results nor work units.
- Denial classes and messages for another tenant's objects are identical
  whether those objects exist or not. Work-unit and budget outcomes are
  independent of the other tenant's population.
- The chunk store may deduplicate identical content across tenants. This is
  allowed **only because each tenant independently holds an authorized object
  with that content root**: addressing the shared `ContentId` returns the
  caller's own authorized object. Deduplication MUST NOT be used to enumerate
  names, counts, or existence across tenants. A content root that only another
  tenant holds remains `NotFound`.
- **No timing claim.** v1 does not claim constant-time authorization, cache
  behaviour, or absence of timing side channels across tenants (§6.6). The
  isolation court proves the logical/non-disclosure contract, not a timing
  bound.

### 9.7 CLI

`weftfs capability sign --envelope FILE --key FILE --out FILE [--attach]`
signs a canonical JSON/hex envelope and writes the stored blob form; `--attach`
requires `--store`. `weftfs capability key --key FILE` prints the public key for
a seed. `weftfs capability verify --envelope FILE [--revocations FILE]
[--grantee HEX] [--now-ns N]` verifies offline. `weftfs capability list` and
`capability show ID` read stored attachments. Commands that resolve or read
(`context resolve`, `context materialize`, `cat`) accept `--envelope FILE`
(plus `--revocations FILE`) and enforce the verified grant.

Seed files contain 32 raw bytes or 64 hex digits. Envelope files are JSON or
canonical hex. Revocation files are hex lines, comma-separated hex, or a JSON
array.

### 9.8 Compatibility

`weft.capability.envelope.v1` freezes the schema string, field set, WCB types,
the three domain strings, the signature scheme (Ed25519), the verification
order, and the rejection names. Additive optional fields require a new schema
string; they MUST NOT be inferred by old readers. Historical
`weft.capability.v1` records, hashes, relations, and fixtures are unchanged.
