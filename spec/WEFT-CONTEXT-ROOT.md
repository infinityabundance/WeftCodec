# WEFT-CONTEXT-ROOT v1.0.0 — the primary immutable public address

**Status:** Normative for `weft-context-root 1.0`. Part of the `weft` v1
identity family. Implemented in Rust `weftcodec::context` and Go
`WeftCodec-Go/context` as of F1.
**Depends on:** `WEFTCODEC-FORMAT.md` §1 (hash domains) and §8 (WCB),
`WEFT-SELECTORS.md`, `WEFT-CAUSAL.md`, `WEFT-CAPABILITIES.md`.
**Source of truth:** `spec/WEFT-CONTEXT-ROOT.md` in `infinityabundance/WeftFS`.

A **Context Root** is the canonical, immutable public address of a *selection*
of store state. It binds a selector expression to a pinned snapshot, a resolver
(policy) version, an ordered causal closure, and optionally a capability scope
and a resource budget. It is not a tag, a path, a live query, or a result set.

The primary public identity in WeftFS is the **Context Root**, not the path and
not the content hash. Paths are a mutable compatibility projection. Content
hashes identify bytes; Context Roots identify *which state is selected, under
which rules, at which snapshot*.

---

## 1. Terms

| Term | Meaning |
|---|---|
| Context Root | The typed record defined in §2 |
| `ContextRootId` | `H("weft/context-root/v1", WCB(context-root record))` |
| authoritative URI | `weft://<namespace>/c/<ContextRootId-hex>` |
| descriptor URI | a human-writable `weft://` form that canonicalizes to the authoritative form (§5) |
| pinned snapshot | the commit root hash named by the record; never "latest" |
| resolver version | the text name of the deterministic evaluation policy |
| causal closure | ordered, unique ancestor set (see `WEFT-CAUSAL.md`) |
| capability hash | optional hash binding the authorized view (see `WEFT-CAPABILITIES.md`) |
| budget | optional resource bound applied before evaluation |

---

## 2. Record schema `weft.context-root.v1`

The canonical value is a WCB map with exactly these keys. The **logical** field
order below is documentation only; the wire order is always WCB encoded-key
order (`WEFTCODEC-FORMAT.md` §8.2). Unknown keys are rejected.

| Key | Type | Required | Rule |
|---|---|---|---|
| `schema` | text | yes | exactly `weft.context-root.v1` |
| `namespace` | text | yes | §3.1 |
| `selector` | bytes | yes | canonical selector WCB bytes (`weft.selector.v1`) |
| `selector_id` | bytes(32) | yes | `H("weft/selector/v1", selector)` |
| `snapshot` | bytes(32) | yes | nonzero pinned commit root hash |
| `resolver` | text | yes | §3.2 |
| `causal` | array of bytes(32) | yes | strictly ascending, unique, nonzero; may be empty |
| `ordering` | text | yes | exactly `selector_then_object_id` in v1 |
| `capability_hash` | bytes(32) | no | nonzero; omitted means "no capability binding" |
| `budget` | map | no | §4; omitted means "unbounded within store policy" |

There is no `created_at`, no owner, no mutable name, and no "latest" marker. A
record that needs mutable metadata is a different (versioned) schema.

### 2.1 Selector binding

The `selector` field carries the **canonical selector bytes**, and
`selector_id` carries their hash. Decoders MUST:

1. decode `selector` as canonical `weft.selector.v1` using the selector rules;
2. require that re-encoding the decoded AST produces exactly the same bytes;
3. require `selector_id == H("weft/selector/v1", selector)`;
4. require the selector's namespace equals the record `namespace`.

Violating any rule is `NonCanonical`. Carrying both fields is deliberate: the
record is self-describing for resolution and evidence, while `selector_id`
allows indexing without decoding. They are not independent identities; the
second MUST equal the hash of the first.

### 2.2 Why the snapshot is mandatory

A selector alone is a live query: its results change when the store commits. A
Context Root names an *immutable* view. The snapshot is the commit root hash;
readers MUST validate that the named commit exists and is complete. They MUST
NOT substitute the current head, a tag, a path, or an "equivalent" snapshot. A
missing or different snapshot is a typed error (§8), never a fallback.

### 2.3 Why the resolver version is mandatory

Evaluation rules are code. Two resolver versions may order, fail, or filter
differently. The text `resolver` field makes the policy explicit and part of the
identity. It does not replace the store format version; it names the selection
policy. Unknown resolver names are not silently interpreted by a default.

---

## 3. Text fields, limits, and strict validation

### 3.1 Namespace

Namespaces are ASCII and URI-safe. The grammar is:

```
namespace := first rest*
first     := ALPHA / DIGIT / "_"
rest      := ALPHA / DIGIT / "_" / "." / "-"
```

Length 1..=128 bytes. The names `.` and `..` are rejected. No percent-encoding
appears in the namespace segment of a URI because every allowed byte is an
unreserved URI character. A namespace that violates this grammar is
`NonCanonical` in a record and `Parse` in a URI.

### 3.2 Resolver version

```
resolver := ALPHA (ALPHA / DIGIT / "." / "-" / "_")*
```

Length 1..=64 bytes. The v1 default and the only name frozen by this document is
`weft-resolver-v1`. Implementations MAY know additional names, but a name they
do not know MUST be rejected at resolution time (`UnsupportedVersion`); they
MUST NOT evaluate it under v1 semantics.

### 3.3 Limits

| Limit | Default | Applies to |
|---|---:|---|
| `max_causal_entries` | 4,096 | `causal` array, closure arrays |
| `max_capability_entries` | 4,096 | capability lists |
| `max_uri_bytes` | 65,536 | any `weft://` URI |
| `max_string_bytes` | 256 MiB | namespace/resolver/selector text |
| `max_structured_depth` | 64 | WCB nesting |
| `max_structured_items` | 1,000,000 | WCB items |

Exceeding a limit is `LimitExceeded` before allocation or evaluation.

### 3.4 Required validation checklist

A decoder MUST reject with the listed class:

| Condition | Class |
|---|---|
| not canonical WCB, unknown key, wrong type, trailing bytes | `NonCanonical`/`Malformed` per WCB |
| `schema` not `weft.context-root.v1` | `UnknownSchema` |
| invalid namespace or resolver text | `NonCanonical` |
| `snapshot` all zero | `NonCanonical` |
| `causal` not strictly ascending, duplicate, or zero entry | `NonCanonical` |
| `selector` not canonical selector bytes | `NonCanonical`/`UnknownSchema` |
| `selector_id` mismatch | `NonCanonical` |
| selector namespace ≠ record namespace | `NonCanonical` |
| `ordering` not `selector_then_object_id` | `UnsupportedVersion` |
| `capability_hash` zero | `NonCanonical` |
| `budget` empty, unknown key, zero value, wrong type | `NonCanonical` |
| any count/length over a configured limit | `LimitExceeded` |

Decoders MUST NOT repair. There is no "normalize and accept" path for canonical
record bytes; normalization exists only at the descriptor-URI boundary (§5).

---

## 4. Budget

`budget` is a WCB map, omitted when absent. Keys are:

| Key | Type | Rule |
|---|---|---|
| `max_results` | u64 | ≥ 1; maximum authorized results returned |
| `max_bytes` | u64 | ≥ 1; maximum materialized plaintext bytes |
| `max_depth` | u64 | ≥ 1; maximum relation/dependency traversal depth |
| `max_work` | u64 | ≥ 1; maximum implementation-defined work units |

At least one key MUST be present. Each key appears at most once (WCB enforces
map-key uniqueness). Unknown keys are `NonCanonical`. A budget is a *ceiling*:
enforcement may stop earlier; it MUST NOT exceed any stated bound and MUST NOT
silently drop results to meet a bound. Exceeding a bound is
`BudgetExceeded` (see `WEFT-CAPABILITIES.md`), never a truncated success that
claims completeness.

Budgets participate in `ContextRootId`. Raising a budget changes the address
even though it does not change the selected set; this is intentional. The
budget describes the authorized evaluation, not just a client preference. A
caller that wants an unbounded view uses a record without `budget`.

---

## 5. `weft://` grammar

### 5.1 Authoritative form (canonical)

```
weft-uri   := "weft://" namespace "/c/" hex64
hex64      := 64 lowercase hexadecimal digits
```

This is the only canonical spelling. It carries the `ContextRootId`; it does not
repeat the selector, snapshot, or policy. It is the address used in storage,
logs, APIs, and cross-references.

Rules:

- scheme is exactly lowercase `weft`;
- no userinfo, port, query, fragment, empty path segment, or trailing slash;
- `namespace` follows §3.1 and is case-sensitive;
- the 64 hex digits are lowercase; uppercase is `Parse`;
- the all-zero value is `Parse` (it is never a valid hash);
- the ID MUST be resolvable by the local store/index before resolution; an
  unknown ID is `NotFound`, not "create something equivalent".

### 5.2 Descriptor form (human-writable, canonicalizes to §5.1)

```
weft://<namespace>/c/<pct-selector>@<snapshot-hex>?<query>
query     := param ("&" param)*
param     := "closure=" hex64 ("," hex64)*
           | "cap=" hex64
           | "budget=" budget-entry ("," budget-entry)*
           | "resolver=" resolver
budget-entry := ("max_results" / "max_bytes" / "max_depth" / "max_work") "=" u64
```

- `<pct-selector>` is the UTF-8 selector source text, percent-encoded per §5.3.
- The `@` separating selector text from the snapshot MUST be a raw `@`. Because
  a raw `@` cannot occur inside the percent-encoded selector, the **first** raw
  `@` in the path after `/c/` is the separator.
- `<snapshot-hex>` is exactly 64 lowercase hex digits and MUST be nonzero.
- Query parameters are optional; when present they must be non-empty. Parameter
  order is free when parsing; duplicate parameter names are `Parse`.
- `resolver` defaults to `weft-resolver-v1` when absent.
- `closure` entries are joined with `,`; they are normalized by sorting and
  removing duplicates before the record is built (the record itself is strict).
- `budget` entries use the §4 names and canonical decimal (`0` or
  `[1-9][0-9]*`); zero and duplicate names are `Parse`.
- No other query parameters exist in v1; unknown parameters are `Parse`.
- A fragment (`#`) is never permitted.
- Percent-encoding is not applied inside query values in v1; all their bytes are
  from the unreserved set or are `,`/`=` delimiters.

Canonical descriptor emission orders query parameters as
`closure`, `cap`, `budget`, `resolver`, omitting absent ones, and always emits
`resolver` when it differs from `weft-resolver-v1`.

### 5.3 Percent-encoding

For the selector text only:

- **Encode** each UTF-8 byte: emit it raw if it is an unreserved character
  (`ALPHA / DIGIT / "-" / "." / "_" / "~"`); otherwise emit `%` followed by two
  **uppercase** hex digits.
- **Decode** strictly: `%` MUST be followed by two uppercase hex digits; any
  other raw byte outside the unreserved set is `Parse`; an escape whose decoded
  byte is unreserved (for example `%41` for `A`) is non-canonical and is
  `Parse`; the decoded byte sequence MUST be valid UTF-8.
- This subset is stricter than RFC 3986 `pchar`. Being stricter is deliberate:
  every selector text has exactly one descriptor spelling, so descriptor URIs
  can be compared, signed, logged, and diffed.

### 5.4 Canonicalization

Given a descriptor URI, the canonical address is:

1. parse the namespace, selector text, snapshot, resolver, closure, capability
   hash, and budget under the rules above;
2. parse the selector text with the namespace (selector grammar, limits);
3. build the record (§2), including canonical selector bytes, `selector_id`,
   sorted/unique causal entries, and validated budget;
4. compute `ContextRootId`;
5. emit `weft://<namespace>/c/<ContextRootId>`.

Two descriptors that denote the same selection under the same snapshot and
policy produce the **same** authoritative URI. A descriptor is not itself an
identity; it is a spelling that reduces to one.

### 5.5 Why the canonical URI is not the selector text

- Selector text is not unique: whitespace, quoting, and equivalent spellings
  differ while the canonical AST is identical.
- Selector text does not bind a snapshot, resolver version, closure, capability,
  or budget; the same text can denote unboundedly many views.
- Selector text is unbounded in length, may contain arbitrary UTF-8, and is
  unsafe as a filename, log key, or cache key.
- Hashing the selector text would hash a *spelling*, not the expression. The
  selector's own identity is already `SelectorId`, which is intentionally
  snapshot-free (`WEFT-SELECTORS.md` §4).
- The authoritative URI must be fixed-size, canonical, and comparable. The
  descriptor exists for humans and tools; it is never stored as authority.

---

## 6. Resolution semantics

`Resolve(context_root, principal_context) -> ResultSet` is defined normatively.
The implementation arrives in F2/F3; this document defines the contract.

1. **Verify identity.** Recompute `ContextRootId` from the record; if it differs
   from the requested address, `Corrupt`. Do not resolve under a different ID.
2. **Verify the snapshot.** The named commit MUST exist, verify, and be
   complete. Missing: `SnapshotMissing`. Present but failing integrity:
   `Corrupt`. Never fall back to another generation, tag, or live head.
3. **Verify the resolver.** Evaluate only under the named resolver version.
   Unknown version: `UnsupportedVersion`.
4. **Authorize first.** Compute the authorized universe for the capability
   context (or the store principal when no capability is bound) *before*
   evaluation and before counting. Unauthorized objects are absent, not errors,
   where non-disclosure applies.
5. **Evaluate.** Apply the pinned selector to the pinned snapshot. Returns
   0, 1, or n results. Every API MUST expose `matched_count` (authorized only)
   and MUST NOT collapse cardinality to "the first match".
6. **Order.** Apply the selector's `order` list; break ties by `object_id`
   ascending. The v1 `ordering` policy is `selector_then_object_id`; it is fixed
   and part of the identity. Ordering is unsigned byte-wise over UTF-8 for text
   fields, then `object_id` bytes.
7. **Limit/offset.** Apply the selector's `offset`, then `limit`, after ordering
   and after authorization.
8. **Budget.** Apply the record budget as a ceiling over the authorized result
   set, materialization bytes, traversal depth, and work. Exceeding it is
   `BudgetExceeded`; partial results are returned only with an explicit,
   machine-readable truncation flag and never as a complete set.
9. **Evidence.** Per-result evidence names only authorized attributes/edges.
   Evidence, counts, and error detail MUST NOT leak unauthorized existence.
10. **Materialization.** Each result's `content_id` (and version) is verified
    exactly at materialization: every chunk hash, manifest hash, and whole
    content hash. Mismatch fails closed (`MaterializationMismatch`/`Corrupt`).
    There is no approximation, no "best effort", and no silent re-resolution at
    a newer snapshot.
11. **Pinned vs live.** An authoritative Context Root always denotes the pinned
    snapshot, even after the store advances. A live query is a different
    construct: it has no `ContextRootId` until a snapshot is chosen. Callers
    MUST NOT present a live result as the content of a pinned Context Root.
12. **No "first match" semantics.** A dependency edge is created only between a
    `ContextRootId` and a concrete result set (or a singleton only when the
    authorized count is exactly one). A Context Root never implies "the first
    thing that matched".

### 6.1 Cardinality

| Authorized matches | Result |
|---:|---|
| 0 | empty set, `matched_count = 0` |
| 1 | singleton set; singular accessors may succeed |
| n > 1 | full ordered set; singular accessors MUST fail with `Ambiguous` |

Selectors that were written with `limit 1` are still n-capable: their result is
one *after* ordering, and `matched_count` before the limit MUST be exposed when
the store can compute it. An implementation that cannot compute the pre-limit
count MUST say so; it MUST NOT claim the limit is the cardinality.

---

## 7. Versioning and compatibility

`weft-context-root` has its own version line, independent of the frame format
and the store format. v1 freezes:

- the schema string `weft.context-root.v1`;
- the field set and their semantics in §2;
- the `selector_then_object_id` ordering policy;
- budget key names;
- the authoritative URI shape;
- the descriptor grammar as the *only* non-authoritative spelling.

Additive changes (new optional field with a safe default, new ordering policy
name that old readers reject with `UnsupportedVersion`, new budget key that old
readers reject) require a minor version and new vectors. Any change that alters
the canonical bytes for an existing record requires a major version and a new
domain string. Existing vectors are append-only.

### 7.1 Selection identity is separate from content identity

- `ContentId`, `ManifestId`, `ObjectId`, `VersionId` identify bytes or objects.
- `SelectorId` identifies an expression, without a snapshot.
- `ContextRootId` identifies *an expression pinned to a snapshot under a
  policy*, optionally narrowed by a capability and a budget.
- A `ContextRootId` does not imply that any object exists. It can resolve to an
  empty set.
- Two different Context Roots may select byte-identical objects. That does not
  make them the same root, and dedup MUST NOT merge them.
- A `ContextRootId` is not a capability, a signature, or an authorization
  decision. See `WEFT-CAPABILITIES.md`.

---

## 8. Error classes

| Class | Cause |
|---|---|
| `Parse` | URI syntax, percent-encoding, bad query, bad hex case/length |
| `NonCanonical` | record invariants violated (types, sortedness, binding) |
| `Malformed` | truncated/non-WCB bytes |
| `UnknownSchema` | schema string mismatch |
| `UnsupportedVersion` | unknown ordering policy or resolver version |
| `LimitExceeded` | configured bound exceeded |
| `NotFound` | authoritative ID unknown to the resolver |
| `SnapshotMissing` | pinned snapshot absent (F2/F3) |
| `Corrupt` | snapshot/identity/materialization integrity failure |
| `Unauthorized` / `Capability*` | capability enforcement (`WEFT-CAPABILITIES.md`) |
| `BudgetExceeded` | a stated budget would be exceeded |
| `Ambiguous` | singular access with authorized count ≠ 1 |
| `MaterializationMismatch` | content hash mismatch during materialization |

Error strings are human text; the class is the interoperable contract.

---

## 9. Worked example (normative shape)

Selector source:

```
select attrs.project == "demo" limit 10
```

Namespace `default`, snapshot `11...11` (64 hex), resolver `weft-resolver-v1`,
causal entries empty, budget `max_results=10`. The record is a WCB map; the
authoritative URI is:

```
weft://default/c/<ContextRootId>
```

A descriptor spelling is (the `=` inside the query value is a query delimiter,
not part of the path, so it is written raw):

```
weft://default/c/select%20attrs.project%20%3D%3D%20%22demo%22%20limit%2010@1111111111111111111111111111111111111111111111111111111111111111?budget=max_results=10
```

Concrete golden bytes and IDs for this and other cases live in
`conformance/v1/context/`. They are authoritative over this illustrative prose.

---

## 10. Non-goals in v1

- No mutable Context Root names. A name may point at a root, but the name is not
  the root and re-pointing it is a separate, auditable operation.
- No implicit "latest" or "current" resolution.
- No selector-text address, no semantic equivalence of selectors, no query
  optimizer that changes operand order.
- No cryptographic authentication in the URI. See `WEFT-CAPABILITIES.md` for
  what is explicitly not claimed.
- No distributed namespace resolution. v1 resolution is store-local.
