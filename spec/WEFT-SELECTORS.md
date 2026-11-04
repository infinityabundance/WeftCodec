# WEFT-SELECTORS v1.0.0 — WeftQL and canonical selector identity

**Status:** Normative. Frozen for `weft-selectors 1.0`.
**Depends on:** `WEFTCODEC-FORMAT.md` §8 (WCB) and §9 (SelectorId).

A **selector** is a deterministic query over a namespace snapshot. It is not an
object. It may match zero, one, or many objects, and every API must expose that
cardinality. The canonical bytes of the AST are hashed to produce `SelectorId`;
the result set is never hashed into an address.

---

## 1. Grammar (v1)

```
query      := "select" expr order? limit? offset? EOF
order      := "order" "by" order_item ("," order_item)*
order_item := field ("asc" | "desc")?
limit      := "limit" int
offset     := "offset" int
expr       := or_expr
or_expr    := and_expr ("or" and_expr)*
and_expr   := unary ("and" unary)*
unary      := "not" unary | primary
primary    := "(" expr ")"
            | "true"
            | "false"
            | "has" "(" field ")"
            | "tag" "(" name ")"
            | "relation" "(" name "," object_ref ")"
name       := string | IDENT
            | comparison
comparison := field cmp literal
cmp        := "==" | "!=" | "<=" | ">=" | "<" | ">"
field      := IDENT ("." IDENT)*
object_ref := "obj:" HEX64 | HEX64
literal    := string | int | "true" | "false" | "null"
```

- Keywords are lowercase and reserved: `select`, `and`, `or`, `not`, `true`,
  `false`, `null`, `has`, `tag`, `relation`, `order`, `by`, `asc`, `desc`,
  `limit`, `offset`.
- `IDENT` is ASCII `[A-Za-z_][A-Za-z0-9_]*`; field paths join segments with `.`
  and are limited to 16 segments.
- `HEX64` is exactly 64 ASCII hex digits (`0-9a-fA-F`, normalized to lowercase).
- Integer literals are decimal digits with an optional leading `-` that must
  be immediately followed by digits. Non-negative values are in `0..=2^64-1`
  and encode as WCB `u64`; negative values are in `-1..=-(2^63)` and encode as
  WCB `i64`. A `-` anywhere else is a parse error; `-0` normalizes to `u64` 0.
- Whitespace (space, tab, LF, CR) is insignificant between tokens. Comments are
  not part of v1.
- Input length is bounded (default 64 KiB) and nesting depth is bounded
  (default 64); exceeding either is `LimitExceeded`.

### 1.1 Strings and escapes

String literals are double-quoted UTF-8. Escapes: `\"`, `\\`, `\/`, `\b`,
`\f`, `\n`, `\r`, `\t`, and `\uXXXX` (exactly four hex digits, UTF-16 code unit;
a high surrogate must be followed by a low surrogate escape). A lone surrogate,
invalid UTF-8, raw control characters below `0x20`, and unterminated strings are
parse errors. **No Unicode normalization is performed.** Byte-distinct strings
remain distinct.

---

## 2. AST and canonical WCB bytes

The canonical form is a WCB map:

```
map {
  "schema":    "weft.selector.v1"        # text, required
  "namespace": <text>                    # required, non-empty
  "predicate": <expr>                    # required
  "order":     [ <order_item>, ... ]     # optional; omitted when empty
  "limit":     <u64>                     # optional; omitted when absent
  "offset":    <u64>                     # optional; normalized: 0 is omitted
}
```

`order_item` is the map `{"field": <text>, "dir": <text>}` with `dir` either
`"asc"` or `"desc"`; `asc` is the default and MUST be encoded explicitly in the
canonical form (both written forms normalize to `"asc"`). Duplicate order fields
are rejected.

`expr` is an array whose first element is a text discriminator:

| Canonical array | Meaning |
|---|---|
| `["true"]` | always true within the namespace |
| `["false"]` | always false |
| `["has", field]` | attribute path exists and is non-null |
| `["tag", name]` | object has relation `tagged_as` with text value `name` |
| `["relation", predicate, bytes32]` | object is subject of an edge to the target object |
| `["attr_eq", field, value]` | equality |
| `["attr_ne", field, value]` | inequality, defined only when both sides exist |
| `["attr_lt", field, value]` / `_le` / `_gt` / `_ge` | ordering comparison |
| `["and", [expr, ...]]` | conjunction, 2 or more operands |
| `["or", [expr, ...]]` | disjunction, 2 or more operands |
| `["not", expr]` | closed-world complement within the namespace |

The parser flattens nested `and`/`or` of the same operator into one operand
array and sorts **nothing**: operand order is semantic for evidence and is
preserved. `not true` is canonicalized to `["false"]`; `not false` to
`["true"]`; double negation is removed. These are the only normalizations.

`value` is a WCB scalar: text, u64, i64, bool, or null. Timestamps are encoded
with tag `0x09` when the runtime value is temporal; the text grammar has no
timestamp literal in v1.

### 2.1 Field names

v1 defines these root fields for object records:

| Field | Type | Notes |
|---|---|---|
| `object_id` | bytes32 rendered as lowercase hex for comparison | stable identity |
| `content_id` | bytes32 hex | exact content root |
| `namespace` | text | namespace name |
| `media_type` | text | declared MIME type |
| `size` | u64 | plaintext bytes |
| `mtime` | timestamp | version mtime |
| `kind` | text | `file`, `dir`, `symlink` |
| `attrs.<name>` | any | application attributes |

Unknown root fields and unknown attribute paths are **not** errors: they simply
do not match (closed-world). A schema/version change that alters this rule
requires a new selector schema string.

### 2.2 Object references in `relation`

The textual `relation(pred, obj:<hex>)` form parses the target into 32 raw bytes
before hashing, so `obj:AB...` and `obj:ab...` canonicalize identically. The
target need not exist at parse time; resolution simply yields no matches.

---

## 3. Evaluation semantics

`Resolve(selector, snapshot, principal, limits) -> ResultSet`.

1. Compute the **authorized universe**: objects visible to `principal` in the
   selector namespace at the pinned snapshot. Rule out unauthorized objects
   before evaluation, including in counts and evidence.
2. Evaluate predicates with three-valued-observable behavior reduced to a
   two-valued result: a comparison whose field is absent or whose types differ
   is **false**; `attr_ne` is false when either side is absent. Use
   `NOT has(...)` to test absence. `not` complements within the authorized
   universe.
3. `relation` is a one-hop edge lookup in v1 (no transitive closure). Cycles in
   the relation graph are legal and have no special meaning.
4. `tag` is sugar for an edge `(subject, "tagged_as", text:value)`.
5. Order results by the `order` list, then always by `object_id` ascending as a
   deterministic tie-break. String comparison is unsigned byte-wise over UTF-8.
6. Apply `offset`, then `limit`.
7. Return ordered `ResolvedRef` values whose content roots are integrity
   verified at materialization time.

The result includes `selector_id`, `snapshot_id`, `matched_count` (authorized
only), and optional per-result evidence naming the edges/attributes that
matched. Evidence MUST be filtered by the same authorization predicate.

`require_one` APIs MUST reject zero matches (`NotFound`) and more than one
(`Ambiguous`) with the full authorized count; they MUST NOT pick one silently.

---

## 4. Identity

```
selector_bytes = WCB(selector map)
selector_id    = H("weft/selector/v1", selector_bytes)   # 32 bytes, hex for display
```

Properties:

- Whitespace, comments (none in v1), numeric formatting, hex case, and the
  normalizations in §2 do not change the identity.
- Operand reordering does change identity even when logically equivalent.
- `selector_id` does **not** encode the snapshot, the principal, or the
  outcome. A live selector may return different results after a commit.
- The hash does not hide the expression. Confidential selectors must not be
  exposed to principals who cannot see the namespace; hashing is not encryption.
- A pinned view identity is separate: `ViewId = H("weft/view/v1",
  selector_id || snapshot_id || resolver_version)` (informative in v1).

---

## 5. Errors

| Class | Cause |
|---|---|
| `Parse` | grammar, escape, integer, or token error (position reported) |
| `LimitExceeded` | input size, nesting, operand count, or path length bound |
| `NonCanonical` | AST bytes are not canonical WCB or violate §2 |
| `UnknownSchema` | schema string is not `weft.selector.v1` |
| `NamespaceMismatch` | resolution requested for another namespace |
| `NotFound` | `require_one` with zero results |
| `Ambiguous` | `require_one` with more than one result |
| `Unauthorized` | the principal cannot read the namespace |

---

## 6. Pagination

An offset cursor is allowed in v1 but is O(offset) to evaluate. The opaque
cursor form is:

```
WCB({"schema":"weft.cursor.v1","selector_id":bytes32,"snapshot":bytes32,"offset":u64})
```

Clients MUST treat the base64 encoding as opaque. A cursor whose `selector_id`
or snapshot differs from the request is rejected (`CursorMismatch`). Cursor
replay is deterministic for the same snapshot and authorized view.

---

## 7. Worked canonical example

Written:

```
select attrs.project == "demo" and relation(derived_from, obj:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff) order by attrs.seq desc limit 10
```

Canonical WCB (pseudo, not literal bytes):

```
{"namespace":"default",
 "predicate":["and",[
   ["attr_eq","attrs.project","demo"],
   ["relation","derived_from",bytes32(001122...eeff)]
 ]],
 "order":[{"field":"attrs.seq","dir":"desc"}],
 "limit":10,
 "schema":"weft.selector.v1"}
```

The map is then encoded with WCB's encoded-key ordering (§8.2 of
`WEFTCODEC-FORMAT.md`): keys sort by their **full encoded bytes**, which means
shorter encoded key strings sort before longer ones, and byte-wise comparison
breaks ties. The order shown above is illustrative, not the wire order.
Golden bytes live in `conformance/v1/selectors/`.

---

## 8. Non-goals in v1

No joins, transitive closure, aggregation, full-text search, embeddings,
arithmetic expressions, or user-defined predicates. These may be added only
under a new selector schema with explicit cost and determinism analysis.

---

## 9. Cross-reference: binding to a pinned Context Root (additive)

This section is additive and does not change any v1 selector byte or
`SelectorId`. It records how selectors are used by `spec/WEFT-CONTEXT-ROOT.md`.

- `SelectorId` identifies the expression only. It never includes the snapshot,
  resolver version, causal closure, capability scope, or budget.
- A **Context Root** (`weft.context-root.v1`) embeds the canonical selector
  bytes and their `SelectorId`, and adds the pinned snapshot, resolver version,
  causal closure, optional capability hash, optional budget, and the ordering
  policy `selector_then_object_id`.
- The Context Root's embedded selector MUST be canonical bytes that re-encode
  identically, and its `SelectorId` MUST equal
  `H("weft/selector/v1", selector bytes)`. A mismatch is `NonCanonical`.
- The Context Root namespace MUST equal the selector namespace. A selector
  framed under another namespace is rejected; it is not re-scoped silently.
- Resolution of a Context Root evaluates this selector at the pinned snapshot
  with the pinned resolver version, applies authorization before evaluation and
  counting, then applies the selector's `order` list and always the
  `object_id` ascending tie-break (§3 step 5).
- A live result of the selector text MUST NOT be presented as the result of a
  Context Root, and vice versa. They are different identities with different
  stability guarantees.
- Selector equivalence is not claimed. Two logically equivalent expressions
  remain distinct identities; the Context Root inherits that distinction.

Error classes remain those of §5; a Context Root layer adds the classes in
`WEFT-CONTEXT-ROOT.md` §8 (snapshot, budget, materialization, capability) and
never reinterprets a selector error as success.
