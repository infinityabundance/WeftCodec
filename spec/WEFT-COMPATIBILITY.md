# WEFT-COMPATIBILITY v1.0.0

**Status:** Normative policy. Applies to `weftcodec-format` and
`weftfs-store` version lines.

---

## 1. Independent version lines

| Line | Where recorded | v1 value |
|---|---|---|
| `weftcodec-format` | frame bytes 4–5 | `1.0` |
| `weft-selectors` | selector schema string | `weft.selector.v1` |
| `weftfs-store` | `FORMAT` file, superblock | `1` |
| Library semver | package metadata | unrelated to the above |

A library release NEVER implies a format change. A format change NEVER hides in
a patch release.

---

## 2. Read/write matrix (v1)

| Reader | Writer | Behavior |
|---|---|---|
| 1.x | 1.0 | MUST read all frozen v1 vectors; unknown critical bits fail closed |
| 1.x | 1.y (y>x) | MUST reject if a new critical feature is required; otherwise read |
| 2.x | 1.x | MAY provide migration; MUST NOT silently reinterpret |

Both Rust and Go publish this matrix with their actual test results. A row is
only "supported" when a conformance test names the vector set.

---

## 3. Change classes

**Patch (1.0.z):** documentation, bug fixes that do not change accepted bytes,
performance. A bug fix that changes emitted bytes for the same logical value
requires a frozen-vector addition and a note; it must not mutate existing
vectors.

**Minor (1.x.0):** new optional field with an existing safe default, new
non-critical feature bit that old readers may ignore, new error detail. Old
readers must already behave correctly for the default.

**Major (2.0.0):** changed hash domain, changed canonical encoding, changed
frame layout, removal of a mode, changed selector semantics. Requires new
golden vectors, a migration tool, and a compatibility report.

---

## 4. Hash and algorithm agility

- v1 uses BLAKE3-256 with `H(domain,data) = BLAKE3(u32le(|domain|)||domain||data)`.
- A future algorithm is introduced by a new major version and a new domain
  string; existing ContentIds remain valid under their recorded algorithm.
- No implementation may re-hash existing content under a new algorithm and
  present it as the same identity.
- Unkeyed hashes are integrity identifiers, never MACs, signatures, or
  authorization capabilities.
- Compression algorithm changes are mode-version changes. Zstd frames are
  identified by `mode=ZSTD` and decode per RFC 8878; encoder byte identity is
  NOT promised because compressors evolve. Conformance for ZSTD/DICTIONARY is
  **decode equivalence plus recorded encoder identity**, while RAW, REFERENCE,
  DELTA, and STRUCTURED promise byte-identical encoding.

---

## 5. Unknown data

- Unknown critical frame flag/feature/mode: reject with `UnsupportedFeature` or
  `UnsupportedMode`.
- Unknown optional selector fields: not present in v1; would require a minor
  selector schema (`weft.selector.v1.1`) and a documented default.
- Unknown WCB tags: reject (`NonCanonical`).
- Unknown commit ops: the whole commit is rejected; no partial application.
- Unknown attributes and unknown root fields in selectors: evaluate to no match
  (closed world), never an error.

---

## 6. Frozen vectors

`conformance/v1/` is append-only and contains:

- `codec/` frame hex + expected plaintext + expected error class;
- `structured/` canonical WCB hex and non-canonical rejects;
- `selectors/` source text, canonical bytes, SelectorId;
- `store/` a deterministic mini-store whose commit hashes are fixed.

Rules:

1. Existing vector files are never edited; a correction adds a new vector file
   with an explanation and a compatibility note.
2. Every vector records the tool and version that produced it.
3. Invalid vectors record the required error **class**, not wording.
4. Cross-language tests must consume the same files, not regenerated copies.

---

## 7. Migration

`weftfs migrate --from <format> --to <format> --dry-run` must:

1. validate the source with read-only verification;
2. produce a plan and byte/object counts;
3. write a new store, never modify the source;
4. verify the destination fully;
5. emit a machine-readable receipt naming the source and destination roots.

v1 implements only `1 -> 1` (copy/verify). No migration is claimed for formats
that do not exist.

---

## 8. Deprecation

A feature is deprecated only after a replacement is specified, tested, and
documented. Deprecated readers remain for at least one major version. Removing
read support is a major change and requires an archival note naming the last
version that could read the data.

---

## 9. Context Root, causal, and capability version lines (additive)

These lines are additive to the v1 lines in §1. They introduce no change to
`weftcodec-format`, `weft-selectors`, or `weftfs-store` v1 bytes.

| Line | Where recorded | v1 value |
|---|---|---|
| `weft-context-root` | record `schema` string | `weft.context-root.v1` |
| `weft-causal` | record `schema` string | `weft.causal.v1` |
| `weft-capabilities` | record `schema` string | `weft.capability.v1` |

### 9.1 Read/write matrix (v1)

| Reader | Writer | Behavior |
|---|---|---|
| root 1.x | root 1.0 | MUST read all frozen context/causal vectors; unknown ordering policy is `UnsupportedVersion` |
| root 1.x | root 1.y (y>x) | MUST reject unknown critical fields/policies; additive optional fields require a documented default |
| causal 1.x | bytes_only | MUST accept; it is the v1 default and byte-identical to the frozen version record |
| causal 1.x | causal_exact | MUST either implement the mode or reject with `UnsupportedVersion`; MUST NOT ignore the fields |

### 9.2 Change classes

- **Patch:** documentation, bug fixes that do not change accepted canonical
  bytes. A fix that changes emitted bytes for the same logical record is a minor
  change plus new vectors; existing vectors are never mutated.
- **Minor:** new optional Context Root field with a safe default, new budget key
  that old readers reject explicitly, new resolver version name, new canonical
  ordering policy name that old readers reject with `UnsupportedVersion`.
- **Major:** changed schema string for the same meaning, changed hash domain,
  changed canonical encoding, changed closure semantics, changed URI grammar in
  a way that alters accepted strings. Requires new vectors, a migration tool,
  and a compatibility report.

### 9.3 Frozen vectors

New append-only sets:

- `conformance/v1/context/` — canonical context-root bytes, `ContextRootId`,
  authoritative URI, descriptor spellings, and rejection vectors;
- `conformance/v1/causal/` — canonical closure records, `CausalClosureId`, and
  rejection vectors.

Rules in §6 apply unchanged: existing files are never edited, every file records
who produced it, invalid vectors record the error **class**, and cross-language
replay consumes the same files rather than regenerated copies.

### 9.4 Identity modes and compatibility

The `bytes_only` mode is the v1 default and leaves every existing identity
frozen. `causal_exact` uses the same `weft/version/v1` domain over an extended
record; it is an opt-in identity mode, not a reinterpretation of existing bytes.
An implementation MUST NOT silently downgrade a `causal_exact` artifact to
`bytes_only`, and MUST NOT present a `bytes_only` identity as if it were causal.

---

## 10. AI codec mode allocation (additive, v1)

`WEFT-AI-CODEC.md` v1.2.0 allocates previously reserved payload modes:

| Value | Name | Body |
|---:|---|---|
| 6 | `TENSOR` | `u32le(meta_len) || WCB(weft.tensor.v1) || payload` |
| 7 | `TOKEN_SEQUENCE` | `u32le(meta_len) || WCB(weft.tokens.v1) || payload` |
| 8 | `QUANTIZED_TENSOR` | `u32le(meta_len) || WCB(weft.quantized-tensor.v1) || payload` |
| 9 | `FLOAT_TENSOR` | `u32le(meta_len) || WCB(weft.ai.float_tensor.v1) || huffman-canonical-v1 exponent plane || raw sign+mantissa plane` |
| 10 | `TENSOR_SHARDED` | `u32le(meta_len) || WCB(weft.ai.tensor_sharded.v1) || huffman-canonical-v1 stream` |
| 11 | `WEIGHT_DELTA` | `u32le(meta_len) || WCB(weft.ai.weight_delta.v1) || huffman-canonical-v1 planes`; one dependency |
| 12 | `KV_CACHE` | `u32le(meta_len) || WCB(meta) || huffman-canonical-v1 planes`; one page per frame |
| 13 | `ACTIVATION_STREAM` | `u32le(meta_len) || WCB(meta) || huffman-canonical-v1 planes`; one chunk per frame |
| 14 | `SEMANTIC_DELTA` | `u32le(meta_len) || WCB(weft.ai.semantic_delta.v1) || huffman-canonical-v1 planes`; one dependency |
| 15 | `TENSOR_TILED` | `u32le(meta_len) || WCB(meta) || tile payload` |
| 16 | `SPARSE_DELTA` | `u32le(meta_len) || WCB(weft.ai.sparse_delta.v1) || u32le(run_count) || zero-run records`; one dependency |

This is an additive extension of the `weftcodec-format 1.0` mode space, not a
change to existing modes. Rules:

- A reader that does not implement these modes MUST reject them as
  `UnsupportedMode`. It MUST NOT treat the payload as opaque bytes.
- Mode values `17..=127` stay reserved and `128..=255` stay experimental.
- The new hash domains `weft/tensor/payload/v1`,
  `weft/tokens/payload/v1`, `weft/quantized-tensor/payload/v1`,
  `weft/float-tensor/payload/v1`, and `weft/tensor-sharded/payload/v1` are
  additive; existing domains are unchanged. Modes 10, 11, 14, 15, and 16 also
  bind logical identity under `weft/tensor/values/v1`, mode 15 adds
  `weft/tensor/tile/v1`, and mode 14 adds `weft/tensor/semantic/v1` and
  `weft/tensor/semantic-values/v1`. Mode 16 introduces no hash domain of its
  own: it reuses `weft/tensor/values/v1` for `delta_root` and
  `weft/chunk/v1` for the reconstructed content.
- Neither mode changes RAW, ZSTD, REFERENCE, DELTA, DICTIONARY, STRUCTURED, the
  header, the tables, WCB, or selector bytes.
- Historical vector `017_unknown_mode` asserted that value 6 was unknown. It is
  retained byte-for-byte and marked `superseded_by`
  `017_unknown_mode_reserved8`; both replay suites skip superseded vectors.
  The authoritative reserved-value reject is now mode 17.
- Modes 6–16 are deterministic. Entropy-coded planes use
  `huffman-canonical-v1` (WEFT-AI-CODEC §5.3.1), not a library zstd frame, so
  cross-language conformance requires byte-identical frames **and** decode
  equivalence in both directions. Mode 16 is a varint run stream with no
  entropy coder; it is byte-identical for the same input by construction. This
  includes `FLOAT_TENSOR`; the historical
  "byte identity depends on the two zstd encoders" note is superseded.
