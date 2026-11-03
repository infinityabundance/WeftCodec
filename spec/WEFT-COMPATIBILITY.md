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
