# WeftCodec v1 — implementation notes

Status: implemented and tested for `weftcodec-format 1.0` and
`weft-selectors 1.0`. All library, integration, CLI, and conformance-vector
tests pass in the OOM-protected Rust Docker court. Nothing in this file is a
substitute for running the tests; the commands and observed results are listed
below.

## What is implemented

- **Framing (`frame`)**: exact 96-byte WFT1 header, 48-byte dependency and chunk
  entries, table length, header CRC32C over `[0,72)` + `[76,96)`, trailing frame
  CRC32C, reserved-field and mode/flag/feature validation, checked arithmetic.
- **Hash (`hash`)**: `H(domain, data) = BLAKE3-256(u32le(|domain|) || domain ||
  data)`, all v1 domains exported, lowercase hex parsing/display.
- **Varints (`varint`)**: minimal LEB128, at most 10 bytes, tenth byte
  `0x00`/`0x01`, non-minimal rejected as `NonCanonical`.
- **WCB (`wcb`)**: tags 0x00–0x09, minimal varints, zigzag i64/timestamp,
  text/bytes/array/map, map keys sorted by full encoded key bytes, duplicate and
  unsorted keys rejected, trailing bytes and non-text keys rejected, configurable
  depth/item/string limits.
- **Modes (`modes`)**:
  - RAW (chunked and unchunked),
  - ZSTD (chunked and unchunked; output capped at declared `decoded_len`; zstd
    rejects frames whose output exceeds the capacity),
  - REFERENCE (input must equal base at encode time),
  - DELTA with a bounded hash-chain matcher (sampled base positions, up to four
    candidates per 4-byte key, greedy longest match, minimum 8-byte match),
    COPY/INSERT decoder bounded by `max_delta_ops` and `max_decoded_bytes`,
  - DICTIONARY (raw-content zstd dictionary; dependency domain 2),
  - STRUCTURED (canonical WCB body).
- **Dependencies (`dependency`)**: `Resolver`, `MapResolver`,
  `DirectoryResolver` (lowercase-hex filenames, length + hash verified),
  `ArchiveResolver`, `FrameSource`/`MapFrameSource`, recursive
  `decode_frame_dag` with a visiting set (`DependencyCycle`), depth/fetch/
  cumulative-decoded limits, and content-hash binding of fetched frames.
- **Archive (`archive`)**: WFA1 container creation with deterministic Kahn
  topological order, missing-dependency and cycle rejection, duplicate-hash
  consistency, CRC, truncation and non-topological-order rejection, full decode
  without external resolution.
- **Selectors (`selector`)**: lexer, recursive-descent parser, all v1 grammar
  forms, string escapes and surrogate pairs, normalization (`and`/`or`
  flattening, double-negation removal, `not true/false`, offset-0 omission,
  explicit `asc`), canonical WCB bytes, `SelectorId`, canonical-bytes decoder
  with strict non-canonical rejection.
- **Streaming primitives (`stream`)**: `encode_chunked`, `decode_chunked`, and
  `read_range` (decodes only intersecting chunks and verifies their hashes;
  unchunked frames are decoded fully then sliced).
- **CLI**: `encode`, `decode`, `inspect --json`, `verify`, `archive
  create/decode`, `selector id/canonicalize`, plus `conformance` generation.
  `auto` builds real candidate frames (RAW, ZSTD, REFERENCE only when the base
  equals the input, DELTA when a base is supplied, DICTIONARY when a dictionary
  is supplied) and selects the smallest full frame, breaking ties toward the
  lower mode value (RAW first). Errors exit nonzero and print the typed class.

## Limits

`Limits::default()` uses every default in WEFTCODEC-FORMAT §10, plus selector
limits from WEFT-SELECTORS §1/§5 and `max_archive_entries = 1_000_000`.
`with_*` builders exist for each field. Callers decoding untrusted input should
only lower these values (the builders do not clamp upward).

## Deliberate interpretations, ambiguities, and deviations

The specification is authoritative. Where it is internally inconsistent or
silent, the most literal reading consistent with the rest of the format was
chosen and is recorded here.

1. **Chunked RAW.** §3 says `RAW` has `chunk_count == 0`, while §6 says "For
   `RAW`, chunks are literals" and defines chunk-table rules generally. The
   deliverable also requires chunked RAW. We allow chunked RAW; the §3 line is
   read as applying to the unchunked form. `body_len == decoded_len` still holds.
2. **Structured `content_hash` domain.** §2.1 defines the header field as
   `H("weft/chunk/v1", plaintext)` unconditionally. §1.1 also lists
   `weft/structured/v1` but does not assign it to the header. We use
   `weft/chunk/v1` for every frame header and export `DOMAIN_STRUCTURED` for
   callers; it is not used in the header.
3. **WCB map ordering vs §7's worked selector.** §8.2 mandates ordering by full
   encoded key bytes (tag + varint length + bytes). §7's prose claims the example
   map keys appear as `namespace, order, predicate, limit, schema`, which is
   lexicographic key-string order and contradicts §8.2 for keys of different
   lengths (e.g. `schema` length 6 sorts before `namespace` length 9). We follow
   §8.2; conformance vectors freeze that ordering.
4. **Selector integer sign.** §1 says "Integer literals are decimal without sign
   (sign is a parse error)" and then gives ranges for a leading `-`. We accept an
   optional leading `-` only immediately before digits; non-negative values
   encode as WCB `u64`, negative values as WCB `i64` (`-0` normalizes to `u64`
   0). A `-` anywhere else is a `Parse` error.
5. **Selector `tag`/`relation` arguments.** The grammar spells them as string
   literals, but §7's worked example writes `relation(derived_from, ...)` bare.
   We accept a non-keyword identifier as an equivalent text argument; canonical
   bytes are identical. Quoted strings remain the primary form.
6. **Minor version.** Only `format_minor == 0` is accepted; other minors are
   `UnsupportedVersion`. The format freezes v1 at minor 0 and no optional-field
   minor version exists yet.
7. **Reserved `ORIGINAL_HASH` (flag bit 3).** Treated as `UnsupportedFeature`
   when set. `SELF_CONTAINED` (bit 1) is also `UnsupportedFeature`, as §7 says
   implementations must reject rather than guess.
8. **WCB error classes.** Non-minimal varints, unsorted/duplicate/non-text map
   keys, trailing bytes, and unknown tags are `NonCanonical`; truncated values
   and overlong varints that overflow are `Malformed`; depth/item/string
   violations are `LimitExceeded`.
9. **Chunk `encoded_len == 0`.** Rejected as `Malformed` (zero-length extents
   cannot satisfy strict start ordering and contiguity in a useful way).
10. **Archive ordering errors.** A dependency that exists later in the archive
    is `Malformed` ("not topological"); a dependency absent entirely is
    `DependencyMissing`. Duplicate content hashes with different bytes are
    `Malformed` at creation and `Corrupt` at parse.
11. **Range reads.** `read_range` clamps the requested length to EOF; an offset
    at or beyond EOF returns an empty vector. For chunked frames only
    intersecting chunks are decoded and hash-verified, so the whole-content hash
    is not checked on a partial read (it is checked on full decode).
12. **Selector duplicate order fields.** Rejected by the parser as `Parse`
    (the spec says "rejected" without assigning a class).
13. **Selector canonical `i64(0)`.** The text grammar never produces it, but
    `Selector::from_canonical_bytes` accepts a WCB `i64(0)` literal because the
    WCB value is well formed and re-encodes identically; all canonical forms
    produced by the grammar use `u64(0)`.
14. **Error classes beyond §11.** `ErrorClass` also carries selector `Parse` and
    `UnknownSchema` so all APIs share one error type. Codec callers can ignore
    them.
15. **DICTIONARY dependency domain.** A DICTIONARY frame's dependency must have
    `dep_domain == 2`; mismatch is `Malformed`.
16. **Negative zstd levels.** The header `level` is an informational `u16`; a
    negative level cannot be represented and is rejected at encode time with
    `LimitExceeded` rather than silently clamped.
17. **Zstd window log.** The Rust `zstd` crate's decompression context defaults
    to `ZSTD_WINDOWLOG_LIMIT_DEFAULT = 27` (confirmed in the vendored
    `zstd-sys` sources), satisfying the §3 limit. Output is additionally capped
    at the declared `decoded_len` and a length mismatch is `Corrupt`.

## Known limitations

- There is no true streaming file API (no incremental reader/writer for files
  larger than memory). The chunked encoder/decoder and `read_range` are real and
  tested; they are not a substitute for a streaming pipeline.
- The CLI reads whole inputs into memory, bounded by the relevant limit
  (`max_decoded_bytes` for encode inputs, `max_frame_bytes` for frames,
  `max_total_fetched_bytes` for archives, `max_selector_bytes` for selectors).
  It does not exploit chunked encoding to process files larger than memory.
- Archive decode materializes all entry plaintexts and enforces
  `max_decoded_bytes` cumulatively; a single archive whose total plaintext
  exceeds that limit is rejected even though each frame is individually legal.
- The delta matcher is a bounded heuristic. It produces correct streams but not
  necessarily the smallest possible one; matched regions shorter than 8 bytes
  become INSERTs.
- `DirectoryResolver` trusts filesystem metadata for the pre-read length check,
  then verifies length and hash again after reading.
- Selector evaluation is intentionally absent; the filesystem layer evaluates
  the public AST. Selector source is limited to 64 KiB and 64 nesting levels by
  default.
- The CLI's `--mode structured` input is canonical WCB bytes (validated before
  framing); it is not JSON.
- ZSTD/DICTIONARY frames are decode-equivalent across zstd library versions, not
  byte-identical; conformance records the encoder identity implicitly (crate
  `zstd 0.13.4`, zstd 1.5.7).
- The crate forbids `unsafe` in its own code. Third-party crates (`zstd`,
  `blake3`, `crc32c`) may use `unsafe` internally.

## Verification record

Environment: Docker court `weft/rust-court:1` (from `rust:1-slim-bookworm`),
launched only through `scripts/docker-court.sh rust`. Default hard bounds:
`--memory=3g --memory-swap=3g --pids-limit=1024 --cpus=4`, bounded `/tmp`,
`no-new-privileges`, `--cap-drop=ALL`. No build or test was run on the host.

Because the court runs as uid 0 with `--cap-drop=ALL` (no
`CAP_DAC_OVERRIDE`), host-created 0755 directories under `.docker-cache` and the
working tree are not writable by the container. Their permissions were widened
locally (`chmod -R a+rwX` on `.docker-cache` and on the WeftCodec working tree,
excluding `.git`). File modes other than the executable bit are not tracked by
git; no parent-repository file or git state was changed.

Commands actually run, with observed results:

```sh
# Formatting (rustfmt is installed into the throwaway container).
scripts/docker-court.sh rust "rustup component add rustfmt && cd WeftCodec && cargo fmt --all"
# -> completed; `cargo fmt --all -- --check` later printed FMT_CHECK_OK.

# Full test suite.
scripts/docker-court.sh rust "cd WeftCodec && cargo test --all-targets"
# Observed:
#   unittests src/lib.rs      28 passed; 0 failed
#   unittests src/main.rs      0 passed; 0 failed
#   tests/cli.rs               8 passed; 0 failed
#   tests/conformance.rs       4 passed; 0 failed
#   tests/dependencies.rs     12 passed; 0 failed
#   tests/malformed.rs         8 passed; 0 failed
#   tests/modes.rs             9 passed; 0 failed
#   tests/roundtrip.rs         7 passed; 0 failed
#   tests/selectors.rs         7 passed; 0 failed
#   tests/stream.rs            4 passed; 0 failed
#   total 87 passed, 0 failed (10 test binaries)

# Release build.
scripts/docker-court.sh rust "cd WeftCodec && cargo build --release"
# Observed: Finished `release` profile [optimized] target(s) in 16.55s
# (the final incremental release build after the arithmetic-hardening change
# finished in 2.24s; both were run in the court).

# Conformance vector generation (files committed under conformance/v1).
scripts/docker-court.sh rust "cd WeftCodec && cargo run --quiet -- conformance --dir conformance/v1"
# Observed: wrote conformance vectors to conformance/v1
```

Vector counts: 29 codec vectors (12 successful decodes covering every mode and
error vectors spanning `Malformed`, `Corrupt`, `UnsupportedVersion`,
`UnsupportedMode`, `UnsupportedFeature`, `LimitExceeded`,
`DependencyMismatch`, `NonCanonical`), 20 structured vectors, 22 selector
vectors. `tests/conformance.rs` replays all of them.

No benchmark or performance claim is made here.

---

## F1 — Context Root, causal closure, and `weft://` (implemented)

Status: **implemented and tested** in this crate for the data layer. Resolver
and capability *enforcement* are specified but not implemented here (F2/F3).
The AI codec is design-only and has no code.

### What is implemented

- **`context` module** (`src/context.rs`):
  - `Budget` with strict canonical parsing (`max_results`, `max_bytes`,
    `max_depth`, `max_work`; non-empty, ≥1, canonical decimal), WCB encode/
    decode, and query rendering.
  - `IdentityMode` (`bytes_only`, `causal_exact`).
  - `CausalClosure` (`weft.causal.v1`): explicit parents, closure superset,
    strictly ascending/unique/nonzero, canonical bytes, `CausalClosureId =
    H("weft/causal/v1", bytes)`.
  - `CapabilityScope` (`weft.capability.v1`): principal, sorted namespaces,
    sorted roots, predicate allowlist, optional budget/expiry, `CapabilityHash =
    H("weft/capability/v1", bytes)`.
  - `ContextRoot` (`weft.context-root.v1`): canonical selector bytes + selector
    id binding, nonzero pinned snapshot, resolver version, causal closure, fixed
    `selector_then_object_id` ordering, optional capability hash and budget.
    `ContextRootId = H("weft/context-root/v1", bytes)`.
  - Authoritative `weft://<namespace>/c/<64hex>` formatting/parsing and
    descriptor URIs with strict uppercase percent-encoding, canonical query
    parameter order, and rejection of non-canonical spellings.
- **Hash domains** `weft/context-root/v1`, `weft/causal/v1`,
  `weft/capability/v1` exported from `hash`.
- **Limits** `max_causal_entries` (4,096), `max_capability_entries` (4,096),
  `max_uri_bytes` (64 KiB), all configurable downward.
- **CLI**: `context id`, `context uri [--descriptor]`, `context parse <uri>
  [--json]`, `causal id --parents a,b [--json]`.
- **Vectors**: `conformance/v1/context/` (6 canonical + 10 rejection = 16) and
  `conformance/v1/causal/` (4 canonical + 6 rejection = 10), replayed by
  `tests/context.rs`.

Golden `ContextRootId` values (also in the vectors):

| Vector | ContextRootId |
|---|---|
| `001_root_true` | `53f70ff51a49496af01dc87e7002fa7ef7fe1ab5c192bdb326d0868b1f01a41a` |
| `002_root_filter_budget` | `190b59a2510cf402fe0e4c8a3e8bc8a72d2b60e5c0ee4ed9ca8ce2787bc56af2` |
| `003_root_causal_capability` | `ce702e74220a98d9b3b36e007860b20ee2404ef0f735085c8f942c9b29acfc02` |
| `004_root_descriptor_escaping` | `dc6433a13d94842a972b37ab141f1911d02f533a043e7f662c57571b403b7e5a` |
| `005_root_custom_resolver` | `f4d84a7e21f100e7e01d4788bd538b0a50db11b1b0b9a4293e510c12e31bbc26` |
| `006_root_empty_causal` | `b61f96f46c6231f057b2dc64fcb602c5a48622a855d99fb4b600e075e10209be` |

Golden causal closure hashes:

| Vector | CausalClosureId |
|---|---|
| `001_single_parent` | `77475fa35f898c07f43c6ca1b6a8b30da44468825e0dcd8a0249442ba86ec15c` |
| `002_two_parents_unsorted_input` | `58f09b15991e2de83a8f689327a928d31e3df8c80cf44562293377316b6c92ae` |
| `003_three_parents` | `41c5d31f9a62e7bc57e52a5aafbac806830aae08265b8f99ea330a70c46f2100` |
| `004_empty_closure` | `fe2d8422572489a70dc533dffa9a891fd36d99f496f16aab85254fff147ab7cc` |

### Rules chosen where the spec leaves room (recorded as normative here)

1. **Causal array is always present.** An empty closure encodes as `array(0)`;
   the encoder never omits the field. Decoders require it.
2. **`ordering` is required and fixed** to `selector_then_object_id` in v1.
   An unknown value is `UnsupportedVersion`, not a default.
3. **`resolver` is required**; `weft-resolver-v1` is the default the CLI uses.
   Descriptor URIs omit `resolver` only when it is the default.
4. **Descriptor normalization is one-way.** Descriptor parsing sorts and
   deduplicates `closure`; canonical record decoding rejects unsorted/duplicate
   arrays. This is deliberate: human spellings normalize, machine records are
   strict.
5. **Lowercase hex in URIs only.** Authoritative URIs and descriptor snapshot/
   closure/capability hex require lowercase; uppercase is `Parse`. Record bytes
   are unaffected (hex is only a display form outside WCB).
6. **Percent-encoding is canonical only.** Raw bytes outside the unreserved set,
   lowercase escapes, and escapes of unreserved characters are `Parse`.
7. **Selector binding is checked twice:** the embedded selector must decode as
   canonical `weft.selector.v1`, re-encode identically, hash to `selector_id`,
   and have the same namespace as the record.

### Verification record (exact commands and observed results)

Environment: Docker court `weft/rust-court:1` (image id
`sha256:c96472367ec767612a78d1809c2a8c8cf9244c4ed9757dbe1bd821eb3a723b9d`),
launched only through `scripts/docker-court.sh rust`; hard limits unchanged
(3 GiB memory/swap, 1024 pids, 4 CPUs, bounded `/tmp`, no-new-privileges,
`--cap-drop=ALL`).

```sh
scripts/docker-court.sh rust "cd WeftCodec && cargo test --all-targets --quiet"
# Observed: 36 + 0 + 8 + 4 + 5 + 12 + 8 + 9 + 7 + 7 + 4 = 100 passed, 0 failed
# (the new 5 are tests/context.rs; 8 new unit tests are in src/context.rs)

scripts/docker-court.sh rust "cd WeftCodec && cargo run --quiet -- conformance --dir /work/.docker-cache/vectors"
# Observed: wrote conformance vectors to /work/.docker-cache/vectors
# Only the new context/ and causal/ directories were copied into the repo;
# existing codec/structured/selectors vectors were not touched.

scripts/docker-court.sh rust "cd WeftCodec && cargo fmt --all && cargo test --all-targets --quiet"
# Observed: same 100 passed, 0 failed after formatting.
```

Cross-language Context Root gate (run in the courts, exact commands):

```sh
scripts/docker-court.sh rust "cd WeftCodec && cargo build --release --quiet && cd /work/WeftCodec-Go && bash hack/context-cross.sh rust-gen"
# -> rust-gen: wrote 4 context cases and one causal case
scripts/docker-court.sh go "cd WeftCodec-Go && bash hack/context-cross.sh go-build && bash hack/context-cross.sh go-gen && bash hack/context-cross.sh go-parse-rust"
# -> go-build: wrote /work/.docker-cache/weftcodec-go
# -> go-gen: wrote 4 context cases and one causal case
# -> ... parsed 4 rust descriptors -> go-parsed
scripts/docker-court.sh rust "cd /work/WeftCodec-Go && bash hack/context-cross.sh rust-parse-go"
# -> ... parsed 4 go descriptors -> rust-parsed
scripts/docker-court.sh go "cd WeftCodec-Go && bash hack/context-cross.sh compare"
# Observed: 4/4 ContextRootId PASS, 4/4 URI PASS, 4/4 descriptor PASS,
#           4/4 go-parse-rust PASS, 4/4 rust-parse-go PASS, causal hash PASS,
#           causal canonical bytes PASS. All checks passed.
```

Earlier existing codec cross-language gate re-run after these changes: all 8
modes decoded in both directions, all 6 deterministic frame modes byte-identical,
ZSTD/DICTIONARY differing bytes allowed by format §12. No regression.

`bash scripts/check-spec-drift.sh` in the rust court: `spec drift: OK (24 file
comparisons across 3 repos)`.

### Not implemented in this crate (labelled precisely)

- Resolver execution, snapshot existence/organization checks, authorization
  enforcement, budgets enforcement, evidence filtering, and materialization:
  **specified, not implemented** here (F2/F3; the store already has its own
  selector/materialization paths).
- Capability revocation records and enforcement points: **specified, not
  implemented** (F2). The `CapabilityScope` type and hash are implemented.
- AI tensor/token/KV/delta modes: **specified, not implemented**
  (`spec/WEFT-AI-CODEC.md`); no wire mode values were allocated.
- No performance claim and no model-checking claim for the new identities.

## F5 — decoder limits, DAG frame source, adversarial vectors (2026-11-03)

- `decode`, `archive decode`, and `selector id|canonicalize` accept the ten
  requested limit flags: `--max-frame-bytes`, `--max-decoded-bytes`,
  `--max-dependency-depth`, `--max-dependency-count`,
  `--max-total-fetched-bytes`, `--max-chunks`, `--max-dictionary-bytes`,
  `--max-delta-ops`, `--max-structured-depth`, and `--max-selector-bytes`.
  Additional `--max-structured-items`, `--max-selector-depth`,
  `--max-selector-operands`, and `--max-archive-entries` bound the remaining
  spec limits. Defaults are the normative v1 values.
- `decode --frame-source DIR` resolves dependency *frames* from a directory
  named by lowercase content hash and walks the DAG with the existing depth,
  cycle, total-fetch, and cumulative-decoded bounds.
- **Allocation fix:** zstd and dictionary decompression previously allocated
  the full declared output length via `zstd::bulk::Decompressor`. A tiny frame
  claiming 4 GiB could therefore force a 4 GiB allocation. Both paths now
  stream with an exact `declared_len + 1` read bound; the 4 GiB-declared vector
  returns `Corrupt` with a few MiB RSS.
- `conformance --adversarial --dir conformance/v1` writes 36 vectors plus
  `PROVENANCE.json`. They are copied byte-identically into WeftCodec-Go and
  replayed by `bench/scripts/run_decoder_bomb_court.sh`; expected classes are
  identical in both implementations. This is a curated corpus, not fuzzing.

---

## F6 — TENSOR (6) and TOKEN_SEQUENCE (7), 2026-11-03

The earlier statement that AI modes were "specified, not implemented" is
historical. F6 allocates exactly two modes, both implemented and frozen:

- `TENSOR` (6): `u32le(meta_len) || WCB(weft.tensor.v1) || payload`; dtype
  `f16|bf16|f32|f64|i8|u8|i16|i32|i64`; row-major little-endian only; checked
  shape product, element count, payload bytes, and `weft/tensor/payload/v1`
  hash binding.
- `TOKEN_SEQUENCE` (7): `u32le(meta_len) || WCB(weft.tokens.v1) || payload`;
  first token absolute minimal LEB128, later ids wrapping 64-bit zigzag deltas
  in minimal LEB128; exact count/trailing checks; `weft/tokens/payload/v1`
  binding.

Every other brainstormed layout/stream/delta mode is `specified, not
implemented` with no allocated wire value (`spec/WEFT-AI-CODEC.md` §4).

Verification in the Rust court:

```
$ scripts/docker-court.sh rust "cd WeftCodec && cargo test --all-targets --quiet"
# 110 tests pass, 0 failed (includes tests/ai_modes.rs: 9 tests)
$ scripts/docker-court.sh rust "cd WeftCodec && cargo run --quiet -- conformance --dir conformance/v1"
# 49 codec vectors, including 100-104 valid and 120-135 rejects
```

Cross-language gate (`WeftCodec-Go/hack/cross.sh`):

```
PASS rust->go tensor / tokens
PASS go->rust tensor / tokens
PASS identical tensor / tokens
```

Historical `017_unknown_mode` asserted mode 6 was reserved. It is retained
byte-for-byte with `superseded_by: 017_unknown_mode_reserved8`; the parser now
rejects mode 8 as `UnsupportedMode`. Both replay suites skip superseded
vectors.

CLI additions:

```
weftcodec encode --mode tensor --input PAYLOAD --dtype f32 --shape 2,3
weftcodec encode --mode tensor --input PAYLOAD --metadata META.json
weftcodec encode --mode tokens --input TOKENS.json
```

Limits added: `max_tensor_elements` (2^32), `max_tensor_rank` (8), `max_tokens`
(2^24).
