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
