# WeftCodec (Rust)

WeftCodec v1: a deterministic, lossless, context-conditioned codec.

- Normative format: `../spec/WEFTCODEC-FORMAT.md`
- Normative selectors: `../spec/WEFT-SELECTORS.md`
- Implementation notes, limits, ambiguities, and the verification record:
  [`NOTES.md`](NOTES.md)
- Frozen conformance vectors: [`conformance/v1/`](conformance/v1/) (append-only;
  see [`conformance/README.md`](conformance/README.md))

## Highlights

- WFT1 frames with a 96-byte header, dependency/chunk tables, CRC32C, and
  domain-separated BLAKE3 hashes.
- Payload modes: RAW, ZSTD, REFERENCE, DELTA (bounded COPY/INSERT matcher),
  DICTIONARY, STRUCTURED (canonical WCB).
- WCB canonical bytes with encoded-key map ordering.
- Self-contained WFA1 archives with enforced topological order.
- WeftQL selector parser, normalization, canonical bytes, and `SelectorId`.
- Chunked RAW/ZSTD plus random-range reads that decode only intersecting chunks.

## CLI

```sh
weftcodec encode --input FILE --output FRAME --mode raw|zstd|reference|delta|dictionary|structured|auto
weftcodec decode --input FRAME --output FILE [--deps DIR]
weftcodec inspect FRAME --json
weftcodec verify FRAME [--deps DIR]
weftcodec archive create --output BUNDLE.wfa FRAME...
weftcodec archive decode --input BUNDLE.wfa --out-dir DIR
weftcodec selector id --input QUERY [--json]
weftcodec selector canonicalize --input QUERY
```

Dependency directories contain plaintext files named by the lowercase hex
digest (`H("weft/chunk/v1", bytes)`). For `--mode structured`, the input must
already be canonical WCB bytes; the CLI validates it and rejects non-canonical
input.

## Build and test

All builds and tests run in the OOM-protected Docker court; never on the host:

```sh
cd /path/to/WeftFS
scripts/docker-court.sh rust "cd WeftCodec && cargo test --all-targets"
scripts/docker-court.sh rust "cd WeftCodec && cargo build --release"
```

No `unsafe` in this crate. Untrusted parsing returns typed `Result` errors;
limits are checked before allocating.
