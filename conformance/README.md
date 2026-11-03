# WeftCodec v1 conformance vectors

These files freeze the `weftcodec-format 1.0` and `weft-selectors 1.0` bytes.
They are **append-only**: never regenerate, rewrite, reorder, or "fix" an
existing vector to match a later implementation change. If an implementation
disagrees with a committed vector, the implementation is wrong (or the vector
must be superseded by a new schema version with new files).

The vectors were produced by:

```sh
scripts/docker-court.sh rust "cd WeftCodec && cargo run --quiet -- conformance --dir conformance/v1"
```

`cargo test --all-targets` replays every vector (`tests/conformance.rs`).

## Layout

### `codec/*.json`

One complete WFT1 frame per file.

```json
{
  "name": "006_raw_chunked_1024",
  "mode": "raw",
  "mode_value": 0,
  "format_major": 1,
  "format_minor": 0,
  "domain": 0,
  "flags": 1,
  "features": 2,
  "decoded_len": 5000,
  "body_len": 5000,
  "frame_len": 5396,
  "chunk_count": 5,
  "content_hash": "<64 hex>",
  "plaintext_hex": "<hex>",
  "plaintext_hash": "<64 hex>",
  "frame_hex": "<hex>",
  "dependencies": [
    {
      "hash": "<64 hex>",
      "decoded_len": 123,
      "dep_domain": 0,
      "bytes_hex": "<hex>"
    }
  ],
  "expect": { "ok": true }
}
```

Error vectors use `"expect": { "error": "<Class>" }` and may include a
`"limits"` object with lowered limits (for example `{"max_decoded_bytes":
1024}`). Decoders MUST fail with exactly that class. Dependency entries provide
the bytes a resolver should return for the listed hash; for deliberately wrong
hash/length vectors the listed hash/bytes are the wrong values supplied to the
resolver.

`mode` names are `raw`, `zstd`, `reference`, `delta`, `dictionary`, `structured`.

### `structured/*.json`

Canonical WCB values and non-canonical rejects.

```json
{ "name": "map", "description": "...", "canonical_hex": "<hex>", "expect": "ok" }
{ "name": "unsorted_map_keys", "hex": "<hex>", "expect_error": "NonCanonical" }
```

### `selectors/*.json`

WeftQL source, canonical WCB bytes, and selector id.

```json
{
  "name": "1_select_true",
  "source": "select true",
  "namespace": "default",
  "canonical_hex": "<hex>",
  "selector_id": "<64 hex>"
}
```

## Notes

- WCB map ordering is by **full encoded key bytes** (tag + varint length +
  bytes), per WEFTCODEC-FORMAT §8.2. This is not lexicographic key-string
  ordering; see `../NOTES.md`.
- ZSTD and DICTIONARY frames are frozen as decode-equivalence vectors. Their
  bytes depend on the zstd library version and level; RAW, REFERENCE, DELTA, and
  STRUCTURED encodings are byte-deterministic and are re-encoded by the tests.
- Plaintext is arbitrary bytes. Nothing here implies UTF-8 except WCB text and
  selector source strings.
