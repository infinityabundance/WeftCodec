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

`mode` names are `raw`, `zstd`, `reference`, `delta`, `dictionary`,
`structured`, `tensor`, `tokens`, `quantized_tensor`, `float_tensor`,
`tensor_sharded`, `weight_delta`, `kv_cache`, and `activation_stream`.

### AI modes (`tensor`, `tokens`, `quantized_tensor`, `float_tensor`)

WEFT-AI-CODEC v1.1.0 allocates mode 6 `TENSOR`, mode 7 `TOKEN_SEQUENCE`,
mode 8 `QUANTIZED_TENSOR`, and mode 9 `FLOAT_TENSOR`. Their bodies are
`u32le(meta_len) || WCB(meta) || payload`; the frame `decoded_len` and
`content_hash` cover the decoded bytes, and the WCB descriptor binds the stored
payload under `weft/tensor/payload/v1`, `weft/tokens/payload/v1`,
`weft/quantized-tensor/payload/v1`, or `weft/float-tensor/payload/v1`.
`FLOAT_TENSOR` stores a `huffman-canonical-v1` exponent plane plus a raw
sign+mantissa plane, while `decoded_len` is the reconstructed element length.

Valid vectors: `100_tensor_f32_2x3`, `101_tensor_f16_scalar`,
`102_tensor_u8_zero_element`, `103_tokens_delta_boundaries`,
`104_tokens_empty`, `105_quantized_i8_2x3`,
`106_quantized_u8_zero_element`, `107_quantized_i8_scalar`,
`108_float_bf16_2x2`, `109_float_fp16_3d`, `110_float_fp32_4d`. Reject vectors
`120`–`135` cover shape/payload/hash/layout/dtype/overflow/rank/encoding/
canonicality/count limits; `030`–`037` cover quantized dtype, scale, zero
point, layout, payload-length, and hash rejects; `140`–`153` cover float-tensor
dtype, shape/count, rank 0, zero dimension, field widths, plane-length
overflow, unknown exponent coder, decoded-length, over-subscribed/incomplete/
over-long Huffman tables, non-zero padding, and truncated exponent streams.

`017_unknown_mode` historically asserted that mode value 6 was unknown. It is
retained byte-for-byte with `"superseded_by":
"017_unknown_mode_reserved8"`; replay suites skip superseded vectors.
`017_unknown_mode_reserved8` is retained and marked
`"superseded_by": "017_unknown_mode_reserved9"` because mode 8 is now
`QUANTIZED_TENSOR`; `017_unknown_mode_reserved9` is retained and marked
`"superseded_by": "017_unknown_mode_reserved10"` because mode 9 is now
`FLOAT_TENSOR`. `017_unknown_mode_reserved14` is retained and marked
`"superseded_by": "017_unknown_mode_reserved15"` because mode 14 is now
`SEMANTIC_DELTA`. The current reserved-value vector is
`017_unknown_mode_reserved15` (mode 15); `017_unknown_mode_reserved12` is
retained and marked superseded because mode 12 is now `KV_CACHE` (mode 13 is
`ACTIVATION_STREAM`).

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

### `context/*.json`

Canonical Context Root records (`weft.context-root.v1`), their identity, the
authoritative URI, and the descriptor spelling. Invalid vectors use
`record_hex` plus `expect_error`.

```json
{
  "name": "001_root_true",
  "namespace": "default",
  "selector_source": "select true",
  "selector_canonical_hex": "<hex>",
  "selector_id": "<64 hex>",
  "snapshot": "<64 hex>",
  "resolver": "weft-resolver-v1",
  "causal": [],
  "capability_hash": null,
  "budget": null,
  "ordering": "selector_then_object_id",
  "canonical_hex": "<hex>",
  "context_root_id": "<64 hex>",
  "uri": "weft://default/c/<64 hex>",
  "descriptor": "weft://default/c/select%20true@<64 hex>"
}
```

### `causal/*.json`

Canonical causal closure records (`weft.causal.v1`) and their closure hash.
Invalid vectors use `record_hex` plus `expect_error`.

```json
{
  "name": "001_single_parent",
  "parents": ["<64 hex>"],
  "closure": ["<64 hex>"],
  "canonical_hex": "<hex>",
  "closure_hash": "<64 hex>"
}
```

### `tensor/*.json`

WEFT-TENSOR v1 logical identity, layout descriptors, and shard containers.
These are additive records outside mode 6; the mode-6 `layout` field is
unchanged. Generated by
`cargo run -- conformance --dir conformance/v1 --tensor-only`.

Valid logical vectors carry the element bytes so a decoder can recompute every
derived value:

```json
{
  "name": "001_logical_bf16_2x3x4",
  "kind": "logical",
  "dtype": "bf16",
  "shape": [2, 3, 4],
  "element_count": 24,
  "values_hex": "<hex>",
  "values_hash": "<64 hex>",
  "canonical_hex": "<hex>",
  "logical_root": "<64 hex>"
}
```

Layout vectors use `kind: "layout"`, `dtype`, `shape`, `layout`, the
layout-specific fields, and `canonical_hex`. Shard vectors use
`kind: "shards"`, `dtype`, `shape`, `axis`, `shards`, `values_hex`,
`logical_root`, and the per-shard `shard_hex`, `shard_descriptor_hex`, and
byte-identical `shard_container_hex` arrays. Reject vectors use `record_hex`
plus `expect_error` in the same way as `structured/` and `context/`.

### `sharded/*.json`

TENSOR_SHARDED (mode 10) shard sets. Each valid file carries the whole-tensor
canonical quantized words, the global per-block scale/zero-point arrays, the
expected WeftTensor `layout_root`, and every shard frame (deliberately
shuffled; `shard_index` is authoritative). Generated by
`cargo run -- conformance --dir conformance/v1 --sharded-only`.

```json
{
  "name": "001_q8_0_rank2_2shards",
  "schema": "weft.ai.tensor_sharded.v1",
  "dtype": "q8_0",
  "shape": [4, 8],
  "element_count": 32,
  "axis": 0,
  "shards": 2,
  "block_size": 16,
  "scales": [1056964608, 1048576000],
  "zero_points": [0, 3],
  "words_hex": "<hex>",
  "layout_root": "<64 hex>",
  "frames": [ { "index": 1, "frame_hex": "<hex>" }, ... ],
  "expect": { "ok": true }
}
```

Reject files use `expect_error` and a `frames` array instead of `expect`. The
same files are replayed unchanged by both implementations.

### `kv-cache/*.json`

KV_CACHE (mode 12) page sets. Each valid file carries every page's committed
frame, element bytes, page position, and byte counts.

```json
{
  "name": "001_bf16_two_pages",
  "schema": "weft.ai.kv_cache.v1",
  "model": "llama-4-8b",
  "layer": 3,
  "session": "sess-7",
  "dtype": "bf16",
  "head_count": 2,
  "head_dim": 4,
  "page_count": 2,
  "pages": [
    {
      "page_index": 0,
      "seq_start": 0,
      "seq_end": 4,
      "page_rows": 4,
      "page_bytes": 64,
      "payload_len": 336,
      "elements_hex": "<hex>",
      "frame_hex": "<hex>"
    }
  ],
  "expect": { "ok": true }
}
```

Reject files carry `frame_hex` and `expect_error`.

### `activation-stream/*.json`

ACTIVATION_STREAM (mode 13) chunk sets. Valid files carry every chunk frame,
element bytes, per-chunk `digest_root`, and the assembled `stream_root`.

```json
{
  "name": "001_activation_bf16_two_chunks",
  "schema": "weft.ai.activation_stream.v1",
  "stream": "activation",
  "step": 7,
  "rank": 0,
  "world": 1,
  "chunk_count": 2,
  "dtype": "bf16",
  "shape": [2, 4],
  "element_count": 8,
  "chunks": [
    {
      "chunk_index": 0,
      "digest_root": "<64 hex>",
      "payload_len": 336,
      "elements_hex": "<hex>",
      "frame_hex": "<hex>"
    }
  ],
  "stream_root": "<64 hex>",
  "chunk_digests": ["<64 hex>"],
  "expect": { "ok": true }
}
```

Reject files carry `frame_hex` and `expect_error`.

## Notes

- WCB map ordering is by **full encoded key bytes** (tag + varint length +
  bytes), per WEFTCODEC-FORMAT §8.2. This is not lexicographic key-string
  ordering; see `../NOTES.md`.
- ZSTD and DICTIONARY frames are frozen as decode-equivalence vectors. Their
  bytes depend on the zstd library version and level; RAW, REFERENCE, DELTA, and
  STRUCTURED encodings are byte-deterministic and are re-encoded by the tests.
- Plaintext is arbitrary bytes. Nothing here implies UTF-8 except WCB text and
  selector source strings.
