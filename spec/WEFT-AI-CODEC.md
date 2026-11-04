# WEFT-AI-CODEC v1.1.0 — AI-native codec modes

**Status: PARTIALLY IMPLEMENTED.** Exactly three AI-native modes are normative
and implemented in both Rust `weftcodec` and Go `weftcodec-go`:

| Value | Name | Status | Frozen vectors |
|---:|---|---|---|
| 6 | `TENSOR` | **implemented** | `100_tensor_f32_2x3`, `101_tensor_f16_scalar`, `102_tensor_u8_zero_element` + rejects |
| 7 | `TOKEN_SEQUENCE` (CLI `tokens`) | **implemented** | `103_tokens_delta_boundaries`, `104_tokens_empty` + rejects |
| 8 | `QUANTIZED_TENSOR` (CLI `quantized`) | **implemented** | `105_quantized_i8_2x3`, `106_quantized_u8_zero_element`, `107_quantized_i8_scalar` + rejects |

Every other mode below is `specified, not implemented` and has **no allocated
wire value**. Nothing in this document is a compression or performance claim.

**Depends on:** `WEFTCODEC-FORMAT.md` (framing, hashing, limits, modes),
`WEFT-CONTEXT-ROOT.md`, `WEFT-CAUSAL.md`.
**Source of truth:** `spec/WEFT-AI-CODEC.md` in `infinityabundance/WeftFS`.

---

## 1. Non-negotiable constraints

Every implemented mode satisfies:

1. **Exactness.** Decode(Encode(x)) == x byte-for-byte, or the decoder fails
   closed with a typed error. No lossy fallback, no dtype conversion, no
   rounding.
2. **Self-description.** Every semantic decision (dtype, shape, layout, byte
   order, element count, token count, vocabulary schema, token encoding) is
   carried in the WCB descriptor. Nothing is inferred from byte length or
   implementation defaults.
3. **Frozen identity.** The frame's `decoded_len` is the declared payload byte
   length and `content_hash` is `H("weft/chunk/v1", payload)`. The descriptor
   additionally binds the payload under a mode-specific domain. Changing the
   descriptor without changing the payload is `Corrupt`.
4. **Cross-language equality.** Rust and Go emit byte-identical frames for the
   same payload and descriptor, and decode each other's frames. The
   cross-language gate asserts both directions and byte identity.
5. **Bounded resources.** Element count, rank, token count, descriptor size,
   payload size, WCB depth/items, and frame length are checked before
   allocation. Exceeding a limit is `LimitExceeded`.
6. **No semantic execution.** A codec decodes bytes. It does not evaluate a
   model, tokenize text, dequantize, or interpret tensor values. `f16`/`bf16`
   bit patterns, NaNs, and infinities are preserved exactly.
7. **Vectors or it did not happen.** A mode is implemented only when committed
   vectors include valid cases and adversarial rejects replayable by both
   implementations.
8. **Unknown or unimplemented descriptor values fail closed.** A decoder MUST
   reject a layout, endianness, dtype, schema, or token encoding it does not
   implement. It MUST NOT reinterpret the payload.

---

## 2. Mode 6 — `TENSOR` (implemented)

### 2.1 Frame shape

```
domain        = 0 (payload)
mode          = 6
flags         = 0 (CHUNKED, SELF_CONTAINED, DICT_RAW, ORIGINAL_HASH all clear)
dep_count     = 0
chunk_count   = 0
decoded_len   = payload byte length
content_hash  = H("weft/chunk/v1", payload)
body          = u32le(meta_len) || WCB(meta) || payload
```

`meta_len` is the length of the WCB descriptor only. The decoder checks
`4 + meta_len + decoded_len == body_len` with checked arithmetic; any mismatch
is `Malformed`. `meta_len` MUST be at most the configured string/body limits.

The payload is the exact little-endian element bytes in **row-major** order.
No normalization, endianness conversion, quantization, quantization metadata,
padding, or layout conversion is performed.

### 2.2 Descriptor

The WCB value is a map with exactly these keys and no others. Missing or extra
keys are `Malformed`.

| Key | WCB type | Value |
|---|---|---|
| `schema` | text | exactly `weft.tensor.v1` |
| `dtype` | text | one of `f16`, `bf16`, `f32`, `f64`, `i8`, `u8`, `i16`, `i32`, `i64` |
| `shape` | array of u64 | logical extents, rank 0..=8 (rank 0 is a scalar) |
| `layout` | text | exactly `row_major` in v1 |
| `endianness` | text | exactly `little` in v1 |
| `element_count` | u64 | product of `shape` (rank 0 ⇒ 1) |
| `payload_bytes` | u64 | `element_count * element_size(dtype)` |
| `payload_hash` | bytes(32) | `H("weft/tensor/payload/v1", payload)` |

Element sizes are fixed:

| dtype | bytes | dtype | bytes |
|---|---:|---|---:|
| `f16` | 2 | `i8` | 1 |
| `bf16` | 2 | `u8` | 1 |
| `f32` | 4 | `i16` | 2 |
| `f64` | 8 | `i32` | 4 |
| | | `i64` | 8 |

`shape` with zero in any dimension gives `element_count = 0` and an empty
payload. This is valid and exact. A rank-0 (`shape = []`) tensor has
`element_count = 1`.

### 2.3 Validation order (normative for error classes)

A decoder MUST perform these checks in this order and return the stated class:

1. Body split/`meta_len`/`decoded_len` consistency → `Malformed`.
2. WCB canonical decode → `NonCanonical`, `Malformed`, or `LimitExceeded`.
3. Descriptor key set, `schema` → `Malformed`.
4. `dtype` → `Malformed`; `layout` and `endianness` not implemented → `Malformed`.
5. `shape` array of u64, rank > `max_tensor_rank` → `LimitExceeded`.
6. Shape product checked (`u64` overflow) → `LimitExceeded`; product >
   `max_tensor_elements` → `LimitExceeded`.
7. `element_count != product` → `Malformed`.
8. `payload_bytes != element_count * element_size` (checked) → `Malformed` /
   `LimitExceeded` on overflow.
9. `payload_bytes != actual payload length` → `Malformed`.
10. `payload_hash` length ≠ 32 → `Malformed`; value mismatch → `Corrupt`.

### 2.4 Encoder rules

`EncodeTensor(payload, dtype, shape)` MUST:

- validate dtype, rank, shape product, and payload length exactly as above;
- emit the canonical WCB descriptor (map keys sorted by encoded key bytes, as
  WCB requires);
- not accept a payload whose length is a non-multiple of the element size;
- be deterministic: the same payload/dtype/shape yields identical frame bytes
  in Rust and Go.

### 2.5 CLI (maintainer tool, not a new wire format)

```
weftcodec encode --mode tensor --input PAYLOAD --output FRAME \
    (--dtype f32 --shape 2,3 | --metadata META.json)

META.json := {"dtype":"f32","shape":[2,3]}
```

An omitted `--shape` means rank 0. `layout` and `endianness` may appear in
`META.json` only as `row_major` and `little`; other values are rejected before
encoding. `decode` returns the exact payload bytes. `--shape ""` is the same as
omitting it.

---

## 3. Mode 7 — `TOKEN_SEQUENCE` (implemented)

### 3.1 Frame shape

Identical to §2.1, with `mode = 7` and the token descriptor/payload below.
`decoded_len` is the **encoded token payload** length and `content_hash` is
`H("weft/chunk/v1", payload)`.

### 3.2 Descriptor

| Key | WCB type | Value |
|---|---|---|
| `schema` | text | exactly `weft.tokens.v1` |
| `vocab_schema` | text | non-empty vocabulary/tokenizer schema identifier |
| `encoding` | text | exactly `u64-zigzag-delta-leb128` in v1 |
| `token_count` | u64 | number of token ids |
| `payload_bytes` | u64 | encoded payload byte length |
| `payload_hash` | bytes(32) | `H("weft/tokens/payload/v1", payload)` |

The codec does not interpret text, tokenizer tables, or vocabulary content.
`vocab_schema` is a declaration carried for the consumer; no lookup or remap
occurs. Binding an actual vocabulary artifact is a future dependency mode and
is `specified, not implemented`.

### 3.3 Payload encoding — exact definition

Token ids are unsigned 64-bit integers. For a sequence `t[0..n)`:

- `n = 0` ⇒ the payload is empty.
- `t[0]` is emitted as the minimal unsigned LEB128 encoding of `t[0]`.
- For `i > 0`, let
  `delta = t[i] - t[i-1]` computed in **wrapping 64-bit unsigned arithmetic**
  (equivalently, the two's-complement bit pattern reinterpreted as `i64`).
  Emit `zigzag64(delta)` as minimal unsigned LEB128, where
  `zigzag64(x) = ((x << 1) ^ (x >> 63))` on the 64-bit pattern.
- Decoding reverses this exactly:
  `t[0] = varint[0]`,
  `t[i] = t[i-1] + wrap64(zigzag_decode(varint[i]))`.
- Every varint MUST be minimal (§1 of `WEFTCODEC-FORMAT.md`). Non-minimal is
  `NonCanonical`.
- After decoding exactly `token_count` varints, the cursor MUST equal
  `payload_bytes`; trailing bytes are `Malformed`, and a truncated payload is
  `Malformed`.
- `token_count` MUST be ≤ `max_tokens`. It also MUST be ≤ `payload_bytes`
  because every token consumes at least one byte; this bounds allocation before
  it happens.
- The transform is a bijection over all `u64` ids, including `0` and
  `u64::MAX`. Decrease and sign-crossing sequences round-trip exactly. Golden
  vector `103_tokens_delta_boundaries` freezes the zero, `u64::MAX`, decrease,
  and repeat boundaries.

### 3.4 Validation order (normative for error classes)

1. Body split/`meta_len`/`decoded_len` consistency → `Malformed`.
2. WCB canonical decode → `NonCanonical`, `Malformed`, or `LimitExceeded`.
3. Descriptor key set, `schema`, non-empty `vocab_schema` → `Malformed`.
4. `encoding` not implemented → `Malformed`.
5. `token_count > max_tokens` → `LimitExceeded`.
6. `payload_bytes != actual payload length` → `Malformed`.
7. `payload_hash` length ≠ 32 → `Malformed`; value mismatch → `Corrupt`.
8. Payload decode: non-minimal varint → `NonCanonical`; truncation, count/byte
   mismatch, trailing bytes → `Malformed`; count > payload bytes → `Malformed`.

### 3.5 Attention masks — `specified, not implemented`

A mask would need an exact bit order, byte length, and padding rule. It has no
wire representation in v1 and MUST NOT be inferred from `token_count`. A future
version may add it as an explicit field (for example `mask_bits` plus
`mask_bytes`) with its own vectors. Until then, an object carrying a mask field
is rejected as having an unknown descriptor key.

### 3.6 CLI input format (maintainer tool, not a new wire format)

```
weftcodec encode --mode tokens --input TOKENS.json --output FRAME

TOKENS.json := {"vocab_schema":"weft.test.vocab.v1","token_ids":[1,2,3]}
```

`token_ids` are JSON unsigned integers in `0..=u64::MAX`; JSON negative values,
fractions, strings, or missing keys are rejected. The decoder returns the
canonical encoded payload bytes; token-id extraction is an API operation
(`decode_token_payload` / `DecodeTokenPayload`), not a wire mode.

---

## 4. Mode 8 — `QUANTIZED_TENSOR` (implemented)

### 4.1 Frame shape

Identical to §2.1, with `mode = 8` and the descriptor/payload below.
`decoded_len` is the packed element payload length and `content_hash` is
`H("weft/chunk/v1", payload)`. No dependencies or chunks are permitted.

### 4.2 Descriptor

The WCB descriptor is a map with exactly these keys:

| Key | WCB type | Value |
|---|---|---|
| `schema` | text | exactly `weft.quantized-tensor.v1` |
| `dtype` | text | exactly `i8` or `u8` in v1 |
| `shape` | array of u64 | logical extents, rank 0..=8 |
| `layout` | text | exactly `row_major` in v1 |
| `quantization` | text | exactly `affine-asymmetric-per-tensor` in v1 |
| `scale_bits_le` | bytes(8) | the exact f64 IEEE-754 bit pattern, little-endian |
| `zero_point` | i64 | affine zero point; must fit the dtype range |
| `element_count` | u64 | product of `shape` (rank 0 ⇒ 1) |
| `payload_bytes` | u64 | `element_count * 1` |
| `payload_hash` | bytes(32) | `H("weft/quantized-tensor/payload/v1", payload)` |

The payload is the packed `i8`/`u8` element bytes in row-major order. The codec
never dequantizes, converts dtype, or interprets the affine parameters.
`scale_bits_le` is carried as bytes so the f64 bit pattern is preserved exactly
(including `-0.0` bits if they could occur); `scale` MUST be finite and > 0.
`zero_point` MUST be in `-128..=127` for `i8` and `0..=255` for `u8`.

### 4.3 Validation order

1. Body split/`meta_len`/`decoded_len` consistency → `Malformed`.
2. WCB canonical decode → `NonCanonical`, `Malformed`, or `LimitExceeded`.
3. Descriptor key set, `schema` → `Malformed`.
4. `dtype` not `i8`/`u8` → `Malformed`; `layout` or `quantization` not
   implemented → `Malformed`.
5. `scale_bits_le` length ≠ 8 or non-finite/non-positive scale → `Malformed`.
6. `zero_point` type ≠ i64 or value outside the dtype range → `Malformed`.
7. `shape` rank/product overflow or product > `max_tensor_elements` →
   `LimitExceeded`.
8. `element_count` ≠ shape product → `Malformed`.
9. `payload_bytes` ≠ `element_count` (one-byte elements) → `Malformed`.
10. `payload_bytes` ≠ actual payload length → `Malformed`.
11. `payload_hash` length ≠ 32 → `Malformed`; value mismatch → `Corrupt`.

### 4.4 CLI

```
weftcodec encode --mode quantized --input PAYLOAD --output FRAME \
    --dtype i8 --shape 2,3 --scale 0.25 --zero-point -3
```

`--metadata META.json` accepts the same fields
(`{"dtype":"i8","shape":[2,3],"scale":0.25,"zero_point":-3}`). `--scale` is
required in flag mode. An omitted `--shape` is rank 0; `--zero-point` defaults
to 0.

---

## 5. Specified, not implemented (no wire values)

These designs have **no allocated mode value and no wire representation**.
They are recorded so a future version cannot silently invent them by reusing a
reserved value or an existing descriptor field.

| Family | Status | Notes |
|---|---|---|
| Tensor `column_major` layout | specified, not implemented | Do not infer from shape; reject the descriptor value |
| Tensor `blocked` layout | specified, not implemented | No block extents/padding wire form |
| Tensor `sharded` layout | specified, not implemented | No shard index/count/axis wire form; no assembly semantics |
| Tensor `quantized` storage beyond affine per-tensor i8/u8 | specified, not implemented | i4 nibble packing, per-axis scales, and dequantization are out of codec scope |
| Activation/gradient streams | specified, not implemented | No stream descriptor or element-hash list wire form |
| Embedding / KV-cache pages | specified, not implemented | No layer/head/position wire form |
| Vocabulary artifact binding | specified, not implemented | `vocab_schema` is a declaration only; no dependency hash field |
| Attention masks | specified, not implemented | No mask bits/bytes field |
| Weight deltas / model deltas | specified, not implemented as a typed mode | The generic `DELTA`/`DICTIONARY` modes already carry exact byte deltas against a dependency; there is no dtype/shape-checked weight-delta descriptor |
| Semantic/structured deltas | specified, not implemented | No canonical patch stream or base-hash field |

A consumer MUST NOT persist an object in any of these forms under a WeftCodec
mode. A future implementation requires a new `weftcodec-format` version, frozen
vectors, and the same cross-language gates.

---

## 6. Limits (added by this document)

| Limit | Default | Applies to |
|---|---:|---|
| `max_tensor_elements` | 4,294,967,296 | TENSOR shape product |
| `max_tensor_rank` | 8 | TENSOR shape length |
| `max_tokens` | 16,777,216 | TOKEN_SEQUENCE `token_count` |

Existing limits also apply: `max_decoded_bytes` (payload), `max_frame_bytes`,
`max_structured_depth`, `max_structured_items`, `max_string_bytes` (descriptor
and vocab schema), and `max_dependency_*` (frames have no dependencies, so the
header rules reject them first).

---

## 7. Acceptance tests (both implementations)

The committed suites and vectors cover:

1. **Round-trip exactness** for every dtype, scalar (rank 0), 1-D, 2-D
   non-square, 4-D, 5-D, zero-element, and maximum-rank (8) shapes.
2. **Determinism**: encoding a decoded payload reproduces identical frame bytes.
3. **Descriptor binding**: wrong dtype, shape, layout, endianness, element
   count, payload length, and payload hash are rejected with the classes above;
   a payload bound under the other AI mode's domain is `Corrupt`.
4. **Checked arithmetic**: `u64` shape overflow and element-count overflow are
   `LimitExceeded`.
5. **Token boundaries**: empty, one token, `0`, `u64::MAX`, increases,
   decreases, repeats, sign crossings, and a 4,096-token mixed sequence.
6. **Token rejects**: count mismatch, non-minimal varint, trailing bytes,
   unsupported encoding, hash mismatch, and count limit.
7. **Resource limits**: rank, element count, token count, and payload limits
   fail closed with `LimitExceeded`.
8. **Quantized tensors**: i8 and u8 round-trips, scalar/zero-element shapes,
   exact f64 scale bit preservation, dtype-range zero-point checks, non-finite/
   non-positive scale rejection, payload/hash/length rejects, and checked shape
   limits.
9. **Cross-language**: Rust→Go and Go→Rust decode equivalence plus
   byte-identical frames for all three modes (`WeftCodec-Go/hack/cross.sh`).

Frozen vector names live in both codec repositories under
`conformance/v1/codec/`. The Rust generator is
`cargo run -- conformance --dir conformance/v1`; the Go suite replays the same
files without modification.

---

## 8. Explicit non-goals and non-claims

- No training, inference, gradient computation, tokenization, dequantization,
  layout conversion, or model evaluation.
- No lossy compression claim. Any lossy storage is a different product and must
  not use `weft/chunk/v1` content identity for the lossy form.
- No cross-vendor dtype equivalence. `bf16`, `f16`, and `f32` are distinct.
- No claim that TENSOR or TOKEN_SEQUENCE compresses better than ZSTD or DELTA.
  F7 measures them and reports losses.
- No claim of production data or Meta data. The F7 corpus is synthetic and its
  manifest records that.
