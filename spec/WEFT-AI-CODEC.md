# WEFT-AI-CODEC v1.2.0 — AI-native codec modes

**Status: PARTIALLY IMPLEMENTED.** Exactly ten AI-native modes are normative
and implemented in both Rust `weftcodec` and Go `weftcodec-go`:

| Value | Name | Status | Frozen vectors |
|---:|---|---|---|
| 6 | `TENSOR` | **implemented** | `100_tensor_f32_2x3`, `101_tensor_f16_scalar`, `102_tensor_u8_zero_element` + rejects |
| 7 | `TOKEN_SEQUENCE` (CLI `tokens`) | **implemented** | `103_tokens_delta_boundaries`, `104_tokens_empty` + rejects |
| 8 | `QUANTIZED_TENSOR` (CLI `quantized`) | **implemented** | `105_quantized_i8_2x3`, `106_quantized_u8_zero_element`, `107_quantized_i8_scalar` + rejects |
| 9 | `FLOAT_TENSOR` (CLI `float-tensor`) | **implemented** | `108_float_bf16_2x2`, `109_float_fp16_3d`, `110_float_fp32_4d` + rejects |
| 10 | `TENSOR_SHARDED` (CLI `tensor-sharded`) | **implemented** | `conformance/v1/sharded/001`–`003` + eight rejects |
| 11 | `WEIGHT_DELTA` (CLI `weight-delta`) | **implemented** | `conformance/v1/weight-delta/001`–`003` + eight rejects |
| 12 | `KV_CACHE` (CLI `kv-cache`) | **implemented** | `conformance/v1/kv-cache/001`–`003` + eight rejects |
| 13 | `ACTIVATION_STREAM` (CLI `activation-stream`) | **implemented** | `conformance/v1/activation-stream/001`–`003` + eight rejects |
| 14 | `SEMANTIC_DELTA` (CLI `semantic-delta`) | **implemented** | `conformance/v1/semantic-delta/001`–`003` + eight rejects |
| 15 | `TENSOR_TILED` (CLI `tensor-tiled`) | **implemented** | `conformance/v1/tensor-tiled/001`–`003` + eight rejects |

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

## 5. Mode 9 — `FLOAT_TENSOR` (implemented)

### 5.1 Frame shape

Identical to §2.1, with `mode = 9`. The body is

```
u32le(meta_len) || WCB(weft.ai.float_tensor.v1) || exponent_plane || mantissa_plane
```

`decoded_len` is the length of the **reconstructed element bytes**, not the
stored payload length: `element_count * (sign_bits + exponent_bits +
mantissa_bits) / 8`. `content_hash` is `H("weft/chunk/v1", reconstructed
bytes)`, so decoding yields the original little-endian element bytes exactly.
Frame `domain` is `0`; no dependencies and no chunks are permitted.

The descriptor additionally binds the stored payload under
`weft/float-tensor/payload/v1` (§1.3). Encoding the exponent plane therefore
never changes element identity, and swapping planes or descriptors is caught.
The frame `level` field is 0 for this mode because no zstd frame is used.

### 5.2 Descriptor

The WCB descriptor is a map with exactly these keys:

| Key | WCB type | Value |
|---|---|---|
| `schema` | text | exactly `weft.ai.float_tensor.v1` |
| `dtype` | text | one of `bf16`, `fp16`, `fp32`, `fp8-e4m3`, `fp8-e5m2` |
| `shape` | array of u64 | rank 1..=`max_tensor_rank`; every dimension > 0 |
| `element_count` | u64 | product of `shape` |
| `exponent_bits` | u64 | fixed per dtype, `0..=255` (WCB has no u8 tag) |
| `mantissa_bits` | u64 | fixed per dtype, `0..=255` |
| `sign_bits` | u64 | fixed per dtype, `0..=255` |
| `exponent_coder` | text | exactly `huffman-canonical-v1` |
| `exponent_plane_len` | u64 | length of the **huffman-canonical-v1** exponent plane (256-byte table + bitstream) |
| `mantissa_plane_len` | u64 | length of the raw sign+mantissa plane |
| `payload_hash` | bytes(32) | `H("weft/float-tensor/payload/v1", stored payload)` |

WCB has no unsigned-byte tag, so the three bit-width fields are carried as u64
values restricted to `0..=255`. The fixed widths are:

| `dtype` | `sign_bits` | `exponent_bits` | `mantissa_bits` | element bytes |
|---|---:|---:|---:|---:|
| `bf16` | 1 | 8 | 7 | 2 |
| `fp16` | 1 | 5 | 10 | 2 |
| `fp32` | 1 | 8 | 23 | 4 |
| `fp8-e4m3` | 1 | 4 | 3 | 1 |
| `fp8-e5m2` | 1 | 5 | 2 | 1 |

All five dtypes are byte-aligned (`sign_bits + exponent_bits + mantissa_bits` is
a multiple of 8). A dtype that is not byte-aligned is not implemented and MUST
be rejected.

### 5.3 Payload encoding — exact definition

Input elements are the little-endian element words in row-major order. For
dtype `E = exponent_bits`, `M = mantissa_bits`, `S = sign_bits`:

- **Exponent plane.** `element_count` bytes; byte `i` is element `i`'s exponent
  field, in element order, and MUST be `< 2^E`. This plane is encoded with
  `huffman-canonical-v1` (§5.3.1); `exponent_plane_len` is the encoded length.
  The uncompressed exponent-plane length is `element_count` and is carried by
  the descriptor. The encoder always applies the canonical coder; no zstd
  frame is used for this mode.
- **Mantissa plane.** For each element in order, emit the sign bit (`S` bits)
  followed by the mantissa bits (`M` bits, most significant first). Pack these
  `element_count * (S + M)` bits contiguously, MSB-first: bit 7 of the first
  byte is the first emitted bit. The plane is exactly
  `ceil(element_count * (S + M) / 8)` bytes; unused low bits of the final byte
  MUST be zero. The plane is stored raw.

#### 5.3.1 `huffman-canonical-v1`

This is an order-0 canonical Huffman code over the 256 possible exponent-byte
values. It is normative because Rust and Go MUST emit bit-identical bytes.

**Serialized form.**

```
u8 code_lengths[256]   # 0 = symbol absent
|| bitstream           # MSB-first, zero-padded to a byte boundary
```

**Encoder.**

1. Count byte frequencies over the exponent plane.
2. Build a binary Huffman tree. When selecting the two least-weighted active
   nodes, compare `(weight, minimum symbol byte, node index)` ascending;
   leaves are inserted in ascending symbol order, so the final index fallback
   is deterministic. A parent's minimum symbol is the minimum of its children.
3. Sort the used symbols by `(code length, symbol)` and assign increasing
   canonical codes, starting at 0 and shifting left by the code-length delta.
   Codes are written most-significant bit first.
4. If any code length exceeds 24 bits, replace every non-zero frequency `f`
   with `max(1, f >> 1)` and rebuild the tree; repeat until every length is at
   most 24. Equal weights produce a tree of depth at most 9 for 256 symbols,
   so this procedure terminates.
5. Emit the 256 code lengths as unsigned bytes followed by the bitstream. The
   final byte is zero-padded. The uncompressed length is `element_count`.

**Canonical edge cases.**

- Empty plane: all 256 lengths are 0 and the bitstream is empty. This is valid
  only when the declared plane length is 0.
- Exactly one used symbol: its length is 1 and the bitstream is empty; the
  decoder emits the declared number of copies. A length-0 code is never
  emitted, so the form is total.

**Decoder.** Rebuild the canonical table from the 256 lengths and reject, in
this order:

1. a length greater than 24 → `NonCanonical`;
2. an over-subscribed or incomplete prefix table (`Kraft` sum with 24-bit
   resolution not exactly `2^24`) → `NonCanonical`, except the single-symbol
   form above;
3. a truncated bitstream (no code before the bits are exhausted) → `Corrupt`;
4. extra trailing bytes, or padding bits that are not zero → `Corrupt`;
5. an output length other than the declared `element_count` → `Corrupt`.

The decoder never allocates more than the declared plane length and fails
closed on every condition above. Both implementations use exactly this class
mapping. (The exponent plane never has an empty declared length for a valid
`FLOAT_TENSOR`, because every shape dimension is greater than zero; the empty
form exists for the coder's own tests and for future callers.)

Decoding reverses this exactly: element `i`'s word is
`(sign << (E + M)) | (exponent << M) | mantissa`, written as an `S+E+M`-bit
little-endian word. Every bit pattern round-trips, including NaNs with non-zero
payloads, signalling/quiet bits, infinities, signed zeros, and denormals. The
codec never converts, canonicalizes, or interprets a value.

### 5.4 Validation order (normative for error classes)

1. Body split / `meta_len` bounds → `Malformed`.
2. WCB canonical decode → `NonCanonical`, `Malformed`, or `LimitExceeded`.
3. Descriptor key set, `schema`, and `exponent_coder` → `Malformed`.
4. Unknown `dtype` → `Malformed`; a bit-width field that is not `0..=255` or
   disagrees with the dtype table → `Malformed`; a non-byte-aligned total →
   `Malformed`.
5. `shape` rank 0 or any dimension 0 → `Malformed`; rank > `max_tensor_rank`,
   product overflow, or product > `max_tensor_elements` → `LimitExceeded`.
6. `element_count` ≠ shape product → `Malformed`.
7. `decoded_len` arithmetic overflow or `> max_decoded_bytes` →
   `LimitExceeded`; `decoded_len` ≠ `element_count * element_bytes` →
   `Corrupt`.
8. `mantissa_plane_len` ≠ `ceil(element_count * (S + M) / 8)` → `Malformed`;
   `exponent_plane_len + mantissa_plane_len` overflow → `LimitExceeded`;
   the sum ≠ stored payload length → `Malformed`.
9. The encoded exponent plane shorter than its 256-byte table, a truncated
   bitstream, extra trailing bytes, or non-zero padding → `Corrupt`; a code
   length > 24 or an over-subscribed or incomplete table → `NonCanonical`
   (§5.3.1).
10. An exponent byte ≥ `2^E` → `Corrupt`; non-zero mantissa padding bits →
    `Corrupt`.
11. `payload_hash` length ≠ 32 → `Malformed`; value mismatch → `Corrupt`.

### 5.5 CLI

```
weftcodec encode --mode float-tensor --input PAYLOAD --output FRAME \
    --dtype fp32 --shape 2,3
```

`--metadata META.json` accepts `{"dtype":"fp32","shape":[2,3]}`. `--shape`
is required (rank ≥ 1); an omitted or empty `--shape` is a rank-0 reject.
`inspect` reports `mode=float_tensor` and `exponent_coder=huffman-canonical-v1`
(JSON: `"exponent_coder"`).

---

## 6. Specified, not implemented (no wire values)

These designs have **no allocated mode value and no wire representation**.
They are recorded so a future version cannot silently invent them by reusing a
reserved value or an existing descriptor field.

| Family | Status | Notes |
|---|---|---|
| Tensor `column_major` layout | specified, not implemented in mode 6 | Mode 6 remains `row_major`; the additive layout layer is WEFT-TENSOR §4 |
| Tensor `blocked` layout | specified, not implemented in mode 6 | No block extents in mode 6; WEFT-TENSOR §4 defines the additive block form |
| Tensor `sharded` layout | specified, not implemented in mode 6 | No shard fields in mode 6; WEFT-TENSOR §4–§6 defines shards and containers |
| Tensor `quantized` storage beyond affine per-tensor i8/u8 | implemented as additive mode 10 for per-block q8_0/q4_0 | Mode 10 (§11) implements i4 nibble packing and per-block scales; dequantization remains out of codec scope |
| Activation/gradient streams | specified, not implemented | No stream descriptor or element-hash list wire form |
| Embedding / KV-cache pages | specified, not implemented | No layer/head/position wire form |
| Vocabulary artifact binding | specified, not implemented | `vocab_schema` is a declaration only; no dependency hash field |
| Attention masks | specified, not implemented | No mask bits/bytes field |
| Weight deltas / model deltas | **implemented as mode 11** | `WEIGHT_DELTA` (§12) carries a dtype/shape-checked, dependency-linked word delta against a base tensor frame |
| Semantic/structured deltas | specified, not implemented | No canonical patch stream or base-hash field |

A consumer MUST NOT persist an object in any of these forms under a WeftCodec
mode. A future implementation requires a new `weftcodec-format` version, frozen
vectors, and the same cross-language gates.

---

## 7. Limits (added by this document)

| Limit | Default | Applies to |
|---|---:|---|
| `max_tensor_elements` | 4,294,967,296 | TENSOR/FLOAT_TENSOR/TENSOR_SHARDED/WEIGHT_DELTA shape product |
| `max_tensor_rank` | 8 | TENSOR/FLOAT_TENSOR/TENSOR_SHARDED/WEIGHT_DELTA shape length |
| `max_tokens` | 16,777,216 | TOKEN_SEQUENCE `token_count` |
| `max_structured_items` | 1,000,000 | TENSOR_SHARDED `scales`/`zero_points` block arrays |

Existing limits also apply: `max_decoded_bytes` (payload), `max_frame_bytes`,
`max_structured_depth`, `max_structured_items`, `max_string_bytes` (descriptor
and vocab schema), and `max_dependency_*` (frames have no dependencies, so the
header rules reject them first).

---

## 8. Acceptance tests (both implementations)

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
9. **Float tensors**: every dtype round-trips bit-exactly; the
   `huffman-canonical-v1` exponent-plane binding (empty plane, one symbol, all
   256 symbols, skewed, uniform random, 2,000,000 bytes, and length-limited
   Fibonacci weights); mantissa MSB-first packing and zero padding; NaN
   payloads, infinities, signed zeros, and denormals; unknown dtype, bit-width
   mismatch, rank 0, zero dimension, plane-length overflow, unknown exponent
   coder, decoded-length mismatch, over-subscribed/incomplete/over-long
   Huffman tables, non-zero padding, and truncated streams; element/rank/
   decoded-byte limits.
10. **Cross-language**: Rust→Go and Go→Rust decode equivalence plus
   byte-identical frames for TENSOR, TOKEN_SEQUENCE, QUANTIZED_TENSOR,
   FLOAT_TENSOR, and TENSOR_SHARDED (`WeftCodec-Go/hack/cross.sh`).
   `FLOAT_TENSOR` and `TENSOR_SHARDED` byte identity is mandatory because both
   use the deterministic `huffman-canonical-v1` coder, not a library zstd
   frame. The frozen vectors are generated once and replayed by both
   implementations.
11. **TENSOR_SHARDED**: q8_0 rank-2 and q4_0 rank-3 shard sets round-trip
    exactly; single-shard q8_0; nibble order; shard sets accepted in any order;
    duplicate/missing/out-of-range indices rejected; sharding axis/count do not
    change `layout_root`; unknown dtype, scale-plane mismatch, wrong element
    count, invalid block size, non-divisible axis, out-of-range index, and a
    malformed Huffman stream rejected with the classes of §11.4; rank, element,
    block-array, and plane-length limits fail closed.
12. **WEIGHT_DELTA**: bf16, fp16, fp32, q8_0, and q4_0 round-trip bit-exactly
    (including NaNs, denormals, and signed zeros); an all-zero delta
    reconstructs exactly; base dtype and shape mismatches are
    `DependencyMismatch`; a missing base is `DependencyMissing`; a cyclic base
    is `DependencyCycle`; unknown `op`, inconsistent `word_bits`/`planes`,
    `payload_len` mismatch, and descriptor key/schema errors are `Malformed`;
    a wrong `delta_root` is `Corrupt`; a 4 MiB bf16 pair differing in 1% of
    elements is byte-exact and its logical root matches the original.

Frozen vector names live in both codec repositories under
`conformance/v1/codec/`. The Rust generator is
`cargo run -- conformance --dir conformance/v1`; the Go suite replays the same
files without modification.

---

## 9. Explicit non-goals and non-claims

- No training, inference, gradient computation, tokenization, dequantization,
  layout conversion, or model evaluation.
- No lossy compression claim. Any lossy storage is a different product and must
  not use `weft/chunk/v1` content identity for the lossy form.
- No cross-vendor dtype equivalence. `bf16`, `f16`, and `f32` are distinct.
- No claim that TENSOR or TOKEN_SEQUENCE compresses better than ZSTD or DELTA.
  F7 measures them and reports losses.
- No claim of production data or Meta data. The F7 corpus is synthetic and its
  manifest records that.

---

## 10. Logical tensor identity and region addressing

WEFT-TENSOR v1 (`spec/WEFT-TENSOR.md`) defines the additive layer that makes
mode-6 tensor identity independent of physical layout:

```
values_hash       = H("weft/tensor/values/v1",  row-major element bytes)
LogicalTensorRoot = H("weft/tensor/logical/v1", WCB(weft.tensor.logical.v1))
```

It additionally defines canonical `weft.tensor.layout.v1` descriptors for
`row-major`, `column-major`, `blocked`, and `sharded` arrangements, a half-open
region reader that touches only intersecting physical units, and the
`weft.tensor.shard.v1` container with `shard_root` verification.

The layer is **additive**: it introduces new schema strings outside mode 6 and
does not alter any mode-6 encoding or frozen vector. Mode-6 `TENSOR` frames
continue to require `layout: "row_major"` and remain the frame format for
single-region canonical payloads. See WEFT-TENSOR §1 and §10 for the exact
boundary.

---

## 11. Mode 10 — `TENSOR_SHARDED` (implemented)

Mode 10 carries one shard of a quantized tensor. The descriptor names the whole
logical shape, the shard position, the global quantization blocks, and the
WeftTensor `LogicalTensorRoot` of the unsharded tensor. Decoding reconstructs
the exact quantized words and scale bits; dequantization is out of band.

### 11.1 Frame shape

Identical to §2.1, with `mode = 10`. The body is

```
u32le(meta_len) || WCB(weft.ai.tensor_sharded.v1) || huffman_stream
```

`decoded_len` is the length of the reconstructed planes `P` (§11.4), not the
stored Huffman stream length. `content_hash` is `H("weft/chunk/v1", P)`.
Frame `domain` is `0`; no dependencies and no chunks are permitted; `level` is
`0`. The descriptor additionally binds the stored stream under

```
H("weft/tensor-sharded/payload/v1", huffman_stream)
```

so swapping the encoded stream without changing the descriptor is `Corrupt`.

### 11.2 Descriptor

The WCB descriptor is a map with exactly these keys:

| Key | WCB type | Value |
|---|---|---|
| `schema` | text | exactly `weft.ai.tensor_sharded.v1` |
| `dtype` | text | exactly `q8_0` or `q4_0` in v1 |
| `shape` | array of u64 | whole-tensor extents, rank 1..=`max_tensor_rank`; every dimension > 0 |
| `element_count` | u64 | product of `shape` |
| `shard_axis` | u64 | axis being split, `0 <= shard_axis < rank` (u32 range) |
| `shard_count` | u64 | `1..=u32::MAX`; MUST divide `shape[shard_axis]` |
| `shard_index` | u64 | `0 <= shard_index < shard_count` (u32 range) |
| `block_size` | u64 | exactly 16, 32, 64, 128, or 256 |
| `scales` | array of u64 | one per global block; low 32 bits are the exact f32 bit pattern, high 32 bits MUST be zero |
| `zero_points` | array of i64 | one per global block, in block order |
| `layout_root` | bytes(32) | the WeftTensor `LogicalTensorRoot` of the whole tensor (§11.5) |
| `payload_len` | u64 | length of the stored `huffman-canonical-v1` stream |
| `payload_hash` | bytes(32) | `H("weft/tensor-sharded/payload/v1", stored stream)` |

`WCB` has no `u32` tag, so `shard_axis`, `shard_count`, and `shard_index` are
carried as `u64` restricted to the `u32` range.

Derived quantities (never on the wire):

```
shard_elements = element_count / shard_count
block_count    = element_count / block_size
```

Requirements: `element_count % block_size == 0`; `shard_count` divides
`shape[shard_axis]`, so `element_count % shard_count == 0`; `block_count >=
scales.len() == zero_points.len()`; and every scale is a **finite positive
f32**. The scale/zero-point arrays are global: every shard of the same tensor
MUST carry identical arrays. A block `b` covers the global canonical row-major
element indices `[b * block_size, (b + 1) * block_size)`.

### 11.3 Quantized values and planes

Quantized values are exact integers. The codec never converts, rounds, or
deinterprets them.

- `q8_0`: one signed 8-bit value per element, two's complement, range
  `-128..=127`.
- `q4_0`: one signed 4-bit value per element, two's complement, range
  `-8..=7`. Two values are packed per byte; the even element is the **low**
  nibble and the odd element is the **high** nibble. A `q4_0` shard therefore
  requires an even `shard_elements`; otherwise the descriptor is `Malformed`.

For shard `j`, the quantized plane is the shard's `shard_elements` values in
**shard-local canonical row-major order** (WEFT-TENSOR §4.2), packed as above.
The scale plane is the `block_count` exact f32 scale bits, each as 4 bytes
little-endian, in global block order. Thus the reconstructed plane is

```
P = quantized plane || scale plane
```

and its length is

```
q8_0: shard_elements + 4 * block_count
q4_0: shard_elements / 2 + 4 * block_count
```

The stored stream is `huffman-canonical-v1(P)` (§5.3.1), including its 256-byte
code-length table. `payload_len` is the stored length and `decoded_len` is
`len(P)`. The decoder MUST verify that the scale plane is byte-identical to the
`scales` array (`Corrupt` otherwise). It reconstructs the quantized words and
scale bits exactly; no dequantization is part of the codec or of identity.

### 11.4 Validation order (normative for error classes)

1. Body split / `meta_len` bounds → `Malformed`.
2. WCB canonical decode → `NonCanonical`, `Malformed`, or `LimitExceeded`.
3. Descriptor key set, `schema` → `Malformed`.
4. Unknown `dtype` → `Malformed`.
5. `shape` rank 0 or zero dimension → `Malformed`; rank > `max_tensor_rank`,
   product overflow, or product > `max_tensor_elements` → `LimitExceeded`.
6. `element_count` ≠ shape product → `Malformed`.
7. `shard_axis` ≥ rank, `shard_count` = 0, `shard_index` ≥ `shard_count`, or
   `shape[shard_axis] % shard_count ≠ 0` → `Malformed`.
8. `block_size` not in `{16,32,64,128,256}` → `Malformed`; `element_count %
   block_size ≠ 0` → `Malformed`; `block_count` > `max_structured_items` →
   `LimitExceeded`.
9. `scales` or `zero_points` length ≠ `block_count` → `Malformed`; a scale
   value above `u32::MAX`, a non-finite f32, or a non-positive scale →
   `Malformed`.
10. `q4_0` with odd `shard_elements` → `Malformed`.
11. `payload_len` ≠ actual stored length → `Malformed`; computed plane length
    overflow or above `max_decoded_bytes` → `LimitExceeded`.
12. `decoded_len` ≠ computed plane length → `Corrupt`.
13. Huffman stream rejects → the classes of §5.3.1.
14. Scale plane ≠ descriptor `scales` → `Corrupt`.
15. `content_hash` ≠ `H("weft/chunk/v1", P)` → `Corrupt`.
16. `payload_hash` length ≠ 32 → `Malformed`; value mismatch → `Corrupt`.
17. `layout_root` length ≠ 32 → `Malformed`. A single shard cannot verify the
    root; `shard_set_root` does (§11.6).

### 11.5 Logical root

The root is the WeftTensor `LogicalTensorRoot` (WEFT-TENSOR §3) applied to the
quantized tensor, with the additive quantized dtypes `q8_0` and `q4_0`. Let

- `W` be the whole-tensor canonical row-major sequence of quantized values,
  each **sign-extended to one `u8` byte**: for `q8_0`, the `i8` two's-complement
  byte; for `q4_0`, the `i4` value sign-extended (for example `-8` → `0xF8`).
- `B` be the block records in block order:
  `u32le(scales[b]) || i64le(zero_points[b])`.

Then

```
values_hash = H("weft/tensor/values/v1", W || B)
record      = { "schema": "weft.tensor.logical.v1",
                "dtype":  "q8_0" | "q4_0",
                "shape":  shape,
                "element_count": element_count,
                "values_hash": values_hash }
layout_root = H("weft/tensor/logical/v1", WCB(record))
```

The record uses the canonical WCB map encoding and key ordering of
WEFT-TENSOR §3. Because `W` and `B` are defined over the whole canonical
tensor, `layout_root` does **not** depend on `shard_axis`, `shard_count`, or
`shard_index`. Two artifacts that describe the same quantized tensor with
different sharding produce the same root. Resharding therefore need not change
logical identity.

Zero points are carried exactly and are part of the root, but this document
defines no dequantization rule. A consumer that needs floats applies its own
out-of-band formula; that formula is not part of WeftCodec identity.

### 11.6 Shard-set verification

`shard_set_root(shards)` accepts decoded mode-10 shards in any order and:

1. requires at least one shard (`Malformed` for an empty set);
2. requires every descriptor to agree on `dtype`, `shape`, `element_count`,
   `shard_axis`, `shard_count`, `block_size`, `scales`, `zero_points`, and
   `layout_root` (`NonCanonical` otherwise);
3. requires exactly `shard_count` entries whose indices form a permutation of
   `0..shard_count` (`Malformed` for a wrong count; `NonCanonical` for a
   duplicate, missing, or out-of-range index);
4. reassembles `W` by placing each shard's slab into its logical position
   (WEFT-TENSOR §4.2–§4.3), computes the root above, and requires every
   shard's declared `layout_root` to equal it (`Corrupt` otherwise).

The shard order in the input is not significant: `shard_index` is
authoritative. Both implementations reorder deterministically by index.

### 11.7 Limits

`max_tensor_rank` and `max_tensor_elements` bound the whole shape.
`block_count` is additionally bounded by the WCB `max_structured_items`
(default 1,000,000), because `scales` and `zero_points` are descriptor arrays;
this is the effective mode-10 bound and is deliberately stricter than
`max_tensor_elements` for large tensors. Reconstructed planes, stored streams,
bodies, and frames are bounded by `max_decoded_bytes` and `max_frame_bytes`.
All products and sums are checked; overflow is `LimitExceeded` before
allocation.

### 11.8 API and CLI (maintainer tools, not wire formats)

Rust (`weftcodec::tensor_sharded`):

- `encode_sharded_tensor(input, limits) -> Vec<frame bytes>`
- `decode_shard_frame(frame, limits) -> DecodedTensorShard`
- `assemble_sharded_words(shards, limits) -> Vec<i8>`
- `shard_set_root(shards, limits) -> Hash32`
- `logical_root(dtype, shape, words, scales, zero_points, limits) -> Hash32`

Go (`modes` package) exposes the equivalent `EncodeShardedTensor`,
`DecodeShardFrame`, `AssembleShardedWords`, `ShardedSetRoot`, and
`LogicalRootSharded`.

```
weftcodec encode --mode tensor-sharded --input WORDS --output DIR \
    --dtype q8_0 --shape 4,8 --axis 0 --shards 2 --block-size 16 \
    --scales 1056964608,1048576000 --zero-points 0,3
weftcodec tensor root --shards-dir DIR --frames [--json]
```

`WORDS` contains one byte per whole-tensor quantized value; `q4_0` values are
sign-extended (`0xF8` for `-8`). Encoding writes `shard-<index>.wft1` frames
into `DIR`. `--metadata JSON` accepts the same fields as an object.

### 11.9 Conformance vectors

`conformance/v1/sharded/` is byte-identical in both codec repositories:

- `001_q8_0_rank2_2shards`, `002_q4_0_rank3_4shards`, and
  `003_q8_0_rank1_1shard`: valid sets with `words_hex`, global scales and zero
  points, the expected `layout_root`, and every shard frame (deliberately
  shuffled);
- `101`–`108`: unknown dtype, scale-plane mismatch, wrong element count,
  invalid block size, non-divisible shard axis, out-of-range index, duplicate
  shard, and malformed Huffman stream.

`WeftCodec-Go/hack/cross.sh compare` additionally encodes all three sets with
both CLIs and reports byte-identical frames, decode equivalence, and identical
logical roots.

---

## 12. Mode 11 — `WEIGHT_DELTA` (implemented)

Mode 11 carries the exact word-wise difference between one tensor generation
and a base tensor frame. Every operation is on unsigned words:

```
d_i = (a_i - b_i) mod 2^word_bits
a_i = (d_i + b_i) mod 2^word_bits
```

`a_i` is the target element word and `b_i` the base element word. No
floating-point arithmetic is performed anywhere. Reconstruction is bit-exact
for every pattern, including NaNs with non-zero payloads, signalling/quiet
bits, infinities, denormals, and signed zeros, because the codec treats the raw
element words as unsigned integers.

### 12.1 Frame shape

```
domain        = 0 (payload)
mode          = 11
flags         = 0 (CHUNKED, SELF_CONTAINED, DICT_RAW, ORIGINAL_HASH all clear)
level         = 0
dep_count     = 1
chunk_count   = 0
decoded_len   = element_count * storage_bytes(dtype)
content_hash  = H("weft/chunk/v1", reconstructed element bytes)
body          = u32le(meta_len) || WCB(weft.ai.weight_delta.v1) || payload
```

The single dependency entry is

```
hash         = base            # the base frame's content hash
decoded_len  = decoded_len     # the reconstructed element byte length
dep_domain   = 0               # payload plaintext
dep_flags    = REQUIRED
codec_hint   = the base frame's mode value, or 0 when unknown
```

For a mode-6, mode-9, or mode-11 base frame, the frame's `content_hash` is
`H("weft/chunk/v1", decoded element bytes)`, so the same value is both the
frame content root and the dependency plaintext hash. The decoder therefore
resolves the base through the existing `Resolver` contract with the same
depth, cycle, fetch, and decoded-byte limits as every other dependency mode.

### 12.2 Descriptor

The WCB descriptor is a map with exactly these keys:

| Key | WCB type | Value |
|---|---|---|
| `schema` | text | exactly `weft.ai.weight_delta.v1` |
| `dtype` | text | one of `bf16`, `fp16`, `fp32`, `q8_0`, `q4_0` |
| `shape` | array of u64 | logical extents, rank 1..=`max_tensor_rank`; every dimension > 0 |
| `element_count` | u64 | product of `shape` |
| `word_bits` | u64 | fixed per dtype: 16, 16, 32, 8, 4 |
| `base` | bytes(32) | content root (frame content hash) of the base tensor frame |
| `base_dtype` | text | MUST equal `dtype` |
| `base_shape` | array of u64 | MUST equal `shape` |
| `op` | text | exactly `mod-sub` |
| `planes` | u64 | fixed per dtype: 2, 2, 4, 1, 1 |
| `plane_codec` | text | exactly `huffman-canonical-v1` |
| `delta_root` | bytes(32) | `H("weft/tensor/values/v1", canonical delta words)` |
| `payload_len` | u64 | length of the stored plane payload |

`WCB` has no `u32` tag, so `planes` and `word_bits` are carried as `u64`.
`delta_root` binds the decoded delta words; because `huffman-canonical-v1` is
deterministic and rejects non-canonical tables and padding, it also binds the
stored payload. The frame `content_hash` binds the reconstructed element bytes.

### 12.3 Element words, planes, and `delta_root`

| `dtype` | `word_bits` | `planes` | storage bytes | word representation |
|---|---:|---:|---:|---|
| `bf16` | 16 | 2 | 2 | little-endian u16 |
| `fp16` | 16 | 2 | 2 | little-endian u16 |
| `fp32` | 32 | 4 | 4 | little-endian u32 |
| `q8_0` | 8 | 1 | 1 | one quantized integer byte |
| `q4_0` | 4 | 1 | 1 | one unpacked nibble byte, value `0..=15` |

`q8_0` and `q4_0` reuse the quantized-integer definitions of §11.3: `q8_0`
values are `i8` and `q4_0` values are `i4` in `-8..=7`. For mode 11 the
element bytes are the unsigned word representation: a `q8_0` element is its
two's-complement byte, and a `q4_0` element is one byte holding its nibble in
`0..=15`. The codec never dequantizes, converts, or interprets a value.

For each element, the delta word `d_i` is split into its bytes, least
significant first:

- plane 0 is the least significant byte, plane 1 the next, and so on;
- each plane is the concatenation of that byte over all elements in canonical
  row-major order;
- planes are concatenated in plane order into the payload, and each plane is
  encoded as its own `huffman-canonical-v1` stream (§5.3.1), including its own
  256-byte code-length table. There is no padding between streams; the next
  plane starts at the first byte after the previous stream, whose own padding
  bits MUST be zero.

For `q4_0`, `word_bits` is 4 and there is a single plane of unpacked nibbles,
one byte per element with value `0..=15`.

The delta root is

```
delta_root = H("weft/tensor/values/v1", D)
```

where `D` is the canonical row-major sequence of delta words written as
`word_bits` little-endian bytes (2 bytes for 16-bit words, 4 bytes for 32-bit
words, 1 byte for 8-bit words). For `q4_0`, `D` is one byte per element with
the nibble value in `0..=15`.

### 12.4 Base resolution and verification

The decoder resolves `base` through the caller's dependency resolver. The
existing DAG resolver supplies the base frame bytes as well as the plaintext;
the plaintext resolver supplies only the plaintext. In both cases the decoder:

1. requires the dependency table's single entry to have `hash == base` and
   `decoded_len == decoded_len`;
2. resolves the plaintext through the resolver, which enforces hash, length,
   depth, and cycle rules;
3. resolves the base frame bytes when the resolver can supply them (explicit
   `--base` also supplies them) and verifies:
   - the base frame's `content_hash` equals `base` (`DependencyMismatch`);
   - the base frame's logical dtype and shape equal `dtype` and `shape`
     (`DependencyMismatch`);
   - the base plaintext length equals `element_count * storage_bytes`
     (`DependencyMismatch`).

The base frame's storage dtype maps to the weight dtype as follows:

| weight `dtype` | accepted base frame storage |
|---|---|
| `bf16`, `fp16`, `fp32` | mode 6 / mode 9 / mode 11 with the same logical dtype |
| `q8_0` | mode 6 `i8`, or mode 11 `q8_0` |
| `q4_0` | mode 6 `u8`, or mode 11 `q4_0` |

A missing base is `DependencyMissing`; a re-entered frame is
`DependencyCycle`; a wrong hash, length, dtype, or shape is
`DependencyMismatch`. A `q4_0` base or target byte outside `0..=15` is
`Corrupt`. Decode never substitutes a different base and never falls back to
another mode.

### 12.5 Validation order (normative for error classes)

1. Body split / `meta_len` bounds → `Malformed`.
2. WCB canonical decode → `NonCanonical`, `Malformed`, or `LimitExceeded`.
3. Descriptor key set, `schema` → `Malformed`.
4. Unknown `dtype`, `op` not `mod-sub`, or `plane_codec` not
   `huffman-canonical-v1` → `Malformed`.
5. `word_bits` or `planes` not matching the dtype table → `Malformed`;
   `base_dtype` ≠ `dtype` or `base_shape` ≠ `shape` → `Malformed`.
6. `shape` rank 0, zero dimension, or product overflow → `Malformed` /
   `LimitExceeded`; rank or product above the configured limits →
   `LimitExceeded`.
7. `element_count` ≠ shape product → `Malformed`.
8. `base` or `delta_root` not exactly 32 bytes → `Malformed`.
9. `payload_len` ≠ actual stored payload length → `Malformed`;
   `decoded_len` ≠ `element_count * storage_bytes` → `Corrupt`.
10. Dependency count/table fields inconsistent with the descriptor →
    `Malformed`; dependency resolution failures use the existing dependency
    classes (`DependencyMissing`, `DependencyDenied`, `DependencyMismatch`,
    `DependencyCycle`, `LimitExceeded`).
11. Huffman stream rejects → the classes of §5.3.1.
12. `delta_root` ≠ the reconstructed delta words → `Corrupt`.
13. Base frame metadata or content mismatch → `DependencyMismatch`;
    `q4_0` word out of range → `Corrupt`.
14. Reconstructed `content_hash` mismatch → `Corrupt`.

### 12.6 Limits

`max_tensor_rank` and `max_tensor_elements` bound the shape. `decoded_len`,
the stored payload, the body, and the frame are bounded by
`max_decoded_bytes` and `max_frame_bytes`. The dependency is bounded by
`max_dependency_depth`, `max_dependency_count`, `max_total_fetched_bytes`, and
the cumulative decoded-byte limit. Every product and sum is checked; overflow
is `LimitExceeded` before allocation.

### 12.7 API and CLI (maintainer tools, not wire formats)

Rust (`weftcodec::weight_delta`):

- `encode_weight_delta(target, dtype, shape, base_frame, limits)`
- `encode_weight_delta_with_resolver(target, dtype, shape, base_frame, resolver, limits)`
- `decode_frame(frame, resolver, limits)`
- `decode_with_base_frame(frame, base_frame, resolver, limits)`
- `weight_delta_metadata(frame, limits)`
- `base_frame_descriptor(frame, limits)`

Go (`modes` package) exposes the equivalent `EncodeWeightDelta`,
`EncodeWeightDeltaWithResolver`, `DecodeWeightDeltaFrame`,
`DecodeWeightDeltaFrameWithBase`, `WeightDeltaMetadata`, and
`BaseFrameDescriptor`.

```
weftcodec encode --mode weight-delta --base BASE_FRAME --input TARGET \
    --output FRAME --dtype bf16|fp16|fp32|q8_0|q4_0 --shape 2,3
weftcodec decode --input FRAME --output TARGET [--base BASE_FRAME | --resolve --frame-source DIR]
```

`--base` supplies the base frame explicitly. `--resolve` uses the existing
frame-source DAG resolver and requires `--frame-source`; it verifies the base
frame's own dtype/shape descriptor. `inspect` reports `mode=weight_delta`,
`dtype`, `shape`, `element_count`, `word_bits`, `base`, `base_dtype`,
`base_shape`, `op`, `planes`, `plane_codec`, `delta_root`, and `payload_len`.

### 12.8 Conformance vectors

`conformance/v1/weight-delta/` is byte-identical in both codec repositories:

- `001_bf16_small_changes`: bf16 base with NaN/infinity/signed-zero/denormal
  words and exactly one changed word;
- `002_fp32_zero_delta`: fp32 base and target identical, so every delta word
  is zero and each plane is a single-symbol Huffman table;
- `003_q8_0_mixed_changes`: q8_0 words with mixed wrapped changes, backed by a
  mode-6 `i8` base frame;
- `101`–`108`: base dtype mismatch, base shape mismatch, unknown op,
  `word_bits` inconsistent with dtype, delta-root mismatch, payload-length
  mismatch, missing base, and dependency cycle.

The vector files carry the base frame, the target bytes, the committed delta
frame, the descriptor fields, and the expected error class. Both
implementations replay them, decode the valid cases, re-encode them to the
same bytes, and reject every invalid case with the stated class.

## 13. Mode 12 — `KV_CACHE` (implemented)

### 13.1 Frame shape

```text
domain        = 0 (payload)
mode          = 12
flags         = 0
dep_count     = 0
chunk_count   = 0
decoded_len   = page_bytes
content_hash  = H("weft/chunk/v1", page element bytes)
body          = u32le(meta_len) || WCB(meta) || payload
```

One frame is exactly one page. Pages have no dependencies, no chunk table, and
no decoder state outside the frame. A page is fetched and decoded without
reading, parsing, or decoding its siblings.

### 13.2 Descriptor

Schema: `weft.ai.kv_cache.v1`. Exactly these keys, no others:

| Key | Type | Rule |
|---|---|---|
| `schema` | text | exactly `weft.ai.kv_cache.v1` |
| `model` | text | model identity |
| `layer` | u64 | fits u32 |
| `session` | text | session identity |
| `page_index` | u64 | `< page_count` |
| `page_count` | u64 | `>= 1` |
| `seq_start` | u64 | `< seq_end` |
| `seq_end` | u64 | `> seq_start` |
| `dtype` | text | `bf16`, `fp16`, or `fp32` |
| `head_count` | u64 | fits u32 and `> 0` |
| `head_dim` | u64 | fits u32 and `> 0` |
| `shape` | array(u64) | exactly `[page_rows, head_count, head_dim]` |
| `page_bytes` | u64 | `page_rows * head_count * head_dim * word_bytes` |
| `plane_codec` | text | exactly `huffman-canonical-v1` |
| `payload_len` | u64 | exact stored payload byte length |

`page_rows = seq_end - seq_start`. `word_bytes` is 2 for `bf16`/`fp16` and 4
for `fp32`. Every product and sum is checked; overflow is `LimitExceeded`.
`page_bytes` is bounded by `max_decoded_bytes`, the rank by `max_tensor_rank`,
and the element count by `max_tensor_elements`.

### 13.3 Payload

The payload is the canonical row-major little-endian element bytes split into
byte planes: plane 0 is the least significant byte of each element word.
`bf16` and `fp16` have two planes; `fp32` has four. Each plane is an
independent `huffman-canonical-v1` stream (§5.3.1) with its own 256-byte code
length table, and the streams are concatenated in plane order. Decoding MUST
reconstruct exactly `page_bytes` bytes. No dtype conversion, dequantization, or
numeric interpretation occurs. NaNs, denormals, signed zeros, and infinities
are preserved bit-for-bit.

### 13.4 Validation order (normative for error classes)

1. Unknown key, missing key, wrong WCB type, schema mismatch, unknown dtype →
   `Malformed`.
2. `plane_codec` not `huffman-canonical-v1` → `Malformed`.
3. `page_count == 0`, `page_index >= page_count` → `Malformed`.
4. `seq_start >= seq_end` → `Malformed`; `head_count == 0` or
   `head_dim == 0` → `Malformed`.
5. `shape` not exactly `[page_rows, head_count, head_dim]` → `Malformed`.
6. A checked product overflows u64 → `LimitExceeded`; a rank/element/page
   limit is exceeded → `LimitExceeded`.
7. `page_bytes` differs from the computed byte length → `Malformed`.
8. `payload_len` differs from the stored payload length → `Malformed`;
   frame `decoded_len` differs from `page_bytes` → `Corrupt`.
9. Huffman stream rejects use §5.3.1 classes; trailing payload bytes →
   `Corrupt`.
10. Reconstructed `content_hash` mismatch → `Corrupt`.

### 13.5 API and CLI

Rust (`weftcodec::kv_cache`): `encode_kv_page`, `decode_kv_page`,
`decode_kv_page_range`, `kv_cache_metadata`, `KvPageSet` (which counts
page-level decodes). Go (`modes`): `EncodeKvPage`, `DecodeKvPage`,
`DecodeKvPageRange`, `KvCacheMetadata`, `KvPageSet`.

```sh
weftcodec encode --mode kv-cache --input PAGE --output FRAME --metadata PAGE.json
weftcodec decode --input FRAME --output PAGE
```

`decode_kv_page_range` decodes one page and slices the requested byte range.
Huffman planes are not independently seekable; the bounded guarantee is that
only one page is touched, never that only the requested bytes are entropy
decoded.

### 13.6 Conformance vectors

`conformance/v1/kv-cache/` is byte-identical in both repositories:

- `001_bf16_two_pages`: two bf16 pages of one layer with NaN, denormal, signed
  zero, and infinity patterns;
- `002_fp32_two_pages`: two fp32 pages;
- `003_bf16_seq_boundary`: a late page with `seq_start = 4096`;
- `101`–`108`: unknown dtype, zero dimension, page index out of range, invalid
  sequence range, shape mismatch, page-byte mismatch, sequence product
  overflow, and payload-length mismatch.

Both implementations decode each valid page, re-encode it to the identical
bytes, and reject every invalid frame with the stated class.

## 14. Mode 13 — `ACTIVATION_STREAM` (implemented)

### 14.1 Frame shape

```text
domain        = 0 (payload)
mode          = 13
flags         = 0
dep_count     = 0
chunk_count   = 0
decoded_len   = chunk element byte length
content_hash  = H("weft/chunk/v1", chunk element bytes)
body          = u32le(meta_len) || WCB(meta) || payload
```

One frame is exactly one chunk of one `(stream, step, rank)` stream. Chunks
do not depend on one another: encoding chunk `k+1` does not decode chunk `k`,
and a decoder may materialize one chunk without its siblings.

### 14.2 Descriptor

Schema: `weft.ai.activation_stream.v1`. Exactly these keys:

| Key | Type | Rule |
|---|---|---|
| `schema` | text | exactly `weft.ai.activation_stream.v1` |
| `stream` | text | `activation` or `gradient` |
| `step` | u64 | training/inference step |
| `rank` | u64 | fits u32 and `< world` |
| `world` | u64 | fits u32 and `>= 1` |
| `chunk_index` | u64 | `< chunk_count` |
| `chunk_count` | u64 | `>= 1` |
| `dtype` | text | `bf16`, `fp16`, or `fp32` |
| `shape` | array(u64) | rank `>= 1`, every dimension `> 0` |
| `element_count` | u64 | exactly the shape product |
| `digest_root` | bytes(32) | `H("weft/tensor/values/v1", chunk element bytes)` |
| `plane_codec` | text | exactly `huffman-canonical-v1` |
| `payload_len` | u64 | exact stored payload byte length |

### 14.3 Payload

The byte-plane layout is exactly §13.3. Decoding reconstructs the chunk's
canonical row-major element bytes, verifies `digest_root`, and verifies the
frame `content_hash`. No numeric interpretation occurs.

### 14.4 Validation order (normative for error classes)

1. Unknown key, missing key, wrong WCB type, schema mismatch, unknown stream,
   unknown dtype → `Malformed`.
2. `plane_codec` not `huffman-canonical-v1` → `Malformed`.
3. `world == 0` or `rank >= world` → `Malformed`.
4. `chunk_count == 0` or `chunk_index >= chunk_count` → `Malformed`.
5. Empty shape, zero dimension, rank/element limit, or shape-product overflow
   → `Malformed`/`LimitExceeded` as in §13.4.
6. `element_count` not the shape product → `Malformed`; `digest_root` not 32
   bytes → `Malformed`.
7. `payload_len` differs from the stored payload → `Malformed`;
   `decoded_len` differs from `element_count * word_bytes` → `Corrupt`.
8. Huffman rejects use §5.3.1; trailing payload bytes → `Corrupt`.
9. Reconstructed `digest_root` mismatch → `Corrupt`; reconstructed
   `content_hash` mismatch → `Corrupt`.

### 14.5 Assembly and append

`assemble_stream(chunks)` accepts chunk frames in any order and:

1. decodes every chunk and re-verifies its `digest_root`;
2. requires the same `stream`, `step`, `rank`, `world`, `chunk_count`,
   `dtype`, `shape`, and `element_count` on every chunk;
3. sorts by `chunk_index`, rejects duplicate indices (`NonCanonical`) and
   missing indices (`Malformed`);
4. concatenates the element bytes in index order and returns them with

```text
stream_root = H("weft/ai/stream/v1", WCB({
  stream, step, rank, world, chunk_count, digest_roots
}))
```

where `digest_roots` is the array of per-chunk `digest_root` values in
`chunk_index` order. Duplicate, missing, or inconsistent chunks never produce
a stream root.

`append_chunk` encodes the next `chunk_index` from the elements alone. It does
not accept, read, or decode earlier chunk frames; both implementations expose
an encoded/decoded counter so this property is testable.

### 14.6 API and CLI

Rust (`weftcodec::activation_stream`): `encode_activation_chunk`,
`decode_activation_chunk`, `decode_activation_chunk_range`,
`activation_chunk_metadata`, `assemble_stream`, `stream_root`,
`ActivationStreamWriter`, `ActivationChunkSet`. Go (`modes`): the equivalent
`EncodeActivationChunk`, `DecodeActivationChunk`, `DecodeActivationChunkRange`,
`ActivationChunkMetadata`, `AssembleStream`, `StreamRoot`,
`ActivationStreamWriter`, `ActivationChunkSet`.

```sh
weftcodec encode --mode activation-stream --input CHUNK --output FRAME --metadata CHUNK.json
weftcodec decode --input FRAME --output CHUNK
```

### 14.7 Conformance vectors

`conformance/v1/activation-stream/` is byte-identical in both repositories:

- `001_activation_bf16_two_chunks`: two bf16 activation chunks of one step;
- `002_gradient_fp32_two_chunks`: the gradient variant with fp32;
- `003_activation_fp16_single_chunk`: a single fp16 chunk at a world
  boundary;
- `101`–`108`: unknown dtype, unknown stream, rank not below world, chunk index
  out of range, element-count mismatch, digest-root mismatch, payload-length
  mismatch, and decoded-length mismatch.

Valid vectors carry every chunk frame, the element bytes, per-chunk digest
roots, and the assembled `stream_root`. Both implementations decode, re-encode
to identical bytes, assemble, and recompute the root; every invalid frame is
rejected with the stated class.

## 15. Semantic roots and mode 14 — `SEMANTIC_DELTA` (implemented)

Near-duplicate tensors need a way to be grouped and proposed as delta bases
without weakening exactness. A **semantic root** is metadata for that grouping.
It is **not** a content identity, it never replaces `LogicalTensorRoot`, and it
never replaces a frame `content_hash`. Two objects with the same semantic root
are *candidates*; they are not interchangeable bytes.

### 15.1 Semantic root

```
SemanticRoot = H("weft/tensor/semantic/v1",
                 WCB({schema: "weft.tensor.semantic.v1",
                      dtype, shape, element_count, policy,
                      quantized_values_hash}))
```

`dtype` is one of `bf16`, `fp16`, `fp32`, `q8_0`, `q4_0`; `shape` and
`element_count` follow §12.2. `quantized_values_hash` is:

- for `{"scheme":"none"}`, exactly the logical values hash
  `H("weft/tensor/values/v1", row_major_bytes)`;
- otherwise `H("weft/tensor/semantic-values/v1", transformed_bytes)` where
  `transformed_bytes` is the policy output below.

Policies are canonical WCB maps and are required:

| Policy | Value |
|---|---|
| identity | `{"scheme":"none"}` |
| mantissa truncation | `{"scheme":"mantissa_truncate","bits":N}` |
| block quantization | `{"scheme":"block_quantize","block_size":B,"levels":L}` |

`mantissa_truncate` is defined only for `bf16` (7 mantissa bits), `fp16` (10),
and `fp32` (23). For each element word, clear the `N` least-significant mantissa
bits (`word & ~((1<<N)-1)`). A word whose exponent field is all ones (NaN or
infinity) is stored unchanged, so sign, exponent, NaN payloads, and infinities
are preserved verbatim. `1 <= N <= mantissa_bits(dtype)`; `N = 0` is rejected
because `none` is the identity.

`block_quantize` interprets each raw word as a signed two's-complement integer
`s` of the dtype's word width and splits the element sequence into blocks of
`B` consecutive row-major elements, where `B` MUST divide `element_count` and
`1 <= L <= 2^24`. For a block with `amax = max |s_i|`:

- if `amax == 0` or `L == 1`, every reconstructed word is signed zero;
- otherwise the reconstruction grid is

  ```
  level_k = -amax + floor((2*amax*k + (L-1)/2) / (L-1)),   k = 0..L-1
  ```

  and element `s` maps to

  ```
  k = clamp(floor(((s + amax)*(L-1) + amax) / (2*amax)), 0, L-1)
  ```

  storing `level_k` in two's complement. Every intermediate fits `u64` because
  `amax <= 2^31` and `L <= 2^24`. The grid spans `[-amax, amax]`, so the
  quantizer is symmetric about zero within each block.

Unknown schemes, wrong WCB types (including null in place of a numeric field),
out-of-range `bits`, non-dividing `block_size`, `levels == 0`, and
`element_count != product(shape)` are `Malformed`.

### 15.2 Mode 14 — `SEMANTIC_DELTA`

Mode 14 is the exact WEIGHT_DELTA reconstruction scheme plus a verified
semantic-root binding. The payload is identical to §12.3: modular element-wise
subtraction on raw words, one `huffman-canonical-v1` stream per byte plane, and
the same `delta_root`. Frame shape, dependency shape, and content hash follow
§12.1 with `mode = 14`.

The WCB descriptor is `weft.ai.semantic_delta.v1` with exactly these keys:

| Key | WCB type | Value |
|---|---|---|
| `schema` | text | exactly `weft.ai.semantic_delta.v1` |
| `dtype` | text | `bf16`, `fp16`, `fp32`, `q8_0`, or `q4_0` |
| `shape` | array of u64 | logical extents |
| `element_count` | u64 | product of `shape` |
| `word_bits` | u64 | fixed per dtype (§12.3) |
| `base` | bytes(32) | base frame content hash |
| `base_dtype` | text | MUST equal `dtype` |
| `base_shape` | array of u64 | MUST equal `shape` |
| `base_semantic_root` | bytes(32) | semantic root of the base bytes under `policy` |
| `policy` | map | one canonical policy from §15.1 |
| `delta_root` | bytes(32) | `H("weft/tensor/values/v1", delta words)` |
| `plane_codec` | text | exactly `huffman-canonical-v1` |
| `payload_len` | u64 | stored payload length |

Decoding verifies, in order: descriptor structure and policy validity; frame
`decoded_len`, `payload_len`, dependency count/domain, and dependency hash;
`delta_root` against the reconstructed delta words; the base content hash,
dtype, and shape; and finally the semantic root recomputed from the actual base
bytes under the declared policy. Any mismatch is `DependencyMismatch` (base
metadata or semantic root) or `Corrupt` (`delta_root`, reconstructed content
hash). A missing base is `DependencyMissing`.

`base_semantic_root` is an **input to base selection**, never a substitute for
verification. The semantic root is not recomputed from the target for
verification: mode 14 proves the base, and the frame content hash proves the
target.

### 15.3 No lossy mode

Approximate decoding is **not implemented**. No mode, flag, or decoder path
silently reconstructs approximate bytes. `SEMANTIC_DELTA` reconstruction is
bit-exact; its descriptor policy only selects and verifies a base, and an
approximate base is a hard error, not a fallback. The policy transformations
exist solely to compute a grouping hash; they are never emitted as
reconstructed element bytes.

### 15.4 Candidate selection

The candidate helper takes a target, its dtype/shape, one policy, and a set of
candidate base **frames whose bytes are present**. It computes the target's
semantic root, skips candidates whose dtype/shape differ, decodes each remaining
candidate, recomputes its semantic root under the same policy, and skips
candidates whose root differs. Among matches it computes the exact modular
difference and its exact `huffman-canonical-v1` per-plane size and chooses the
smallest. Ties are broken by the candidate frame content hash. The result is
deterministic for a fixed candidate order, and a pair with different semantic
roots is never offered as a candidate. Because both tensors must be present,
this is a local selection helper, not a remote index query.

### 15.5 API and CLI (maintainer tools, not wire formats)

Rust (`weftcodec::semantic`, `weftcodec::semantic_delta`):

- `semantic_root(dtype, shape, bytes, policy)` and `semantic_root_with_limits`
- `semantic_record(...)`, `SemanticRootRecord::from_canonical_bytes`
- `semantic_matches(a, b)`
- `select_semantic_base(target, dtype, shape, policy, candidates, limits)`
- `encode_semantic_delta(target, dtype, shape, base_frame, policy, limits)`
- `decode_frame(frame, resolver, limits)` and `decode_with_base_frame(...)`
- `semantic_delta_metadata(frame, limits)`

Go (`modes` package) exposes `SemanticRoot`, `SemanticRootWithLimits`,
`NewSemanticRecord`, `ParseSemanticRecord`, `SemanticMatches`,
`SelectSemanticBase`, `EncodeSemanticDelta`, `DecodeSemanticDeltaFrame`,
`DecodeSemanticDeltaFrameWithBase`, and `SemanticDeltaMetadata`.

```
weftcodec encode --mode semantic-delta --base BASE_FRAME --input TARGET \
    --output FRAME --dtype bf16|fp16|fp32|q8_0|q4_0 --shape 2,3 \
    --semantic-policy none|mantissa_truncate:3|block_quantize:128:256
weftcodec decode --input FRAME --output TARGET [--base BASE_FRAME | --resolve --frame-source DIR]
```

`inspect` reports `base_semantic_root` and the canonical `policy` in addition
to the mode-11 fields.

### 15.6 Conformance vectors

`conformance/v1/semantic-root/` and `conformance/v1/semantic-delta/` are
byte-identical in both codec repositories:

- roots: `001` none, `002` bf16 `mantissa_truncate:3`, `003` fp32
  `block_quantize` `B=128 L=256`; rejects `101`–`106` unknown scheme, bits out
  of range, block size not dividing, shape mismatch, levels zero, and a
  non-numeric (null) policy field;
- deltas: `001` bf16 near-duplicate, `002` fp32 small change, `003` q8_0;
  rejects `101`–`106` base semantic mismatch, policy mismatch, delta-root
  mismatch, missing base, base shape mismatch, and payload-length mismatch.

Both implementations replay every valid vector, decode, and re-encode to the
same bytes; invalid vectors are rejected with the stated class. The
cross-language gate reports byte-identical mode-14 frames in both directions.

---

## 16. Mode 15 — `TENSOR_TILED` (implemented)

Mode 15 stores one tensor as **independently decodable tiles**. A region read
decodes only the tiles it intersects; it never decodes the whole tensor. Mode
10 (`TENSOR_SHARDED`) is unchanged and remains the default checkpoint shard
representation.

### 16.1 Frame shape

```
domain        = 0 (payload)
mode          = 15
flags         = 0
dep_count     = 0
chunk_count   = 0
decoded_len   = product(shape) * element_size
content_hash  = H("weft/chunk/v1", canonical row-major element bytes)
body          = u32le(meta_len) || WCB(meta) || stored_payload
```

`meta_len` is the descriptor length only. The decoder checks the descriptor
prefix and `stored_payload.len() == payload_len` with checked arithmetic. The
stored payload is `sum(tile.compressed_len)` bytes; it is normally smaller than
`decoded_len`.

### 16.2 Descriptor

Schema `weft.ai.tensor_tiled.v1`. Exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `schema` | text | `"weft.ai.tensor_tiled.v1"` |
| `dtype` | text | one of the v1 element dtypes (`WEFT-TENSOR` §2) |
| `shape` | `[u64]` | logical shape; rank ≥ 1, every extent > 0, rank ≤ `max_tensor_rank` |
| `element_count` | u64 | must equal `product(shape)` |
| `layout` | text | `"row_major"` (the only layout implemented in v1) |
| `tile_shape` | `[u64]` | rank-equal to `shape`; every extent > 0 and divides the corresponding shape extent |
| `tile_counts` | `[u64]` | `shape[i] / tile_shape[i]` |
| `tile_codec` | text | `"huffman-canonical-v1"` |
| `tiles` | array | one entry per tile, in canonical order |
| `payload_len` | u64 | exact stored payload length |
| `logical_root` | bytes32 | `LogicalTensorRoot` of the whole tensor (`WEFT-TENSOR` §3) |

A tile entry has exactly `index`, `dims`, `offset`, `compressed_len`,
`element_len`, `tile_root`:

* `index` is the canonical grid position (row-major over `tile_counts`); the
  array position MUST equal `index`.
* `dims` MUST equal `tile_shape`. v1 defines full tiles only; partial boundary
  tiles are not defined. A non-dividing `tile_shape` is `Malformed`.
* `offset` is the byte offset of the stored tile inside `stored_payload`.
  Offsets MUST be contiguous: the first is `0`, each next offset equals the
  previous `offset + compressed_len`, and the last end equals `payload_len`.
  A gap, overlap, or non-monotonic offset is `Malformed`.
* `compressed_len` is the stored tile length; zero is `Malformed`.
* `element_len` MUST equal `product(dims) * element_size`.
* `tile_root` is bytes32 (see §16.4).

### 16.3 Tile payload

A tile's decoded bytes are its canonical row-major element bytes. The tile is
split into `element_size` **byte planes**: plane `p` is byte `p` of each
element, in row-major element order. Each plane is coded independently with
`huffman-canonical-v1` (its own 256-byte table plus bitstream). A stored tile is
the concatenation of its planes in plane order. The decoder decodes each plane
with `huffman-canonical-v1` expecting `product(dims)` bytes; it consumes exactly
the plane's own table and stream and rejects trailing bytes inside the declared
tile range. This makes every tile independently decodable and verifiable.

### 16.4 Tile identity

```
values_hash = H("weft/tensor/values/v1", tile canonical row-major bytes)
tile_root   = H("weft/tensor/tile/v1",
                WCB({dtype, dims, element_len, values_hash}))
```

`WCB` keys are exactly `dtype` (text), `dims` (`[u64]`), `element_len` (u64),
and `values_hash` (bytes32).

### 16.5 Decoding and region reads

Full decode assembles every tile into the canonical row-major tensor, verifies
each `tile_root`, then recomputes `LogicalTensorRoot` and requires it to equal
`logical_root`. The frame content hash is verified as usual.

A region read (`read_region`) takes a half-open hyper-rectangle in logical
coordinates and:

1. Computes the tile-index range intersecting the region on every axis.
2. Decodes only those tiles, verifying each `tile_root`.
3. Copies the intersection box from each decoded tile into the region's
   canonical row-major output.
4. Returns the bytes and the number of tiles decoded (`tiles_touched`).

Tiles outside the region are never read or decoded. A range reader that does
not materialise the whole frame must read the fixed header, the descriptor, and
only the stored ranges of intersecting tiles. In that mode the frame trailer
and whole-frame content hash cannot be verified; the per-tile roots are the
exactness guarantee. Full decode continues to verify both.

### 16.6 Validation order (normative for error classes)

1. Frame header/tables (`Malformed`, `UnsupportedVersion`, `UnsupportedFeature`).
2. Descriptor WCB canonical decode (`Malformed`, `NonCanonical`).
3. Schema, dtype, layout, tile codec (`Malformed`; unknown values fail closed).
4. Shape/tile arithmetic: zero dimension, rank, non-dividing tile shape,
   `element_count`, `tile_counts` (`Malformed`, `LimitExceeded`).
5. `decoded_len` consistency (`Corrupt`).
6. Tiles array length, canonical index order, `dims`, `element_len`, offsets,
   `payload_len` (`Malformed`, `LimitExceeded`).
7. Tile plane decode (`Corrupt`, `NonCanonical`) and `tile_root` (`Corrupt`).
8. `logical_root` and frame content hash (`Corrupt`).

### 16.7 Limits

`max_tensor_elements`, `max_tensor_rank`, `max_decoded_bytes`,
`max_structured_items` (tile count), `max_string_bytes` (descriptor), and
`max_frame_bytes` all apply. Tile count is checked before allocating per-tile
state. A prefix reader additionally bounds the descriptor read by
`max_string_bytes`.

### 16.8 API and CLI (maintainer tools, not wire formats)

Rust: `modes::encode_tiled_tensor`, `modes::parse_tiled_tensor`,
`modes::parse_tiled_descriptor`, `modes::decode_tiled_tensor`,
`modes::tiled_tensor_metadata`, `modes::tensor_tile_root`,
`modes::read_region`, and `modes::read_region_with` (caller-supplied stored
tile fetch for a range reader).

Go: `modes.EncodeTiledTensor`, `modes.ParseTiledTensor`,
`modes.DecodeTiledTensor`, `modes.TiledTensorMetadata`,
`modes.TensorTileRoot`, and `modes.ReadRegion` (the stored-tile fetch is the
caller's in the store-side range reader).

CLI: `weftcodec encode --mode tensor-tiled --input PAYLOAD --dtype D
--shape A,B,... --tile-shape X,Y,...`; decode uses the ordinary `decode`
verb.

The WeftFS store writes mode 15 through `checkpoint write --tile-shape` and
reads it through per-tile backend range reads; mode 10 remains the default.

### 16.9 Conformance vectors

`conformance/v1/tensor-tiled/` (byte-identical in both codec repositories):

* `001_tiled_bf16_4x4` — bf16 4×4, tile 2×2, with a region read.
* `002_tiled_fp32_2x2x2` — fp32 rank 3, tile 1×2×2, with a region read.
* `003_tiled_f16_single_tile` — f16 2×3, single tile.

Rejects: `101_non_dividing_tile_shape`, `102_tile_count_mismatch`,
`103_overlapping_tile_offsets`, `104_tile_root_mismatch`,
`105_payload_length_mismatch`, `106_unknown_tile_codec`, `107_zero_dimension`,
`108_logical_root_mismatch`.

The cross-language gate asserts byte-identical frames for mode 15 and
decode-equivalence in both directions.
