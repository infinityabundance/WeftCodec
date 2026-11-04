# WEFT-AI-CODEC v1.1.1 — AI-native codec modes

**Status: PARTIALLY IMPLEMENTED.** Exactly four AI-native modes are normative
and implemented in both Rust `weftcodec` and Go `weftcodec-go`:

| Value | Name | Status | Frozen vectors |
|---:|---|---|---|
| 6 | `TENSOR` | **implemented** | `100_tensor_f32_2x3`, `101_tensor_f16_scalar`, `102_tensor_u8_zero_element` + rejects |
| 7 | `TOKEN_SEQUENCE` (CLI `tokens`) | **implemented** | `103_tokens_delta_boundaries`, `104_tokens_empty` + rejects |
| 8 | `QUANTIZED_TENSOR` (CLI `quantized`) | **implemented** | `105_quantized_i8_2x3`, `106_quantized_u8_zero_element`, `107_quantized_i8_scalar` + rejects |
| 9 | `FLOAT_TENSOR` (CLI `float-tensor`) | **implemented** | `108_float_bf16_2x2`, `109_float_fp16_3d`, `110_float_fp32_4d` + rejects |

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

## 7. Limits (added by this document)

| Limit | Default | Applies to |
|---|---:|---|
| `max_tensor_elements` | 4,294,967,296 | TENSOR/FLOAT_TENSOR shape product |
| `max_tensor_rank` | 8 | TENSOR/FLOAT_TENSOR shape length |
| `max_tokens` | 16,777,216 | TOKEN_SEQUENCE `token_count` |

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
   byte-identical frames for TENSOR, TOKEN_SEQUENCE, QUANTIZED_TENSOR, and
   FLOAT_TENSOR (`WeftCodec-Go/hack/cross.sh`). `FLOAT_TENSOR` byte identity is
   mandatory because its exponent plane uses the deterministic
   `huffman-canonical-v1` coder, not a library zstd frame. The frozen vectors
   are generated once and replayed by both implementations.

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
