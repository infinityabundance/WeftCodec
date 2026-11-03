# WEFT-AI-CODEC v0.1.0 — AI-native codec modes (design status)

**Status: DESIGN STATUS. Every mode in this document is `specified, not
implemented`.** There is no AI codec code in F1, no vectors, and no performance
claim. Nothing here may be cited as implemented. When a mode gains an
implementation and frozen vectors, this status changes in a separate commit and
the mode's row is updated to `implemented`.
**Depends on:** `WEFTCODEC-FORMAT.md` (framing, hashing, limits, modes),
`WEFT-CONTEXT-ROOT.md`, `WEFT-CAUSAL.md`.
**Source of truth:** `spec/WEFT-AI-CODEC.md` in `infinityabundance/WeftFS`.

WeftCodec already provides exact, deterministic framing for opaque bytes. AI
workloads add *typed* payloads whose semantics are known: tensors, gradients,
token sequences, KV-cache pages, and weight deltas. The goal of this design is
not "compression wins"; it is **exactness plus a declared layout**, so two
independent implementations can produce and verify the same object, and a
consumer can refuse an object whose semantics it cannot reproduce.

This document defines constraints and acceptance tests for future modes. It
deliberately does not define JSON schemas as wire formats; all structured values
are WCB.

---

## 1. Non-negotiable constraints for every mode

A future AI mode MUST satisfy all of the following before it may be called
implemented:

1. **Exactness.** Decode(Encode(x)) == x byte-for-byte, or the decoder fails
   closed with a typed error. No lossy fallback, no approximate layout, no
   "close enough" dtype conversion.
2. **Self-description.** Every semantic decision (dtype, shape, layout, shard
   index/count, endianness, padding, quantization parameters, vocabulary hash)
   is carried in the object or bound by a dependency hash. It is never inferred
   from the byte length or from implementation defaults.
3. **Frozen identity.** The plaintext bytes fed to `H("weft/chunk/v1", ...)` are
   exactly the declared canonical payload. Compression is an encoding choice; it
   never changes `content_hash`.
4. **Cross-language equality.** For deterministic modes, Rust and Go MUST emit
   byte-identical frames on the same input and parameters, and decode each
   other's output. For modes that use a third-party compressor, conformance is
   decode-equivalence plus recorded encoder identity (the existing
   `WEFTCODEC-FORMAT.md` §12 rule).
5. **Bounded resources.** Every mode declares limits checked before allocation:
   element count, byte length, shard count, token count, page count, patch op
   count, depth, and total decoded bytes. Exceeding a limit is `LimitExceeded`.
6. **Hash binding.** Any base tensor/weight/vocabulary a mode depends on is a
   codec dependency with a hash and length; a mismatch is
   `DependencyMismatch`, never a substitute base.
7. **No semantic execution.** A codec decodes bytes. It does not evaluate a
   model, apply gradients, or interpret tokenizer text. Semantic checks are
   limited to declared metadata consistency.
8. **Vectors or it did not happen.** A mode is implemented only when committed
   vectors include valid cases and adversarial rejects, replayable by both
   implementations. Until then the mode's status is `specified, not
   implemented` and its numeric value is reserved (see §2.10).

---

## 2. Mode family: canonical tensor (design)

Reserved mode names are given numeric values only when implemented. Proposed
values below are **not allocated** and MUST NOT be used on the wire.

### 2.1 Tensor descriptor

```
{
  "schema": "weft.tensor.v1",
  "dtype":  "f16" | "bf16" | "f32" | "f64" | "i8" | "u8" | "i16" | "u16"
          | "i32" | "u32" | "i64" | "u64" | "bool",
  "shape":  [ u64, ... ],            # row-major logical extents, 0..=8 dims
  "layout": "row" | "col" | "blocked" | "sharded" | "quantized",
  "layout_params": { ... },          # layout-specific, see §2.2–§2.5
  "byte_order": "le"                # only "le" in v1 design
}
```

- Element type size is fixed by `dtype` (bool is one byte, value 0 or 1; any
  other value is `Corrupt`).
- `shape` length 0 is a scalar; product is checked for overflow.
- The canonical payload is the concatenation of elements in the declared layout
  order. Its length MUST equal `product(shape) * elem_size`, checked before
  allocation.
- Padding bytes, if any, are part of the declared layout and MUST be zero. A
  non-zero padding byte is `Corrupt`.
- NaN/Inf bit patterns are preserved exactly. No canonicalization of floating
  point values is permitted.

### 2.2 `row` and `col`

- `row`: last dimension contiguous.
- `col`: first dimension contiguous.
- `layout_params` MUST be an empty map in v1 design.
- Testable constraint: transposing a 2-D tensor and encoding it as `col` yields
  the same payload as encoding the transposed tensor as `row`.

### 2.3 `blocked`

```
layout_params := { "block": [ u64, ... ] }   # one per dimension, ≥ 1
```

- Block extents are declared, not inferred. The iteration order is block index
  in row-major order over blocks, then element index in row-major order within
  the block.
- Blocks at the boundary are **dense**: the payload contains the full block with
  zero padding. Partial blocks are not packed.
- The decode result after removing declared padding MUST equal the logical
  tensor exactly.

### 2.4 `sharded`

```
layout_params := { "shard": u64, "shards": u64, "shard_axis": u64 }
```

- `shards ≥ 1`, `shard < shards`, `shard_axis < len(shape)`.
- Each shard is a complete object with its own `content_hash`; the full tensor is
  reconstructed only by explicit assembly, never implicitly by a decoder.
- The shard's logical shape is `shape` with `shard_axis` divided by `shards`.
  Divisibility is required; otherwise `Malformed`.
- A set of shards is bound by a manifest whose identity includes `(shards,
  shard_axis, full_shape, dtype, per-shard hashes)`. A missing shard is an
  explicit error; no "use what is present".

### 2.5 `quantized`

```
layout_params := {
  "scheme": "affine",
  "scale":  { "num": u64, "den": u64 },   # positive rational, exact
  "zero_point": i64,
  "storage": "i8" | "u8" | "i4"
}
```

- Dequantization is a **semantic** operation and is out of codec scope. The codec
  stores integer codes plus exact scale/zero-point metadata. It does not claim a
  floating-point result.
- `i4` packs two codes per byte, low nibble first. The declared element count
  determines the expected size; a trailing half-byte MUST be zero.
- `den != 0`, `scale` positive, zero-point range checked against storage.
- Rounding is not a codec concern; encoders that produce codes are responsible
  for their own exactness policy, which MUST be recorded outside the codec.

### 2.6 Activation and gradient streams

- A stream is an ordered sequence of tensor records with a declared count.
- Each element carries its own descriptor and hash; the stream descriptor binds
  the ordered list of element hashes.
- Partial streams are explicit: a stream object states `expected_count` and
  `present_count`; a reader MUST NOT treat a truncated stream as complete.

### 2.7 Embedding / KV-cache pages

```
{
  "schema": "weft.kvpage.v1",
  "layer": u64, "head": u64,
  "position_start": u64, "position_count": u64,
  "dtype": "f16" | "bf16" | "f32",
  "layout": "row" | "col",
  "shape": [ u64, ... ]
}
```

- A page is exact bytes with its position range bound in the descriptor.
- Pages from different layers/heads/positions are distinct identities even if
  their bytes are equal, because the descriptor is part of the canonical
  payload binding. Deduplicating across them MUST NOT merge descriptors.
- A KV cache set is reconstructed only through an explicit manifest listing
  pages and their expected ranges.

### 2.8 Token sequences and masks

```
{
  "schema": "weft.tokens.v1",
  "encoding": "u32" | "u16",
  "count": u64,
  "vocabulary": bytes(32) | null,   # hash of the exact vocabulary artifact
  "mask": { "bits": u64, "bytes_hex": ... } | null
}
```

- Token IDs are little-endian unsigned integers, densely packed.
- `count` is elements, not bytes; the byte length is checked.
- `vocabulary` binds the tokenizer/token table by content hash when present. A
  lookup mismatch is `DependencyMismatch`, never a remap.
- Mask bits are MSB-first within each byte (declared explicitly). Padding bits
  beyond `count` MUST be zero.
- The codec does not interpret token text. Detokenization is not a codec
  operation.

### 2.9 Weight deltas and semantic deltas

- **Weight delta**: dependency 0 is the base tensor/weight object; the body is
  the existing DELTA op stream (`COPY`/`INSERT`) over the base's exact bytes. The
  descriptor declares base hash, dtype/shape/layout of both sides, and requires
  equality of semantics metadata. Reconstruction verifies the base hash and the
  resulting whole-object hash.
- **Semantic delta** (structured): a WCB patch stream with ops
  `{"op":"set","path":[...],"value":...}`, `{"op":"delete","path":[...]}`,
  `{"op":"insert","path":[...],"index":u64,"value":...}` and a declared
  `base_hash`. Paths are canonical text segments. Application is left-to-right;
  a failed path is `Malformed`; the result must re-encode to the declared target
  hash. Patches that are not canonical are rejected.
- Neither delta mode may fall back to a full object silently. A base mismatch is
  an error.

### 2.10 Reserved numeric values

No numeric `mode` value is allocated by this document. A future allocation MUST
be recorded in `WEFTCODEC-FORMAT.md` §3 with a new minor/major version, and MUST
ship vectors in the same change. Until then, encoders MUST NOT emit them and
decoders MUST reject candidate values as `UnsupportedMode`.

---

## 3. Acceptance tests a future implementation must pass

For each mode that moves from design to implemented:

1. **Round-trip exactness** on at least: scalar, 1-D, 2-D non-square, 4-D,
   zero-element, maximum-dimension, and adversarial-shape cases.
2. **Descriptor binding**: mutate dtype, shape, layout params, byte order, and
   padding; each mutation is rejected or yields different bytes, never silently
   reinterpreted.
3. **Padding discipline**: non-zero padding rejected; zero padding round-trips.
4. **Shard discipline**: missing/duplicate/out-of-range shards rejected; the
   assembled result matches the monolithic encoding under an explicit assembly
   oracle.
5. **Quantized metadata**: bad scale/zero-point/storage rejected; `i4` nibble
   order frozen by vector.
6. **Token/mask bounds**: count/byte-length mismatch, padding bits set, and
   vocabulary hash mismatch rejected.
7. **Delta exactness**: base mismatch, op-stream truncation, path failure, and
   re-encoded target mismatch rejected.
8. **Cross-language**: Rust and Go produce identical bytes for deterministic
   modes and decode each other's vectors; the reverse direction is tested, not
   assumed.
9. **Limits**: each declared bound is exceeded in at least one adversarial vector
   and fails closed with `LimitExceeded` under the standard hard limits.
10. **No semantic execution**: a vector whose metadata is internally consistent
    but whose tensor values are nonsense still round-trips; the codec must not
    "fix" values.

---

## 4. Explicit non-goals and non-claims

- No training, inference, gradient computation, tokenization, or model
  evaluation.
- No lossy compression claim. Any lossy storage is a different product and MUST
  NOT use `weft/chunk/v1` content identity for the lossy form.
- No cross-vendor dtype equivalence. `bf16` and `f16` are distinct.
- No automatic layout conversion. A consumer asks for the declared layout or
  converts outside the codec.
- No performance or compression numbers in this document. None have been
  measured.
- No claim that any mode here is implemented. The status is `specified, not
  implemented` for all of §2.
