# WEFTCODEC-FORMAT v1.0.0

**Status:** Normative specification. Frozen for `weftcodec-format 1.0`.
**License:** CC BY 4.0.
**Source of truth:** `spec/WEFTCODEC-FORMAT.md` in `infinityabundance/WeftFS`.
Implementations: Rust `weftcodec`, Go `weftcodec-go`.

This document defines exact bytes. Where prose and test vectors disagree, the
vectors in `conformance/v1/` win and this document MUST be corrected. No
implementation-defined structs, Go `gob`, Rust `bincode`, or default CBOR
encoders may appear on the wire.

---

## 1. Conventions

- `u8`, `u16`, `u32`, `u64`, `i64` are little-endian fixed-width integers.
- `varint` is unsigned LEB128 encoding of a `u64`: 7 bits per byte, low group
  first, high bit set means another byte follows. **Minimal encoding is
  mandatory**: the last byte MUST NOT be `0x00` when more than one byte is
  present, and at most 10 bytes are allowed; the tenth byte MUST be `0x00` or
  `0x01`. Non-minimal varints are rejected (`NonCanonical`).
- `bytes(N)` is exactly N raw bytes.
- `str(N)` is N raw bytes that MUST be valid UTF-8 where the field is declared
  a string. Names on POSIX paths are `bytes`, never implicitly UTF-8.
- `H(domain, data) = BLAKE3-256( u32le(|domain|) || domain || data )`, where
  `domain` is ASCII without a NUL. This construction is the only hash form in
  v1. Domain strings are versioned.
- `CRC32C` is Castagnoli (polynomial `0x1EDC6F41`, reflected input/output,
  init `0xFFFFFFFF`, final XOR `0xFFFFFFFF`), stored as `u32le`.
- "MUST reject" means return the typed error class; implementations MUST NOT
  repair malformed input silently.
- All size arithmetic uses checked arithmetic; overflow or exceeding a limit is
  `LimitExceeded` or `Malformed`, never a panic.

### 1.1 Hash domains (v1)

| Purpose | Domain string |
|---|---|
| Raw chunk/plaintext bytes | `weft/chunk/v1` |
| Whole-file content manifest | `weft/manifest/v1` |
| Canonical structured value identity (not the frame header) | `weft/structured/v1` |
| Selector canonical expression | `weft/selector/v1` |
| Snapshot record | `weft/snapshot/v1` |
| Commit record | `weft/store/commit/v1` |
| Relation assertion | `weft/relation/v1` |
| Object version record | `weft/version/v1` |
| Object identity seed | `weft/object/v1` |
| Directory entry identity | `weft/entry/v1` |
| Tensor element payload binding | `weft/tensor/payload/v1` |
| Token-sequence payload binding | `weft/tokens/payload/v1` |

A digest is content identity only. It is not a capability, signature, or
authorization decision.

---

## 2. Frame layout

A frame is:

```
header (96 bytes) || dependency table || chunk table || body || trailer (4 bytes)
```

`body` starts at `96 + table_len`. `table_len` is explicitly stored so readers
do not guess. `trailer` is the last 4 bytes of the frame.

### 2.1 Header (96 bytes, offsets from frame start)

| Offset | Size | Field | Rule |
|---:|---:|---|---|
| 0 | 4 | `magic` | ASCII `WFT1` |
| 4 | 1 | `format_major` | `1` |
| 5 | 1 | `format_minor` | `0` |
| 6 | 1 | `domain` | `0` payload, `1` structured |
| 7 | 1 | `mode` | see §3 |
| 8 | 2 | `flags` | see §2.2 |
| 10 | 2 | `level` | informational encoder level, `0` if none |
| 12 | 8 | `decoded_len` | plaintext length |
| 20 | 8 | `body_len` | encoded body bytes after tables |
| 28 | 4 | `dep_count` | dependency table entries |
| 32 | 4 | `chunk_count` | chunk table entries |
| 36 | 2 | `features` | see §2.2; unknown bits rejected in v1 |
| 38 | 2 | `reserved0` | MUST be `0` |
| 40 | 32 | `content_hash` | `H("weft/chunk/v1", plaintext)` |
| 72 | 4 | `header_crc32c` | CRC of header bytes `[0,72)` and `[76,96)` |
| 76 | 8 | `table_len` | `dep_count*48 + chunk_count*48` |
| 84 | 4 | `reserved1` | MUST be `0` |
| 88 | 8 | `reserved2` | MUST be `0` |

`header_crc32c` is computed over the 92-byte header with bytes `[72,76)` treated
as zero. Readers MUST verify it before trusting any length field.

### 2.2 Flags and features

`flags` bits:

| Bit | Name | Meaning |
|---:|---|---|
| 0 | `CHUNKED` | chunk table present and body is concatenated chunk encodings |
| 1 | `SELF_CONTAINED` | dependency bytes are appended after the trailer (archive form, §7) |
| 2 | `DICT_RAW` | dependency `0` is used as a raw-content zstd dictionary |
| 3 | `ORIGINAL_HASH` | reserved in v1; MUST be `0` |
| 4..15 | critical | unknown set bit ⇒ `UnsupportedFeature` |

`features` bits in v1:

| Bit | Name | Meaning |
|---:|---|---|
| 0 | `DEPENDENCIES` | dependency table may be non-empty |
| 1 | `CHUNK_HASHES` | per-chunk hashes are present (implies `CHUNKED`) |
| 2..15 | critical | unknown set bit ⇒ `UnsupportedFeature` |

### 2.3 Trailer

`frame_crc32c`: `CRC32C` over all bytes from offset 0 through the byte before the
trailer. A frame whose stored trailer differs is `Corrupt`.

A reader MUST reject:
- total length `< 100` or `!= 96 + table_len + body_len + 4`;
- `body_len` that overflows `96 + table_len + body_len + 4`;
- `table_len != dep_count*48 + chunk_count*48`;
- `CHUNKED` unset with `chunk_count != 0`;
- `CHUNKED` set with `chunk_count == 0`;
- `SELF_CONTAINED` unless the extended archive layout of §7 is used;
- reserved fields non-zero;
- any magic, major, or critical-feature mismatch.

---

## 3. Modes

| Value | Name | Body interpretation |
|---:|---|---|
| 0 | `RAW` | body is the plaintext, byte for byte |
| 1 | `ZSTD` | body is one standard Zstandard frame (RFC 8878) |
| 2 | `REFERENCE` | body is empty; plaintext is dependency 0 |
| 3 | `DELTA` | body is a copy/insert op stream over dependency 0 |
| 4 | `DICTIONARY` | body is one zstd frame using dependency 0 as raw dictionary |
| 5 | `STRUCTURED` | body is canonical WCB (see §6); domain MUST be 1 |
| 6 | `TENSOR` | body is `u32le(meta_len) || WCB(meta) || element payload`; WEFT-AI-CODEC §2 |
| 7 | `TOKEN_SEQUENCE` | body is `u32le(meta_len) || WCB(meta) || token payload`; WEFT-AI-CODEC §3 |
| 8 | `QUANTIZED_TENSOR` | body is `u32le(meta_len) || WCB(meta) || packed element payload`; WEFT-AI-CODEC §4 |
| 9 | `FLOAT_TENSOR` | body is `u32le(meta_len) || WCB(meta) || huffman-canonical-v1 exponent plane || raw sign+mantissa plane`; WEFT-AI-CODEC §5 |
| 10 | `TENSOR_SHARDED` | body is `u32le(meta_len) || WCB(meta) || huffman-canonical-v1 stream`; WEFT-AI-CODEC §11 |
| 11 | `WEIGHT_DELTA` | body is `u32le(meta_len) || WCB(meta) || huffman-canonical-v1 planes`; one dependency; WEFT-AI-CODEC §12 |
| 12 | `KV_CACHE` | body is `u32le(meta_len) || WCB(meta) || huffman-canonical-v1 planes`; one page per frame; WEFT-AI-CODEC §13 |
| 13 | `ACTIVATION_STREAM` | body is `u32le(meta_len) || WCB(meta) || huffman-canonical-v1 planes`; one chunk per frame; WEFT-AI-CODEC §14 |
| 14 | `SEMANTIC_DELTA` | body is `u32le(meta_len) || WCB(meta) || huffman-canonical-v1 planes`; one dependency; WEFT-AI-CODEC §15 |
| 15 | `TENSOR_TILED` | body is `u32le(meta_len) || WCB(meta) || tile payload`; WEFT-AI-CODEC §16 |
| 16 | `SPARSE_DELTA` | body is `u32le(meta_len) || WCB(meta) || u32le(run_count) || zero-run records`; one dependency; WEFT-AI-CODEC §17 |
| 17 | `FLOAT_TILED` | body is `u32le(meta_len) || WCB(meta) || tile payload`; WEFT-AI-CODEC §18 |
| 18 | `FLOAT_TILED_STREAMS` | body is `u32le(meta_len) || WCB(meta) || per-tile interleaved stream payload`; WEFT-AI-CODEC §19 |
| 19..127 | — | reserved; `UnsupportedMode` |
| 128..255 | — | experimental; MUST be rejected for persistence |

Global mode rules:

- `RAW`: `body_len == decoded_len`. Without `CHUNKED`, `chunk_count == 0`;
  with `CHUNKED`, chunks are literal byte ranges and the chunk table rules of
  §6 apply.
- `ZSTD`: `chunk_count == 0` unless `CHUNKED`; `dep_count == 0` unless chunk
  dictionaries are used (not v1).
- `REFERENCE`: `body_len == 0`, `dep_count == 1`, `chunk_count == 0`;
  `decoded_len == dependency.decoded_len`; plaintext is the dependency bytes.
- `DELTA`: `dep_count == 1`; `chunk_count == 0`; body is §4.
- `DICTIONARY`: `dep_count == 1`, `flags.DICT_RAW` set, `chunk_count == 0`;
  body decodes with the dependency bytes as a raw content dictionary.
- `STRUCTURED`: `domain == 1`, `mode == 5`, `body_len == decoded_len`; body is
  canonical WCB; `dep_count` MUST be `0` in v1; `chunk_count` MUST be `0`.
- AI modes 6–17: `domain == 0`, `chunk_count == 0`, `flags.DICT_RAW` clear,
  `body_len >= 4`. The frame's `decoded_len` is the reconstructed payload
  length and `content_hash` is `H("weft/chunk/v1", reconstructed payload)`.
  Modes 6, 7, 8, 9, 10, 12, 13, and 15 have `dep_count == 0`; modes 11, 14, and
  16 have `dep_count == 1` naming the base tensor. The body is exactly
  `u32le(meta_len) || WCB(meta) || payload`, where the WCB descriptor is
  canonical and binds the payload with the mode-specific domain from §1.1.
  Every entropy-coded plane uses `huffman-canonical-v1` (WEFT-AI-CODEC
  §5.3.1), never a library zstd frame, so these modes are deterministic and
  Rust and Go MUST emit byte-identical frames. Mode 16 is not entropy-coded at
  all: its payload is a varint run stream (WEFT-AI-CODEC §17.3). Full
  descriptor, payload, and validation rules are normative in
  `WEFT-AI-CODEC.md` §2–§17. A decoder that does not implement one of these
  modes MUST reject it as `UnsupportedMode`; it MUST NOT treat the payload as
  opaque bytes.

Zstd decoding limits: window log ≤ 27, decoded output ≤ `decoded_len`, and
`decoded_len` MUST be stated before decompression. Implementations MUST fail
closed if the decompressor yields a different length.

Zstd framing rules (apply to `ZSTD`, `DICTIONARY`, and every chunk of a
`CHUNKED` ZSTD frame): the body MUST be a complete, well-formed zstd stream. An
empty body is `Malformed`, even when `decoded_len == 0`; the encoder emits a
valid minimal zstd frame for empty plaintext. If the zstd frame header declares
a content size, that size MUST equal `decoded_len` (`Corrupt` otherwise), even
when the actual decompressed byte count equals `decoded_len`. The actual
decompressed length MUST equal `decoded_len` (`Corrupt` otherwise).

---

## 4. DELTA op stream (mode 3)

The delta is a sequence of operations until the body is exhausted:

| Opcode | Name | Payload |
|---:|---|---|
| `0x00` | `COPY` | `varint offset`, `varint length` (both in dependency 0) |
| `0x01` | `INSERT` | `varint length`, then `length` literal bytes |
| `0x02..0xFF` | — | `Malformed` |

Rules:
- `offset + length` MUST NOT exceed `dependency.decoded_len` (checked).
- A zero-length `COPY` or `INSERT` is `Malformed`.
- Total inserted + copied bytes MUST equal `decoded_len` exactly.
- Total emitted bytes are bounded by the decoder's `max_decoded_bytes`.
- The decoder MUST verify the dependency hash before using it.
- The encoder MUST NOT emit overlapping COPY semantics beyond simple
  offset/length reads; each COPY reads the original dependency, not output.

Deltas are exact: `decoded == expected` byte for byte or the frame is rejected.

---

## 5. Dependency table (48 bytes per entry)

| Offset | Size | Field |
|---:|---:|---|
| 0 | 32 | `hash` = `H("weft/chunk/v1", dependency plaintext)` |
| 32 | 8 | `decoded_len` of the dependency |
| 40 | 1 | `dep_domain` (`0` payload, `1` structured, `2` dictionary bytes) |
| 41 | 1 | `dep_flags` (bit 0 `REQUIRED`; all v1 deps MUST set it) |
| 42 | 2 | `reserved` MUST be `0` |
| 44 | 4 | `codec_hint` (informational mode of the dependency frame) |

Entries are ordered: dependency 0 is the delta/dictionary base. Other entries
are permitted only for future modes and MUST be rejected by v1 decoders when the
mode does not define them.

Resolution contract:

```
resolve(hash, decoded_len, dep_domain, limits, seen) -> bytes | typed_error
```

The resolver MUST return bytes whose length equals `decoded_len` and whose
`H("weft/chunk/v1", bytes)` equals `hash`. Missing, denied, stale, or mismatched
dependencies are typed errors (`DependencyMissing`, `DependencyDenied`,
`DependencyMismatch`). **No decoder may consult live graph context, infer a
substitute base, or fall back to another representation.**

---

## 6. Chunk table (48 bytes per entry, `CHUNKED`)

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | `encoded_off` (offset within body) |
| 8 | 4 | `encoded_len` |
| 12 | 4 | `decoded_len` |
| 16 | 32 | `hash` = `H("weft/chunk/v1", chunk plaintext)` |

Rules:
- `encoded_off` starts at 0 and is strictly increasing; chunk extents MUST be
  contiguous with no gaps or overlap and MUST end exactly at `body_len`.
- Sum of `decoded_len` equals header `decoded_len`.
- Each chunk is decoded independently **as if** it were the body of a frame
  with the same mode and the same dependency 0 (if any). For `RAW`, chunks are
  literals. For `ZSTD`/`DICTIONARY`, each chunk is its own zstd frame.
- Per-chunk hash is mandatory when `CHUNKED` is set; mismatch is `Corrupt`.
- Random access reads only the chunks intersecting the requested range.

Chunking is an encoding choice, not a semantic change: a chunked frame and an
unchunked frame with the same plaintext have the same `content_hash`, and both
decode to identical bytes.

---

## 7. Self-contained archive (v1, experimental)

An archive is a separate container, not a frame:

```
"WFA1" (4) || u32le version=1 || u32le entry_count ||
  entry_count * ( u64le frame_len || frame_bytes ) ||
  u32le archive_crc32c
```

- Entries MUST be topologically ordered: every dependency appears in an earlier
  entry, so in-archive resolution never cycles.
- `entry_count` is bounded (default 1,000,000).
- `frame_len` MUST be ≥ 100 and the total MUST fit the input.
- Duplicate frame hashes MAY appear only if byte-identical.
- The archive CRC covers everything before it.
- A resolver for an archive indexes entries by `content_hash` and by dependency
  hashes; external resolution MUST NOT happen for `SELF_CONTAINED` frames.
- An archive is not a substitute for a snapshot; it is a transport container.

`SELF_CONTAINED` + appended bytes (§2.2 bit 1) is reserved and not produced in
v1. Implementations MUST reject it rather than guess.

---

## 8. Structured domain: WCB canonical bytes

`STRUCTURED` frames carry a single WCB value. WCB is a strict subset of
deterministic CBOR concepts with explicit bytes, not RFC 8949 defaults.

### 8.1 Tags

| Tag | Type | Encoding after tag |
|---:|---|---|
| `0x00` | null | nothing |
| `0x01` | false | nothing |
| `0x02` | true | nothing |
| `0x03` | unsigned int | `varint` value |
| `0x04` | signed int | `varint` zigzag: `(v << 1) ^ (v >> 63)` on 64-bit two's complement |
| `0x05` | text | `varint len`, UTF-8 bytes |
| `0x06` | bytes | `varint len`, raw bytes |
| `0x07` | array | `varint count`, then count values |
| `0x08` | map | `varint count`, then count `(key,value)` pairs |
| `0x09` | timestamp | zigzag i64 nanoseconds since Unix epoch, UTC |
| `0x0A..0xFF` | — | `NonCanonical` in v1 |

Floating point, indefinite lengths, tags, and duplicate keys are not in v1.

### 8.2 Canonical map ordering

- Keys MUST be text (`0x05`) in v1.
- Entries are sorted by the **full encoded key bytes** (tag + varint length +
  bytes), compared as unsigned byte strings, ascending.
- Duplicate encoded keys are rejected.
- Missing and null are distinct.

### 8.3 Canonical validation

A decoder MUST reject: non-minimal varints, unsorted map keys, duplicate keys,
non-text map keys, trailing bytes after the value, nesting deeper than 64,
element counts above the configured limit, and strings longer than the limit.
Canonical bytes are unique for a given value: two encoders given the same value
MUST produce identical bytes.

---

## 9. Selector canonicalization

The selector AST and its canonical WCB form are defined in
`spec/WEFT-SELECTORS.md`. This format defines only the identity derivation:

```
selector_bytes  = WCB(selector map)
selector_id     = H("weft/selector/v1", selector_bytes)
```

`SelectorId` identifies the expression. It MUST NOT be treated as an object ID,
a snapshot, or evidence that any object matches.

---

## 10. Limits (decoder defaults, all configurable downward)

| Limit | Default |
|---|---:|
| `max_frame_bytes` | 1 GiB |
| `max_decoded_bytes` | 4 GiB |
| `max_dependency_depth` | 8 |
| `max_dependency_count` | 64 |
| `max_total_fetched_bytes` | 8 GiB |
| `max_chunks` | 1,048,576 |
| `max_dictionary_bytes` | 64 MiB |
| `max_structured_depth` | 64 |
| `max_structured_items` | 1,000,000 |
| `max_string_bytes` | 256 MiB |
| `max_delta_ops` | 16,777,216 |
| `max_tensor_elements` | 4,294,967,296 |
| `max_tensor_rank` | 8 |
| `max_tokens` | 16,777,216 |
| `max_sparse_runs` | 16,777,216 |

A decoder MUST enforce every applicable limit before allocating. A frame that
would exceed a limit is `LimitExceeded`, even if it is otherwise well formed.

---

## 11. Error taxonomy (normative classes)

| Class | Meaning |
|---|---|
| `Malformed` | structural/byte error, including bad varints and bad tables |
| `Corrupt` | checksum, hash, or length mismatch |
| `UnsupportedVersion` | major/minor not supported |
| `UnsupportedMode` | mode not implementable |
| `UnsupportedFeature` | unknown critical flag/feature |
| `LimitExceeded` | a configured resource bound would be exceeded |
| `DependencyMissing` | dependency cannot be found |
| `DependencyDenied` | authorization refuses the dependency |
| `DependencyMismatch` | resolved bytes fail hash or length |
| `DependencyCycle` | reconstruction dependency graph is cyclic |
| `NonCanonical` | structured bytes violate WCB canonical form |
| `Io` | transport/read failure |

Error strings are human text; the class is the interoperable contract.

---

## 12. Determinism requirements

Given the same plaintext, mode, level, dependency bytes, and chunking profile,
both implementations MUST produce byte-identical frames for `RAW`,
`REFERENCE`, `DELTA`, and `STRUCTURED`. `ZSTD` and `DICTIONARY` frames are
byte-identical only when both use the same zstd library version and level;
therefore conformance requires **decode equivalence** and recorded encoder
identity, not byte equality, for those modes. This is a deliberate, documented
asymmetry, not a hidden fallback.

`STRUCTURED` canonical bytes MUST be identical across languages for the same
typed value. Golden vectors freeze them.

`TENSOR` and `TOKEN_SEQUENCE` are deterministic: canonical WCB descriptors and
minimal-LEB128 payloads mean both implementations MUST emit byte-identical
frames for the same payload and descriptor, and MUST decode each other's
frames. Their `content_hash` is over the declared payload bytes, not over the
compressed or descriptor bytes.

---

## 13. Compatibility policy

- Adding an optional field requires a minor version and a default that older
  readers already implement; otherwise it requires a major version.
- Allocating a previously reserved mode value is an additive extension that
  old readers must reject as `UnsupportedMode` (fail closed). `TENSOR` (6),
  `TOKEN_SEQUENCE` (7), `QUANTIZED_TENSOR` (8), `FLOAT_TENSOR` (9),
  `TENSOR_TILED` (15), `SPARSE_DELTA` (16), `FLOAT_TILED` (17), and
  `FLOAT_TILED_STREAMS` (18) were allocated this way;
  `WEFT-AI-CODEC.md` is normative for their bodies. Historical vector
  `017_unknown_mode` asserted that value 6 was unknown; it is retained
  byte-for-byte and marked `superseded_by` the current reserved-value vector
  `017_unknown_mode_reserved19`.
- Unknown critical flags/features/modes fail closed.
- Frozen v1 vectors are append-only; they are never regenerated to hide a
  change.
- See `spec/WEFT-COMPATIBILITY.md`.
