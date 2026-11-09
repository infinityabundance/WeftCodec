# WEFT-KV-PROVENANCE v1.0.0 — provenance-bound KV cache pages

**Status:** Normative for `weft.ai.kv_prov.v1` and mode 19 `KV_CACHE_PROV`.
Implemented in Rust `weftcodec::kv_prov` and Go `modes` (see §9 for the exact
surface). The store-side scope and capability path (§10) is implemented in the
Rust store only; that gap is stated, not hidden.

**Depends on:** `WEFTCODEC-FORMAT.md` (framing, `H(domain,data)`, WCB, limits),
`WEFT-AI-CODEC.md` §13 (mode 12 `KV_CACHE`, whose page payload this mode reuses
verbatim), `WEFT-CAPABILITIES.md` (capability envelopes).

---

## 1. The defect this document closes

Mode 12 `KV_CACHE` freezes the page bytes and the page geometry. It does **not**
bind the page to the model weights, adapters, tokenizer, attention
configuration, positional treatment, cache layout, or sharing domain that
produced it. Provenance therefore lived *outside* the frame: in store
attributes (`weft.kv.descriptor`, mode 12) and in the prefix record
(`weft.kv.prefix.v1`).

That is not enough. A consumer that resolves a KV page by **content hash
alone** — a shared cache, a content-addressed object store, a bundle of frames,
a third-party index — cannot tell two pages apart when their token bytes and
geometry coincide but their provenance does not. Reusing such a page is silent
and catastrophic: the model attends with a foreign adapter, a foreign tokenizer,
or a foreign attention window, and nothing in the bytes refuses it.

Mode 19 closes this structurally. The provenance dimensions are **inside the
frame descriptor**, the frame's identity is a hash **over** those dimensions and
the payload hash, and a decoder refuses any frame whose dimensions are
internally inconsistent or inconsistent with the frozen page metadata.

Mode 12 remains frozen and byte-identical. Mode 19 is additive: the payload is
the exact mode-12 page payload, so the only difference on the wire is the
envelope of the descriptor.

---

## 2. The provenance dimension registry

The registry is **canonical and ordered**. The order below is normative for
validation and for this document; the wire carrier is a WCB map, whose keys are
already sorted by encoded key bytes.

| # | Dimension | Carrier | Canonical rule |
|---:|---|---|---|
| 1 | model weights root | `model_root` (bytes) | exactly 32 bytes, not all zero |
| 2 | adapter/LoRA set root | `adapter_roots` (array of bytes) + `adapter_set_root` (bytes) | each root exactly 32 bytes and non-zero; the list is sorted ascending by raw bytes and contains no duplicate; `adapter_set_root = H("weft/kv/adapters/v1", WCB(adapter_roots))`; the empty list is allowed |
| 3 | tokenizer root | `tokenizer` (map) + `tokenizer_root` (bytes) | `tokenizer` is exactly `{scheme: text, vocab_size: u64}` with non-empty `scheme` and `vocab_size >= 1`; `tokenizer_root = H("weft/kv/tokenizer/v1", WCB(tokenizer))` |
| 4 | attention configuration | `attention` (map) | exactly `{head_count: u64, head_dim: u64, sliding_window: u64, rope_theta_bits: u64, rope_scale_bits: u64}`; `head_count` and `head_dim` fit `u32` and are `> 0`; `rope_*_bits` are IEEE-754 binary64 bit patterns and MUST be finite; `sliding_window = 0` means "no sliding window" |
| 5 | positional treatment | `positional` (text) | exactly `rope`, `alibi`, or `none` |
| 6 | cache layout | `layout` (map) | exactly `{dtype: text, page_rows: u64, layer_count: u64}`; `dtype` is `bf16`, `fp16`, or `fp32`; `page_rows >= 1`; `layer_count >= 1` |
| 7 | sharing-domain salt | `salt` (bytes) | exactly 32 bytes; required; all-zero is legal and documented as offering **no** unlinkability |
| 8 | extension dimensions | `extensions` (array of map) | each entry is exactly `{name: text, value: <any canonical WCB value>}`; `name` is non-empty; entries are sorted strictly ascending by `name` (byte-wise); names are unique; the list may be empty |

There is no ninth dimension in v1. A new dimension is a new schema version,
never a new key in `weft.ai.kv_prov.v1`.

### 2.1 Cross-checks against the frozen page

The provenance dimensions are not free-floating. A mode-19 descriptor also
carries the complete frozen mode-12 page descriptor (§3). Decoding requires:

* `layout.dtype` equals the page `dtype`;
* `layout.page_rows` equals the page `shape[0]` (`seq_end - seq_start`);
* `attention.head_count` equals the page `head_count`;
* `attention.head_dim` equals the page `head_dim`;
* the page `layer` is below `layout.layer_count`.

Any disagreement is `Malformed`. These checks are what make "the descriptor
says one thing and the payload was produced another way" unrepresentable
without a hash change, and they are why the four `*_mismatch` conformance
rejects exist.

---

## 3. Wire shape

### 3.1 Frame

```text
domain        = 0 (payload)
mode          = 19
flags         = 0
dep_count     = 0
chunk_count   = 0
decoded_len   = page_bytes
content_hash  = H("weft/chunk/v1", page element bytes)   # identical to mode 12
body          = u32le(descriptor_len) || WCB(descriptor) || payload
```

`payload` is **byte-for-byte the mode-12 page payload**: the canonical
row-major little-endian element bytes split into byte planes (plane 0 = least
significant byte), each plane coded with its own `huffman-canonical-v1` stream
and concatenated in plane order (`WEFT-AI-CODEC.md` §13.3). No new coder, no
new plane order, no new table. `page_bytes`/`decoded_len` and `content_hash`
follow mode 12 unchanged.

### 3.2 Descriptor

Schema: `weft.ai.kv_prov.v1`. Exactly these keys, no others:

| Key | Type | Rule |
|---|---|---|
| `schema` | text | exactly `weft.ai.kv_prov.v1` |
| `page` | map | the frozen mode-12 descriptor of `WEFT-AI-CODEC.md` §13.2 **verbatim**, including its `schema` key: `schema` (`weft.ai.kv_cache.v1`), `model`, `layer`, `session`, `page_index`, `page_count`, `seq_start`, `seq_end`, `dtype`, `head_count`, `head_dim`, `shape`, `page_bytes`, `plane_codec`, `payload_len` (fifteen keys, no others) |
| `model_root` | bytes | 32 bytes, non-zero |
| `adapter_roots` | array(bytes) | sorted, unique, non-zero, may be empty |
| `adapter_set_root` | bytes | `H("weft/kv/adapters/v1", WCB(adapter_roots))` |
| `tokenizer` | map | `{scheme, vocab_size}` |
| `tokenizer_root` | bytes | `H("weft/kv/tokenizer/v1", WCB(tokenizer))` |
| `attention` | map | see §2 |
| `positional` | text | `rope` / `alibi` / `none` |
| `layout` | map | see §2 |
| `salt` | bytes | 32 bytes |
| `extensions` | array(map) | sorted by `name`, unique names |
| `payload_hash` | bytes | `H("weft/chunk/v1", page element bytes)`; MUST equal the frame `content_hash` |

`page.payload_len` is the exact stored payload length in bytes.
`page.page_bytes` is the exact decoded page length.

The `page` map is the canonical mode-12 descriptor value, so the frozen
mode-12 validation is reused unchanged (including its `schema` check). The
provenance dimensions and `payload_hash` sit beside it.

### 3.3 Identity

```text
descriptor_root = H("weft/kv/prov/v1", WCB(descriptor))
```

`descriptor_root` is the KV provenance identity. It is **derived, not stored**
(a field holding its own hash would be circular); the decoder recomputes it and
returns it alongside the decoded page.

Because the descriptor includes every dimension, the layout cross-checks, and
`payload_hash`, this holds by construction:

> Two KV pages with identical token bytes and geometry but different model,
> adapter set, tokenizer, attention configuration, positional treatment, layout,
> salt, or extension dimensions necessarily have different `descriptor_root`s.

The only way to obtain the same root is to present the same descriptor, which
means the same dimensions and the same payload hash.

---

## 4. Fail-closed resolution rule

> A resolver that is asked for a page under a scope that does not match the
> page's descriptor MUST fail closed. It MUST NOT return bytes.

"Scope" is whatever the caller's authorization names: for the store path
(§10) it is the provenance descriptor id plus the tenant namespace. Concretely:

* The codec emits **zero bytes** before every structural check passes. Decode
  returns either a complete `DecodedKvProvPage` or an error.
* The store checks authorization **before** it reads or decodes the frame; a
  scope that does not name the descriptor id is `Unauthorized` with the single
  non-disclosing text already used by the KV surface, and no byte is emitted.
* A page whose descriptor matches its own bytes but not the requested scope is
  the **same denial** as a page that does not exist. Existence is not disclosed.

There is no similarity rule, no nearest-neighbour fallback, no "close enough"
provenance, and no path that decodes first and authorizes afterwards.

---

## 5. The sharing-domain salt

`salt` is a 32-byte dimension included in `descriptor_root`. Its purpose is
**cross-tenant unlinkability of identifiers**.

What it does:

* Two sharing domains (tenants, reservations, jobs, trust boundaries) that
  otherwise hold the *same* KV page derive *different* `descriptor_root`s.
* A resolver or index that keys only on `descriptor_root` therefore cannot
  silently reuse a page across domains, and cannot infer from a root list that
  two domains hold the same page.
* Because the salt is part of the identity, rotating the salt is a new
  identity — old roots never validate a new-domain page.

What it does **not** do (non-claims):

* It does not encrypt or redact the descriptor. The salt and every dimension
  are visible in the frame body to anyone who can read the frame.
* It does not hide timing, object sizes, access patterns, request rates, or the
  fact that KV storage and reuse occurred.
* It does not hide equality *inside* one domain: a holder of both frames sees
  both salts.
* It is not a key, a nonce, a secret, or a substitute for capability
  authorization. A guessed or shared salt is not an exploit: the frame still
  must pass every structural check and the store still checks the scope.
* An all-zero salt is legal but provides no unlinkability; it is recorded
  explicitly so that "no salt" is representable without being invisible.

---

## 6. Descriptor validation order (normative for error classes)

Decoding a frame:

1. Frame header mode is not 19 → `Malformed`. Body shorter than its 4-byte
   descriptor length → `Malformed`; descriptor extends past the body →
   `Malformed`.
2. WCB decode of the descriptor: non-minimal varints, unsorted or duplicate map
   keys, non-text map keys, trailing bytes, invalid UTF-8, unknown tags →
   `NonCanonical`; truncation → `Malformed`.
3. Descriptor is not a map, has an unknown key, is missing a key, or a value has
   the wrong WCB type → `Malformed`; `schema` differs → `Malformed`.
4. The `page` map is decoded and validated exactly as mode 12
   (`WEFT-AI-CODEC.md` §13.4); its errors keep their mode-12 classes
   (`Malformed` / `LimitExceeded`).
5. A provenance value has the wrong type or length (including `salt` or any
   root not exactly 32 bytes) → `Malformed`; a root that is all zero where §2
   forbids it → `Malformed`.
6. `adapter_roots` is not sorted, has a duplicate, or `adapter_set_root` differs
   from the derived root → `Malformed`.
7. `tokenizer` is malformed or `tokenizer_root` differs from the derived root →
   `Malformed`.
8. `attention.head_count`/`head_dim` are zero, exceed `u32`, or a `rope_*_bits`
   value is NaN or infinite → `Malformed`; `layout.dtype` is unknown or any §2.1
   cross-check fails → `Malformed`.
9. `extensions` is unsorted by `name` or has a duplicate name → `Malformed`.
10. `page.payload_len` differs from the stored payload length → `Malformed`;
    frame `decoded_len` differs from `page.page_bytes` → `Corrupt`.
11. `payload_hash` differs from the frame `content_hash` → `Corrupt`.
12. Huffman plane errors keep `WEFT-AI-CODEC.md` §5.3.1 classes; payload bytes
    left over after the declared planes → `Corrupt`.
13. The reconstructed element bytes hash differently from `payload_hash` (and
    therefore from the frame `content_hash`) → `Corrupt`.

`H("weft/kv/prov/v1", …)` is computed only after the descriptor is fully valid,
so a caller never sees a root for a descriptor that cannot decode.

---

## 7. Versioning and compatibility

* `weft.ai.kv_prov.v1` is the only schema this document defines. `weft.ai.kv_prov.v2`
  would be a new mode value, not a new key.
* Mode 12 stays byte-identical. A mode-12 frame is never accepted as mode 19 and
  a mode-19 frame is never accepted as mode 12 (the mode byte is not ignored).
* A value of 20 is reserved (`UnsupportedMode`); the reserved-mode reject vector
  moved from 19 to 20, and the historical 19 vector is retained with
  `superseded_by`.
* No new frame-domain, chunk-table, dependency, or coder semantics are added.
  The frozen v1 format is untouched.

---

## 8. Conformance vectors

`conformance/v1/kv-prov/` is byte-identical in `WeftCodec` and `WeftCodec-Go`:

* valid: `001_bf16_single_layer`, `002_fp16_adapter_set`,
  `003_fp32_multi_layer`, `004_empty_adapter_set`, `005_tenant_salt_a`,
  `006_tenant_salt_b`, `007_extension_dimension`, `008_two_pages`;
* rejects: `101_missing_salt`, `102_unknown_dimension`,
  `103_duplicate_dimension`, `104_unsorted_extensions`,
  `105_adapter_root_mismatch`, `106_tokenizer_mismatch`,
  `107_attention_mismatch`, `108_layout_mismatch`, `109_payload_truncated`,
  `110_payload_hash_mismatch`, `111_descriptor_non_canonical`,
  `112_reserved_mode_20`.

Each valid vector carries the full descriptor JSON, the exact element bytes,
and the committed frame. Both implementations decode it, re-encode it to the
identical bytes, recompute `descriptor_root`, and check it against the committed
value. Each reject vector states its error class and both implementations must
produce that class.

---

## 9. Codec surface

Rust (`weftcodec::kv_prov`):

```text
KvProvenanceInput::{new, with_adapter, with_extension, canonicalize}
KvProvenance, KvAttention, KvPositional, KvLayout, KvExtension
KvProvDescriptor::{from_page, canonical_bytes, root, model_root, salt, ...}
encode_kv_prov_page(input, elements, provenance, limits) -> Vec<u8>
decode_kv_prov_page(frame, limits) -> DecodedKvProvPage
kv_prov_metadata(frame, limits) -> KvProvDescriptor
```

`DecodedKvProvPage { descriptor, root, bytes }`. `root` is
`H("weft/kv/prov/v1", WCB(descriptor))`; `bytes` are the exact element bytes.

Go (`modes`): `EncodeKvProvPage`, `DecodeKvProvPage`, `KvProvMetadata`,
`KvProvenanceInput`, `KvProvenance`, `KvProvDescriptor`, and the same helper
types; `DecodeKvProvPage` returns `(DecodedKvProvPage, error)` with `Root`
populated.

CLI (both languages):

```text
weftcodec encode --mode kv-prov --input PAGE --output FRAME --metadata PAGE.json --provenance PROV.json
weftcodec decode --input FRAME --output PAGE
```

`PROV.json` carries the eight dimensions plus the derived roots (they are
recomputed and must agree). `PAGE.json` is the mode-12 metadata file.

---

## 10. Store scope and capability binding (Rust store)

`src/kv_capability.rs` accepts mode-19 objects through a distinct descriptor
schema, `weft.kv.prov_capability.v1`, whose canonical WCB map is the mode-12
capability descriptor plus one field:

| field | type | meaning |
|---|---|---|
| `provenance_root` | hex text | the `descriptor_root` of the mode-19 frame |

Its id is `H("weft/kv/prov_capability/v1", canonical_bytes)`. A capability
envelope must name that id in `roots`, grant `kv.<tenant>`, and name the same
principal. The uniform denial text
(`kv page is not available to this principal`) is unchanged.

Read ordering is unchanged and normative: object record → descriptor → derived
descriptor id → envelope validity → tenant/namespace/root grant → POSIX
authorization → **only then** read and decode. After decode the descriptor is
re-derived from the frame and compared field by field, including
`provenance_root` (which must equal the recomputed descriptor root). A mismatch
is `Corrupt` and is reported only to an already-authorized reader.

Consequence: knowing a `ContentId`, an `ObjectId`, or the raw frame bytes yields
no decode. The object cannot be fetched "by content hash alone" into a decode
path, because the only decode path checks the provenance-bound descriptor id
first and emits zero bytes on any mismatch.

---

## 11. Non-claims

* No inference-engine integration (vLLM, SGLang, TensorRT-LLM); pages are
  synthetic/deterministic fixtures in the courts.
* No claim that provenance identity improves model output correctness by
  measurement. The claim is structural: a mismatched scope cannot produce bytes.
* No timing, throughput, or fleet claim from this document.
* No encryption, confidentiality, or access-pattern privacy; see §5.
* Go has no store-side KV capability path (the whole KV capability surface is
  Rust-only today), so §10 is Rust-only. The Go codec implements §3–§7 and the
  vectors.
* The `weft.kv.prefix.v1` record still exists and its `compatibility()` scan is
  unchanged; it is not a substitute for a mode-19 descriptor and this document
  does not claim it is.
