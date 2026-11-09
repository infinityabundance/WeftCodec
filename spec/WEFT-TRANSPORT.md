# WEFT-TRANSPORT v1.0.0 — transport-aligned frame container

**Status:** Normative for `weft.transport.v1`. Implemented in Rust
(`weftcodec::transport`) and in Go (`WeftCodec-Go/transport`), with a court in
`examples/transport_court.rs` (`weft.transport.v1` receipt).
**License:** CC BY 4.0.
**Source of truth:** `spec/WEFT-TRANSPORT.md` in `infinityabundance/WeftFS`.

**Depends on:** `WEFTCODEC-FORMAT.md` (the WFT1 frame, `H(domain, data)`,
limits), `WEFT-AI-CODEC.md` (per-tile and per-group roots in modes 15–20).

---

## 1. What this format is

`weft.transport.v1` is an **additive envelope**. It carries one or more
existing, unmodified WeftCodec frames so that a *section* of the container can
be fetched with one aligned range read and handed to a transport that requires
alignment.

The enclosed bytes are unchanged:

* a frame's header, tables, body, and trailer are byte-for-byte the frame that
  would be stored on its own;
* a frame's mode number is unchanged and no mode is renumbered;
* a frame's own content root (`frame.header.content_hash`) is unchanged and is
  copied into the section table;
* padding added by the envelope is outside every frame and is not hashed.

The envelope adds no compression, no encryption, and no transform. It is a
layout.

### 1.1 Non-claims

This document and its court make **only** layout and parse claims:

* this is a **layout, not a transport**;
* **no DMA, GPU memory, RDMA, kernel-bypass, or deployed-3FS claim** is made;
* alignment is **necessary but not sufficient** for zero-copy. A decoder that
  materialises an owned buffer still copies, whatever the alignment;
* the court measures container parse cost, range-read bytes, wall time on one
  host with a warm page cache, and padding overhead. It is not a throughput or
  latency claim about any storage system;
* a single section can be *fetched* without reading the other sections, but a
  reader that must verify the container-level `payload_hash` reads every
  section. Both costs are reported separately.

---

## 2. Byte layout

```text
offset 0                      offset 4096
│                             │
▼                             ▼
┌─────────────────────────────┬──────────────────┬─────────┬─────────┬─────────┐
│ transport header (4096 B)   │ section table    │ padding │ section │ padding │ …
└─────────────────────────────┴──────────────────┴─────────┴─────────┴─────────┘
                               table_offset = 4096   section[0].offset
```

Total container length is `total_len` and MUST equal the physical byte length
of the object.

### 2.1 Transport header (4096 bytes)

All integers are little-endian. The header is fixed-width so a reader can
issue exactly one 4 KiB range read and then one range read of
`section_count * 56` bytes for the table.

| Offset | Size | Field | v1 rule |
|---:|---:|---|---|
| 0 | 4 | `magic` | exactly ASCII `WFTT` |
| 4 | 2 | `format_major` | exactly `1` |
| 6 | 2 | `format_minor` | exactly `0` |
| 8 | 4 | `flags` | exactly `0` in v1; any other value is `UnsupportedFeature` |
| 12 | 4 | `section_count` | `1 ..= 4096`; `0` is `Malformed`; `> 4096` is `LimitExceeded` |
| 16 | 8 | `section_table_offset` | exactly `4096` in v1; anything else is `Malformed` |
| 24 | 8 | `total_len` | exact physical length, `>= 4096 + 56*section_count` |
| 32 | 32 | `payload_hash` | `H("weft/transport/v1", enclosed bytes)` — see §2.3 |
| 64 | 4032 | reserved | MUST be zero; non-zero is `Malformed` |

`payload_hash` covers the enclosed frame bytes, concatenated in section-table
order, and **not** any padding.

### 2.2 Section table

The table starts at byte 4096 and has exactly `section_count` entries of 56
bytes each. Entries are in strictly increasing `offset` order.

| Offset | Size | Field | Rule |
|---:|---:|---|---|
| 0 | 8 | `offset` | absolute byte offset in the container; a multiple of `alignment` |
| 8 | 8 | `length` | exact enclosed frame length, `> 0` |
| 16 | 4 | `alignment` | one of `512`, `4096`, `65536`; any other value is `Malformed` |
| 20 | 1 | `mode` | the enclosed frame's mode byte; must equal it (`Corrupt` otherwise) |
| 21 | 1 | `role` | `0` payload, `1` dependency, `2` metadata, `3` provenance; other values are `Malformed` |
| 22 | 2 | reserved | MUST be zero; non-zero is `Malformed` |
| 24 | 32 | `root` | must equal the enclosed frame's `content_hash` (`Corrupt` otherwise) |

`role` is a coarse, non-normative hint for a consumer ("why is this frame
here"). It never changes identity, ordering, decoding, or GC of the enclosed
frame. Two containers with the same frames and different roles have different
container bytes and the same enclosed frame roots.

### 2.3 `payload_hash`

```text
payload_hash = H("weft/transport/v1", frame_bytes[section 0] || … || frame_bytes[section n-1])
```

It binds the set and order of enclosed frames at the container level. It does
**not** replace any frame's own content hash: a frame is still verified against
its own root, and a section is still verified against `section.root`.

---

## 3. Encoder rules (deterministic)

Given an ordered list of complete frames and one alignment `A` from the allowed
set:

1. Parse every frame (the encoder rejects bytes that are not a valid frame) and
   record its `content_hash` and mode.
2. `payload_hash` is computed over the frames in the supplied order.
3. `table_end = 4096 + 56 * section_count`.
4. For each section in order: `offset = align_up(cursor, A)`, then the frame
   bytes occupy `[offset, offset + length)`, then `cursor = offset + length`.
5. `total_len = align_up(cursor, A)`.
6. Every byte not occupied by a section is **zero**.

There is no table sorting, no deduplication, and no frame reordering. The
output is a pure function of (ordered frames, alignment). Two encoders that
implement this document produce byte-identical containers for the same input.

The container is a single object; a caller that wants frame-level deduplication
across containers already has content addressing for that.

---

## 4. Reader rules

### 4.1 Strict validation order

A reader that validates a complete container MUST check, in this order, and
MUST return the stated class:

| # | Check | Class |
|---:|---|---|
| 1 | length `>= 4096` | `Malformed` |
| 2 | magic | `Malformed` |
| 3 | major/minor | `UnsupportedVersion` |
| 4 | `flags == 0` | `UnsupportedFeature` |
| 5 | `section_count` in `1..=4096` | `Malformed` / `LimitExceeded` |
| 6 | `section_table_offset == 4096` | `Malformed` |
| 7 | table fits before `total_len`, checked arithmetic | `Malformed` |
| 8 | `total_len == physical length` | `Malformed` |
| 9 | header reserved bytes zero | `Malformed` |
| 10 | per entry: `alignment` in the allowed set | `Malformed` |
| 11 | per entry: `offset % alignment == 0` | `Malformed` |
| 12 | per entry: `offset >= table_end` and `length > 0` | `Malformed` |
| 13 | per entry: `offset + length <= total_len`, checked | `Malformed` |
| 14 | entry offsets strictly increasing and non-overlapping | `Malformed` (duplicate offsets are also `Malformed`) |
| 15 | per entry: table reserved field zero | `Malformed` |
| 16 | per entry: `role` in `0..=3` | `Malformed` |
| 17 | per section: enclosed bytes parse as a WFT1 frame prefix | the frame parser's class |
| 18 | per section: `root == frame.content_hash` | `Corrupt` |
| 19 | per section: `mode == frame.mode` | `Corrupt` |
| 20 | every padding byte is zero | `Malformed` |
| 21 | `total_len == align_up(last offset + last length, last alignment)` | `Malformed` |
| 22 | `payload_hash` over the enclosed bytes | `Corrupt` |

Checks 1–16 and 20–21 are structural. Check 17 reads each section's frame
header (96 bytes plus its tables); checks 18–19 need only that header. Check
22 reads every section body. A metadata-only parse performs 1–21 without
reading whole frame bodies; a full verification performs 1–22 and then a full
frame parse per section (frame CRC, dependency table, chunk table, trailer).

### 4.2 Single-section read

A reader MAY read one section without touching the others:

```text
1. read [0, 4096)                    -> header
2. read [4096, 4096 + 56*n)          -> section table
3. select entry i
4. read [entry.offset, entry.offset + entry.length) -> exactly the frame
5. verify: length, frame prefix/full parse, section.root, section.mode
```

Step 4 is one aligned range read when `entry.offset` and `entry.length` are
aligned to the transport; `length` is the exact frame length and therefore NOT
padded inside the section. A transport that requires a multiple-of-alignment
length must round the *request* up to the next block, which reads padding bytes
that are outside the section and MUST NOT be interpreted as frame data. The
court reports both the exact-section request and the container total so the
difference is explicit.

A single-section reader cannot verify `payload_hash` (that is a whole-container
property) and MUST NOT claim it did. It can and MUST verify `section.root`.

### 4.3 What a section root proves

`section.root` is the enclosed frame's `content_hash`, i.e.
`H("weft/chunk/v1", decoded_plaintext)`. Comparing it at parse time proves only
that the table and the frame header agree. Proving that the bytes decode to
that root requires decoding the frame, which is exactly the existing frame
verification and is unchanged. A reader that only fetches a section has
verified that the section is the frame the table names; it has not decoded it.

---

## 5. Sub-section reads

A section is padded to its alignment **outside** its frame; no padding is
inside a frame. A slice of a section that is not the whole frame is therefore
not an independently rooted object:

* a 4096-aligned slice inside a section is aligned with respect to the
  container only because the section offset is aligned; its `offset` is
  `section.offset + k*4096`;
* hashing that slice does **not** reproduce `section.root`, and no byte in the
  container claims that it should;
* data inside the slice is verifiable only through the inner roots the enclosed
  frame defines: mode 15/17/18 tile roots (`tile_root`), mode 20 group roots
  (`group_root`), and mode 16 delta roots. Those roots are unchanged by this
  envelope.

This is the intended composition: the transport container makes the *outer*
fetch aligned, and the codec's per-tile/per-group roots make an *inner* fetch
verifiable. A consumer must not treat the section root as a root for a slice.

---

## 6. Byte cost

For `n` sections of lengths `L[i]` and alignment `A`:

```text
container = align_up(4096 + 56*n + sum(L[i] + padding_i), A)
overhead  = container - sum(L[i])
```

The overhead is the 4 KiB header, 56 bytes per section, and alignment padding.
It is deterministic and can be computed before writing. Consequences:

* one large frame (MiB-scale) at 4096 alignment pays one 4 KiB header plus at
  most 4 KiB of tail padding;
* many small frames (a few KiB each) at 65536 alignment pay up to 64 KiB **per
  section**; the court measures this case explicitly and publishes the
  percentage. 64 KiB alignment is expensive for small frames and SHOULD be used
  only when the consumer's registration granularity requires it.

Padding is zero and is not hashed, so it does not change any encodable value.

---

## 7. Limits

* `section_count <= 4096`.
* Every section is a single complete frame; a frame is bounded by the existing
  `max_frame_bytes`, and the container by `max_total_fetched_bytes`.
* All arithmetic is checked; overflow is `Malformed` or `LimitExceeded`, never
  a panic.
* `total_len` and every table-derived offset are validated before any
  allocation sized from them: the reader allocates one `Section` per
  `section_count` entry only after the table is proved to fit inside the
  container.

---

## 8. Conformance vectors

`conformance/v1/transport/` (byte-identical in `WeftCodec` and
`WeftCodec-Go`):

* valid: 512 alignment (one section), 4096 alignment (one and eight
  sections), 65536 alignment (one section), and a 512-alignment multi-role
  container;
* rejects: bad magic, bad version, non-zero flags, zero sections, wrong table
  offset, misaligned offset, overlapping sections, out-of-bounds section,
  duplicate offsets, table count mismatch, non-zero padding, wrong section
  root, truncated container, mode mismatch, disallowed alignment value,
  unknown role, and non-zero header reserved bytes.

The container itself is `weft.transport.v1`; the enclosed frames remain the
`WFT1` vectors they already are. Enclosing a frame never changes its
conformance result.

---

## 9. Relationship to the self-contained WFA1 archive

`WFA1` (`WEFTCODEC-FORMAT.md` §7) is a dependency-closed archive: it reorders
frames topologically, deduplicates them, and is decoded as a set. `WFTT` is not
an archive: it preserves the caller's order, does not deduplicate, does not
require dependency closure, and exists so one section can be fetched alone.
Neither format replaces the other, and a frame may be enclosed by both without
changing.
