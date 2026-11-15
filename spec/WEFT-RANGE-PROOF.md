# WEFT-RANGE-PROOF — portable authenticated range proofs

Status: **implemented in the Rust store (T073)**. Normative for
`weft.range-proof.v1` and `weft.range-anchor.v1`. A Go implementation does not
exist; a Go store MUST reject both schemas rather than guess. No frozen byte,
mode number, root or fixture is changed by this document: both records are new,
additive, content-addressed schemas.

## 1. What a range proof is

A range proof answers one question, for a verifier that holds a proof bundle
(or a store directory) and a checkpoint root:

> are these bytes the declared canonical region of tensor `T`, and is that
> region committed to by the checkpoint's authenticated root?

It is **not** a new tree. The leaf is the frozen Merkle leaf over a chunk-table
entry (`WEFT-MODEL.md` §2.3.2):

```
chunk_hash_i = H("weft/chunk/v1", chunk_bytes_i)
leaf_i       = H("weft/merkle/leaf/v1", u64le(len_i) || chunk_hash_i)
parent(l, r) = H("weft/merkle/node/v1", l || r)
```

with the frozen pairing rule: nodes pair left to right, an odd trailing node is
promoted unchanged to the next level (never duplicated, never hashed with
itself), and a one-chunk table's root is that chunk's leaf. The proof is the
**authentication path over the chunk table the manifest already carries**. No
sibling table is written to disk, and no new leaf or node domain is introduced.

## 2. Why an anchor record exists (the smallest additive change)

A `weft.checkpoint.v1` shard entry records
`content_root = H("weft/chunk/v1", whole shard bytes)` — a **whole-file**
digest. Checking a range proof against a whole-file digest requires reading
every byte, which is the cost a range proof exists to avoid. The chunk table's
Merkle root is therefore not authenticated by the checkpoint alone, and no
amount of proof structure can fix that: the root has to come from somewhere
signed or committed.

The smallest additive change is one canonical, content-addressed record,
`weft.range-anchor.v1`, which an operator publishes (signed with Ed25519) next
to the checkpoint. It binds the checkpoint manifest root, the tensor entry, the
shard entry, the shard's manifest id, and the **Merkle root of that shard's
chunk table**. Stated plainly: without a signed anchor, a chunk-table root is
only as trustworthy as whoever produced the chunk table.

## 3. `weft.range-anchor.v1`

One record covers **one shard** of one tensor. The key set is exact:

```
{
  "schema": "weft.range-anchor.v1",
  "checkpoint_name": text,
  "checkpoint_root": bytes32,          # H("weft/checkpoint/manifest/v1", canonical bytes)
  "checkpoint_manifest_id": bytes32,   # store blob id of the manifest object
  "tensor": text,
  "dtype": text,
  "shape": [u64, ...],
  "logical_root": bytes32,             # LogicalTensorRoot of the whole tensor
  "content_root": bytes32,             # H("weft/chunk/v1", whole canonical tensor bytes)
  "element_bytes": u64,
  "chunk_size": u64,
  "bytes": u64,                        # byte length of THIS shard
  "chunk_count": u64,
  "chunk_table_root": bytes32,         # Merkle root over this shard's chunk table
  "shards": [ {
      "index": u64,
      "object_name": text,
      "object_id": bytes32,
      "content_root": bytes32,
      "logical_root": bytes32,
      "byte_len": u64,
      "manifest": bytes32,             # manifest id of the retained chunk table
      "chunk_table_root": bytes32,
      "chunk_count": u64
  } ],
  "load_attestation": bytes32 | null,  # weft.load-attestation.v1 root, when present
  "issued_at": timestamp,
  "signature": {"algorithm": "ed25519", "public_key": bytes32, "signature": bytes64} | null
}
```

Rules:

* `shards` is non-empty, sorted by ascending `index`, with distinct indices.
  A writer emits one entry; a reader accepts any sorted subset whose
  `byte_len` values sum to `bytes`.
* Every `shard.chunk_count` MUST equal `ceil(shard.byte_len / chunk_size)`.
* `RangeAnchorId = H("weft/range-anchor/v1", WCB(record))`, over the record
  **including** `signature`. The signature covers the canonical bytes of the
  record **without** the `signature` key, using the Ed25519 construction shared
  with `weft.retention.v1` and `weft.load-attestation.v1`.
* `load_attestation`, when present, MUST name the same checkpoint root and a
  tensor entry with the same `logical_root`.
* Bounds: `shards` ≤ 4096; `bytes` > 0; `chunk_size` > 0. Violations are
  `LimitExceeded` or `Malformed`.

## 4. `weft.range-proof.v1`

```
{
  "schema": "weft.range-proof.v1",
  "anchor": bytes32,                   # RangeAnchorId
  "manifest": bytes32,                 # manifest id of the shard's chunk table
  "content_root": bytes32,             # the checkpoint shard entry's content root
  "tensor": text,
  "shard_index": u64,
  "chunk_size": u64,
  "chunk_count": u64,
  "size": u64,                         # the shard's byte length
  "offset": u64,                       # region offset INSIDE the shard
  "length": u64,
  "region_root": bytes32,              # H("weft/range/v1", region bytes)
  "leaves": [ {"index": u64, "offset": u64, "len": u64, "hash": bytes32}, ... ],
  "paths":  [ [ {"side": "left"|"right", "hash": bytes32}, ... ], ... ]
}
```

* `leaves` are **whole chunks**, sorted by `index`, and MUST be exactly the
  chunks the declared region intersects. The expected set is derived from
  `offset`, `length`, `chunk_size`, `size` and `chunk_count` alone:
  `first = offset / chunk_size`, `last = (offset + length - 1) / chunk_size`,
  chunk `i` starts at `i * chunk_size` and has `chunk_size` bytes except the
  final chunk, which has `size - i * chunk_size`.
* `paths` carries exactly one authentication path per leaf, ordered from the
  leaf upward. A path contains nodes only for levels where the leaf has a
  sibling. **The verifier derives each level's width from `chunk_count`**, so a
  skipped, extra, or wrongly sided node is a rejection, never an alternative
  reading.
* `region_root` is the digest of exactly the requested bytes, sliced out of the
  concatenated chunk bytes: `H("weft/range/v1", bytes[offset..offset+length])`.
* `RangeProofId = H("weft/range-proof/v1", WCB(record))`. A proof carries no
  signature of its own; it is authenticated by the anchor it names.
* Canonical WCB, exact key sets, no unknown fields. A record that does not
  decode, re-encode and satisfy §4 is `Malformed`, `NonCanonical` or
  `Corrupt` — the decoder never interprets a byte string that is not exactly
  canonical.

## 5. Verification

A verifier performs, in order:

1. **Canonicality.** Decode each record and require byte-identical
   re-encoding.
2. **Anchor identity.** `RangeAnchorId` of the supplied anchor equals
   `proof.anchor`.
3. **Signature and trust.** If the anchor is signed, the signature verifies.
   If a trust set is configured, the signer must be a member. An unsigned
   anchor is `Unauthorized` when a trust set is configured.
4. **Checkpoint binding.** The checkpoint manifest bytes hash to
   `anchor.checkpoint_root`, and every anchor field equals the manifest's
   tensor and shard entry (`dtype`, `shape`, `logical_root`, `content_root`,
   `element_bytes`, shard `object_id`, `content_root`, `logical_root`,
   `byte_len`). This is what prevents replaying an anchor against another
   checkpoint, tensor or shard.
5. **Dependency re-read** (when a store is available). The shard manifest is
   read from `manifests/<ab>/<id>` and its chunk table re-derived: `size`,
   `chunk_size`, chunk count, the Merkle root and the manifest's whole-file
   `content_hash` must all agree with the anchor and the checkpoint entry.
   Without a store this check is reported as `manifest_rechecked=false` and
   nothing is claimed about it.
6. **Geometry.** The leaf set equals the expected set of §4; each leaf's
   `hash`/`len` equals the chunk table entry at that index (when the table is
   available).
7. **Paths.** Every leaf folds to `chunk_table_root` under the frozen rule.
8. **Chunk bytes.** Each supplied or re-read chunk hashes to `leaf.hash` with
   the chunk domain and has exactly `leaf.len` bytes.
9. **Region.** The sliced region bytes hash to `region_root`.

A failure is a typed error naming the step. There is no partial acceptance and
no approximate path.

## 6. Boundaries (stated, not approximated)

* **Chunk-table shards only.** A shard whose retained representation is a
  frame (`weft.manifest.v2` `kind:"codec"` or `kind:"tensor"`, modes 9/10/15/17,
  or a `weft.manifest.v4` overlay) has no chunk table and is **refused** with
  `UnsupportedRepresentation`. The
  writer reports the refusal with its reason; it never substitutes a
  whole-file digest. Producing chunk tables for checkpoints is a
  writer-strategy decision (`--shard-strategy raw`), not something a proof can
  retrofit.
* **One shard per proof.** Shards split along an axis; only an axis-0 split
  yields a contiguous byte interval per shard. A region that is not contained
  in one shard, and any proof request on an axis != 0 multi-shard tensor, is
  refused with a typed error rather than reinterpreted.
* **The anchor is the trust root.** A verifier that accepts an unsigned anchor
  with no trust set proves internal consistency, not authorship. The receipt
  records `anchor_signature_present` for exactly this reason.
* **Not a privacy mechanism.** A proof reveals the region bytes, the chunk
  hashes of every intersecting chunk, and the path. It reveals nothing about
  chunks outside the path beyond their sibling hashes.
* **No cross-language implementation.** The independent verifier in
  `verifier/range_proof_check.py` re-implements the construction from this
  document using only the Python standard library plus the repository's
  clean-room BLAKE3, WCB and Ed25519. It shares no code with WeftFS.
