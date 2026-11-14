# Context-plane freeze matrix (Phase 0)

Status: **Phase 0 reconciliation record** for draft RFC 0001. This document is
an audit of the *existing* frozen identities, not a new wire format. It assigns
no new wire value and changes no canonical byte.

Placed at WeftFS `staging` revision `f8f352ba9db37dfa08d9305fe3ba7f9288d6be46`
(the revision this file was added on top of), with the mode-registry
reconciliation exported to all three submodule mirrors.

**Incorporation by reference.** Every identity in §2 below is frozen as
published in its normative document. RFC 0001 incorporates those records by
reference and does not restate, re-derive, renumber or rehash them. A future
profile that needs a different field set, serializer or hash domain must use a
new schema string and a new domain; it must not reinterpret these rows.

---

## 1. Mode-registry reconciliation (editorial only)

### 1.1 The contradiction as found

`spec/WEFTCODEC-FORMAT.md` §3 (the framing document) carried a derived table
that ended:

```
| 18 | `FLOAT_TILED_STREAMS` | ... |
| 19..127 | — | reserved; `UnsupportedMode` |
| 128..255 | — | experimental; MUST be rejected for persistence |
```

and its §13 compatibility policy listed allocations only through `FLOAT_TILED_STREAMS` (18).

`spec/WEFT-AI-CODEC.md` §0 (the AI-mode document) listed:

```
| 19 | `KV_CACHE_PROV` (`kv-prov`) | `weft.ai.kv_prov.v1` | ... | v1 | normative | `conformance/v1/kv-prov/001–008` + twelve rejects |
| 20 | `ENGRAM_SHARD` (`engram-shard`) | `weft.ai.engram_shard.v1` | ... | v1 | normative | `conformance/v1/engram/001–006` + fourteen rejects |
| 21 | — | — | — | — | **reserved** | ... |
```

with the status column also marking 19 and 20 **implemented**. The two
statements cannot both be true: one says values 19 and 20 are reserved and
rejected, the other says they are allocated, normative and implemented.

### 1.2 What was changed

Editorial only, in `spec/WEFTCODEC-FORMAT.md`:

1. A new authority paragraph at the head of §3 states that
   `WEFT-AI-CODEC.md` §0 is the normative mode registry for values 6–20 and for
   the reserved/experimental ranges, and that `WEFTCODEC-FORMAT.md` §3 is a
   derived summary that MUST be corrected if it disagrees.
2. Two rows were added to the §3 table: `19 | KV_CACHE_PROV | ...` and
   `20 | ENGRAM_SHARD | ...`.
3. The reserved row became `21..127`; `128..255` is unchanged.
4. The global mode rules now say "AI modes 6–20"; the zero-dependency set gained
   17, 18, 19 and 20; the normative-reference range became §2–§21.
5. §13 now lists every allocated value (6–20) and names
   `017_unknown_mode_reserved21` as the current reserved-value vector. It also
   records that the earlier vectors in the chain
   `017_unknown_mode_reserved8..20` are retained with `superseded_by`.

**Not changed:** no frame byte, no mode number, no hash domain, no descriptor
key, no conformance vector, no manifest digest. `conformance/manifest.json`
still reports the same aggregate (see §5), and all three submodule mirrors were
re-exported byte-for-byte.

### 1.3 Other registry statements checked

| Document | Statement | Verdict |
|---|---|---|
| `spec/README.md` | "Modes 0–20 are allocated … modes 21–127 are reserved and 128–255 experimental" | consistent |
| `spec/WEFT-AI-CODEC.md` §0 | modes 0–20 allocated, 21 reserved, >21 `UnsupportedMode` | **authoritative** |
| `spec/WEFTCODEC-FORMAT.md` §3 | 0–20 allocated, 21–127 reserved | corrected |
| `docs/WRITE-YOUR-OWN-DECODER.md` §1 | "Mode values 21–127 Reserved" | consistent |

---

## 2. Freeze matrix — identity-bearing canonical records

Legend: **Rust** = `WeftCodec` and/or `WeftFS`; **Go** = `WeftCodec-Go` and/or
`WeftFS-Go`; **Py** = the independent reader in `python/weftcodec`;
**Verifier** = `verifier/verify_corpus.py` (Phase 1, same organisation,
implementation-independent). "Identity" is the value the record is addressed
by; "domain(s)" are the exact ASCII domain strings passed to
`H(domain, data) = BLAKE3-256(u32le(len(domain)) ‖ domain ‖ data)`.

| # | Canonical record / schema | Identity | Hash domain(s) | Version | Normative document | Rust | Go | Py | Verifier |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | `weft.selector.v1` | `SelectorId` | `weft/selector/v1` | v1 | WEFT-SELECTORS §4 | impl | impl | none | **validated** (22 vectors) |
| 2 | `weft.context-root.v1` | `ContextRootId` | `weft/context-root/v1` | v1 | WEFT-CONTEXT-ROOT §2 | impl | impl | none | **validated** (6 valid + 10 refusals) |
| 3 | `weft.causal.v1` | `CausalClosureId` | `weft/causal/v1` | v1 | WEFT-CAUSAL §1.2 | impl | impl | none | **validated** (4 + 6) |
| 4 | `weft.tensor.logical.v1` | `LogicalTensorRoot` | `weft/tensor/values/v1` → `weft/tensor/logical/v1` | v1 | WEFT-TENSOR §3 | impl | impl | partial: root helper, no container reader | **validated** (3 + 10) |
| 5 | `weft.tensor.layout.v1` | none (self-describing descriptor) | — | v1 | WEFT-TENSOR §4 | impl | impl | none | **canonical bytes only** (4) |
| 6 | `weft.tensor.semantic.v1` | semantic root | `weft/tensor/semantic/v1`; values via `weft/tensor/semantic-values/v1` | v1 | WEFT-AI-CODEC §15.1 | impl | impl | partial (mode-14 path) | not implemented |
| 7 | `weft.provenance.v1` | `ProvenanceId` | `weft/provenance/v1` | v1 | WEFT-PROVENANCE §3 | impl | **write-only** (rebase receipts; no reader/verify verb) | none | **identity + schema self-test only; no frozen corpus** |
| 8 | `weft.capability.v1` | `CapabilityHash` | `weft/capability/v1` | v1 | WEFT-CAPABILITIES §1.1 | impl | impl | none | **validated** (identity) |
| 9 | `weft.capability.envelope.v1` | `EnvelopeId`, `BlobId` | `weft/capability/envelope/v1`; sign input `weft/capability/sign/v1`; blob `weft/capability/blob/v1` | v1 | WEFT-CAPABILITIES §9.1 | impl (Ed25519) | impl (Ed25519) | none | **identity only**; signature/grantee/expiry/revocation out of scope |
| 10 | `weft.ai.kv_prov.v1` | `descriptor_root` | `weft/kv/prov/v1`; also `weft/kv/adapters/v1`, `weft/kv/tokenizer/v1`, `weft/chunk/v1` | v1 | WEFT-KV-PROVENANCE §3 | impl (mode 19) | impl (mode 19) | **none** (mode 19 not implemented) | **validated** (8 valid + 12 refusals) |
| 11 | `weft.kv.prefix.v1` | prefix record identity | `weft/kv/prefix/v1` | v1 | WEFT-KV-PROVENANCE §1, `src/kv_prefix.rs` | impl | none | none | not implemented |
| 12 | `weft.checkpoint.v1` | manifest content root | `weft/checkpoint/manifest/v1` (**code only; absent from FORMAT §1.1**) | v1 | `src/checkpoint.rs`; WEFT-TRAINING-READY | impl | none | only as a CLI caller | not implemented |
| 13 | `weft.training_ready.v1` / `.manifest.v1` | none (output record; carries ids) | n/a (values are ids computed elsewhere) | v1 | WEFT-TRAINING-READY §4 | impl | none | none | not implemented |
| 14 | `weft.manifest.v1` | `ManifestId` / whole-file `ContentId` | `weft/manifest/v1` | v1 | WEFT-MODEL §2.3 | impl | impl | none | not implemented |
| 15 | `weft.manifest.v3` | `content_root` (Merkle) | `weft/merkle/leaf/v1`, `weft/merkle/node/v1`, `weft/merkle/empty/v1` | v3 | WEFT-MODEL §2.3.2 | impl | impl | none | not implemented |
| 16 | `weft.manifest.v4` | `content_root` (equal to a canonical v3 root) | derived from the same Merkle domains | v4 | WEFT-MODEL §2.3.3 | impl (writer + compaction) | impl (writer + compaction, R20) | none | not implemented |
| 17 | `weft.transport.v1` | `payload_hash` | `weft/transport/v1` | v1 | WEFT-TRANSPORT | impl | impl | none | not implemented |
| 18 | `weft.engram.v1` | `descriptor_root`, `group_root` | `weft/engram/v1`, `weft/engram/group/v1`, `weft/engram/payload/v1`, `weft/tensor/values/v1` | v1 | WEFT-ENGRAM; WEFT-AI-CODEC §21 | impl (mode 20) | impl (mode 20) | none (mode 20 not implemented) | not implemented |
| 19 | `weft.retention.v1` | `RetentionId` | `weft/retention/v1` | v1 | WEFT-RETENTION | impl | impl | none | not implemented |
| 20 | `weft.tx.v1` | transaction id | `weft/tx/v1` digest; `weft/mw2/key/v1` serialisation key | v1/v2 | WEFT-MULTIWRITER §1–§14 | impl | impl | none | not implemented |
| 21 | `weft.domain.v1` / `weft.domains.v1` | `DomainId` | `weft/domain/v1` | v1 | WEFT-MULTIWRITER §15 (MW3) | impl | none | none | not implemented |
| 22 | `weft.writers.v1` | none (policy) | — | v1 | WEFT-MULTIWRITER §9 | impl | impl | none | not implemented |
| 23 | `weft.buck2.action.v1` | `ActionKey` | `weft/buck2/action/v1` | v1 | WEFT-BUCK2 §1 | oracle CLI only | impl (`adapters/buck2`) | none | not implemented |
| 24 | `weft.load-attestation.v1` | attestation digest | `weft/load/attestation/v1`; ranges `weft/load/range/v1` | v1 | `docs/LOAD-ATTESTATION.md`; `src/load_attestation.rs` | impl | **signature verification only; prints `byte_replay=not_performed`** | none | not implemented |

Cross-check: row 12's domain `weft/checkpoint/manifest/v1` is defined in
`src/checkpoint.rs` and is used by every checkpoint vector, but it is **not**
listed in `WEFTCODEC-FORMAT.md` §1.1 (the frozen hash-domain table). The same is
true of `weft/index/state/v1`, `weft/index/candidates/v1`, `weft/relation/v1`
(which *is* listed), `weft/pack/*`, `weft/load/*` and `weft/mw2/key/v1`. That is
a documentation gap, not a byte ambiguity: the code that writes the bytes and
the code that reads them agree, and no vector changed.

### 2.1 Explicit divergences and specified-only rows

- **Checkpoint (row 12):** Rust-only. Go has no `weft.checkpoint.v1` reader or
  writer; `WeftFS-Go/cmd/attest-verify` reads a load-attestation record, not a
  checkpoint manifest.
- **Provenance (row 7):** Rust has record, attachment, GC and planner. Go
  writes one provenance record from `store/rebase.go` and has no
  `provenance show`, closure or verify verb. Python has neither.
- **KV prefix (row 11) and retention (row 19):** Rust writes and reads; Go
  implements retention (`store/retention.go`) but not `weft.kv.prefix.v1`.
- **MW3 domains (row 21):** specified and Rust-implemented; Go refuses.
  `weft.domain.link.v1` is specified, not implemented, in every language.
- **KV provenance (row 10):** Rust and Go implement mode 19; the independent
  Python reader explicitly refuses mode 19 as unimplemented. That is a declared
  skip, not a pass.
- **BUCK2 (row 23):** the Go adapter implements the store-backed cache; the
  Rust CLI is the independent oracle; this is not a second implementation of
  the action-key semantics in Rust.

---

## 3. Descriptor-only schemas (no standalone identity)

These are canonical WCB descriptors carried *inside* a frame or record. They
are validated field-by-field, but they are not addressed by their own hash; the
enclosing root is.

| Schema | Carried by | Payload hash domain | Normative document |
|---|---|---|---|
| `weft.tensor.v1` | mode 6 frame | `weft/tensor/payload/v1` | WEFT-AI-CODEC §2 |
| `weft.tokens.v1` | mode 7 frame | `weft/tokens/payload/v1` | WEFT-AI-CODEC §3 |
| `weft.quantized-tensor.v1` | mode 8 frame | `weft/quantized-tensor/payload/v1` | WEFT-AI-CODEC §4 |
| `weft.ai.float_tensor.v1` | mode 9 frame | `weft/float-tensor/payload/v1` | WEFT-AI-CODEC §5 |
| `weft.ai.tensor_sharded.v1` | mode 10 frame | `weft/tensor-sharded/payload/v1`, `weft/tensor/shard-payload/v1` | WEFT-AI-CODEC §11 |
| `weft.ai.weight_delta.v1` | mode 11 frame | `weft/tensor/values/v1` (`delta_root`) | WEFT-AI-CODEC §12 |
| `weft.ai.kv_cache.v1` | mode 12 frame (and mode 19 `page`) | `weft/chunk/v1` | WEFT-AI-CODEC §13 |
| `weft.ai.activation_stream.v1` | mode 13 frame | `weft/ai/stream/v1`, `weft/tensor/values/v1` | WEFT-AI-CODEC §14 |
| `weft.ai.semantic_delta.v1` | mode 14 frame | `weft/tensor/values/v1` (`delta_root`) | WEFT-AI-CODEC §15 |
| `weft.ai.tensor_tiled.v1` | mode 15 frame | `weft/tensor/tile/v1`, `weft/tensor/values/v1` | WEFT-AI-CODEC §16 |
| `weft.ai.sparse_delta.v1` | mode 16 frame | `weft/tensor/values/v1` (`delta_root`) | WEFT-AI-CODEC §17 |
| `weft.ai.float_tiled.v1` | mode 17 frame | `weft/tensor/float_tile/v1`, `weft/tensor/values/v1` | WEFT-AI-CODEC §18 |
| `weft.ai.tensor_tiled_streams.v1` | mode 18 frame | `weft/tensor/float_tile_streams/v1`, `weft/tensor/values/v1` | WEFT-AI-CODEC §19 |
| `weft.ai.kv_prov.v1` | mode 19 frame | `weft/kv/prov/v1` (descriptor root) | WEFT-KV-PROVENANCE §3 |
| `weft.ai.engram_shard.v1` | mode 20 frame | `weft/engram/v1`, `weft/engram/group/v1` | WEFT-ENGRAM; WEFT-AI-CODEC §21 |

---

## 4. The tokenizer identity gap (confirmed in code)

RFC 0001 §6.1 and §18 state the gap. It is accurate.

`WEFT-KV-PROVENANCE.md` §2 row 3 requires:

```
tokenizer      = exactly {scheme: text, vocab_size: u64}
tokenizer_root = H("weft/kv/tokenizer/v1", WCB(tokenizer))
```

`WeftCodec/src/kv_prov.rs` implements exactly that, with no other input:

```rust
/// `H("weft/kv/tokenizer/v1", WCB({scheme, vocab_size}))`.
pub fn tokenizer_root(scheme: &str, vocab_size: u64) -> Result<Hash32> {
    let value = Value::map([
        ("scheme", Value::Str(scheme.to_owned())),
        ("vocab_size", Value::U64(vocab_size)),
    ]);
    let bytes = wcb::encode(&value, &Limits::default())?;
    Ok(hash_domain(DOMAIN_KV_TOKENIZER, &bytes))
}
```

`WeftCodec-Go/modes/kv_prov.go` defines the same constant and the same
`{scheme, vocab_size}` input. The descriptor validator recomputes the root and
refuses a mismatch (`kv-prov/106_tokenizer_mismatch.json`), so the record is
internally consistent. It does **not** bind:

- the vocabulary contents (token → id map),
- the merge table or BPE rules,
- the normalizer / pre-tokenizer configuration,
- special-token ids and their roles,
- the chat template or any preprocessing applied before tokenisation.

Therefore two tokenizer implementations with different contents but the same
`scheme` string and `vocab_size` produce the **same** `tokenizer_root` and can
share a mode-19 descriptor. The mode-19 descriptor remains a correct provenance
record for the dimensions it names; it is not proof of complete tokenizer
identity. RFC 0001 §6.1 requires a future execution-contract profile to bind a
digest of the actual tokenizer implementation/configuration before claiming
exact cross-runtime reuse.

**No frozen mode-19 byte was changed.** `conformance/v1/kv-prov/` is untouched;
the manifest aggregate below is unchanged.

---

## 5. Two-language (and Python) vector results recorded for this freeze

Commands run at parent `f8f352b` with the codec submodules at
`WeftCodec 6cd0f6d`, `WeftCodec-Go b280d94`, `WeftFS-Go 3ce1d8b`, after building
each CLI inside its Docker court (`rust` and `go` courts; 3 GiB / 4 CPU /
1024 pids, limits not raised) and running the host-only reference runner:

| Adapter | Receipt | Total | Pass | Fail | Skip | Result |
|---|---|---:|---:|---:|---:|---|
| Rust `weftcodec` | `bench/receipts/conformance-rust-20261114T062338Z.json` | 420 | 288 | 0 | 132 | pass |
| Go `weftcodec` | `bench/receipts/conformance-go-20261114T062338Z.json` | 420 | 288 | 0 | 132 | pass |
| Python reader | `bench/receipts/conformance-python-20261114T062338Z.json` | 420 | 229 | 0 | 191 | pass |

All three receipts name the same manifest aggregate:

```
conformance/manifest.json vector_count = 420
aggregate_sha256 = 6287b817b94d2cc805abe23561d3d4270cd46b3c2e4f4e598fa673dfe48dfbe8
```

Skips are explicit and reasoned by the runner (for example, `kind
'checkpoint' has no adapter`, `context subkind 'causal_versions' has no
adapter`). A skip is not a pass. The 132 Rust/Go skips are the surfaces those
CLIs do not expose; the extra Python skips are its unimplemented kinds and
modes 19–20.

This run is the "full two-language vector run" that the adoption plan's Phase 0
ship gate requires. It is a *reference* run of two implementations derived from
the same algorithm plus an independent reader, **not** an external third-party
submission.

---

## 6. Findings from Phase 0

1. **The mode table was contradictory** (§1.1) and is now editorially
   reconciled; `WEFT-AI-CODEC.md` §0 is named authoritative.
2. **The tokenizer gap is real and confirmed in both codec implementations**
   (§4). It cannot be closed without a new execution-contract schema; mode-19
   bytes must not be mutated.
3. **Several identity domains are documented only in code**, including
   `weft/checkpoint/manifest/v1`, `weft/index/state/v1`,
   `weft/index/candidates/v1`, `weft/pack/*`, `weft/load/*` and
   `weft/mw2/key/v1`. They are not in `WEFTCODEC-FORMAT.md` §1.1. This is a
   documentation gap, not a byte disagreement, and is left unfixed here because
   changing the frozen domain table is a normative change, not an editorial
   one.
4. **Provenance has no frozen conformance vector.** `weft.provenance.v1` has a
   normative document, Rust implementation and tests, but no
   `conformance/v1/provenance/` family in `manifest.json`. The independent
   verifier's provenance checks are therefore self-tests (see §7), not corpus
   replay.
5. **The corpus has no selector refusal vector.** All 22 selector vectors are
   valid expressions. The selector error taxonomy is exercised only by
   implementation tests.
6. **The cross-layout equal-root claim has no portable vector.** The four
   `tensor/201..204` layout vectors carry `content_root: null`; the equality is
   asserted by Rust/Go tests, not by a frozen corpus case.
7. **No snapshot-selection or ambiguity vector exists.** `context/103` covers a
   zero snapshot only; missing snapshot and singular ambiguity need a portable
   request/response family.
8. **Rust/Go divergence survives** for checkpoint, provenance reading, KV
   prefix and MW3 domains (see §2.1). None of it affects a frozen identity; it
   limits what each language can *read*.

---

## 7. What the Phase 1 verifier adds to this matrix

`verifier/` is a clean-room Python implementation written from the published
text. It validates the canonical bytes and identities of rows 1–5, 8–10 of §2
(selector, Context Root, causal closure, logical tensor, capability identity,
KV provenance descriptor) and implements the `weft.provenance.v1` schema and
identity as a self-test because no frozen corpus exists for it. It does not
link, import or shell out to the Rust, Go or Python Weft packages.

Its per-case outcomes, refusals and the list of rules found under-specified in
the spec text are in `bench/receipts/context-plane-verifier-<stamp>.json` and
`bench/results` prose (`bench/RESULTS-context-plane-verifier.md`). A negative
suite proves it rejects a flipped byte, noncanonical map order, duplicate key,
extra unknown key, wrong hash domain, truncated record and oversize record.

This verifier is **independent of the implementation code path but authored by
the same organisation**; it is not yet an external third-party implementation,
and this document does not claim otherwise.

---

## 8. Reproduce

```sh
# Mode-registry mirror equality (host-only).
bash scripts/check-spec-drift.sh

# Codec adapters (build in court, run the host-only reference runner).
scripts/docker-court.sh rust "cd WeftCodec && cargo build --release --quiet"
scripts/docker-court.sh go   "cd WeftCodec-Go && go build -o /work/.docker-cache/weftcodec-go ./cmd/weftcodec"
scripts/run-conformance.sh scripts/conformance-adapter-rust.sh --json OUT.json
scripts/run-conformance.sh scripts/conformance-adapter-go.sh   --json OUT.json
scripts/docker-court.sh python "WEFT_PYTHONPATH=/work/python PYTHONPATH=/work/python \
  bash scripts/run-conformance.sh scripts/conformance-adapter-python.sh --json OUT.json"

# Independent verifier.
python3 verifier/verify_corpus.py --json bench/receipts/context-plane-verifier-<stamp>.json
```
