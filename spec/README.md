# WeftFS normative specification

This directory is the **single source of truth** for the WeftFS v1 formats. It
is exported byte-for-byte into `WeftCodec/spec/`, `WeftCodec-Go/spec/`, and
`WeftFS-Go/spec/` by `scripts/export-spec.sh`. Subproject copies are mirrors;
they are never edited in place. `scripts/check-spec-drift.sh` fails on any
difference.

## Where to start

Links beginning `../docs/` and `../bench/` refer to the program-home
repository (`infinityabundance/WeftFS`). The codec and filesystem subprojects
mirror `spec/` only; the normative text is identical there.

| You want to… | Read |
|---|---|
| Implement a decoder | [`../docs/WRITE-YOUR-OWN-DECODER.md`](../docs/WRITE-YOUR-OWN-DECODER.md), then `WEFTCODEC-FORMAT.md` and `WEFT-AI-CODEC.md` |
| Run the conformance corpus | [`../docs/CONFORMANCE.md`](../docs/CONFORMANCE.md) and `scripts/run-conformance.sh` |
| Find the status, version, modes, and vectors for every document | [`../docs/RFC.md`](../docs/RFC.md) |
| Evaluate the design and its measurements | [`../docs/PAPER.md`](../docs/PAPER.md) |
| See what is implemented versus specified | [`../docs/STATUS.md`](../docs/STATUS.md) |

## Documents

- [`WEFT-MODEL.md`](WEFT-MODEL.md) — identity, state, transactions, durability.
- [`WEFTCODEC-FORMAT.md`](WEFTCODEC-FORMAT.md) — frame, WCB, modes 0–5, error classes.
- [`WEFT-COMPATIBILITY.md`](WEFT-COMPATIBILITY.md) — version lines and change classes.
- [`WEFT-SELECTORS.md`](WEFT-SELECTORS.md) — WeftQL and `SelectorId`.
- [`WEFT-CONTEXT-ROOT.md`](WEFT-CONTEXT-ROOT.md) — the immutable public address.
- [`WEFT-CAUSAL.md`](WEFT-CAUSAL.md) — causal closure and identity.
- [`WEFT-CAPABILITIES.md`](WEFT-CAPABILITIES.md) — capability contexts and budgets.
- [`WEFT-AI-CODEC.md`](WEFT-AI-CODEC.md) — modes 6–20.
- [`WEFT-TENSOR.md`](WEFT-TENSOR.md) — logical tensor identity and region reads.
- [`WEFT-PROVENANCE.md`](WEFT-PROVENANCE.md) — receipts and recovery planning.
- [`WEFT-TRAINING-READY.md`](WEFT-TRAINING-READY.md) — materialization as a refusal.
- [`WEFT-QUOTA.md`](WEFT-QUOTA.md) — write budgets and branching.
- [`WEFT-BUCK2.md`](WEFT-BUCK2.md) — action keys and a store-backed cache.
- [`WEFT-TRANSPORT.md`](WEFT-TRANSPORT.md) — `weft.transport.v1` containers.
- [`WEFT-KV-PROVENANCE.md`](WEFT-KV-PROVENANCE.md) — mode 19 provenance-bound KV pages.
- [`WEFT-ENGRAM.md`](WEFT-ENGRAM.md) — mode 20 row-group tables.
- [`WEFT-RETENTION.md`](WEFT-RETENTION.md) — retention bindings and class-aware GC.
- [`WEFT-MULTIWRITER.md`](WEFT-MULTIWRITER.md) — MW0–MW3 writer contracts (MW3: conflict domains and independent heads).

## Status in one line

Modes 0–20 are allocated. Modes 0–17 and 19–20 are frozen for the fields their
documents list; mode 18 is frozen on the wire but remains opt-in and
experimental in selection; modes 21–127 are reserved and 128–255 experimental.
Every frozen change is append-only in `conformance/`; a correction adds a new
vector, never edits an existing one. A reader that does not implement a mode
MUST reject it as `UnsupportedMode`, never treat its payload as opaque bytes.

This index is normative only for navigation. The cited document is normative
for its own subject.
