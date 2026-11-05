# WEFT-PROVENANCE — receipts and provenance-guided recovery

Status: **implemented in the Rust store (D1)**. The record format, attachment
rule, retention rule, verification rule, and planner below are normative.
A Go implementation does not exist yet; a Go store MUST reject an unknown
`weft.provenance.v1` blob rather than guess.

## 1. Two orthogonal descriptions

- A **representation** (`weft.manifest.v2`, `spec/WEFT-MODEL.md` §2.3.1)
  answers *how these bytes are physically stored*: raw chunks, a codec frame,
  or a tensor frame, plus any dependency content roots needed to decode it.
- A **provenance record** (`weft.provenance.v1`) answers *how these bytes were
  produced*: the operation, parameters, verified inputs, declared outputs,
  environment, determinism class, principal, and time.

Neither replaces the other. Deleting bytes can still leave a representation
unrecoverable while recomputation is possible, and a preserved representation
can still be non-reproducible. The planner in §8 uses both facts.

## 2. Record schema

A record is one canonical WCB map:

```
{
  "schema": "weft.provenance.v1",
  "operation": text,
  "params": map,
  "inputs": [ {"role": text, "content": bytes32, "object": bytes32|null}, ... ],
  "outputs": [ bytes32, ... ],
  "environment": {
    "tool": text, "version": text, "env_hash": bytes32|null,
    <optional canonical WCB fields>
  },
  "determinism": "exact" | "declared" | "unknown",
  "principal": {"uid": u64, "gid": u64},
  "timestamp_ns": timestamp
}
```

- `content` is the **content root** used everywhere else in the store: the
  manifest id of the input/output bytes (`ObjectRecord.content`). It is not a
  representation-independent `ContentId`; the manifest carries that digest in
  `content_hash`. Recording a receipt therefore names the exact retained
  physical representation that the computation consumed or produced.
- `object` is an optional ObjectId hint that resolved to `content`. A reader
  MAY use it to continue a closure walk; it is not part of identity validation
  beyond the record bytes themselves.
- `environment.tool` and `environment.version` MUST be non-empty text.
  `env_hash` is `bytes32` or null. Extra fields are preserved with their WCB
  types and MUST NOT shadow `tool`, `version`, or `env_hash`.
- `determinism` is the producer's declaration:
  - `exact`: re-running the operation on the verified inputs reproduces the
    exact declared outputs;
  - `declared`: the producer asserts determinism but no mechanism enforces it;
  - `unknown`: determinism has not been established.
  Only `exact` makes recomputation eligible in §8.

### 2.1 Canonical ordering and duplicates

- `inputs` MUST be strictly ordered by `(role, content)` using byte order:
  text bytes first, then the 32 content-root bytes. Equal `(role, content)`
  pairs are forbidden, including pairs whose optional `object` differs.
- `outputs` MUST be strictly ascending by the 32 content-root bytes.
  Duplicate output roots are forbidden. Several retained objects MAY share one
  output content root; the record names the root once.
- `params` map keys are sorted by canonical WCB (the encoder enforces this).
- A constructor that receives unordered inputs/outputs canonicalises them
  before computing identity. A decoder MUST NOT reorder: unsorted or duplicate
  arrays are `NonCanonical`.

### 2.2 Bounds

| Field | Bound |
|---|---|
| `params` canonical bytes | 65,536 |
| `environment` canonical bytes | 16,384 |
| inputs | 1,024 |
| outputs | 1,024 |
| any text field (`operation`, role, tool, version) | 1,024 bytes |
| whole canonical record | 1,048,576 bytes |

Every violation is `LimitExceeded`; the zero hash is never a valid content
root, object id, or environment hash (`Malformed`).

## 3. Identity

```
ProvenanceId = H("weft/provenance/v1", WCB(record))
```

The record bytes include `schema`. The identity is content-addressed: any
change to any field produces a different identity. A reader MUST verify
`H(domain, stored_bytes) == id` and MUST re-encode the decoded record and
confirm it reproduces the same id. A mismatch is `Corrupt`; there is no
approximate or partial record.

## 4. Storage

The record is stored as a content-addressed blob in the manifest namespace:

```
manifests/<ab>/<provenance-id>     # bytes are WCB(record), domain weft/provenance/v1
```

It is **not** a `weft.manifest.v1`/`v2` manifest. The namespace is shared only
as a physical directory; the domain-separated hash distinguishes the two. A
reader MUST hash with `weft/provenance/v1`, never with `weft/manifest/v1`.
Because the blob is content-addressed and published atomically, it is written
before the relation commit; a crash in between leaves an orphan blob that GC
collects, never a relation pointing at missing bytes.

## 5. Attachment

Every output object named by a receipt carries one relation:

```
subject:   output ObjectId (bytes32)
predicate: "provenance"
kind:      "value"
object:    lowercase 64-hex ProvenanceId (text)
provenance: null
attrs:     {}
```

The relation value target is text because v1 relation value targets admit
text, u64, i64, bool, timestamp, or null, but not raw `bytes32`
(`WEFT-MODEL.md` §2.2). Lowercase hexadecimal is the canonical text encoding
of a hash and is the same spelling used for filenames and CLI output. A
non-lowercase or non-32-byte target is `Corrupt`; there is no alternate
spelling.

`record_provenance(outputs, record)` validates and publishes atomically:

1. Every supplied output must exist and be a regular file.
2. The sorted, deduplicated set of `ObjectRecord.content` values for the
   supplied outputs MUST equal `record.outputs`. A declared root that does not
   match an object's actual content, a missing output, or an extra output is
   rejected with `InvalidArgument` before any write.
3. The record blob is published content-addressed.
4. All output relations are published in **one** commit.

Several receipts MAY be attached to one object (for example after a re-run).
`provenance_of` returns the lowest ProvenanceId deterministically;
`provenance_records_of` returns all of them ordered by identity.

## 6. Retention

GC marks provenance records that are reachable from a retained output:

- the record blob itself;
- every declared input content root that has a local manifest (external inputs
  are skipped, not errors);
- every declared output content root;
- transitively, each marked manifest's chunks, frame, and dependency content
  roots, under the frozen dependency depth bound.

Deleting an object removes every relation whose **subject** is that object in
the same commit as the `delete_object` op (explicit ops, so replay and
cross-language state stay identical). When the last output relation naming a
receipt is gone, the receipt and its otherwise-unreferenced inputs become
collectable. A snapshot or checkpoint state that still contains an output
object and its relation keeps marking the receipt as a root.

## 7. Verification

`verify --full` validates every `provenance` relation in the current state:

- the value target is a lowercase 64-hex 32-byte identity;
- the subject object exists;
- the blob exists, hashes with the provenance domain, and decodes to a
  canonical record whose re-encoded id matches;
- the record still declares the subject object's current content root.

A stale relation (the object was rewritten after the receipt was recorded) is
reported as a problem; write a new receipt rather than rewriting one. The
report counts validated relations in `checked_provenance`.

## 8. Recovery planner

The planner is pure and deterministic. Its input is a target content root and
a set of candidates. A candidate has:

- `strategy` ∈ `fetch`, `decode`, `recompute`;
- `name` (deterministic tie-break name);
- `available` (the bytes or representation the strategy needs are present);
- `determinism` (from the receipt; used by `recompute`);
- `missing_inputs` (declared inputs that could not be verified);
- three declared cost components: `fetch_cost`, `decode_cost`,
  `recompute_cost`. A candidate declares a non-zero component only for work it
  actually performs (a decode may include fetching a dependency chain; a
  recompute may include verified input reads).

Eligibility:

| Strategy | Eligible when |
|---|---|
| `fetch` | `available` |
| `decode` | `available` |
| `recompute` | `available` **and** `determinism == "exact"` **and** `missing_inputs` is empty |

The cost of a candidate is the saturating sum of its three declared
components. The planner chooses the eligible candidate minimising

```
(total_cost, strategy rank, name)
```

where rank is `fetch < decode < recompute`. The order is total: equal cost
prefers fetch over decode over recompute, and equal strategy prefers the
lexicographically smaller name. The result therefore does not depend on input
order. There is no fallback: if no candidate is eligible, the planner returns
`NoEligibleStrategy`, naming every candidate and every reason
(`unavailable`, `non-exact determinism (<class>)`, or `missing input(s) ...`).

Recompute is never silently downgraded to an approximate or partial result.
The executor MUST verify every declared input before running the operation and
MUST verify the recomputed bytes against the target content root.

## 9. CLI

```
weftfs provenance record --record RECORD.json --output PATH_OR_ID [--output ...]
weftfs provenance show PATH_OR_ID
weftfs provenance closure PATH_OR_ID [--depth N]
weftfs recover --plan PATH_OR_ID [--fetch-cost N] [--decode-cost N] [--recompute-cost N]
```

`provenance record` parses the JSON form of §2, canonicalises order, validates
outputs, and prints the ProvenanceId. `recover --plan` builds fetch/decode
candidates from the live representation and recompute candidates from every
attached receipt, runs the §8 planner, and prints the chosen strategy with all
three cost components. Cost units are caller-defined; nanoseconds are the
convention used by the benchmark.

## 10. Conformance vectors (Rust)

The D1 tests are `tests/provenance.rs` (19 tests): record round trip and
canonicalisation; unsorted/duplicate rejection; output-content mismatch
rejection without writing; bounded params; atomic multi-output attachment;
planner cost choice, component summing, exact-determinism and missing-input
exclusion, tie-break order, and no-eligible error text; transitive closure and
depth bound; cycle rejection; GC retention and reclamation; tampered-record,
stale-output, and malformed-target verification. The benchmark is
`examples/provenance_bench.rs` with `bench/scripts/run_provenance_bench.sh`.
