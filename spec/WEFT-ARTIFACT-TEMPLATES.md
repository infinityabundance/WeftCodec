# WEFT-ARTIFACT-TEMPLATES — evaluation and synthetic-data run records

Status: **implemented in the Rust store (T077)**. Normative for
`weft.eval-run.v1` and `weft.synth-data-run.v1`. A Go implementation does not
exist; a Go store MUST reject both schemas rather than guess. No frozen byte,
mode or root changes: both records are new, additive and content-addressed.

## 1. Why a template and not just a provenance receipt

A provenance receipt (`weft.provenance.v1`) answers *how bytes were produced* in
a general shape: operation, params, inputs, outputs, environment, determinism.
It is deliberately generic. An evaluation run and a synthetic-data generation
run have **obligatory structure** — an evaluation has a dataset, a model, a
config and metric values; a generation run has a config, a generator and a
seed — and a reader should be able to check that structure without guessing from
free-form params. The template is that structure; the run also publishes a
provenance receipt with the same edges, so the taint tracer walks through it
like any other operation.

## 2. `weft.eval-run.v1`

```
{
  "schema": "weft.eval-run.v1",
  "run": text,                         # caller-declared name; not identity
  "inputs": [ {"role": text, "content": bytes32} ],   # sorted by (role, content)
  "outputs": [bytes32],                # strictly ascending content roots
  "metrics": [ {"name": text, "value_bits": u64, "kind": text} ],
  "seed": bytes,                       # MUST be empty for this schema
  "generator": bytes32 | null,         # MUST be null for this schema
  "environment": { "tool": text, "version": text, "env_hash": bytes32|null, ... },
  "issued_at": timestamp
}
```

* `inputs` MUST contain exactly the roles `config`, `dataset` and `model`.
  (The store writes them in that order; a decoded record is validated for
  strict `(role, content)` ordering and for role membership.)
* `metrics` values are recorded as the **IEEE-754 binary64 bit pattern** of the
  value the producer held. A metric is never recorded as a decimal string, so a
  recorded value is exact even when the decimal form is not (`0.1`).
  `kind` is a caller-declared label (`ratio`, `scalar`, `loss`, ...).
* Metric names and kinds are 1..1024 bytes; at most 4096 metrics; at most 4096
  outputs.

## 3. `weft.synth-data-run.v1`

```
{
  "schema": "weft.synth-data-run.v1",
  "run": text,
  "inputs": [ {"role": "config"|"generator", "content": bytes32} ],  # sorted
  "outputs": [bytes32],                # the produced dataset shards
  "metrics": [],                       # MUST be empty for this schema
  "seed": bytes,                       # 1..4096 bytes; MUST be present
  "generator": bytes32,                # MUST equal the `generator` input root
  "environment": { ... },
  "issued_at": timestamp
}
```

The seed is recorded **verbatim**, not hashed: a seed is the smallest possible
description of the generation, and a reader that wants to attempt a replay needs
it. Recording it is a deliberate disclosure; a caller that will not disclose a
seed must not record a synthetic-data template.

## 4. Identity, storage, attachment

* `TemplateId = H("weft/artifact-template/v1", WCB(record))`. The record bytes
  include `schema`; any change to any field changes the identity.
* The record is stored as a content-addressed blob in the manifest namespace
  under the template domain. It is **not** a manifest.
* Every output object gets two edges in the same publication:
  * a `weft.provenance.v1` receipt whose `operation` is the template schema,
    whose `inputs` are the template inputs and whose `outputs` are the template
    outputs. Its `params` carry `template` (the template id) and `run`.
  * a relation `artifact_template` from the output object to the template id, so
    a reader holding an output can find the declaration.
* **Declared inputs must exist.** Recording resolves every declared input content
  root against live state. A root the store cannot produce is `NotFound` and
  **nothing is written** — a template never declares an input the store cannot
  show. Several live objects may share one content root (a copy is the same
  artifact); existence, not uniqueness, is required. The zero hash is
  `Malformed`.

## 5. Replay

`replay(template_id)` re-reads every declared input **by content root** through
the store's verified read path and re-reads every recorded output, then reports:

```
{ template, schema,
  inputs:  [ { role, content, objects, bytes, exact } ],
  outputs: [ { content, objects, bytes, exact } ],
  exact: bool,
  note: text }
```

`exact` is true only when every declared input and every recorded output
resolves to at least one live object whose bytes verify. A deleted output makes
replay `exact=false`; it does not make replay fail silently.

### What replay establishes

* every declared input is present and intact in this store;
* every recorded output is present and intact;
* the record is canonical and content-addressed, so the declaration itself has
  not been altered;
* the graph edges exist, so the taint tracer reaches the same closure.

### What replay does **not** establish

* it does **not** re-run the evaluation, the training step or the generator;
* it does **not** establish numerical or behavioural determinism, bitwise
  reproducibility of any kernel, floating-point associativity, or that the
  recorded metrics followed from the recorded inputs;
* the `environment` block is a declared claim about the producer's environment,
  not a proof of it;
* `determinism` recorded on the linked provenance receipt is the producer's
  declaration (`exact` / `declared` / `unknown`); nothing here verifies it.

A template is a record of an experiment, not a proof that the experiment is
reproducible. Reproducibility is an environment property and this document makes
no claim about it.

## 6. Bounds and refusals

| Field | Bound | Violation |
|---|---|---|
| metrics | 4,096 | `LimitExceeded` |
| outputs | 4,096 | `LimitExceeded` |
| seed | 4,096 bytes | `LimitExceeded` |
| text fields | 1,024 bytes | `LimitExceeded` / `Malformed` |
| unknown key | — | `Malformed` |
| evaluation schema with a seed/generator | — | `Malformed` |
| synthetic schema without a seed/generator, or with metrics | — | `Malformed` |
| declared input absent | — | `NotFound` |
| declared input is the zero hash | — | `Malformed` |
| output object absent | — | `NotFound` |
