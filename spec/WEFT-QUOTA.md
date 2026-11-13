# WEFT-QUOTA v1 — write budgets and branching time travel

Status: **normative for the store-level quota accounting and policy record and
the branch/merge surface, both languages.** It does not change any frozen v1
content, manifest, checkpoint, or commit-record schema; quota-enabled commits
carry one additive `set_quota_usage` op (WEFT-MODEL §2.2). Existing commits
without that op replay unchanged.

## 1. Write budgets (quotas)

A budget is an **admission** budget, not a footprint measurement:

```
budget   = { principal, max_new_bytes: u64, max_objects: u64 }
principal = posix { uid: u64, gid: u64 } | tenant { name: text 1..=256 bytes }
```

CLI forms: `--quota uid:GID:BYTES:OBJECTS`, `--quota
tenant:NAME:BYTES:OBJECTS`, or `--quota-file FILE` (one directive per line,
`#` comments and blank lines ignored). Supplying any budget on a writable open
installs or replaces the persisted policy; omitting all budgets enforces the
persisted policy unchanged. Read-only opens refuse a policy install.

### 1.1 Charging

Before a commit is published, the store computes the additions it would make:

- `new_objects` — every object record whose id is absent from the pre-commit
  state. Changed records (a new version, attributes, or mode) are not new
  objects.
- `new_bytes` — for every new or changed object record:
  - the canonical encoded length of the object record itself, plus
  - for a new or changed file content root, the on-disk byte length of every
    manifest, codec frame, and chunk blob reachable from it that is **not**
    already reachable from committed state. A blob reachable from committed
    state costs zero. Blobs are counted at most once per commit.

The namespace root object (created by `init`) is never charged. Commit files,
checkpoint state manifests, and the quota record itself are not charged.

Attribution:

- **POSIX**: a record is charged to its own `(uid, gid)`.
- **Tenant**: a record that carries the reserved attribute `weft.tenant` (text)
  is also charged to that tenant. The attribute is ordinary committed object
  state, so tenant usage is rebuildable exactly like POSIX usage.

Cumulative usage is charged even after an object is deleted (admission, not
footprint). Only an explicit admin operation resets it.

### 1.2 Admission and race safety

The check happens inside the store's exclusive writer lock, before commit-file
or superblock bytes exist. A commit that would exceed **any** budget is
rejected with class `BudgetExceeded` and the fixed message
`write budget exceeded`. Nothing is published: generation, head, and all four
state maps are unchanged. Content blobs already written for the rejected
commit are unreferenced and reclaimable by GC.

Denial text is **non-disclosing**: it names neither the current usage, nor the
limit, nor the principal, nor any other principal's data. `quota show` returns
only the caller's own counters and own limits: the caller is the invoking
process's real `(uid, gid)`, and an explicit `--uid`/`--gid` (or `--tenant`)
selects that principal instead. A principal with no configured budget is
unconstrained; budgets are per-principal and opt-in.

### 1.3 Committed usage (`set_quota_usage`)

Usage is committed state. When a policy is configured, every commit appends
one `set_quota_usage` op as its **last** op:

```
{
  "op": "set_quota_usage",
  "reset": bool,
  "usage": [
    { "kind":"posix",  "uid": u64, "gid": u64, "used_bytes": u64, "used_objects": u64 },
    { "kind":"tenant", "tenant": text,        "used_bytes": u64, "used_objects": u64 }
  ]
}
```

- `reset == false`: the array is the cumulative admitted usage **after** every
  other op in the same commit. It is sorted by principal key and contains no
  duplicates.
- `reset == true`: the explicit admin reset. It MUST be the only op in its
  commit and MUST declare an empty array. It establishes a new accounting
  baseline for later commits and for `quota rebuild`.

The op changes no namespace state; checkpoints therefore do not carry usage.
Usage is recovered from the commit log (the head assertion, or replay when the
head has none), so it survives reopen without any separate durable write. The
commit schema remains `weftfs.commit.v1`: the op is an additive extension, and
commits written before it existed replay unchanged (usage is derived from
their object ops).

On replay, the store recomputes the admission totals for every commit. A
`reset:false` assertion that disagrees with those totals is `Corrupt` (fail
closed). A `reset:true` assertion that is not alone or is non-empty is also
`Corrupt`. `verify --full` and `quota rebuild` perform this replay; the
recomputed totals are also checked against the in-memory counters.

### 1.4 Policy file and migration

Policy is configuration, not accounting. It is written once to the store-root
`quota` file when a policy is installed or replaced:

```
WFQ2 || u64le(len) || WCB || BLAKE3("weft/quota/v2", WCB)
WCB  = { "schema":"weftfs.quota.v2", "budgets": [ ... ] }
```

Budget arrays are sorted by principal key with no duplicates. Decoding
rejects an unsorted array, an unknown key, a wrong schema, a bad digest, or a
length mismatch (`Corrupt` / `UnsupportedVersion` / `NonCanonical`). This file
is never rewritten by an ordinary commit.

`WFQ1` (schema `weftfs.quota.v1`) is the legacy combined policy+usage record.
It is read only for migration. On a **writable** open:

1. the legacy policy is installed in memory;
2. usage is recomputed from the published commit chain (a committed usage op,
   when present, is validated first);
3. the exact legacy bytes are preserved at `quota.migrated`;
4. `quota` is replaced by the `WFQ2` policy-only record;
5. if the recomputed usage is non-empty and the head has no usage op, one
   corrective `set_quota_usage` op is published.

A read-only open imports the legacy policy and recomputed usage in memory and
writes nothing.

Rules when records coexist:

- `quota` is `WFQ1` and `quota.migrated` exists with **different** bytes: the
  migration is ambiguous; open marks quota invalid and writes fail closed with
  `Corrupt`.
- `quota` is `WFQ2` and `quota.migrated` exists: the preserved record is
  historical. If it names the **current** head and its usage differs from the
  committed usage for that head, open marks quota invalid and writes fail
  closed. An older-generation record is evidence only and is not compared.
- A legacy record that names the current head with a different usage is
  `Corrupt` during migration. A stale legacy record (different generation or
  commit) has its usage recomputed from the authoritative chain; the repair is
  recorded in the store notes.

### 1.5 Consistency and rebuild

On open, usage is the head commit's `reset:false` assertion. If the head has no
assertion (older chain, or a checkpoint boundary), the chain is replayed,
validated, and used. Any inconsistent assertion fails closed for writes.

`quota rebuild` replays the chain, validates every assertion, and publishes a
corrective `set_quota_usage` op. It respects an earlier `reset` baseline. If
the policy is unknown (unusable legacy/record), rebuild is refused
(`Corrupt`); re-supply `--quota` instead. `quota reset` publishes one reset op
and zeroes usage; both are explicit admin operations and are the only ways
usage changes other than an admitted commit. `quota show` returns only the
caller's own counters and own limits (the caller's real `(uid, gid)` unless
`--uid`/`--gid`/`--tenant` names another principal).

## 2. Branching time travel

### 2.1 Snapshot branch

`snapshot branch <source-snapshot> <new-name>` publishes one commit that adds a
snapshot ref with the source snapshot's commit hash. It never rewrites history.
Because commits and snapshot refs are immutable, later writes cannot move
either ref. `<source>` may also be a published commit hex.

### 2.2 Context branch

`context branch <source-context> <new-name> [--task T] [--model M]` persists a
new Context Root:

- the selector is the source root's selector, conjoined (in order) with
  `attrs.task == "T"` and/or `attrs.model == "M"` when supplied; an existing
  `and` node is flattened so the AST stays canonical;
- the pinned snapshot is the **source root's** commit (the fork commit);
- `--replace` replaces an existing same-name root;
- lineage is recorded as a root-subject `derived_from` relation whose target is
  the source context name and whose attributes are `weft.context` (new name),
  `weft.context_id`, and `weft.source_context_id`.

Rust publishes the new root and its lineage relation in one group commit; Go
publishes two consecutive commits (root, then lineage). Both are visible in the
same store state.

### 2.3 Union merge

`merge <left> <right>` publishes a commit on the live head that unions the path
sets of two snapshot refs (or published commits).

- A path present in both inputs must have identical content, else `Conflict`
  naming the path. Regular files compare the frozen `ContentId` and size;
  symlinks compare the target bytes; directories always match.
- A path already present at the live head must have identical content, else
  `Conflict` naming the path. Metadata at the head is never overwritten.
- Only missing paths are added. A source object record is reused verbatim when
  its id is free or names an identical record; otherwise it is copied under
  `H("weft/object/v1", "weft/merge/v1" || path || content_root)`.
- Nothing is published for a conflicting merge. Merging the same pair again is
  idempotent.

This is explicitly a **union merge**, not a three-way content merge: there is
no merge base analysis, no per-hunk resolution, and no conflict marker output.
