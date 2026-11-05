# WEFT-QUOTA v1 — write budgets and branching time travel

Status: **normative for the store-level quota record and the branch/merge
surface, both languages.** It does not change any frozen v1 content, commit,
manifest, or commit-chain byte.

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
only the caller's own counters and own limits.

### 1.3 Persisted record

Key `quota` at the store root. Bytes:

```
WFQ1 || u64le(len) || WCB || BLAKE3("weft/quota/v1", WCB)
```

WCB map (`weftfs.quota.v1`):

```
{
  "schema": "weftfs.quota.v1",
  "generation": u64,                # committed head this record describes
  "commit": bytes32,                # committed head hash
  "budgets": [ principal_fields..., "max_new_bytes": u64, "max_objects": u64 ],
  "usage":   [ principal_fields..., "used_bytes": u64,   "used_objects": u64 ]
}
principal_fields = { "kind":"posix", "uid":u64, "gid":u64 }
                 | { "kind":"tenant", "tenant":text }
```

Arrays are sorted by principal key and contain no duplicates:

```
posix  -> "p:%020u:%020u"
tenant -> "t:<name>"
```

POSIX principals sort before tenant principals. Decoding rejects an unsorted
array, an unknown key, a wrong schema, a bad digest, or a length mismatch
(`Corrupt` / `UnsupportedVersion` / `NonCanonical`).

The record is written atomically after the commit (`tmp` + fsync +
`rename(2)` + directory fsync). It is never authoritative over content.

### 1.4 Consistency and rebuild

On open, the record is compared with the committed head. If it names a
different generation or commit, or cannot be decoded:

- quota-enforcing writes fail closed with `Corrupt`;
- `verify` reports the problem;
- `quota rebuild` recomputes cumulative usage from the **published** commit
  chain (never from unpublished commit files or process memory) and rewrites
  the record. If the record is unusable and no policy is known, rebuild is
  refused (re-supply `--quota`), because rebuilding an empty policy would
  silently disable enforcement.

`quota reset` zeroes every counter; the policy is unchanged. `quota rebuild`
and `quota reset` are the only operations that rewrite usage other than an
admitted commit. A missing record means "no policy, no usage"; creating one is
an explicit store-local admin act.

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
