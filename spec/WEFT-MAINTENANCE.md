# WEFT-MAINTENANCE — durable progress records for bounded maintenance

Status: **implemented in the Rust store (T080)**. Normative for
`weft.maintenance-state.v1` and the plan key of `weft.maintenance-plan.v1`. A Go
implementation does not exist. No frozen byte, mode or root changes.

## 1. Two records, two lifetimes

```
weft.maintenance-plan.v1 {
  schema, candidates: [ {kind: "object"|"manifest", hash: bytes32, bytes: u64} ],
  reachable_manifests, reachable_chunks, orphan_records, issued_at
}

weft.maintenance-state.v1 {
  schema, plan_root, generation, head,
  candidates, cursor, reclaimed_blobs, reclaimed_bytes, deferred, passes,
  status: "planned"|"running"|"complete",
  updated_at, last_rate_bytes_per_second, last_duty_percent,
  budget: {bytes_per_second, cpu_share_percent, max_batch_bytes, max_pass_bytes}
}
```

* The **plan** is written once per plan. It is a snapshot of the unreachable
  set at the time it was taken.
* The **state** is written after every batch, so a `SIGKILL` loses at most one
  batch of progress.
* Both are ordinary namespace files under `/maintenance/`, so they are reachable
  state: snapshots, checkpoints and pins retain them and GC collects them with
  the tree. A write **truncates** the file to the new record length; a range
  write would leave the previous record's tail and produce a record that does
  not decode (found by the SIGKILL arm of the court).

### 1.1 `plan_root` is a key, not a record identity

```
plan_root = H("weft/maintenance/plan/v1", WCB({schema, candidates}))
```

Only the candidate list is hashed. Reachable counts and the timestamp are not,
so re-planning over the same garbage yields the same key and a persisted cursor
stays meaningful. It is deliberately **not** the content address of the plan
record.

## 2. Budget

```
bytes_per_second     > 0     minimum wall time per batch (pacing ceiling)
cpu_share_percent    1..=100 maximum share of wall time maintenance may use
max_batch_bytes      > 0     hard cap on one batch
max_pass_bytes       > 0     hard cap on one candidate; a larger blob is a refusal
min_batch_bytes      > 0     do not start a batch smaller than this
```

A zero rate, a zero duty cycle, `min_batch_bytes > max_batch_bytes`, or
`max_batch_bytes > max_pass_bytes` is `InvalidArgument`. A candidate larger than
`max_pass_bytes` is `BudgetExceeded` naming the blob: a blob cannot be split, so
the budget as declared cannot be met, and the driver refuses instead of
exceeding it.

`bytes_per_second` is an **interference ceiling used to pace batches**, not a
throughput promise. A pass whose work is scan-bound will run slower than the
declared rate and the state records what it actually achieved.

## 3. Resume

`maintenance_run` reuses the persisted plan while its state is not `complete`.
A stale plan is safe because every candidate is re-verified (§4). A new plan is
made only when the previous one completed or none exists, so garbage created
during a run waits for the next run instead of reshaping the cursor mid-flight.
This is what makes a restart resume rather than restart.

## 4. Safety

* `maintenance_reclaim` acquires a fresh GC epoch, removes unlinked object
  records (as `gc --apply` does; otherwise it could delete a blob an orphan
  record still names and leave a store that `verify` reports as corrupt), then
  **re-derives reachability** and refuses to delete a candidate that is
  reachable now. A stale plan therefore wastes work but cannot reclaim live
  data.
* The grace gate still applies: a blob younger than `gc_grace_secs` is skipped
  and reported.
* A second concurrent pass is refused by the epoch CAS (`Busy`) or serialises
  through it; `delete_if_present` makes a lost race a no-op, so two passes
  together can never delete a candidate twice.
* Maintenance is data-only: it plans `objects/` and `manifests/`. Advisory
  index segments are left to `gc`.

## 5. Ops surface

`weftfs ops` is read-only unless `--apply`:

```text
weftfs ops [--json]                       # counters, reachability, dry-run plan
weftfs ops --apply [--bytes-per-second N] [--cpu-share P]
           [--max-batch-bytes N] [--max-pass-bytes N] [--min-batch-bytes N]
           [--max-passes K]
```

The read-only form reports generation, head, index bytes, quarantine count,
reachable manifests/chunks, orphan records, the candidate count and bytes, the
first 64 candidates, and the persisted state (cursor, passes, reclaimed bytes,
deferred work, last measured rate and duty). It creates nothing and deletes
nothing. The `--apply` form runs at most `--max-passes` batches and prints the
same counters plus the achieved rate.

## 6. What this is not

* Not a scheduler integration: the duty cycle is enforced by sleeping, not by
  cgroup CPU shares. Concurrent contention on the machine is not compensated.
* Not a throughput claim: the achieved rate is reported, not guaranteed.
* Not a replacement for `gc --apply`: a full pass over a cold tree still costs
  one scan, and `gc` reclaims advisory index segments and transaction records
  that maintenance leaves alone.
* Not cross-language: WeftFS-Go has no maintenance driver.
