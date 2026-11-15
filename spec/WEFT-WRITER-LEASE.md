# WEFT-WRITER-LEASE — durable writer session leases

Status: **implemented, opt-in (X09)**. Normative for the records and the refusal
rules below. This document adds control-plane records only; it changes no frozen
frame byte, object byte, manifest byte, mode number or content root. Multi-writer
mode MW1 stays the default, and a store with no lease policy behaves exactly as
it did before this document existed.

## 1. Why

WeftFS already has two writer modes: MW1 (one exclusive kernel lock per store)
and MW2 (`spec/WEFT-MULTIWRITER.md`, optimistic transactions). Neither has a
bounded **session lifetime**. Without one, a live process that stops writing —
paused, wedged, or simply idle — can hold the store against a second writer
indefinitely, and the only release is the holder's own next operation.

A writer session lease is a durable, renewing grant with an absolute deadline.
It gives three things that a kernel lock cannot:

1. **Expiry.** A session that stops running loses the store at a known instant,
   not when somebody kills it.
2. **Fencing.** A resumed, stale session is refused by a typed error and cannot
   publish, so two writers can never both write.
3. **Fairness.** Waiters take a monotone ticket, so a stream of newcomers cannot
   overtake a writer that is already queued.

## 2. Records

Both records are canonical WCB (`weftcodec::wcb`) and are stored on the commit
plane, never mixed with object or manifest bytes.

### 2.1 `weft.writer.lease.v1` — `writer/lease`

Exactly these keys, in canonical order:

| key | type | meaning |
|---|---|---|
| `schema` | str | `weft.writer.lease.v1` |
| `owner` | str | session identity, 1–128 bytes |
| `fence` | u64 | fencing token, `>= 1`, monotone across takeovers |
| `sequence` | u64 | renewal counter within one tenure |
| `acquired_ns` | timestamp | instant the tenure began |
| `renewed_ns` | timestamp | instant of the most recent renewal |
| `deadline_ns` | timestamp | absolute deadline; a contender may take over at or after it |
| `ttl_ns` | u64 | lifetime granted by the acquisition that produced this record |
| `state` | str | `held` or `released` |
| `released_from` | str or null | token of the bytes a release tombstone replaced |
| `head_generation` | u64 | advisory: a generation this session published |
| `head_commit` | hash or null | advisory: that generation's commit |
| `head_commit_len` | u64 | advisory: that commit record's length |

Validation is mandatory on both encode and decode:

* `owner` is 1–128 bytes; `fence >= 1` (`0` means "no record");
* `deadline_ns >= acquired_ns`;
* `held` must not carry `released_from`; `released` must carry a non-empty
  token of at most 128 bytes.

A `held` record is **live** iff `now < deadline_ns`. A contender may take over
iff the record is `released` or `now >= deadline_ns`.

The `head_*` fields are advisory only. They are never a root, never compared
against `HEAD`, and never used to decide a publication.

### 2.2 `weft.writer.ticket.v1` — `writer/tickets/<16 lowercase hex>`

| key | type | meaning |
|---|---|---|
| `schema` | str | `weft.writer.ticket.v1` |
| `ticket` | u64 | queue position; ascending order is grant order |
| `owner` | str | session identity, 1–128 bytes |
| `deadline_ns` | timestamp | after this instant the ticket blocks nobody |

A ticket is **live** iff `now < deadline_ns`. A dead waiter eventually stops
blocking without any garbage-collection pass, which is the point of the deadline.

## 3. Control-plane I/O

* `writer/lease` is replaced only by a conditional write (compare-and-swap): the
  expected token is `sha256` of the bytes the writer last read or wrote. On a
  POSIX commit plane the compare-and-swap runs inside the short publication
  critical section (`LOCK.publish`); on an object-store plane the native
  conditional write does the same job. There is no unconditional replace and no
  unconditional delete of this record.
* `writer/tickets/<n>` is created once (`put_if_absent`), never overwritten, and
  deleted when its holder leaves. Absence during a delete is success.
* A writer's lease handle is a **separate** backend handle from the store's. The
  keeper thread must renew a session that is doing nothing else, and the store's
  backend sits behind a non-reentrant mutex. Both handles address the same record
  with the same compare-and-swap, so they cannot disagree about the holder.

## 4. Acquisition

Enabled with a non-zero lease lifetime, only together with the optimistic writer
mode (MW2). A lease requested with MW1 is refused `InvalidArgument` at open,
before any byte is written. Bounds: `1 s <= ttl <= 86,400 s`;
`1 ms <= ticket_ttl <= 3,600 s`; at most `4,096` acquisition attempts.

Protocol, repeated until it succeeds or the wait budget is spent:

1. If fairness is enabled, join the queue: list the existing tickets, take
   `highest + 1`, and create it once. A create that loses a race re-lists and
   retries; it never overwrites. Then, while a **live lower ticket** exists,
   sleep with capped exponential backoff (200 µs, doubling, capped at 25 ms).
2. Read `writer/lease`.
   * absent → this writer installs `fence = 1`;
   * live → wait for the deadline (or the wait budget) and retry;
   * released or expired → take over with `fence = previous + 1`.
3. Write the record with a compare-and-swap on the bytes just read (or on
   absence). A lost race is `Busy` and is retried.
4. Leave the queue (delete the ticket) and install the local view.

Terminal outcomes are typed, and none mutates state:

* `Busy` — a live holder, a live lower ticket, or lost compare-and-swaps still
  own the store after the wait budget;
* `LeaseExpired` — never from acquisition (an expired *own* lease is a later
  operation's refusal);
* `LimitExceeded` — the fence would overflow, or the queue exceeds 4,096 tickets.

## 5. Renewal

Renewal is one conditional write of the same record with:

* the same `fence` and `owner`,
* `sequence + 1`,
* `renewed_ns = now` and `deadline_ns = now + ttl`,
* the advisory head fields updated when the caller published something.

Renewal happens when the remaining lifetime is at most **half the TTL** (or on an
explicit request). A keeper thread renews an idle session at `ttl / 3` (capped at
200 ms per sleep). Renewal is refused `LeaseExpired` when this handle's deadline
has passed, when the record is absent, or when the record names another owner or
a higher fence. A live, renewing session therefore cannot be taken over: a
takeover requires the record to be `released` or past its absolute deadline, and
a renewal moves the deadline forward before that.

## 6. Release

Release writes a **tombstone**, never an unconditional delete: a `released`
record carrying `released_from` = the token of the bytes it replaced, written
with a compare-and-swap on those bytes. If a successor already replaced the
record, the successor's bytes are left exactly as they are and the release is a
no-op. A crashed session leaves a `held` record that expires at its deadline; no
tombstone is needed for correctness.

## 7. Publication fence

Every publication checks the lease **inside** the publication critical section,
before the commit record is written and before the head compare-and-swap:

* if there is no lease policy, this check is a no-op (MW1 / unleased MW2);
* otherwise the record is read and must name this handle at this `fence`, with
  `now < deadline_ns`.

Any other outcome is `LeaseExpired`, and because the check precedes both writes,
a refused publication **writes nothing**: no commit record, no head change. The
ordinary write path then renews (usually as a read and no write at all), so the
cost per publication is one extra control-record read.

`Store::require_writable` additionally rejects any mutating operation with
`LeaseExpired` when this handle's own installed deadline has passed — the local
fast check that a resumed, suspended process hits before doing any work.

## 8. Fairness guarantee

Tickets are numbered `highest + 1` and created once. Therefore:

> A ticket number assigned to a waiter is strictly greater than the number of
> every ticket present when it was assigned, and no later assignment can produce
> a lower number. A writer therefore never sees a **newer** arrival ahead of it
> in the queue: every writer that joins after it holds a higher ticket and waits
> behind it.

The guarantee is FIFO **by ticket**, subject to three stated limits:

1. **The lease holder is not queued.** A writer that already holds the lease
   finishes (or its lease expires / is released) before any queued writer runs.
   Fairness orders the *queue*, it does not preempt a holder.
2. **A ticket expires.** A waiter whose ticket deadline passes stops blocking
   others (that is how a dead waiter is forgotten) and forfeits its position.
   The default ticket lifetime (10 s) is twice the default wait budget (5 s), so
   a waiter that is still inside its budget cannot expire its own ticket.
3. **A dead waiter is forgotten after its ticket TTL**, not immediately.

## 9. Refusal classes

| situation | class |
|---|---|
| lease requested with MW1, or impossible lifetime | `InvalidArgument` |
| live holder / live lower ticket / lost CAS after the wait budget | `Busy` |
| this handle's deadline passed, or its fence was superseded | `LeaseExpired` |
| fence overflow, ticket-queue bound | `LimitExceeded` |
| record is not canonical, or fields violate §2.1 | `Malformed` |
| record schema is not this schema | `UnsupportedVersion` |

`LeaseExpired` is a new error class; see `spec/ERROR-CLASSES.md`.

## 10. What this does not do

* It is not a multi-writer protocol. At most one session holds the lease; MW2's
  transaction loop is unchanged and remains the way several processes write.
* It does not make clock skew safe. Deadlines are wall-clock instants compared by
  every participant; a badly skewed clock can expire a live session early. The
  refusal is still typed and still prevents a second writer, but liveness is not
  guaranteed across a clock jump.
* It does not bound the *work* a live holder may do. A session that keeps
  renewing keeps the store; that is the intended meaning of a live writer.
* The advisory `head_*` fields are not an attestation of durability.
