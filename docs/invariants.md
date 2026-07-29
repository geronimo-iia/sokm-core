# sokm-core invariants

Invariants that must hold across the entire codebase. Each entry names the
invariant, states where it is enforced, and explains what breaks if it is
violated.

---

## 1. Same-class strengthening filter

**Location:** `sokm/src/ops.rs` — `strengthen()`

**Rule:** `strengthen` is class-agnostic. It never inspects kernel class labels.
The caller — `KernelGraph::tick` in `sokm-kernel` — is solely responsible for
pre-filtering `activated` to same-class pairs before passing them to
`strengthen`.

**Why it exists:** Hoya Eqs 4.6–4.7 specify that Hebbian strengthening applies
only between kernels of the same class. The `sokm` crate has no knowledge of
classes — enforcement lives one layer up in `sokm-kernel`.

**What breaks:** Passing cross-class pairs or unlabelled kernels (class = None)
to `strengthen` produces incorrect Hebbian strengthening with no error or
warning. The edge store silently accumulates weights between unrelated concepts.

**Where enforced in code:**
- `sokm-kernel/src/graph.rs` — Step 8 of `tick()`, the `same_class_activated`
  filter. The `matches!` pattern explicitly excludes `None == None` pairs.
- `sokm/src/ops.rs` — `strengthen()` doc comment.

---

## 2. compact() must only be called between ticks

**Location:** `sokm-kernel/src/graph.rs` — `compact()` and `compact_with_map()`

**Rule:** `KernelGraph::compact` (and `compact_with_map`) must never be called
during a tick — not from a tick callback, not concurrently with tick.

**Why it exists:** Compaction remaps all kernel indices. Mid-tick, the following
structures hold live kernel indices that would be silently invalidated:
- `prop_scratch` / `prop_touched` — indexed by kernel position
- `stm` — holds kernel indices
- any in-flight `activated` or `fired` list

**What breaks:** Calling compact mid-tick corrupts propagation accumulation,
STM state, and the activated list for the current tick. The corruption is
silent — no panic, wrong learning.

**Where enforced in code:**
- `sokm-kernel/src/graph.rs` — `compact()` and `compact_with_map()` doc
  comments.
- `sokm-kernel/src/stm.rs` — `reindex()` doc comment (called only by
  `compact_with_map`).

---

## 3. CSR and pending are mutually exclusive per edge

**Location:** `sokm/src/sparse.rs` — `SparseEdgeStore`

**Rule:** For any canonical edge key `(a, b)` with `a <= b`, the key is present
in at most one of `csr_index` or `pending` at any time. Never both.

**Why it exists:** `get_weight` returns `csr + pending` when pending is
non-zero. If an edge were non-zero in both, `get_weight` would double-count it.

**How it is maintained:**
- `set_weight`: checks `csr_index` first; writes to CSR if found, otherwise
  writes to pending.
- `apply_increments`: same check — CSR path or pending path, never both.
- `compact()`: merges pending into CSR, then clears pending entirely.

**What breaks:** Double-counted edge weights in `get_weight`, `neighbors`, and
all propagation computations. Strengthening would overshoot `w_max`.

**Where enforced in code:**
- `sokm/src/sparse.rs` — struct-level `INVARIANT` comment.
- `sokm/src/sparse.rs` — `set_weight` and `apply_increments` implementations.

---

## 4. p1 boundary semantics: strict greater-than

**Location:** `sokm/src/ops.rs` — `prune()` / `prune_inactive()`

**Rule:** The inactivity pruning condition is `current_tick - last_active > p1`,
not `>= p1`. An edge touched at tick T survives for exactly p1 inactive ticks
and is pruned on tick `T + p1 + 1`.

**Why it exists:** Hoya specifies extinction *after* p1 ticks of inactivity.
Strict `>` means the edge is alive at `current_tick - T == p1` (the p1-th
inactive tick) and dead at `current_tick - T == p1 + 1`.

**What breaks:** Using `>=` would prune one tick too early, shortening edge
lifetime by one tick relative to the algorithm specification.

**Where enforced in code:**
- `sokm/src/ops.rs` — `prune()` doc comment.
- `sokm/src/sparse.rs` — `prune_inactive()` implementation.
- `sokm/src/store.rs` — `HashEdgeStore::prune_inactive()` implementation.
- Locked by test: `prune_exact_p1_boundary_not_pruned` in `sokm/src/ops.rs`.

---

## 5. prop_scratch is fully zeroed at the end of every tick

**Location:** `sokm-kernel/src/graph.rs` — `KernelGraph::tick()`

**Rule:** `prop_scratch` must contain only zeros when `tick()` returns. The
zero-pass at the end of tick clears only the indices recorded in `prop_touched`
(O(active × degree), not O(num_nodes)).

**Why it exists:** `prop_scratch` is reused across ticks to avoid per-tick
allocation. Any non-zero value left from tick N would be read as propagated
activation in tick N+1, causing phantom excitation.

**Corollary:** `prop_scratch.len()` is at most 1 behind `kernels.len()`. Step
1.5 resizes scratch to `kernels.len()` before propagation. On a grow tick, the
new kernel is added after Step 1.5 — scratch covers `kernels.len() - 1` for
that tick only. The new slot is initialised to 0.0 by `Vec::resize` and is
never written during that tick.

**What breaks:** Stale values in `prop_scratch` cause kernels to appear
propagation-excited on the next tick even when no firing kernel has an edge to
them. This suppresses growth incorrectly and corrupts the excited check.

**Where enforced in code:**
- `sokm-kernel/src/graph.rs` — field-level `INVARIANT` comments on
  `prop_scratch`.
- `sokm-kernel/src/graph.rs` — zero-pass at end of `tick()` and at end of
  `compact_with_map()`.
- Locked by test: `prop_scratch_does_not_leak_between_ticks`.
