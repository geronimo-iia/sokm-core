# Decision: compact() must be called between ticks; kernel indices are invalidated

**Date:** 2026-07-30
**Status:** accepted

## Problem

`GestaltKernelGraph::compact()` removes extinct kernels from both modalities and reindexes
the `CrossEdgeStore` to match. This is a destructive operation: kernel indices are contiguous
before and after compact, but the mapping from old to new indices is arbitrary — old index `i`
may become new index `j ≠ i`, or disappear entirely if extinct.

The question was: when is it safe to call compact(), and what guarantees does the API make
about kernel indices after the call?

## Decision

`compact()` is safe to call only between ticks — never during a tick (e.g. inside a training
loop that calls `tick()` and `compact()` in the same iteration without a tick boundary between
them). After `compact()` returns, all previously captured kernel indices for both modalities
are invalid. Callers must re-query kernel state via the graph's accessors.

## Why

`tick()` computes activation scores at the start of each tick against the current kernel store.
If `compact()` is called mid-tick (conceptually), the index mapping used for cross-edge
strengthen would reference old indices while the store has already been reindexed — producing
silently wrong cross edges or out-of-bounds panics.

Between ticks, no activation state is live. The reindex map returned by `compact_with_map()`
is applied atomically to `CrossEdgeStore::reindex()` before the next tick begins. This is
the only safe window.

## Borrow safety

`compact()` avoids a double-mutable-borrow by sequencing: `compact_with_map()` on each
modality returns an owned `Vec<Option<usize>>` and ends the mutable borrow on that modality's
store before `self.cross.reindex()` is called. No unsafe code, no workaround — this is the
correct ownership split.

## Consequences

- Callers must not store kernel indices across `compact()` calls.
- `compact()` is cheap when the extinction fraction is high — `CrossEdgeStore::reindex`
  scales with survivors, not total edges. Bench: 1000 edges, 75% extinct → 27 µs;
  0% extinct → 62 µs. Call compact() eagerly after extinction-heavy training phases.
- The `compact_lifecycle` example demonstrates the full pattern.

## What was rejected

- Automatic compact inside `tick()` after each extinction event: would invalidate indices
  mid-tick and make the tick-level API unpredictable.
- Returning a reindex map from `compact()` for callers to remap their own indices: adds
  caller burden; the correct pattern is to not hold raw indices across compact() at all.
