# Decision: CrossEdgeStore sources() — O(1) reverse index instead of O(E) scan

**Date:** 2026-07-30
**Status:** accepted

## Problem

`CrossEdgeStore::sources(j)` returns all modal1 sources that reach modal2 kernel `j`.
The original implementation scanned the full `weights` HashMap on every call — O(E) per query.
`recall_from_modal2` calls `sources(j)` for every active modal2 kernel, making reverse recall
O(k × E) where k is the number of active kernels. At 1000 kernels and typical edge densities,
this is a linear scan over the entire edge store per recall call.

## Decision

Add a `reverse: HashMap<usize, Vec<usize>>` field to `CrossEdgeStore`, maintained in lockstep
with `weights` across all mutation paths: `set`, `apply_increments`, `prune_below`,
`prune_inactive`, `reindex`. `sources(j)` becomes an O(1) lookup returning a `Vec<usize>` of
source indices, then a secondary lookup per source into `weights` — O(k) where k is the actual
fanin of target j, not the total edge count.

## Why

Benchmarks confirm parity: `recall_from_modal2 ≈ recall_from_modal1` at all (n, d) pairs
(e.g. 210 µs vs 210 µs at n=1000, d=358). The O(E) scan was the only asymmetry between
forward and reverse recall. Memory cost is one extra `HashMap<usize, Vec<usize>>` — negligible
compared to `weights` and `ticks`.

## What was rejected

- **Lazy construction at query time:** would require a full scan on first call after any mutation.
  Defeats the purpose and complicates invalidation logic.
- **Keeping O(E) scan with a note:** acceptable for toy scale, wrong for production use.
  The bench note `.claude/note-sources-o-e-reverse-scan.md` explicitly flagged this for fixing.

## Invariants

The reverse index is correct iff `reverse[j]` contains `i` exactly when `weights[(i,j)]` exists.
Three tests enforce this: consistency after `prune_below`, after `set(w=0)`, and after `reindex`.
A dedup guard is not added — `apply_increments` checks `weights.get_mut` before inserting, so
duplicate entries are not possible via the public API.
