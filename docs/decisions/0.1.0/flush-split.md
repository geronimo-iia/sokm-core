# Decision: Split `flush` into `compact` + `snapshot`

## Problem

`MemoryStore::flush` bundled two distinct operations:
1. Graph compaction — removes extinct kernels, shifts indices (irreversible state mutation)
2. Snapshot write — writes `.bin` + `.tvim` (I/O only, no state mutation)

Callers wanting periodic persistence (e.g. `snapshot_interval_ticks`) were forced
to also trigger compaction, which is not always desirable mid-session.

## Decision

Three-method API on both `MemoryStore<B>` and `EmotionalMemoryStore<B>`:

```rust
pub fn compact(&mut self)  -> anyhow::Result<FlushResult>;  // compaction only, no I/O
pub fn snapshot(&self)     -> anyhow::Result<FlushResult>;  // snapshot write, no compaction
pub fn flush(&mut self)    -> anyhow::Result<FlushResult>;  // compact then snapshot
```

`flush` is kept as a convenience delegate — removing it would be a breaking change
with no benefit. `snapshot` takes `&self` because extinct kernels are serializable
as-is; no state normalization is required before writing.

## Why

Prerequisite for `snapshot_interval_ticks` auto-flush: auto-flush should write
a snapshot without forcing compaction on every tick interval. The split makes
both operations independently composable.

## What was rejected

Making `snapshot` require `&mut self`: unnecessary — snapshot is pure I/O and the
current graph state (including extinct kernels) is valid to serialize. Requiring
`&mut self` would have prevented calling it from read-heavy contexts.
