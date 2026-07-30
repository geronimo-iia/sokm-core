# Decision: integrate sokm-multimodal into sokm-core

**Date:** 2026-07-30
**Status:** accepted

## Context

sokm-multimodal exists in a separate repository (sokm-rs). It implements Gestalt K³ —
two SOKM modalities coupled via a directed bipartite cross-edge store. The implementation
follows Hoya Eqs. 4.1, 4.3, 4.7 applied cross-modally ([INFERRED] — not verbatim Hoya).

## Decision

Copy the crate as-is into sokm-core. No logic changes. Rename `AosGestaltGraph` →
`DefaultGestaltGraph` for naming consistency.

## Limitations acknowledged at integration time

- `CrossEdgeStore::sources()` is O(E) full scan. Acceptable at current scale.
  See `.claude/note-sources-o-e-reverse-scan.md`.
- No benchmarks at scale. See `.claude/note-no-benchmark-no-large-scale.md`.

## Consequences

- sokm-core now has a cross-modal memory primitive.
- v0.3.0 bumps workspace version for all four crates.
- Publish order: `sokm` → `sokm-kernel` → `sokm-emotion` → `sokm-multimodal`.
