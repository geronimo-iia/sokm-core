# Decision: sokm-multimodal is exactly two modalities — this is the spec, not a limitation

**Date:** 2026-07-30
**Status:** accepted

## Context

Hoya (2005) defines Gestalt K³ for exactly two modalities. The formalism is symmetric:
two independent `KernelGraph` instances coupled by a directed bipartite cross-edge store.
Equations 4.1, 4.3, 4.7 are written for this two-modality case — no generalisation to N
is given or implied.

## Decision

`sokm-multimodal` implements Hoya's Gestalt K³ verbatim for two modalities.
`GestaltKernelGraph<S1, S2, K1, K2>` is a concrete generic type — no dynamic dispatch,
no vtable, no stored config, fully serialisable. This is the correct representation for
a fixed two-modality structure.

## Why two modalities

- **Covers the majority of real associative memory tasks:** audio↔visual, text↔image,
  sensor↔label, stimulus↔response. Most biological cross-modal binding is pairwise.
- **Zero overhead:** both modalities tick in sequence, cross edges update once.
  No topology map, no trait objects, no registration. The compiler sees concrete types.
- **Fully validated:** 72 tests, criterion benchmarks at 1000 kernels × 358d,
  convergence example with 2-class separation verified in both directions.
- **Serialisable end-to-end:** `GestaltKernelGraph`, `CrossEdgeStore`, both `KernelGraph`
  instances all serialise independently under `--features serde`. Reconstruction is explicit
  and type-safe.

## What was considered

Generalising `GestaltKernelGraph` to N modalities via a `Vec<Box<dyn Modality>>` was
considered and rejected for this crate. It would add trait object overhead and topology
management to a type that has no need for them. The two-modality case is closed — adding
a third modality is a different design problem, not a parameter change.

## Consequences

- `sokm-multimodal` is production-quality for the two-modality case.
- Callers needing more than two modalities need a different crate.
- `GestaltKernelGraph` will not gain a `N` parameter — that would break the existing API
  and impose overhead on users who only need two modalities.
