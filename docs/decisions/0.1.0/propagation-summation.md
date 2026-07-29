# Decision: Propagation is summation (Eq 4.3), not max

## Problem

Two interpretations exist for combining propagated activations from multiple
source kernels: take the maximum (winner-takes-all) or accumulate the sum.
Hoya Eq 4.3 is unambiguous, but an early implementation used max.

## Decision

Propagation is summation: `K_j += γ·w_ij·K_i(x)` [Eq 4.3].

Two forms:
- `propagate` [Eq 4.4]: binary gate, `I_i = 1 if K_i(x) >= θ_k`. Used in construction (`tick`).
- `propagate_soft` [Eq 4.3]: graded form. Used in retrieval/recall.

## Why

Verified bug fix: max-propagation produced wrong recall results compared to
Hoya's worked examples. Summation matches the paper and fixed the regression.

`KernelGraph::tick` uses binary (`propagate`). Recall paths use soft
(`propagate_soft`, `cross_propagate_soft`).

## What was rejected

Max propagation: confirmed wrong, removed.
