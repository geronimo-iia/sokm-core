# Decision: SIMD feature not recommended for sokm-multimodal

**Date:** 2026-07-30
**Status:** accepted

## Problem

`sokm-kernel` exposes a `simd` feature that applies explicit SIMD (via `wide::f64x4`) to
`compute_scores` — the Gaussian activation loop over all kernels. At 358d / 10k kernels it
delivers ~2.35× speedup. `sokm-multimodal` exposes the same feature flag, forwarding it to
`sokm-kernel`.

The question was whether enabling `--features simd` in `sokm-multimodal` benches or production
builds delivers the same benefit.

## Decision

Do not recommend `--features simd` for `sokm-multimodal`. The feature flag is preserved for
compatibility with `sokm-kernel` but is documented as providing no benefit and causing a
regression.

## Evidence

Criterion benchmarks on Apple Silicon (release profile):

| bench | scalar | simd | Δ |
|-------|--------|------|---|
| gestalt_tick 1000×16 | 91.5 µs | 112.8 µs | +23% |
| gestalt_tick 1000×358 | 912 µs | 1112 µs | +22% |
| recall_from_modal1 1000×358 | 209.7 µs | 207.4 µs | −1% |

## Why

The Rust compiler (LLVM backend) already applies automatic vectorisation to tight loops over
contiguous memory at release optimisation level (`-O3`). The Gaussian scoring loop in
`compute_scores` is a simple dot-product pattern — LLVM auto-vectorises it without
explicit SIMD intrinsics.

Enabling `--features simd` reorganises the compile unit (new code paths, different inlining
decisions), which disrupts LLVM's auto-vectorisation of the existing scalar hot path in
`gestalt_tick`. The regression is a codegen layout effect, not an algorithmic one — `tick`
does not call `batch_gaussian_simd` directly.

The ~2.35× speedup in `sokm-kernel` at 358d was on the kernel growth scoring loop where LLVM
did not auto-vectorise due to the more complex control flow around growth decisions. That
condition does not apply to `sokm-multimodal`'s recall path, which uses `cross_propagate_soft`
— a simpler loop shape that LLVM handles well without help.

## What was rejected

- Wiring `batch_gaussian_simd` directly into `recall_from_modal1/2`: recall already matches
  auto-vectorised scalar performance; explicit SIMD would add overhead, not remove it.
- Removing the `simd` feature entirely: breaks the `sokm-kernel` feature graph for consumers
  who want SIMD at the kernel layer while using multimodal on top.
