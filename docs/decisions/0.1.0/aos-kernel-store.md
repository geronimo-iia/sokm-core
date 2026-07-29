# Decision: AosKernelStore only — SoaKernelStore dropped

## Problem

Array-of-Structs vs Struct-of-Arrays layout is a classic performance trade-off.
`SoaKernelStore` was prototyped to test whether vectorized memory access on
`centers` and `weights` arrays would accelerate the sq_dist hot loop.

## Decision

Ship only `AosKernelStore`. Remove `SoaKernelStore`.

## Why

Benchmark result: `SoaKernelStore` was 1.07× faster — within noise. The bottleneck
is arithmetic in the sq_dist loop (98.8% of wall time), not memory layout. The
throughput ceiling is the FPU, not the cache or memory bus.

Maintaining two store implementations adds complexity and binary size for no
measurable gain.

## What was rejected

`SoaKernelStore`: prototyped, benched, dropped. Results documented in
`docs/performance.md`.
