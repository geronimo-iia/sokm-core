# Decision: Float type — f64 everywhere

## Problem

Choosing `f32` would halve memory bandwidth on kernel coordinate and weight
storage and is common in ML codebases. SOKM's activation functions
(`exp(-‖x−c‖²/σ²)`) accumulate many multiply-add chains; rounding error
compounds across propagation steps.

## Decision

`f64` everywhere. No `f32`.

## Why

Hoya's reference implementation uses double precision. SOKM propagation chains
(`K_j += γ·w_ij·K_i(x)`) accumulate many floating-point operations; f32 rounding
error caused reproducibility failures in early prototyping. The bottleneck is
arithmetic (sq_dist loop 98.8%) not memory layout — saving bandwidth via f32
would not move the needle on throughput.

## What was rejected

`f32` storage with f64 computation: adds conversion overhead, complicates the
type system, and still accumulates f32 rounding in stored weights.
