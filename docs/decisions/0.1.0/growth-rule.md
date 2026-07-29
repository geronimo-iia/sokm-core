# Decision: Growth rule — excited = direct OR propagated ≥ θ_k

## Problem

Hoya Step 2.1 specifies that the growth check includes propagated activations,
not just direct activation. An implementation using only direct activation
(`should_grow_direct`) inside `tick` would silently deviate from the spec.

## Decision

`KernelGraph::tick` uses the full propagation-gated growth check (Step 2.1):
growth fires only if NO kernel is excited, where excited = direct OR propagated
activation ≥ θ_k.

`should_grow_direct` exists as a separate exported helper for callers (e.g. SAA,
Bevy ECS) that only have direct activations available. It is **never** called
inside `tick`.

## Why

Using `should_grow_direct` inside `tick` is a spec deviation: it ignores
propagated kernels, allowing spurious growth when associated kernels are already
active. This was confirmed as a bug during development.

## What was rejected

`should_grow_direct` inside `tick`: verified bug, removed.
