# Decision: Link lifecycle — decay, prune, and scale_all are three distinct operations

## Problem

Hebbian link maintenance involves three operations that could be collapsed:
1. Weight decay — attenuate all weights each tick
2. Pruning — remove edges below threshold or inactive kernels
3. Cross-modal scale — attenuate cross-modal edge weights

Early designs mixed these: pruning inside decay, or pruning inside scale_all.

## Decisions

**Decay formula:** `w *= exp(-ξ)` [Hoya Eq 4.1]. Not linear subtraction.

**prune() is two-phase:** weight threshold first (remove edges below `min_weight`),
then p1 inactivity extinction (remove kernels below inactivity threshold).

**scale_all is pure multiply:** no pruning. All pruning goes through `prune_below`
/ `prune_inactive` explicitly.

## Why

**Decay formula:** Hoya Eq 4.1 is explicit. Exponential decay is always positive —
weights asymptotically approach zero rather than going negative. Linear subtraction
could produce negative weights and requires clamping logic.

**prune() phase order:** weight cleanup must happen first so extinction operates on
a fully pruned graph. Doing extinction first could leave orphaned edges for kernels
that should have been weight-pruned — the reverse order is semantically wrong.

**scale_all separation:** decay and pruning are distinct operations. Mixing them
makes decay a non-obvious state mutation. Callers control the pruning schedule
explicitly.

## What was rejected

- Linear subtraction for decay: wrong per spec, requires negative-weight guards.
- Pruning inside scale_all: violates single-responsibility.
- Single-pass combined prune: order-dependent bugs.
