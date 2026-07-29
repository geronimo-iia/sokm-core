# Decision: scale_all is pure multiply — no pruning

## Problem

Cross-modal decay (`scale_all`) multiplies edge weights. The question is whether
it should also prune edges that fall below minimum weight after scaling.

## Decision

`scale_all` multiplies weights only. No pruning.

All pruning goes through `prune_below` / `prune_inactive`.

## Why

Separation of concerns: decay and pruning are distinct operations. Mixing them
in `scale_all` would make decay observable-state-changing (removing edges) when
callers may only want to attenuate weights.

Callers control the pruning schedule explicitly by calling `prune_below` /
`prune_inactive` when appropriate.

## What was rejected

Pruning inside `scale_all`: violates single-responsibility, makes decay a
non-obvious state mutation.
