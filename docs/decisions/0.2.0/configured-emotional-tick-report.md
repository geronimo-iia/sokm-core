# Decision: `salience_scores` in `EmotionalTickReport`; config stored in `DefaultEmotionalGraph`

## Problem

`salience` is a pure function not called by `tick`. Callers wanting per-kernel
salience scores after each tick must call it manually with the right `alpha`,
threading config through every call site. `EmotionalTickReport` cannot carry
salience scores because `tick` has no `alpha` parameter.

## Decision

- `salience_scores: Vec<f64>` added directly to `EmotionalTickReport`.
  Empty (`len == 0`) when `alpha == 0.0` — zero allocation for the common case.
- `DefaultEmotionalGraph` (`EmotionalKernelGraph<HashEdgeStore>`) stores
  `EmotionalGraphConfig` internally. `tick` uses the stored config and computes
  salience post-step-3 without any extra parameter at call sites.
- `salience_scores` length = `kernel_count()` — one entry per kernel.
  Activated kernel's salience reflects post-update vars; all others pre-tick vars.
- `EmotionalKernelGraph` itself remains config-free for callers that manage
  config externally; `DefaultEmotionalGraph` is the convenience wrapper.

## Why

Callers need salience scores without manually threading `alpha` through every
tick call site. `salience_scores` is a point-in-time snapshot of post-tick
salience for the current stimulus — not a cache.

## What was rejected

- Adding `salience_scores` to `EmotionalTickReport` with `alpha` as a `tick`
  parameter: would break all existing tick call sites and couples the core tick
  signature to a diagnostic concern.
- A separate `ConfiguredEmotionalTickReport` wrapper type: extra indirection with
  no benefit once config storage moved into `DefaultEmotionalGraph`.
- Extending `EmotionalKernelGraph` (inheritance-style): composition is safer,
  avoids interior mutability concerns.
