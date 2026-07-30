# Decisions

Architectural decisions and their rationale, grouped by release.

This file covers decisions intrinsic to the `sokm`, `sokm-kernel`, and `sokm-emotion` crates.
Decisions for upper layers (memory, store, encoder, MCP) are out of scope for this repo.

## v0.1.0 — 2026-07-22

### Numerics

| Decision | Summary |
| -------- | ------- |
| [float-type-f64](0.1.0/float-type-f64.md) | f64 everywhere — propagation chains accumulate rounding error; bottleneck is arithmetic not memory |

### Data Structures

| Decision | Summary |
| -------- | ------- |
| [default-kernel-store](0.1.0/default-kernel-store.md) | DefaultKernelStore only — SoaKernelStore was 1.07× faster, within noise; bottleneck is FPU not layout |

### Algorithm Fidelity

| Decision | Summary |
| -------- | ------- |
| [growth-rule](0.1.0/growth-rule.md) | Full Step 2.1 growth check in tick; should_grow_direct exported but never used inside tick |
| [propagation-summation](0.1.0/propagation-summation.md) | Summation Eq 4.3 not max; binary form for construction, soft form for recall |
| [stm-eviction](0.1.0/stm-eviction.md) | Min excitation count ε per p.164 — not LIFO |

### Architecture

| Decision | Summary |
| -------- | ------- |
| [link-lifecycle](0.1.0/link-lifecycle.md) | decay/prune/scale_all are three separate operations — exponential decay, two-phase prune, scale_all pure multiply |
| [class-semantics](0.1.0/class-semantics.md) | sokm class-agnostic; class filtering in KernelGraph::tick; require_class_match = true by default |

## v0.2.0 — 2026-07-30

### Architecture

| Decision | Summary |
| -------- | ------- |
| [configured-emotional-tick-report](0.2.0/configured-emotional-tick-report.md) | `DefaultEmotionalGraph` stores config internally; `tick` returns `salience_scores` in report — empty when alpha==0.0, zero allocation |
