# Decisions

Architectural decisions and their rationale, grouped by release.

This file covers decisions that are intrinsic to the `sokm` and `sokm-kernel` crates.
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
| [decay-formula](0.1.0/decay-formula.md) | w *= exp(-ξ) per Eq 4.1 — not linear subtraction |
| [prune-two-phase](0.1.0/prune-two-phase.md) | prune() is weight threshold first, then p1 inactivity extinction |

### Architecture

| Decision | Summary |
| -------- | ------- |
| [class-agnostic-link-layer](0.1.0/class-agnostic-link-layer.md) | sokm has no class concept; class filtering in KernelGraph::tick before strengthen |
| [scale-all-pure-multiply](0.1.0/scale-all-pure-multiply.md) | scale_all multiplies only; pruning is always explicit via prune_below / prune_inactive |
| [require-class-match-default](0.1.0/require-class-match-default.md) | require_class_match = true by default; co-occurrence alone is not association |
| [shared-params-modalities](0.1.0/shared-params-modalities.md) | Both K¹ and K² share SokmConfig + KernelConfig per Hoya's symmetric treatment |
| [flush-split](0.1.0/flush-split.md) | compact/snapshot/flush split — snapshot(&self) enables periodic persistence without forced compaction |
