# Decision: Class semantics — agnostic link layer, explicit gating in KernelGraph

## Problem

Class constraints appear in two places:
1. Same-class strengthening — should `sokm` (link layer) or `sokm-kernel` filter pairs?
2. Cross-modal edge formation — should co-occurrence alone be sufficient, or is a matching class signal required?

## Decisions

**sokm is class-agnostic.** `KernelGraph::tick` filters same-class pairs before
calling `sokm::strengthen`. `sokm` operates on raw indices and weights only.

**require_class_match defaults to true.** Cross-modal edge formation requires a
matching class signal. Callers who want class-free binding must opt in explicitly.

## Why

**Class-agnostic link layer:** `sokm` has no concept of classes. Embedding class
logic there would break the crate's responsibility boundary and force all callers
to carry class metadata. Class filtering belongs in `sokm-kernel` where kernel
class labels live.

**require_class_match = true:** Co-occurrence alone does not imply semantic
association. Without class gating, any simultaneous activation in both modalities
strengthens cross-modal edges — producing spurious associations that pollute recall.
The class signal provides the semantic gating that makes links meaningful.

## What was rejected

- Class filtering in `sokm`: layer violation, forces class metadata into a
  class-agnostic abstraction.
- `require_class_match = false` as default: too permissive — spurious cross-modal
  edges from coincidental co-activation degrade recall precision.
