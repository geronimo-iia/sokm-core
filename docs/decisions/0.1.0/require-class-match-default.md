# Decision: require_class_match defaults to true

## Problem

Cross-modal edge formation (`cross_strengthen_deltas`) can be gated on
co-occurring kernels sharing the same class label, or allowed for any
co-occurring pair.

## Decision

`require_class_match` defaults to `true`: cross-modal edge formation requires
a matching class signal.

## Why

Co-occurrence alone does not imply semantic association. The class signal
provides the semantic gating that makes cross-modal links meaningful.

Without class gating, any simultaneous activation in both modalities would
strengthen cross-modal edges — this produces spurious associations that pollute
recall.

## What was rejected

`require_class_match = false` as default: too permissive — spurious cross-modal
edges from coincidental co-activation degrade recall precision. Callers who
genuinely want class-free binding can opt in explicitly.
