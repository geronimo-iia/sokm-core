# Decision: STM eviction = min excitation count (p.164)

## Problem

Short-term memory (STM) eviction needs a tie-breaking policy when the STM
buffer is full. LIFO (last-in-first-out) is a common default but is not what
Hoya specifies.

## Decision

STM eviction evicts the kernel with the minimum excitation count ε (Hoya p.164).

## Why

The paper is explicit. LIFO would evict the most recent input regardless of
activation history, which contradicts Hoya's intent: rarely-excited kernels
(low ε) are the least relevant and should be evicted first.

## What was rejected

LIFO: incorrect per spec. Not used, not exported. The label "LIFO" must not
appear in docs or comments — it would mislead future readers into thinking LIFO
is an option.
