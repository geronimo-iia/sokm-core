# Decision: sokm crate is class-agnostic; class constraint in KernelGraph

## Problem

Class-constrained strengthening (same-class pairs only) is a SOKM property.
The question is at which layer to enforce it: in `sokm` (the link layer) or
in `sokm-kernel` (the kernel layer).

## Decision

`sokm` (link layer) is fully class-agnostic. `KernelGraph::tick` filters
same-class pairs before calling `sokm::strengthen`.

## Why

`sokm` operates on raw indices and weights — it has no concept of classes.
Embedding class logic there would break the crate's responsibility boundary
and force all callers to carry class metadata.

Class filtering belongs in `sokm-kernel` where kernel class labels live.
`sokm::strengthen` receives only pre-filtered pairs.

## What was rejected

Class filtering in `sokm`: rejected — layer violation, forces class metadata
into a class-agnostic abstraction.
