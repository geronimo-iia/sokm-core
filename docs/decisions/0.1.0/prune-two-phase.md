# Decision: prune() is two-phase — weight threshold then p1 inactivity extinction

## Problem

`prune()` combines two distinct operations. A single-pass implementation
ordering matters: doing extinction before weight pruning could remove a kernel
whose links hadn't been weight-pruned yet.

## Decision

`prune()` is two-phase:
1. Weight threshold: remove edges below minimum weight
2. p1 inactivity extinction: remove kernels below inactivity threshold

## Why

Phase ordering ensures weight cleanup happens first, so extinction operates
on a fully pruned graph. Doing extinction first could leave orphaned edges
for kernels that should have been weight-pruned — the reverse order is
semantically wrong.

## What was rejected

Single-pass combined prune: order-dependent bugs, rejected.
