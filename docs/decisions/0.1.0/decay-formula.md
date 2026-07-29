# Decision: Decay is w *= exp(-ξ) — not linear subtraction

## Problem

Weight decay can be implemented as linear subtraction (`w -= ξ`) or exponential
decay (`w *= exp(-ξ)`). Linear is simpler to reason about; exponential matches
the paper.

## Decision

`w *= exp(-ξ)` [Hoya Eq 4.1]. No linear subtraction.

## Why

Hoya Eq 4.1 is explicit. Exponential decay has the property that weights
asymptotically approach zero rather than going negative, which is semantically
correct for connection strength.

Linear subtraction could produce negative weights and requires clamping logic;
exponential decay is always positive.

## What was rejected

Linear subtraction: wrong per spec, would require negative-weight guards.
`CrossEdgeStore::scale_all` uses the same exponential form for cross-modal decay.
