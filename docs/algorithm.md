# SOKM Algorithm Reference

Core algorithm for the Self-Organizing Kernel Machine, as implemented in `sokm` and `sokm-kernel`.
Equations reference Hoya (2005) — *Artificial Mind System: Kernel Memory Approach*, Springer.

Upper-layer algorithms (emotion, multimodal, episodic memory) are out of scope for this crate.

## 1. Kernel Activation [Eq 3.8]

Gaussian:

```
K_i(x) = exp(-‖x − c_i‖² / σ_i²)
```

`c_i` = centroid, `σ_i` = bandwidth. Result ∈ (0,1]. Equals 1.0 when `x == c_i`.

Compact approximation [Eq 3.10] — cheaper, finite support:

```
K_i(x) = (1 − ‖x − c‖² / (q · σ²))²   if ‖x − c‖² < q · σ²
        = 0.0                             otherwise
```

`q = 2.67` — Hoya's only concrete numeric constant.

Code: `activation::gaussian`, `activation::compact` in `sokm-kernel`.

## 2. Growth Rule [Hoya Step 2.1]

New kernel added only when no existing kernel is excited. A kernel is excited if:
- direct activation `K_i(x) >= θ_k`, OR
- propagated activation `spread_i >= θ_k`

`should_grow_direct` — direct check only, no edge access. For ECS callers.
`KernelGraph::tick` — full check: direct + propagated. The correct Hoya check.

## 3. Hebbian Decay [Eq 4.1]

Applied every tick before strengthen:

```
w *= exp(-ξ)
```

Exponential decay, not linear. `ξ = xi` in `SokmConfig`.

## 4. Strengthen [Eqs 4.6–4.7]

Co-activated same-class kernel pairs strengthen their link:
- New edge: `w = w_init`
- Existing: `w += δ · score_a · score_b`, clamped to `w_max`

`sokm` is class-agnostic — caller (`KernelGraph::tick`) pre-filters to same-class pairs before calling `sokm::tick`.

## 5. Prune

Two sequential phases per tick:

1. Weight threshold: remove edges where `w < min_weight`
2. Inactivity extinction: remove edges where `(tick − last_active) > p1`

## 6. Propagation

Two forms with different semantics.

### Binary [Eq 4.4] — construction path

```
I_i = 1 if K_i(x) >= θ_k, else 0
spread[j] += γ · w_ij · I_i
```

Only `θ_k`-fired kernels contribute. Uniform signal — score not scaled.
Used in: `KernelGraph::tick` dense scratch loop, `sokm::propagate`, `KernelGraph::propagate`.

### Soft / Graded [Eq 4.3] — retrieval path

```
spread[j] += γ · w_ij · K_i(x)
```

All kernels with `K_i(x) > 0` contribute proportional to score. Partial activations propagate.
Used in: `sokm::propagate_soft`, `KernelGraph::propagate_soft`.

Construction uses binary to match Hoya's growth gating — threshold is a hard gate. Retrieval uses soft to produce a graded similarity landscape even when no single kernel scores `>= θ_k`.

## 7. tick() — Full Construction Loop

Order per Hoya Step 2.1:

1. Compute direct scores: `score_i = K_i(x)` for all `i`
2. Build `fired` (score `>= θ_k`) — for binary propagation [Eq 4.4]
3. Build `direct_activated` (score `> 0.0`, with score) — for strengthen [Eqs 4.6–4.7]
4. Compute `spread` via dense scratch buffer (binary form, iterates `fired`)
5. Growth check: if no kernel has `score_i >= θ_k` AND no kernel has `spread_i >= θ_k` → grow
6. Best-match: kernel with highest direct score; increment its excitation ε
7. STM update with best-match index [p.164]
8. Filter `direct_activated` to same-class pairs → `sokm::tick` (decay + strengthen + prune)
9. Label inheritance pass [Hoya §4.3]: unlabelled kernels co-activated with a labelled kernel
   increment a coactivation counter; when counter reaches `label_inherit_threshold` the
   unlabelled kernel inherits the label. Disabled by default (`label_inherit_threshold = u32::MAX`).
10. Kernel extinction [Hoya Rule 3]: kernels inactive for more than `p1_kernel` ticks are
    marked extinct and score 0.0 on all future ticks. Disabled by default (`p1_kernel = u64::MAX`).

`fired` and `direct_activated` both derive from `scores` — gaussian computed once, not twice.

## 8. Short-Term Memory [p.164, Eq 10.5]

Capacity-bounded working memory. Eviction target: kernel with lowest excitation count ε. (Hoya describes "LIFO-like" behaviour — the mechanism is min-ε eviction.)

Blend output [Eq 10.5]:

```
o_STM[i] = λ · c_k[i] + (1 − λ) · x[i]
```

`c_k` = best-match centroid, `x` = current input.

## 9. Parameters

| Param | Symbol | Default | Config |
|---|---|---|---|
| Growth threshold | θ_k | 0.1 | `KernelConfig` |
| Initial bandwidth | σ_0 | 1.0 | `KernelConfig` |
| STM blend | λ | 0.7 | `KernelConfig` |
| Compact ratio | q | **2.67** | `KernelConfig` — Hoya's only concrete constant |
| STM capacity | N_{s,max} | 16 | `KernelConfig` |
| Label inherit threshold | — | u32::MAX (disabled) | `KernelConfig` |
| Kernel inactivity extinction | p1_kernel | u64::MAX (disabled) | `KernelConfig` |
| Decay constant | ξ | 0.01 | `SokmConfig` |
| Propagation attenuation | γ | 0.9 | `SokmConfig` |
| Edge inactivity extinction | p1 | u64::MAX (disabled) | `SokmConfig` |
| Initial link weight | w_init | 0.1 | `SokmConfig` |
| Link weight ceiling | w_max | 1.0 | `SokmConfig` |
| Strengthen increment | δ | 0.05 | `SokmConfig` |
| Weight prune floor | min_weight | 0.001 | `SokmConfig` |

All parameters except `q` require calibration against experiments — Hoya provides no concrete values for them.

## Reference

Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*, Springer.
Equations 3.8, 3.10, 4.1, 4.3–4.7, 10.5; §3.4, §4.3; pp. 40–99, 164.
