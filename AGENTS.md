# sokm-core — Agent Context

Core primitives for SOKM (Self-Organizing Kernel Memory).
Reference: Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*.

**Before changing any algorithm or API:** read `docs/algorithm.md` (equations, tick loop order)
and `docs/invariants.md` (invariants that must hold across all crates).
This file is a quick-reference — those two are authoritative.

## Workspace layout

```
crates/
  sokm/           # link layer — decay, strengthen, prune, propagate
  sokm-kernel/    # kernel layer — activation, growth, STM, KernelGraph
  sokm-emotion/   # emotion layer — per-kernel vars, global state, policy
docs/
  algorithm.md    # Hoya equations, tick loop steps — READ FIRST
  invariants.md   # cross-crate invariants — READ FIRST
  decisions/      # architectural decisions by version
.claude/
  plans/          # implementation plans (YYYY-MM-DD-<feature-name>.md)
```

## Crate layers

`sokm` is the leaf. Changing a public API at layer N breaks all crates above it —
run `cargo build --workspace` after any API change.

```
Layer 0   sokm          — Hebbian link mechanics; no kernel or class knowledge
Layer 1   sokm-kernel   — kernel units, activation, growth, STM, KernelGraph
Layer 2   sokm-emotion  — per-kernel emotion vars, global state, policy
```

Upper layers (`sokm-multimodal`, `sokm-memory`, …) are out of scope for this repo.

## Crate responsibilities

- **`sokm`** — pure Hebbian edge mechanics. Generic over `K: Hash + Eq + Copy`. No knowledge of
  kernel types, class labels, or activation scoring.
- **`sokm-kernel`** — kernel lifecycle: growth, activation scoring, STM, class inheritance.
  `KernelGraph` is the stateful wrapper combining all steps into one `tick()` call.
  Free functions (`compute_scores`, `best_match`, `should_grow_direct`) are pure and testable
  in isolation.
- **`sokm-emotion`** — per-kernel emotion variables and global 2D mood state. Wraps `KernelGraph`
  without modifying it. `EmotionalKernelGraph` fields are `pub(crate)` — use accessors.
  `serde` feature required for snapshot use.

## Design invariants — do not change without understanding these

Full rationale: `docs/invariants.md` and `docs/decisions/`.

- **`SparseEdgeStore` CSR+pending exclusivity**: a key is never non-zero in both CSR and pending
  simultaneously. `apply_increments` routes to CSR if the key exists in `csr_index`, pending
  otherwise. `compact()` merges pending into CSR and clears it. `set_weight` clears pending for
  the key. Guarded by `debug_assert` in `get_weight`.
- **Growth rule**: excited = direct OR propagated >= `theta_k`. Growth fires only when NO kernel
  is excited. `should_grow_direct` checks direct activation only (for ECS callers); `KernelGraph::tick`
  uses the full check [Hoya Step 2.1].
- **Propagation form**: summation `K_j += γ·w_ij·K_i(x)` [Eq 4.3], not max. `tick` uses binary
  gate [Eq 4.4]; `propagate_soft` is the graded form for retrieval.
- **`p1` boundary**: `current_tick.saturating_sub(last_active) > p1` — edge survives the p1-th
  inactive tick. `>` not `>=`.
- **Decay**: `w *= exp(-ξ)` [Eq 4.1]. Not linear subtraction.
- **`KernelGraph::tick` class filter**: `sokm::strengthen` is class-agnostic. `tick` pre-filters
  to same-class pairs before calling it.
- **STM eviction**: min excitation count ε [Hoya p.164]. Not LIFO.
- **`blend_output` centroid dimension**: centroid length must equal `x.len()`. Guarded by
  `assert_eq!` in `blend_output`.
- **`compute_scores` finite inputs**: NaN in `x` propagates silently through Gaussian scoring.
  Guarded by `debug_assert!` in `compute_scores`.
- **`EmotionStore` length == `kernel_count()`**: `emotions.push` is called exactly when
  `report.grew` is true. Mismatch silently corrupts per-kernel emotion reads and salience.
  Guarded by `debug_assert_eq!` at end of `EmotionalKernelGraph::tick`.
- **`KernelTickReport::scores` reuse**: `EmotionalKernelGraph::tick` must use `report.scores`
  for `update_global_emotion` — never recompute gaussian scores on the same input.

## Hoya equation quick-reference

Full derivations: `docs/algorithm.md`.

| Eq | Formula | Code |
|----|---------|------|
| 3.8 | `K_i(x) = exp(-‖x−c‖²/σ²)` | `activation::gaussian` |
| 3.10 | Compact approximation, q=2.67 | `activation::compact` |
| 4.1 | `w *= exp(-ξ)` | `sokm::decay` |
| 4.3 | `K_j += γ·w_ij·K_i(x)` | `propagate_soft` (retrieval) |
| 4.4 | Binary gate: `I_i = 1 if K_i(x) >= θ_k` | `propagate` (construction) |
| 4.6–4.7 | Strengthen piecewise, `w_init`/`w_max` | `sokm::strengthen` |
| 10.5 | `o_STM = λ·c_k + (1-λ)·x` | `Stm::blend_output` |
| p.164 | STM eviction = min ε | `Stm::update` |
| 10.6 | `E_i(n+1) = Σ_j e_i^j · K_j(x)` | `update_global_emotion` |
| 10.7 | `Σ|E_i − E_i*| ≤ θ_E` | `is_attentive` |
| 10.8 | `e_i^j += λ_e·(e_target − e_i^j)` | `update_kernel_emotion_var` |

## Toolchain

- Rust 1.95, edition 2024
- `rustfmt.toml` max_width=100
- `clippy.toml` `avoid-breaking-exported-api = false`

## Commands

```bash
cargo test --workspace                             # full test suite
cargo test -p sokm                                 # link layer only
cargo test -p sokm-kernel                          # kernel layer only
cargo test -p sokm-emotion                         # emotion layer only
cargo test --workspace --features sokm-kernel/simd # with SIMD scoring path
cargo fmt --all -- --check                         # must pass clean
cargo clippy --all-targets -- -D warnings          # must pass clean
cargo doc --workspace --no-deps                    # rustdoc, zero warnings required
cargo bench --workspace --no-run                   # verify benches compile
cargo bench -p sokm-kernel --features simd         # SIMD scoring bench
cargo publish --dry-run -p sokm                    # pre-release gate
```

## Commit convention

Conventional commits, single line: `type(scope): short description`
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `bench`, `ci`, `chore`, `perf`
Scopes: `sokm`, `kernel`, `emotion`, `sparse`, `stm`, `graph`, `growth`, `query`, `ops`, `workspace`
One commit per plan task.
