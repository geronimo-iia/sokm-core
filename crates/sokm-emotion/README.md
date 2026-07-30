# sokm-emotion

[![Crates.io](https://img.shields.io/crates/v/sokm-emotion.svg)](https://crates.io/crates/sokm-emotion)
[![Docs.rs](https://docs.rs/sokm-emotion/badge.svg)](https://docs.rs/sokm-emotion)
[![License](https://img.shields.io/crates/l/sokm-emotion.svg)](LICENSE-MIT)

Per-kernel emotional state variables and 2D global emotion tracking for SOKM (Self-Organizing Kernel Memory).

Implements Hoya's emotion equations from Tetsuya Hoya (2005),
*Artificial Mind System: Kernel Memory Approach*, Eqs. 10.6, 10.7, 10.8; pp. 212–221.

Builds on [`sokm-kernel`](https://github.com/geronimo-iia/sokm-core/tree/main/crates/sokm-kernel) (kernel layer).

## Background

Each kernel accumulates emotional colouring from the inputs that activated it. A 2D global
emotion state `(E₁, E₂)` is recomputed each tick as a weighted sum over per-kernel variables,
weighted by activation strength. An attentive condition gates whether the system is in a
focused or distressed state relative to an optimal target.

`DefaultEmotionalGraph` wraps `KernelGraph` and adds emotion tracking into a single `tick()` call —
the correct Hoya update order (kernel tick → per-kernel update → global state → attentive check).

## What it does

- **Per-kernel emotion variables** `e_i^j` — each kernel accumulates emotional colouring from
  the inputs that activated it [Hoya Eq. 10.8]
- **Global 2D emotion state** `(E₁, E₂)` — weighted sum of per-kernel variables, weighted by
  activation strength [Hoya Eq. 10.6]
  - `E₁`: ecstasy(+3) ↔ misery(−3)
  - `E₂`: rage(+2) ↔ relief(−2)
- **Attentive condition** — gates whether the system is focused or distressed [Hoya Eq. 10.7]
- **Salience scoring** — per-kernel multiplier for emotion-weighted recall (enabled via `alpha > 0`)
- **Pluggable `GlobalEmotionPolicy`** — `IdentityPolicy` (exact Hoya), `ClampPolicy`, `DecayPolicy`

## Usage

```toml
[dependencies]
sokm-emotion = "0.2"
```

```rust
use sokm::{DecayMode, HashEdgeStore, SokmConfig};
use sokm_kernel::KernelConfig;
use sokm_emotion::{DefaultEmotionalGraph, EmotionalGraphConfig, EmotionConfig};

let cfg = EmotionalGraphConfig {
    sokm: SokmConfig::default(),
    kernel: KernelConfig::default(),
    emotion: EmotionConfig::default(),
};
let mut graph = DefaultEmotionalGraph::new(HashEdgeStore::default(), cfg);

// Train: present inputs with an emotional valence target.
// e_target = [E₁_target, E₂_target] — [0.0, 0.0] = neutral.
for t in 0..10u64 {
    let report = graph.tick(
        &[1.0, 0.0],   // input
        Some(1),       // class label
        [0.5, 0.0],    // emotional valence of this input
        t,
        DecayMode::Apply,
    );
    println!("tick {t}: attentive={}, global=({:.3}, {:.3})",
        report.attentive, report.global.e1, report.global.e2);
}
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | off | `Serialize`/`Deserialize` for all public types |
| `simd`  | off | SIMD scoring via `sokm-kernel/simd` |

## Policy types

| Policy | Behaviour | Source |
|--------|-----------|--------|
| `IdentityPolicy` | No bounding, no decay — exact Hoya base equation | [DIRECT] |
| `ClampPolicy` | Clamp E₁ ∈ [−3,3], E₂ ∈ [−2,2] after each update | [INFERRED] |
| `DecayPolicy` | Pre-decay toward zero each tick, then clamp | [INFERRED] |

Default: `IdentityPolicy`. Pass a custom policy via `EmotionalKernelGraph::with_store`.

## Examples

```bash
cargo run -p sokm-emotion --example emotional_learning
```

Trains two concept clusters with opposite emotional valences, then probes
global emotion state and the attentive condition at inference time.

```text
=== Training ===
  tick 0: new kernel — total 1
  tick 3: new kernel — total 2

After training: 2 kernels
Per-kernel emotion vars:
  kernel[0]: e1=0.998  e2=0.662    ← cluster A accumulated positive charge
  kernel[1]: e1=-0.998  e2=-0.662  ← cluster B accumulated negative charge
Global state after training: E1=-0.030  E2=-0.019

=== Inference ===
  cluster A (positive): E1=0.769  E2=0.510  attentive=false  ← positive activation shifts global
  cluster B (negative): E1=-0.030  E2=-0.019  attentive=true  ← near neutral → attentive
```

```bash
cargo run -p sokm-emotion --example policy_comparison
```

Compares `IdentityPolicy`, `ClampPolicy`, and `DecayPolicy` on the same training
stream. Phase 1 drives all three toward positive saturation. Phase 2 uses an
out-of-distribution input (gaussian ≈ 0) to show which policies recover.

```text
=== Phase 1: Training (positive valence) ===
tick       Identity (E1, E2)          Clamp (E1, E2)          Decay (E1, E2)
------------------------------------------------------------------------------
   0  (+0.000, +0.000)          (+0.000, +0.000)          (+0.000, +0.000)
   1  (+1.526, +1.017)          (+1.526, +1.017)          (+1.526, +1.017)
   2  (+3.487, +2.325)          (+3.000, +2.000)          (+2.724, +1.816)  ← Clamp hits ceiling; Decay pre-multiplied so still under
   3  (+5.767, +3.845)          (+3.000, +2.000)          (+3.000, +2.000)
   ...
   9  (+22.270, +14.847)          (+3.000, +2.000)          (+3.000, +2.000)  ← Identity unbound

Attentive after training (theta_e=1.5, optimal=(0,0)):
  Identity : false
  Clamp    : false
  Decay    : false

=== Phase 2: Silence (OOD input, gaussian ≈ 0) ===
tick       Identity (E1, E2)          Clamp (E1, E2)          Decay (E1, E2)
------------------------------------------------------------------------------
  10  (+22.270, +14.847)          (+3.000, +2.000)          (+1.500, +1.000)  ← Decay halves (×0.5)
  11  (+22.270, +14.847)          (+3.000, +2.000)          (+0.750, +0.500)
  12  (+22.270, +14.847)          (+3.000, +2.000)          (+0.375, +0.250)
  ...
  17  (+22.270, +14.847)          (+3.000, +2.000)          (+0.012, +0.008)

Attentive after silence:
  Identity : false   ← no recovery mechanism
  Clamp    : false   ← frozen at boundary, no decay_factor
  Decay    : true    ← recovered to within theta_e of optimal
```

## Crate map

```
sokm              ← link layer                              [sokm-core]
sokm-kernel       ← kernel units, growth, STM               [sokm-core]
sokm-emotion      ← per-kernel emotion variables, global state (this crate) [sokm-core]
sokm-multimodal   ← Gestalt K³ cross-modal memory           [sokm-core]
```

## API

| Symbol | Description |
|--------|-------------|
| `DefaultEmotionalGraph::new(edges, cfg)` | Create graph; takes edge store by move |
| `EmotionalKernelGraph::with_store(edges, kernels, policy, cfg)` | Create with custom kernel store and policy |
| `EmotionalKernelGraph::tick(x, class, e_target, t, decay)` | One learning step: kernel tick → emotion update → global state → attentive check. Returns `EmotionalTickReport` |
| `EmotionalTickReport` | `kernel: KernelTickReport`, `global: EmotionState`, `attentive: bool`, `salience_scores: Vec<f64>` |
| `EmotionalKernelGraph::global_state()` | Current `(E₁, E₂)` global emotion state |
| `EmotionalKernelGraph::emotion_vars(i)` | Per-kernel emotion variables `[e1, e2]` for kernel `i` |
| `EmotionalKernelGraph::is_attentive()` | Whether attentive condition holds |
| `EmotionalKernelGraph::propagate_soft(x)` | Score all kernels then spread activation through edges |
| `EmotionalKernelGraph::compact()` | Merge pending edges into CSR; returns pruned count |
| `salience(emotions, activated, alpha)` | Compute per-kernel salience scores |
| `EmotionConfig` | `lambda_e`, `theta_e`, `optimal`, `alpha` |
| `EmotionalGraphConfig` | Bundles `sokm`, `kernel`, `emotion` configs |

## Design notes

**Goal:** add emotional colouring to SOKM without touching kernel math. Emotion is a layer on
top of `sokm-kernel`, not a modification of it.

**`EmotionStore` grows in lockstep with `KernelGraph`:** `push()` is called exactly when
`report.grew` is true. The invariant `emotions.len() == graph.kernel_count()` is checked via
`debug_assert` after every tick. Violating it silently corrupts salience scores.

**`GlobalEmotionPolicy` is pluggable:** `IdentityPolicy` is the exact Hoya equation with no
bounding — the faithful default. `ClampPolicy` and `DecayPolicy` are pragmatic extensions for
systems where unbounded accumulation is undesirable. New policies implement one method (`apply`)
and compose without touching tick logic.

**Global state is recomputed each tick, not stored per episode:** `(E₁, E₂)` reflects the
current activation pattern. Historical emotional context lives in per-kernel variables `e_i`,
not in the global state.

## MSRV

Rust 1.95 (stable). No nightly required.

## Reference

Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*, Springer.
Equations 10.6, 10.7, 10.8; pp. 212–221, 254–257.

## License

MIT OR Apache-2.0
