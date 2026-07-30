# sokm-core

Core primitives for SOKM (Self-Organizing Kernel Memory) — incremental associative learning without backprop or batch training.

[![CI](https://github.com/geronimo-iia/sokm-core/actions/workflows/ci.yml/badge.svg)](https://github.com/geronimo-iia/sokm-core/actions/workflows/ci.yml)
[![crates.io sokm](https://img.shields.io/crates/v/sokm.svg)](https://crates.io/crates/sokm)
[![crates.io sokm-kernel](https://img.shields.io/crates/v/sokm-kernel.svg)](https://crates.io/crates/sokm-kernel)
[![crates.io sokm-emotion](https://img.shields.io/crates/v/sokm-emotion.svg)](https://crates.io/crates/sokm-emotion)
[![sokm-multimodal crates.io](https://img.shields.io/crates/v/sokm-multimodal.svg)](https://crates.io/crates/sokm-multimodal)
[![sokm-multimodal docs.rs](https://docs.rs/sokm-multimodal/badge.svg)](https://docs.rs/sokm-multimodal)
[![docs.rs sokm](https://docs.rs/sokm/badge.svg)](https://docs.rs/sokm)
[![docs.rs sokm-kernel](https://docs.rs/sokm-kernel/badge.svg)](https://docs.rs/sokm-kernel)
[![docs.rs sokm-emotion](https://docs.rs/sokm-emotion/badge.svg)](https://docs.rs/sokm-emotion)
[![MSRV: 1.95](https://img.shields.io/badge/rustc-1.95+-blue.svg)](https://blog.rust-lang.org/2025/05/15/Rust-1.95.0.html)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

## Background

Most neural networks learn by failing repeatedly and adjusting. Gradient descent computes error,
backpropagates it through layers, nudges weights. Repeat millions of times. This works — but it is not the only way.

In 2005, Tetsuya Hoya published *Artificial Mind System: Kernel Memory Approach*. It describes a
learning algorithm that grows a network one pass at a time, with no gradient, no epochs, no loss function. I found it buried in the literature. It stayed with me — quietly, for years — until I had to build it to know if it was real. `sokm-core` is the result.

### The growth rule

When an input `x` arrives, the network scores it against every existing kernel unit — a Gaussian
radial basis function centred at a learned point:

```
K_i(x) = exp(−‖x − c_i‖² / σ_i²)
```

If nothing scores above threshold `θ_k`, the network grows: a new kernel is added with its
centroid at `x`. That is the entire growth rule. The network remembers what it has not seen
before, and ignores what it already knows.

### Links that live and die

Kernels are not isolated. Co-activated same-class pairs strengthen their link; unused links decay
and eventually disappear. The lifecycle is three rules:

1. **Decay** every tick: `w ← w × exp(−ξ)` — weight erodes whether you use it or not [Eq 4.1]
2. **Strengthen** on co-activation: `w += δ · score_a · score_b`, clamped to `w_max` [Eqs 4.6–4.7]
3. **Prune** after `p₁` inactive ticks or below `min_weight` — the link is gone

The result is a sparse weighted graph that reflects actual co-occurrence history — not a fixed
architecture decided before training.

### Short-term memory

There is also a bounded STM: a small set of recently active kernel indices. When full, the
least-excited kernel is evicted — not the most recent one. The output blends the stored centroid
with the current input: `o = λ·c_k + (1−λ)·x`. Simple, but it means the system has a notion
of working memory built in.

### Emotion

Hoya's system does not separate cognition from affect. Emotion is a layer of the same substrate.

Each kernel accumulates emotional colouring from the inputs that activate it — a 2D variable `(e₁, e₂)` that drifts toward a target valence each time the kernel fires. Persistent activation with positive valence pushes `e₁` and `e₂` up; negative valence pushes them down. The kernel remembers not just what it has seen, but how it felt about it.

A global emotion state `(E₁, E₂)` is recomputed each tick as a weighted sum over all active kernels, weighted by activation strength. `E₁` spans ecstasy to misery; `E₂` spans rage to relief.

The attentive condition is a single inequality: the system is attentive when the global state is close enough to its optimal target. It is distressed when it is not. This gates Phase 5 of the HA-GRNN training schedule — the point where emotion shapes which kernels are selected and reinforced.

`sokm-emotion` implements this layer: per-kernel variables, global state update, attentive check, and three pluggable policies controlling whether the global state accumulates without bound (`IdentityPolicy` — exact Hoya), saturates at a defined range (`ClampPolicy`), or decays toward neutral during silence (`DecayPolicy`).

## When to use

SOKM fits problems where:
- **inputs arrive continuously** and the distribution is not known in advance
- **associations matter** — which inputs co-occur, not just which class they belong to
- **no retraining budget** — the system must learn incrementally in real time

If you have a fixed dataset and a training budget, a neural net will outperform it. SOKM's edge is the stream — inputs that arrive one at a time, without a second pass.

## Crates

| Crate | Role |
|-------|------|
| [`sokm`](crates/sokm/) | Hebbian link layer — decay, strengthen, prune, propagate |
| [`sokm-kernel`](crates/sokm-kernel/) | Kernel layer — activation, growth, STM, KernelGraph |
| [`sokm-emotion`](crates/sokm-emotion/) | Emotion layer — per-kernel vars, global (E₁,E₂) state, attentive condition |
| [`sokm-multimodal`](crates/sokm-multimodal) | Gestalt K³ cross-modal memory — two SOKM modalities coupled via directed bipartite cross-edge store |

Upper layers — episodic memory — are built on top of these primitives and live elsewhere.

## Documentation

- [Algorithm reference](docs/algorithm.md) — equations, parameters, tick() loop order (Hoya 2005)
- [Invariants](docs/invariants.md) — invariants that must hold across the codebase
- [Decisions](docs/decisions/README.md) — architectural decisions and rationale

## Usage

```toml
[dependencies]
sokm = "0.2"
sokm-kernel = "0.2"
sokm-emotion = "0.2"        # optional — emotion layer
sokm-multimodal = "0.3"     # optional — cross-modal Gestalt K³ memory
```

**Kernel layer** — grow and activate kernels:

```rust
use sokm::{HashEdgeStore, SokmConfig};
use sokm_kernel::{DefaultKernelGraph, KernelConfig, KernelGraph};

let mut graph: DefaultKernelGraph = KernelGraph::new(HashEdgeStore::default(), &KernelConfig::default());
let cfg_sokm   = SokmConfig::default();
let cfg_kernel = KernelConfig::default();

let r = graph.tick(&[1.0, 0.0], Some(1), 0, &cfg_sokm, &cfg_kernel, false);
assert!(r.grew); // novel input — kernel grown
```

See also: [category_formation](crates/sokm-kernel/examples/category_formation.rs) — autonomous category formation on two labelled clusters.

**Emotion layer** — add per-kernel emotional colouring:

```rust
use sokm::{DecayMode, HashEdgeStore, SokmConfig};
use sokm_kernel::KernelConfig;
use sokm_emotion::{DefaultEmotionalGraph, EmotionConfig, EmotionalGraphConfig};

let mut graph = DefaultEmotionalGraph::new(
    HashEdgeStore::default(),
    EmotionalGraphConfig { sokm: SokmConfig::default(), kernel: KernelConfig::default(), emotion: EmotionConfig::default() },
);

// e_target = [E₁_target, E₂_target] — emotional valence of this input
let r = graph.tick(&[1.0, 0.0], Some(1), [0.5, 0.0], 0, DecayMode::Apply);
println!("attentive={} E1={:.3} E2={:.3}", r.attentive, r.global.e1, r.global.e2);
```

See also: [emotional_learning](crates/sokm-emotion/examples/emotional_learning.rs) — two clusters with opposite valences, [policy_comparison](crates/sokm-emotion/examples/policy_comparison.rs) — `IdentityPolicy` vs `ClampPolicy` vs `DecayPolicy`.

## MSRV

Rust 1.95 (stable). No nightly required.

## Reference

Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*, Springer.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
