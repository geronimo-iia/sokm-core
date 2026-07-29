# sokm-core

Core primitives for SOKM (Self-Organizing Kernel Memory) — incremental associative learning without backprop or batch training.

[![CI](https://github.com/geronimo-iia/sokm-core/actions/workflows/ci.yml/badge.svg)](https://github.com/geronimo-iia/sokm-core/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

## Background

SOKM is a biologically-inspired learning system from Tetsuya Hoya (2005). Each input activates nearby kernel units (Gaussian radial basis functions). Co-activated units strengthen their Hebbian links; unused links decay and are pruned. Over time the system builds a sparse weighted graph that encodes which inputs tend to co-occur — with no batch pass, no backprop, no explicit loss function.

The key insight: kernel growth is gated by whether any kernel is already excited by the input. Novel inputs grow new kernels; familiar inputs reinforce existing structure. This gives continuous, incremental learning at O(1) per familiar input.

This repo contains the two lowest layers:
- **`sokm`** — Hebbian link mechanics (decay, strengthen, prune, propagate)
- **`sokm-kernel`** — kernel units, activation scoring, one-pass growth, STM, class inheritance

Upper layers (emotion, multimodal, episodic memory) are out of scope here.

## When to use

SOKM fits problems where:
- **inputs arrive continuously** and the distribution is not known in advance
- **associations matter** — which inputs co-occur, not just which class they belong to
- **no retraining budget** — the system must learn incrementally in real time

Not a good fit for: fixed dataset classification (use a neural net), density estimation (use GMM/KDE), or nearest-neighbour lookup (use HNSW/FAISS).

## Crates

| Crate | Role |
|-------|------|
| [`sokm`](crates/sokm/) | Hebbian link layer — decay, strengthen, prune, propagate |
| [`sokm-kernel`](crates/sokm-kernel/) | Kernel layer — activation, growth, STM, KernelGraph |

## Documentation

- [Algorithm reference](docs/algorithm.md) — equations, parameters, tick() loop order (Hoya 2005)
- [Invariants](docs/invariants.md) — invariants that must hold across the codebase
- [Decisions](docs/decisions/README.md) — architectural decisions and rationale

## Usage

```toml
[dependencies]
sokm = "0.1"
sokm-kernel = "0.1"
```

## MSRV

Rust 1.95 (stable). No nightly required.

## Reference

Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*, Springer.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
