# sokm-kernel

[![Crates.io](https://img.shields.io/crates/v/sokm-kernel.svg)](https://crates.io/crates/sokm-kernel)
[![Docs.rs](https://docs.rs/sokm-kernel/badge.svg)](https://docs.rs/sokm-kernel)
[![License](https://img.shields.io/crates/l/sokm-kernel.svg)](LICENSE-MIT)

Kernel unit layer for SOKM — activation scoring, one-pass growth, short-term memory, and class inheritance.

Implements kernel units, activation functions, one-pass growth, STM, and class
inheritance from Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*.

Builds on [`sokm`](https://github.com/geronimo-iia/sokm-core/tree/main/crates/sokm) (link layer).

## Background

Each input vector is scored against every live kernel using a Gaussian (or compact) radial basis function. If no kernel exceeds the excitation threshold `θ_k`, a new kernel is grown at the input location. Co-activated same-class kernels strengthen their Hebbian links via `sokm`; the graph evolves tick by tick without batch training.

`KernelGraph` combines kernel storage, activation, growth, STM, and propagation into one `tick()` call — the correct Hoya check including both direct and propagated activation.

## What it does

- **Kernel units** — centroid `c`, bandwidth `σ`, class label `η`, excitation counter
- **Gaussian / compact activation** — radial basis scoring, result ∈ (0,1] [Hoya Eqs. 3.8, 3.10]
- **One-pass growth** — new kernel grown only when no existing kernel is excited [Hoya §3.4]
- **Early-exit growth check** — O(1) on familiar input, O(N·D) worst case
- **STM** — short-term memory with min-ε eviction [Hoya p. 164, Eq. 10.5]
- **Class inheritance** — unlabelled kernels inherit class from co-activated labelled neighbours [Hoya §4.3]
- **`KernelGraph`** — convenience wrapper combining all of the above into a single `tick()` call

The `Aos` prefix (e.g. `AosKernelGraph`, `AosKernelStore`) denotes
Array-of-Structs kernel storage — the only provided implementation.

Optional SIMD path via `simd` feature (`wide` crate, `f64x4`) for `compute_scores`
in `tick` — ~2.35× at 358d/10k kernels.

## Usage

```toml
[dependencies]
sokm-kernel = "0.1"

# Optional: SIMD batch scoring
sokm-kernel = { version = "0.1", features = ["simd"] }
```

```rust
use sokm::SokmConfig;
use sokm::HashEdgeStore;
use sokm_kernel::{AosKernelGraph, KernelConfig, KernelGraph};

let mut graph: AosKernelGraph = KernelGraph::new(
    HashEdgeStore::default(),
    &KernelConfig::default(),
);

let cfg_sokm   = SokmConfig::default();
let cfg_kernel = KernelConfig::default();

// First tick: novel input → kernel grown
let report = graph.tick(&[1.0, 0.0, 0.5], Some(1), 0, &cfg_sokm, &cfg_kernel, false);
assert!(report.grew);
assert_eq!(graph.kernel_count(), 1);

// Second tick: familiar input → no growth
let report = graph.tick(&[1.0, 0.0, 0.5], Some(1), 1, &cfg_sokm, &cfg_kernel, false);
assert!(!report.grew);
assert_eq!(graph.kernel_count(), 1);
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `simd`  | off | SIMD batch gaussian scoring via `wide::f64x4` (~2.35× at 358d) |
| `serde` | off | `Serialize`/`Deserialize` for all public types |

## Benchmarks

```bash
cargo bench -p sokm-kernel
# With SIMD path:
cargo bench -p sokm-kernel --features simd
```

Covers `gaussian`, `compact`, `should_grow_direct`, and `kernel_graph_tick`
at 16d and 358d across 1k–10k kernels. SIMD bench adds scalar vs `f64x4`
comparison for `compute_scores`.

## Examples

```bash
cargo run -p sokm-kernel --example category_formation
```

Demonstrates autonomous category formation using `SparseEdgeStore`: trains a graph on two labelled
clusters, then classifies novel inputs by nearest kernel.

```text
tick  0: new kernel grown, total = 1  ← novel input: no kernel excited, grow
tick  3: new kernel grown, total = 2  ← second cluster detected, grow again
                                        no further growth: all inputs recognised

Kernels after training: 2             ← exactly 2 kernels for 2 clusters

Classification:
  x=[0.15, 0.25] → predicted=Some(0) (expected 0) ✓
  x=[4.9, 5.1] → predicted=Some(1) (expected 1) ✓
  x=[0.05, 0.05] → predicted=Some(0) (expected 0) ✓
  x=[5.2, 5.3] → predicted=Some(1) (expected 1) ✓

4/4 correct                           ← generalises to unseen inputs
```

```bash
cargo build --release -p sokm-kernel --example profile_gaussian
./target/release/examples/profile_gaussian
```

Profiling harness for `should_grow_direct` — 10k kernels × 358d, 500 iterations.
Intended for use with Instruments CPU Profiler or `perf`.

## Crate map

```
sokm              ← link layer                        [sokm-core]
sokm-kernel       ← kernel units, growth, STM (this crate) [sokm-core]
sokm-multimodal   ← Gestalt K³ cross-modal memory
sokm-emotion      ← emotional state layer
sokm-memory       ← persistent episodic memory store
```

## Design notes

**Goal:** kernel lifecycle — growth, activation scoring, STM, class inheritance — decoupled from edge mechanics and from upper layers (emotion, memory).

**Why separate from `sokm`:** testable in isolation; `sokm-multimodal` owns two `KernelGraph` instances and one cross-modal edge store independently; merging would force emotion and memory to depend on link mechanics they don't need directly.

**`AosKernelStore` only:** Array-of-Structs is the sole provided kernel storage. Struct-of-Arrays was considered for SIMD over all centroids simultaneously but rejected: at typical kernel counts (< 50k) AoS wins on branch predictability and simpler grow/compact code. SIMD is applied per-activation via the `simd` feature, not across the whole store.

**`KernelGraph` takes the edge store by move:** callers choose `HashEdgeStore` for tests or `SparseEdgeStore` for production without changing `KernelGraph` code. The graph owns the store — no shared references across tick boundaries.

**Early-exit growth check:** `should_grow` short-circuits on the first excited kernel — O(1) on familiar inputs, O(N·D) worst case. This is the hottest path in production; profile before changing the exit condition.

## API

| Symbol | Description |
|--------|-------------|
| `KernelGraph::new(edges, cfg)` | Create graph; takes edge store by move |
| `KernelGraph::tick(x, label, t, sokm_cfg, kernel_cfg, dry_run)` | One learning step: activate → grow → strengthen → decay → prune → propagate. Returns `TickReport` |
| `TickReport` | `grew: bool`, `best: Option<usize>`, `activated: Vec<(usize, f64)>` |
| `KernelGraph::propagate_soft(x, cfg)` | Score all kernels then spread activation through edges |
| `KernelGraph::best_match(x, cfg)` | Return kernel index with highest propagated score |
| `AosKernelStore::push(centroid, sigma, label)` | Add kernel manually (bypasses growth rule) |
| `should_grow_direct(store, x, cfg)` | Growth check without edge access — for ECS callers |
| `compute_scores(store, x)` | Raw activation scores for all live kernels |
| `Stm::update(idx, store)` / `Stm::blend_output(x, store, alpha)` | STM: record recent activations; blend centroid mean into output |

## MSRV

Rust 1.95 (stable). No nightly required.

## Reference

Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*, Springer.
Equations 3.8, 3.10, 4.1, 4.3–4.7; pp. 40–99, 164.

## License

MIT OR Apache-2.0
