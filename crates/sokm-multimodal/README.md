# sokm-multimodal

[![Crates.io](https://img.shields.io/crates/v/sokm-multimodal.svg)](https://crates.io/crates/sokm-multimodal)
[![Docs.rs](https://docs.rs/sokm-multimodal/badge.svg)](https://docs.rs/sokm-multimodal)
[![License](https://img.shields.io/crates/l/sokm-multimodal.svg)](LICENSE-MIT)

Gestalt K³ cross-modal associative memory for SOKM (Self-Organizing Kernel Memory).

Implements Hoya's cross-modal extension from Tetsuya Hoya (2005),
*Artificial Mind System: Kernel Memory Approach*, Eqs. 4.1, 4.3, 4.7 \[INFERRED cross-modal extension\].

Builds on [`sokm-kernel`](https://github.com/geronimo-iia/sokm-core/tree/main/crates/sokm-kernel) (kernel layer).

## Background

Two independent SOKM modalities — each a full `KernelGraph` — coupled by a directed bipartite
cross-edge store. Cross-modal Hebbian learning: when kernels from both modalities co-activate
with matching class labels, the edge between them strengthens. Recall works in both directions:
given a modal1 cue, recover modal2 activations; given a modal2 cue, recover modal1 activations.

`DefaultGestaltGraph` wraps two `KernelGraph` instances and a `CrossEdgeStore` into a single
`tick()` call — both modalities tick, then cross edges strengthen, decay, and prune.

## What it does

- **Two coupled modalities** — each a full SOKM `KernelGraph` with independent kernel growth
- **Cross-modal Hebbian edges** — strengthen on co-activation, decay each tick (`xi`), prune by weight (`min_weight`) or inactivity (`p1`)
- **Bidirectional recall** — `recall_from_modal1(x1)` → modal2 scores; `recall_from_modal2(x2)` → modal1 scores
- **Pluggable `CrossStore`** — `CrossEdgeStore` (HashMap-backed) is the default; implement `CrossStore` for custom backends
- **`GestaltConfig`** — single config bundles `SokmConfig`, `KernelConfig`, `CrossSokmConfig`

## Usage

```toml
[dependencies]
sokm-multimodal = "0.3"
```

```rust
use sokm::{DecayMode, HashEdgeStore};
use sokm_multimodal::{DefaultGestaltGraph, GestaltConfig, GestaltKernelGraph};

let cfg = GestaltConfig::default();
let mut g = DefaultGestaltGraph::new(
    HashEdgeStore::new(),
    HashEdgeStore::new(),
    &cfg,
);

// Train: pair modal1 input [1,0] with modal2 input [0,1], class 1
for t in 0..10u64 {
    let report = g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
    println!("tick {t}: cross_strengthened={}", report.cross_strengthened);
}

// Recall: given a modal1 cue, recover modal2 activations
let results = g.recall_from_modal1(&[1.0, 0.0], &cfg);
for (kernel_idx, score) in &results {
    println!("modal2 kernel[{kernel_idx}] score={score:.3}");
}
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | off | `Serialize`/`Deserialize` for all public types |
| `simd`  | off | SIMD scoring via `sokm-kernel/simd` |

## CrossSokmConfig parameters

| Field | Default | Description |
|-------|---------|-------------|
| `gamma` | 0.9 | Propagation attenuation factor γ ∈ (0,1] |
| `delta` | 0.05 | Hebbian increment δ per co-activation |
| `w_init` | 0.1 | Initial weight on first strengthen of a new edge |
| `w_max` | 1.0 | Maximum edge weight |
| `min_weight` | 0.001 | Edges below this are pruned each tick |
| `xi` | 0.01 | Decay factor ξ per tick — weight × exp(−ξ) |
| `p1` | u64::MAX | Inactivity extinction period in ticks (default: disabled) |
| `require_class_match` | true | Only same-class co-activations strengthen; `None`-labelled kernels never strengthen |

## Crate map

```
sokm              ← link layer                              [sokm-core]
sokm-kernel       ← kernel units, growth, STM               [sokm-core]
sokm-emotion      ← per-kernel emotion variables, global state [sokm-core]
sokm-multimodal   ← Gestalt K³ cross-modal memory (this crate) [sokm-core]
```

## API

| Symbol | Description |
|--------|-------------|
| `DefaultGestaltGraph::new(edges1, edges2, &cfg)` | Create graph; takes both edge stores by move |
| `GestaltKernelGraph::with_stores(e1, k1, e2, k2, &cfg)` | Create from pre-populated kernel and edge stores |
| `GestaltKernelGraph::tick(x1, x2, class, t, &cfg, decay)` | One learning step: both modalities tick, cross edges strengthen/decay/prune. Returns `GestaltTickReport` |
| `GestaltTickReport` | `modal1: KernelTickReport`, `modal2: KernelTickReport`, `cross_strengthened: usize`, `cross_pruned: usize` |
| `GestaltKernelGraph::recall_from_modal1(x1, &cfg)` | Score all modal2 kernels from a modal1 cue via cross edges |
| `GestaltKernelGraph::recall_from_modal2(x2, &cfg)` | Score all modal1 kernels from a modal2 cue (O(E) sources scan) |
| `GestaltKernelGraph::compact()` | Prune extinct kernels in both modalities; reindex cross edges. Returns `(pruned1, pruned2)` |
| `GestaltKernelGraph::cross_edge_count()` | Current number of active cross-modal edges |
| `GestaltConfig` | Bundles `sokm: SokmConfig`, `kernel: KernelConfig`, `cross: CrossSokmConfig` |
| `CrossSokmConfig` | Cross-edge Hebbian parameters — see table above |
| `CrossEdgeStore` | HashMap-backed directed bipartite edge store implementing `CrossStore` |
| `CrossStore` | Trait for custom cross-edge backends |

## Design notes

**Both modalities share `SokmConfig` and `KernelConfig`:** the Gestalt assumption is symmetric
learning dynamics. `CrossSokmConfig` governs only the cross-edge lifecycle — it is independent
of the intra-modal configs.

**`require_class_match=true` (default):** cross-modal edges form only between same-class kernels.
`None`-labelled kernels never participate. Set `require_class_match=false` for unsupervised
cross-modal binding where any two labelled kernels from opposite modalities can co-activate.

**`compact()` must reindex cross edges:** call `compact()` after training phases to reclaim memory.
Cross edges referencing extinct kernel indices are dropped automatically during reindex.

## MSRV

Rust 1.95 (stable). No nightly required.

## Reference

Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*, Springer.
Eqs. 4.1, 4.3, 4.7 \[INFERRED cross-modal extension\].

## License

MIT OR Apache-2.0
