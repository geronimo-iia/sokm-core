# sokm-multimodal

[![Crates.io](https://img.shields.io/crates/v/sokm-multimodal.svg)](https://crates.io/crates/sokm-multimodal)
[![Docs.rs](https://docs.rs/sokm-multimodal/badge.svg)](https://docs.rs/sokm-multimodal)
[![License](https://img.shields.io/crates/l/sokm-multimodal.svg)](LICENSE-MIT)

Gestalt K³ cross-modal memory for SOKM (Self-Organizing Kernel Memory).

Two independent SOKM modalities (each a full `KernelGraph`) coupled by a directed bipartite
cross-edge store. Cross-modal Hebbian learning: when kernels from both modalities co-activate
with matching class labels, the edge between them strengthens. Recall in either direction: given
a modal1 cue, recover modal2 activations; given a modal2 cue, recover modal1 activations.

\[INFERRED\] — Hoya Eqs. 4.1, 4.3, 4.7 applied cross-modally.

Builds on [`sokm-kernel`](https://github.com/geronimo-iia/sokm-core/tree/main/crates/sokm-kernel) (kernel layer).

## Background

Hoya's Gestalt K³ architecture extends the single-modality SOKM with a second independent
modality and a bipartite cross-edge layer connecting kernel units across modalities. The
cross-edge layer follows the same Hebbian lifecycle as intra-modality edges: strengthen on
co-activation, decay each tick, prune by weight or inactivity.

`DefaultGestaltGraph` wraps two `KernelGraph` instances and a `CrossEdgeStore` into a single
`tick()` call — both modalities tick in parallel, then cross edges are updated.

## What it does

- Two coupled `KernelGraph` modalities: `modal1` and `modal2`
- Cross-modal Hebbian edges: strengthen on co-activation, decay each tick, prune by weight or inactivity
- `recall_from_modal1(x1)` → modal2 activation scores; `recall_from_modal2(x2)` → modal1 scores
- `GestaltConfig`: single config bundles `SokmConfig`, `KernelConfig`, `CrossSokmConfig`
- `DefaultGestaltGraph` concrete alias — zero boilerplate for standard use

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

// Recall: given modal1 cue, recover modal2 activations
let results = g.recall_from_modal1(&[1.0, 0.0], &cfg);
for (kernel_idx, score) in results {
    println!("modal2 kernel[{kernel_idx}] score={score:.3}");
}
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | off | `Serialize`/`Deserialize` for all public types |
| `simd`  | off | SIMD scoring via `sokm-kernel/simd` |

## API

| Symbol | Description |
|--------|-------------|
| `DefaultGestaltGraph::new(edges1, edges2, &cfg)` | Create graph with two `HashEdgeStore` modalities |
| `GestaltKernelGraph::with_stores(e1, k1, e2, k2, &cfg)` | Create from pre-populated stores |
| `GestaltKernelGraph::tick(x1, x2, class, t, &cfg, decay)` | One learning step: both modalities tick, cross edges strengthen/decay/prune. Returns `GestaltTickReport` |
| `GestaltTickReport` | `modal1: KernelTickReport`, `modal2: KernelTickReport`, `cross_strengthened: usize`, `cross_pruned: usize` |
| `recall_from_modal1(x1, &cfg)` | Score all modal2 kernels from a modal1 cue via cross edges |
| `recall_from_modal2(x2, &cfg)` | Score all modal1 kernels from a modal2 cue (O(E) scan) |
| `compact()` | Prune extinct kernels in both modalities; reindex cross edges. Returns `(pruned1, pruned2)` |
| `cross_edge_count()` | Current number of cross-modal edges |
| `GestaltConfig` | Bundles `sokm: SokmConfig`, `kernel: KernelConfig`, `cross: CrossSokmConfig` |
| `CrossSokmConfig` | `gamma`, `delta`, `w_init`, `w_max`, `min_weight`, `xi`, `p1`, `require_class_match` |
| `CrossEdgeStore` | HashMap-backed directed bipartite edge store implementing `CrossStore` |

## Crate map

```
sokm              ← link layer                              [sokm-core]
sokm-kernel       ← kernel units, growth, STM               [sokm-core]
sokm-emotion      ← per-kernel emotion variables, global state [sokm-core]
sokm-multimodal   ← Gestalt K³ cross-modal memory (this crate) [sokm-core]
sokm-memory       ← persistent episodic memory store
```

## Design notes

Both modalities share the same `SokmConfig` and `KernelConfig` — the Gestalt assumption is that
both modalities operate under identical learning dynamics. `CrossSokmConfig` governs only the
cross-edge lifecycle.

`require_class_match=true` (default) means only same-class co-activations strengthen cross edges
— cross-modal associations are class-gated. `require_class_match=false` allows any two labelled
kernels from opposite modalities to strengthen, which is useful when class labels are unavailable
or when the goal is unsupervised cross-modal binding.

## MSRV

Rust 1.95 (stable). No nightly required.

## Reference

Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*, Springer.
Eqs. 4.1, 4.3, 4.7 \[INFERRED cross-modal extension\].

## License

MIT OR Apache-2.0
