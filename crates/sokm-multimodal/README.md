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
| `simd`  | off | SIMD scoring via `sokm-kernel/simd` — **not recommended**: the Rust compiler already applies automatic vectorisation at release optimisation level (LLVM); enabling this feature regresses `gestalt_tick` by 10–23% due to codegen layout changes with no measurable gain on recall. |

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

## Benchmarks

```bash
cargo bench -p sokm-multimodal
```

Parametrized by `(n, d)` — kernel count × input dimension. Key numbers at n=1000:

| bench | 16d | 358d |
|-------|-----|------|
| `gestalt_tick` | 92 µs | 912 µs |
| `recall_from_modal1` | 5.6 µs | 210 µs |
| `recall_from_modal2` | 5.4 µs | 210 µs |

`recall_from_modal1` and `recall_from_modal2` are symmetric — `CrossEdgeStore` uses a reverse
index (O(1) `sources()` lookup), so reverse recall does not scan all edges.

Additional bench groups:
- `gestalt_tick_sparse` — `SparseEdgeStore` backend (~10–15% slower than Hash for this access pattern)
- `gestalt_tick_no_class_match` — `require_class_match=false` path (no measurable difference at same label set)
- `compact_reindex` — reindex cost vs extinction fraction; scales with survivors, not total edges

## Examples

```bash
cargo run -p sokm-multimodal --example convergence
```

Trains two 4D class clusters (class 0: A1↔A2, class 1: B1↔B2) for 500 ticks and verifies
cross-modal recall separates the two classes in both directions.

```text
After 500 ticks:
  modal1 kernels: 2
  modal2 kernels: 2
  cross edges:    2          ← exactly one edge per class pair

recall_from_modal1(A1 cue):
  modal2 kernel[0] score=0.8910   ← correct class-0 target dominates
  modal2 kernel[1] score=0.1206

recall_from_modal1(B1 cue):
  modal2 kernel[1] score=0.8910   ← correct class-1 target dominates
  modal2 kernel[0] score=0.1206

recall_from_modal2(A2 cue):
  modal1 kernel[0] score=0.8910   ← bidirectional: reverse recall also separates

All convergence checks passed.
```

```bash
cargo run -p sokm-multimodal --example compact_lifecycle
```

Shows the full `compact()` lifecycle: grow kernels → let them age out → compact → verify recall
still works → continue training.

```text
=== Phase 1: grow class-1 kernels (ticks 0–19) ===
  m1_kernels=20 m2_kernels=20 cross_edges=20   ← one kernel + one edge per novel input

=== Phase 2: age out class-1 kernels (ticks 20–34) ===
  m1_kernels=21 m2_kernels=21 cross_edges=21   ← class-2 kernel added; class-1 inactive

=== compact() ===
  pruned: modal1=20 modal2=20                  ← 20 extinct kernels removed from each modality
  m1_kernels=1 m2_kernels=1 cross_edges=1      ← only the live class-2 kernel survives

=== recall after compact() ===
  modal1→modal2 results: 1                     ← recall works; indices remapped internally

=== Phase 3: continue training (ticks 35–44) ===
  m1_kernels=1 m2_kernels=1 cross_edges=1

All checks passed.
```

```bash
cargo run -p sokm-multimodal --example kernel_count
```

Reports actual kernel counts and cross-edge counts vs tick count — useful for calibrating bench
fixtures and understanding cross-edge saturation under decay.

```text
ticks=  100: modal1_kernels=  100 modal2_kernels=  100 cross_edges=  100
ticks=  500: modal1_kernels=  500 modal2_kernels=  500 cross_edges=  460
ticks= 1000: modal1_kernels= 1000 modal2_kernels= 1000 cross_edges=  460
ticks= 2000: modal1_kernels= 2000 modal2_kernels= 2000 cross_edges=  460
                                                        ↑ saturates ~460 — decay prunes faster
                                                          than new edges form beyond this density
```

```bash
cargo run -p sokm-multimodal --example memory_footprint
```

Analytical `CrossEdgeStore` memory estimate at 1k/10k/100k/500k edges (~99 bytes/edge).

## API

| Symbol | Description |
|--------|-------------|
| `DefaultGestaltGraph::new(edges1, edges2, &cfg)` | Create graph; takes both edge stores by move |
| `GestaltKernelGraph::with_stores(e1, k1, e2, k2, &cfg)` | Create from pre-populated kernel and edge stores |
| `GestaltKernelGraph::tick(x1, x2, class, t, &cfg, decay)` | One learning step: both modalities tick, cross edges strengthen/decay/prune. Returns `GestaltTickReport` |
| `GestaltTickReport` | `modal1: KernelTickReport`, `modal2: KernelTickReport`, `cross_strengthened: usize`, `cross_pruned: usize` |
| `GestaltKernelGraph::recall_from_modal1(x1, &cfg)` | Score all modal2 kernels from a modal1 cue via cross edges |
| `GestaltKernelGraph::recall_from_modal2(x2, &cfg)` | Score all modal1 kernels from a modal2 cue via O(1) reverse index |
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
