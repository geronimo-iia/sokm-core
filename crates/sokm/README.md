# sokm

[![Crates.io](https://img.shields.io/crates/v/sokm.svg)](https://crates.io/crates/sokm)
[![Docs.rs](https://docs.rs/sokm/badge.svg)](https://docs.rs/sokm)
[![License](https://img.shields.io/crates/l/sokm.svg)](LICENSE-MIT)

Hebbian link mechanics for SOKM (Self-Organizing Kernel Memory).

Implements the link-weight layer from Tetsuya Hoya (2005),
*Artificial Mind System: Kernel Memory Approach* — decay, strengthen, prune,
and propagate over a sparse weighted graph.

This crate is the foundation layer. For kernel units and growth, see
[`sokm-kernel`](https://github.com/geronimo-iia/sokm-core/tree/main/crates/sokm-kernel).

## What it does

- **Decay** — all edge weights decay by `exp(-ξ)` each tick [Hoya Eq. 4.1]
- **Strengthen** — co-activated node pairs reinforce their edge [Hoya Eqs. 4.6–4.7]
- **Prune** — removes edges below a weight floor or inactive beyond `p1` ticks
- **Propagate** — spreads activation through the graph, binary or soft [Hoya Eqs. 4.3–4.4]
- **`tick`** — runs all four steps in one call; `DecayMode::Skip` bypasses the decay step (use for bulk import to avoid distorting edge weights)

## Usage

```toml
[dependencies]
sokm = "0.1"
```

```rust
use sokm::{HashEdgeStore, SokmConfig, tick};
use sokm::EdgeStore;

let mut store: HashEdgeStore<u32> = HashEdgeStore::new();
let cfg = SokmConfig::default();

// Present two co-activated nodes at tick 1
let activated = vec![(0u32, 1.0), (1, 0.8)];
let report = tick(&mut store, &activated, 1, &cfg, false);

assert_eq!(report.strengthened, 1);
println!("strengthened: {}, pruned: {}", report.strengthened, report.pruned);
```

## Edge store backends

Two backends are provided. Both produce identical results — `SparseEdgeStore`
is a drop-in replacement for `HashEdgeStore` with better performance on large graphs.

| Backend | Use case |
|---------|----------|
| `HashEdgeStore` | Tests and small graphs |
| `SparseEdgeStore` | Production CSR-backed workloads |

Custom backends implement the `EdgeStore` trait. Backends that need to survive
kernel compaction (index remapping after extinct-kernel removal) must also
implement `Reindex`.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | off | `Serialize`/`Deserialize` for all public types |

## Examples

```bash
cargo run -p sokm --example hebbian_link
```

Demonstrates decay, strengthen, prune, and soft propagation on a small graph.
Shows edge weight growth under repeated co-activation, decay when co-activation stops, and cued recall via `propagate_soft`.

## Benchmarks

```bash
cargo bench -p sokm
```

Covers `tick`, `decay`, `strengthen`, `prune`, `propagate_soft`, and
`propagate` on a 1 000-node graph with 50 active nodes.

## Crate map

```
sokm              ← link layer (this crate)          [sokm-core]
sokm-kernel       ← kernel units, growth, STM        [sokm-core]
sokm-multimodal   ← Gestalt K³ cross-modal memory
sokm-emotion      ← emotional state layer
sokm-memory       ← persistent episodic memory store
```

## Design notes

**Goal:** pure Hebbian edge mechanics with no knowledge of kernel units. The link layer is the lowest level in the crate stack.

**Why separate from `sokm-kernel`:** edge logic is generic over any node type `N: Hash + Eq + Copy`; kernel types (usize indices) must not bleed into edge arithmetic. `sokm-kernel` can be tested standalone; `sokm-multimodal` uses both independently with different edge stores.

**Two backends, not one:** `HashEdgeStore` gives predictable iteration order — useful in unit tests. `SparseEdgeStore` uses a CSR layout for cache-friendly traversal on large graphs. Both satisfy the `EdgeStore` + `Reindex` traits so callers are backend-agnostic. A single `HashMap<(N,N), f64>` was considered but rejected: poor cache behavior at scale and no path to CSR.

**`Reindex` is optional:** backends that never survive compaction (tests, ephemeral graphs) implement only `EdgeStore`. Only production backends need `Reindex`.

## Reference

Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*, Springer.
Equations 4.1, 4.3–4.4, 4.6–4.7.

## License

MIT OR Apache-2.0
