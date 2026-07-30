# sokm

[![Crates.io](https://img.shields.io/crates/v/sokm.svg)](https://crates.io/crates/sokm)
[![Docs.rs](https://docs.rs/sokm/badge.svg)](https://docs.rs/sokm)
[![License](https://img.shields.io/crates/l/sokm.svg)](LICENSE-MIT)

Hebbian link mechanics for SOKM (Self-Organizing Kernel Memory) — decay, strengthen, prune, propagate over a sparse weighted graph.

Implements the link-weight layer from Tetsuya Hoya (2005),
*Artificial Mind System: Kernel Memory Approach*.

This crate is the foundation layer. For kernel units and growth, see
[`sokm-kernel`](https://github.com/geronimo-iia/sokm-core/tree/main/crates/sokm-kernel).

## What it does

- **Decay** — all edge weights decay by `exp(-ξ)` each tick [Hoya Eq. 4.1]
- **Strengthen** — co-activated node pairs reinforce their edge [Hoya Eqs. 4.6–4.7]
- **Prune** — removes edges below a weight floor or inactive beyond `p1` ticks [Hoya §4]
- **Propagate** — spreads activation through the graph, binary or soft [Hoya Eqs. 4.3–4.4]
- **`tick`** — runs all four steps in one call; `DecayMode::Skip` bypasses decay (use for bulk import)

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

```text
=== Phase 1: co-activate nodes 0 and 1 for 20 ticks ===
tick   w(0,1)       edges     
1      0.100000     1           ← edge created at w_init
5      0.273410     1
10     0.480643     1
15     0.677768     1
20     0.865280     1           ← weight approaches w_max (1.0) after 20 ticks

=== Phase 2: stop co-activating — decay only for 30 ticks ===
tick   w(0,1)       edges     
25     0.823080     1
30     0.782937     1
...
50     0.641015     1           ← edge survives: w still above min_weight (0.001)

=== Phase 3: rebuild with 3 nodes, then soft propagate ===
edges after 10 ticks: 3
soft propagation from node 0:
  node 1 → 0.393653            ← cued recall: node 1 activated via learned edge
  node 2 → 0.315803            ← weaker recall: node 2 less strongly co-activated
```

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
sokm-emotion      ← per-kernel emotion variables, global state [sokm-core]
sokm-multimodal   ← Gestalt K³ cross-modal memory    [sokm-core]
```

## Design notes

**Goal:** pure Hebbian edge mechanics with no knowledge of kernel units. The link layer is the lowest level in the crate stack.

**Why separate from `sokm-kernel`:** edge logic is generic over any node type `N: Hash + Eq + Copy`; kernel types (usize indices) must not bleed into edge arithmetic. `sokm-kernel` can be tested standalone; `sokm-multimodal` uses both independently with different edge stores.

**Two backends, not one:** `HashEdgeStore` gives predictable iteration order — useful in unit tests. `SparseEdgeStore` uses a CSR layout for cache-friendly traversal on large graphs. Both satisfy the `EdgeStore` + `Reindex` traits so callers are backend-agnostic. A single `HashMap<(N,N), f64>` was considered but rejected: poor cache behavior at scale and no path to CSR.

**`Reindex` is optional:** backends that never survive compaction (tests, ephemeral graphs) implement only `EdgeStore`. Only production backends need `Reindex`.

## MSRV

Rust 1.95 (stable). No nightly required.

## Reference

Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*, Springer.
Equations 4.1, 4.3–4.4, 4.6–4.7.

## License

MIT OR Apache-2.0
