//! Demonstrates Hebbian link mechanics on a small graph.
//!
//! Shows decay, strengthen, prune, and propagation using `SparseEdgeStore`.
//! Each tick co-activates the same pair — their edge weight grows until
//! clamp, then decays when they are no longer presented together.
//!
//! Run: cargo run -p sokm --example hebbian_link

use sokm::{DecayMode, EdgeStore, SokmConfig, SparseEdgeStore, propagate_soft, tick};

fn main() {
    let cfg = SokmConfig::default();
    let mut store = SparseEdgeStore::new(8);

    println!("=== Phase 1: co-activate nodes 0 and 1 for 20 ticks ===");
    println!("{:<6} {:<12} {:<10}", "tick", "w(0,1)", "edges");

    for t in 1u64..=20 {
        let activated = vec![(0usize, 1.0), (1, 0.9)];
        let report = tick(&mut store, &activated, t, &cfg, DecayMode::Apply);
        if t == 1 || t % 5 == 0 {
            println!(
                "{:<6} {:<12.6} {:<10}  (strengthened={}, pruned={})",
                t,
                store.get_weight(0, 1),
                store.edge_count(),
                report.strengthened,
                report.pruned,
            );
        }
    }

    println!("\n=== Phase 2: stop co-activating — decay only for 30 ticks ===");
    println!("{:<6} {:<12} {:<10}", "tick", "w(0,1)", "edges");

    for t in 21u64..=50 {
        // present only node 0 — no co-activation, edge decays
        let activated = vec![(0usize, 1.0)];
        let report = tick(&mut store, &activated, t, &cfg, DecayMode::Apply);
        let w = store.get_weight(0, 1);
        if t % 5 == 0 || report.pruned > 0 {
            println!(
                "{:<6} {:<12.6} {:<10}  pruned={}",
                t,
                w,
                store.edge_count(),
                report.pruned,
            );
        }
        if store.edge_count() == 0 {
            println!("  → edge pruned at tick {t}");
            break;
        }
    }

    println!("\n=== Phase 3: rebuild with 3 nodes, then soft propagate ===");
    let mut store2 = SparseEdgeStore::new(8);
    let activated = vec![(0usize, 1.0), (1, 0.8), (2, 0.6)];
    for t in 1u64..=10 {
        tick(&mut store2, &activated, t, &cfg, DecayMode::Apply);
    }

    println!("edges after 10 ticks: {}", store2.edge_count());

    // Cued recall from node 0 only — soft propagation spreads activation
    let cued = vec![(0usize, 1.0)];
    let spread = propagate_soft(&store2, &cued, &cfg);
    println!("soft propagation from node 0:");
    let mut spread_sorted = spread;
    spread_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (node, score) in spread_sorted {
        println!("  node {node} → {score:.6}");
    }
}
