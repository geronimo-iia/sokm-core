//! Demonstrates the compact() lifecycle: train → inspect → compact → verify → continue.
//!
//! Key points:
//! - compact() must only be called between ticks, never mid-tick
//! - cross edges referencing extinct kernel indices are dropped automatically
//! - kernel indices change after compact() — do not hold stale indices across the call
//! - recall still works correctly after compaction
use sokm::{DecayMode, HashEdgeStore};
use sokm_kernel::KernelConfig;
use sokm_multimodal::{DefaultGestaltGraph, GestaltConfig};

fn main() {
    // p1_kernel=10: kernel goes extinct after 10 ticks without activation.
    let cfg = GestaltConfig {
        kernel: KernelConfig {
            p1_kernel: 10,
            ..KernelConfig::default()
        },
        ..GestaltConfig::default()
    };

    let mut g = DefaultGestaltGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);

    // Phase 1: grow many class-1 kernels using well-separated inputs (large spacing).
    // Each tick sees a novel input → new kernel grown each tick.
    println!("=== Phase 1: grow class-1 kernels (ticks 0–19) ===");
    for t in 0..20u64 {
        let i = t as usize;
        let x1: Vec<f64> = vec![i as f64 * 100.0, 0.0];
        let x2: Vec<f64> = vec![0.0, i as f64 * 100.0];
        g.tick(&x1, &x2, Some(1), t, &cfg, DecayMode::Apply);
    }
    println!(
        "  m1_kernels={} m2_kernels={} cross_edges={}",
        g.kernel_count_modal1(),
        g.kernel_count_modal2(),
        g.cross_edge_count(),
    );

    // Phase 2: stop activating class-1 kernels — let them age out (p1_kernel=10).
    // Train a single class-2 pair repeatedly so it stays alive.
    println!("\n=== Phase 2: age out class-1 kernels (ticks 20–34) ===");
    for t in 20..35u64 {
        g.tick(
            &[9000.0, 0.0],
            &[0.0, 9000.0],
            Some(2),
            t,
            &cfg,
            DecayMode::Apply,
        );
    }
    println!(
        "  m1_kernels={} m2_kernels={} cross_edges={}",
        g.kernel_count_modal1(),
        g.kernel_count_modal2(),
        g.cross_edge_count(),
    );

    // compact(): removes extinct kernels, reindexes cross edges pointing to them.
    // Must be called between ticks. Returns (extinct_from_modal1, extinct_from_modal2).
    // After this call, all previously captured kernel indices are invalid.
    let (pruned1, pruned2) = g.compact();
    println!("\n=== compact() ===\n  pruned: modal1={pruned1} modal2={pruned2}");
    println!(
        "  m1_kernels={} m2_kernels={} cross_edges={}",
        g.kernel_count_modal1(),
        g.kernel_count_modal2(),
        g.cross_edge_count(),
    );
    assert!(
        pruned1 > 0 || pruned2 > 0,
        "expected some kernels to be pruned"
    );

    // Recall still works — indices remapped internally, live kernels unaffected.
    let recall = g.recall_from_modal1(&[9000.0, 0.0], &cfg);
    println!(
        "\n=== recall after compact() ===\n  modal1→modal2 results: {}",
        recall.len(),
    );
    assert!(
        !recall.is_empty(),
        "live class-2 kernel should still recall"
    );

    // Phase 3: continue training after compact — graph fully usable.
    println!("\n=== Phase 3: continue training (ticks 35–44) ===");
    for t in 35..45u64 {
        g.tick(
            &[9000.0, 0.0],
            &[0.0, 9000.0],
            Some(2),
            t,
            &cfg,
            DecayMode::Apply,
        );
    }
    println!(
        "  m1_kernels={} m2_kernels={} cross_edges={}",
        g.kernel_count_modal1(),
        g.kernel_count_modal2(),
        g.cross_edge_count(),
    );

    println!("\nAll checks passed.");
}
