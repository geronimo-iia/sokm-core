//! Demonstrates autonomous category formation over a stream of labelled inputs.
//!
//! The graph grows a kernel for each distinct region of input space it encounters.
//! After training, `predict` classifies novel inputs by nearest kernel.
//!
//! Uses `SparseEdgeStore` — the production backend for large graphs.
//! For tests or small ephemeral graphs, `HashEdgeStore` is simpler.
//!
//! Run: cargo run -p sokm-kernel --example category_formation

use sokm::{DecayMode, SokmConfig, SparseEdgeStore};
use sokm_kernel::store::KernelStore;
use sokm_kernel::{KernelConfig, KernelGraph, compute_scores};

fn main() {
    let sokm_cfg = SokmConfig::default();
    let kernel_cfg = KernelConfig::default();

    // SparseEdgeStore capacity hint = expected max kernel count.
    // Grows automatically if exceeded.
    let mut graph: KernelGraph<SparseEdgeStore> =
        KernelGraph::new(SparseEdgeStore::new(16), &kernel_cfg);

    // Training stream: two classes separated in 2D space.
    // Class 0 clusters around (0, 0); class 1 clusters around (5, 5).
    let training: &[(&[f64], u32)] = &[
        (&[0.1, 0.2], 0),
        (&[0.3, 0.0], 0),
        (&[0.0, 0.4], 0),
        (&[5.1, 4.9], 1),
        (&[4.8, 5.2], 1),
        (&[5.0, 5.0], 1),
        (&[0.2, 0.1], 0),
        (&[5.3, 4.7], 1),
    ];

    for (tick, &(x, class)) in training.iter().enumerate() {
        let report = graph.tick(
            x,
            Some(class),
            tick as u64,
            &sokm_cfg,
            &kernel_cfg,
            DecayMode::Apply,
        );
        if report.grew {
            println!(
                "tick {tick:2}: new kernel grown, total = {}",
                graph.kernel_count()
            );
        }
    }

    println!("\nKernels after training: {}", graph.kernel_count());

    // Classify novel inputs
    let queries: &[(&[f64], u32)] = &[
        (&[0.15, 0.25], 0), // near class 0
        (&[4.9, 5.1], 1),   // near class 1
        (&[0.05, 0.05], 0), // near class 0
        (&[5.2, 5.3], 1),   // near class 1
    ];

    println!("\nClassification:");
    let mut correct = 0;
    for &(x, expected) in queries {
        // Nearest kernel by direct activation score → its class label
        let scores = compute_scores(graph.kernels(), x);
        let predicted = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .and_then(|(i, _)| graph.kernels().class_opt(i));
        let ok = predicted == Some(expected);
        if ok {
            correct += 1;
        }
        println!(
            "  x={x:?} → predicted={predicted:?} (expected {expected}) {}",
            if ok { "✓" } else { "✗" }
        );
    }
    println!("\n{correct}/{} correct", queries.len());
}
