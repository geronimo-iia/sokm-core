use sokm::{DecayMode, HashEdgeStore};
use sokm_kernel::KernelConfig;
use sokm_multimodal::{DefaultGestaltGraph, GestaltConfig};

fn main() {
    // theta_k=0.5: one-hot 4D vectors (distance=sqrt(2), score≈0.37) form separate kernels
    let cfg = GestaltConfig {
        kernel: KernelConfig {
            theta_k: 0.5,
            ..KernelConfig::default()
        },
        ..GestaltConfig::default()
    };
    let mut g = DefaultGestaltGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);

    // Two clusters per modality, well-separated
    let a1 = vec![1.0f64, 0.0, 0.0, 0.0];
    let b1 = vec![0.0f64, 0.0, 1.0, 0.0];
    let a2 = vec![0.0f64, 1.0, 0.0, 0.0];
    let b2 = vec![0.0f64, 0.0, 0.0, 1.0];

    // Train 500 ticks, alternating classes
    for t in 0..500u64 {
        if t % 2 == 0 {
            g.tick(&a1, &a2, Some(0), t, &cfg, DecayMode::Apply);
        } else {
            g.tick(&b1, &b2, Some(1), t, &cfg, DecayMode::Apply);
        }
    }

    println!("After 500 ticks:");
    println!("  modal1 kernels: {}", g.modal1.kernel_count());
    println!("  modal2 kernels: {}", g.modal2.kernel_count());
    println!("  cross edges:    {}", g.cross_edge_count());

    // Recall: modal1 cue A1 → should recover modal2 A2 kernel
    let mut r1 = g.recall_from_modal1(&a1, &cfg);
    r1.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nrecall_from_modal1(A1 cue):");
    for (idx, score) in r1.iter().take(3) {
        println!("  modal2 kernel[{idx}] score={score:.4}");
    }
    assert!(!r1.is_empty(), "recall_from_modal1 returned empty");

    // Recall: modal1 cue B1 → should recover modal2 B2 kernel (different top result)
    let mut r2 = g.recall_from_modal1(&b1, &cfg);
    r2.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nrecall_from_modal1(B1 cue):");
    for (idx, score) in r2.iter().take(3) {
        println!("  modal2 kernel[{idx}] score={score:.4}");
    }
    assert!(!r2.is_empty(), "recall_from_modal1(B1) returned empty");

    // Top results must differ between A1 and B1 cues
    let top_a1 = r1[0].0;
    let top_b1 = r2[0].0;
    assert_ne!(top_a1, top_b1,
        "A1 and B1 cues both resolved to modal2 kernel[{top_a1}] — cross-modal separation failed");

    // Reverse recall: modal2 cue A2 → should recover modal1 A1 kernel
    let mut r3 = g.recall_from_modal2(&a2, &cfg);
    r3.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nrecall_from_modal2(A2 cue):");
    for (idx, score) in r3.iter().take(3) {
        println!("  modal1 kernel[{idx}] score={score:.4}");
    }
    assert!(!r3.is_empty(), "recall_from_modal2 returned empty");

    // Cross edge density check
    assert!(g.cross_edge_count() > 0, "no cross-modal edges formed after 500 ticks");

    println!("\nAll convergence checks passed.");
}
