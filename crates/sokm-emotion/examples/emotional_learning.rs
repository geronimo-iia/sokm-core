//! Demonstrates per-kernel emotional learning and the attentive condition.
//!
//! Two concept clusters are trained with different emotional valences:
//!   cluster A — inputs near [5, 0, 0, 0], positive valence [+1.5, +1.0]
//!   cluster B — inputs near [0, 0, 0, 5], negative valence [−1.5, −1.0]
//!
//! After training:
//!   - Each cluster has its own kernel with accumulated emotion vars.
//!   - The global state reflects the mix of recent activations.
//!   - Presenting a pure cluster-A input puts the system near attentive state.
//!   - Presenting a pure cluster-B input drives the system away from neutral.
//!
//! Run: cargo run -p sokm-emotion --example emotional_learning

use sokm::DecayMode;
use sokm::HashEdgeStore;
use sokm::SokmConfig;
use sokm_emotion::{DefaultEmotionalGraph, EmotionConfig, EmotionalGraphConfig};
use sokm_kernel::KernelConfig;

fn main() {
    let cfg = EmotionalGraphConfig {
        sokm: SokmConfig::default(),
        kernel: KernelConfig::default(),
        emotion: EmotionConfig {
            lambda_e: 0.2, // faster blend for demo
            theta_e: 1.0,
            ..EmotionConfig::default()
        },
    };

    let mut graph = DefaultEmotionalGraph::new(HashEdgeStore::default(), cfg.clone());

    // Training data: (input, class, emotional valence)
    let training: &[(&[f64], u32, [f64; 2])] = &[
        (&[5.0, 0.0, 0.0, 0.0], 1, [1.5, 1.0]),
        (&[4.8, 0.2, 0.0, 0.0], 1, [1.4, 0.9]),
        (&[5.0, 0.0, 0.0, 0.0], 1, [1.5, 1.0]),
        (&[0.0, 0.0, 0.0, 5.0], 2, [-1.5, -1.0]),
        (&[0.1, 0.0, 0.0, 4.9], 2, [-1.4, -0.9]),
        (&[0.0, 0.0, 0.0, 5.0], 2, [-1.5, -1.0]),
        (&[5.0, 0.0, 0.0, 0.0], 1, [1.5, 1.0]),
        (&[0.0, 0.0, 0.0, 5.0], 2, [-1.5, -1.0]),
        (&[4.9, 0.1, 0.0, 0.0], 1, [1.5, 1.0]),
        (&[0.0, 0.0, 0.1, 4.9], 2, [-1.5, -1.0]),
    ];

    println!("=== Training ===");
    for (t, &(x, class, e_target)) in training.iter().enumerate() {
        let report = graph.tick(x, Some(class), e_target, t as u64, DecayMode::Apply);
        if report.kernel.grew {
            println!("  tick {t}: new kernel — total {}", graph.kernel_count());
        }
    }

    println!("\nAfter training: {} kernels", graph.kernel_count());
    println!("Per-kernel emotion vars:");
    for i in 0..graph.kernel_count() {
        let vars = graph.emotion_vars(i);
        println!("  kernel[{i}]: e1={:.3}  e2={:.3}", vars[0], vars[1]);
    }
    let gs = graph.global_state();
    println!(
        "Global state after training: E1={:.3}  E2={:.3}",
        gs.e1, gs.e2
    );

    // Inference: present pure patterns, observe global emotion and attentive condition
    println!("\n=== Inference ===");

    let probes: &[(&[f64], &str, [f64; 2])] = &[
        (&[5.0, 0.0, 0.0, 0.0], "cluster A (positive)", [0.0, 0.0]),
        (&[0.0, 0.0, 0.0, 5.0], "cluster B (negative)", [0.0, 0.0]),
    ];

    let tick_base = training.len() as u64;
    for (i, &(x, label, e_target)) in probes.iter().enumerate() {
        let report = graph.tick(x, None, e_target, tick_base + i as u64, DecayMode::Apply);
        println!(
            "  {label}: E1={:.3}  E2={:.3}  attentive={}",
            report.global.e1, report.global.e2, report.attentive
        );
    }
}
