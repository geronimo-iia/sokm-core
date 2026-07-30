//! Compares `IdentityPolicy`, `ClampPolicy`, and `DecayPolicy` on identical inputs.
//!
//! **Phase 1 — Training** (positive valence, 10 ticks):
//!   - `IdentityPolicy`: unbounded; E₁ grows past +15 by tick 9.
//!   - `ClampPolicy`: saturates at E₁=+3.0, E₂=+2.0 and stays there.
//!   - `DecayPolicy`: also saturates at the clamp boundary (decay pre-multiplies before
//!     clamping; strong activation overcomes the decay factor).
//!
//! **Phase 2 — Silence** (out-of-distribution input, 8 ticks):
//!   An input far from all trained kernels (Gaussian ≈ 0) means activation sum ≈ 0.
//!   - `IdentityPolicy`: frozen at its training-end value — no mechanism pulls it back.
//!   - `ClampPolicy`: frozen at E₁=+3.0, E₂=+2.0 — decay_factor=1.0, no activation.
//!   - `DecayPolicy`: halves each tick (decay=0.5); recovers toward the attentive window.
//!
//! Run: cargo run -p sokm-emotion --example policy_comparison

use sokm::{DecayMode, HashEdgeStore, SokmConfig};
use sokm_emotion::{
    ClampPolicy, DecayPolicy, DefaultEmotionalGraph, EmotionConfig, EmotionalGraphConfig,
    EmotionalKernelGraph,
};
use sokm_kernel::{DefaultKernelStore, KernelConfig};

fn base_cfg() -> EmotionalGraphConfig {
    EmotionalGraphConfig {
        sokm: SokmConfig::default(),
        kernel: KernelConfig {
            sigma_0: 2.0,
            ..KernelConfig::default()
        },
        emotion: EmotionConfig {
            lambda_e: 0.3,
            theta_e: 1.5,
            ..EmotionConfig::default()
        },
    }
}

fn header() {
    println!(
        "{:>4}  {:>22}  {:>22}  {:>22}",
        "tick", "Identity (E1, E2)", "Clamp (E1, E2)", "Decay (E1, E2)"
    );
    println!("{}", "-".repeat(78));
}

fn print_row(
    t: usize,
    ri: &sokm_emotion::EmotionalTickReport,
    rc: &sokm_emotion::EmotionalTickReport,
    rd: &sokm_emotion::EmotionalTickReport,
) {
    println!(
        "{:>4}  ({:+.3}, {:+.3})          ({:+.3}, {:+.3})          ({:+.3}, {:+.3})",
        t, ri.global.e1, ri.global.e2, rc.global.e1, rc.global.e2, rd.global.e1, rd.global.e2,
    );
}

fn main() {
    let cfg = base_cfg();

    let mut identity: DefaultEmotionalGraph =
        DefaultEmotionalGraph::new(HashEdgeStore::default(), cfg.clone());

    let mut clamp: EmotionalKernelGraph<_, DefaultKernelStore, ClampPolicy> =
        EmotionalKernelGraph::with_store(
            HashEdgeStore::default(),
            DefaultKernelStore::default(),
            ClampPolicy::default(),
            cfg.clone(),
        );

    // decay=0.5 — aggressive, makes recovery visible within 8 ticks.
    let mut decay: EmotionalKernelGraph<_, DefaultKernelStore, DecayPolicy> =
        EmotionalKernelGraph::with_store(
            HashEdgeStore::default(),
            DefaultKernelStore::default(),
            DecayPolicy {
                decay: 0.5,
                clamp: ClampPolicy::default(),
            },
            cfg.clone(),
        );

    // Phase 1 — Training: repeated strong positive valence, input near [1.0, 0.0].
    let training: &[(&[f64], u32, [f64; 2])] = &[
        (&[1.0, 0.0], 1, [3.0, 2.0]),
        (&[1.1, 0.0], 1, [3.0, 2.0]),
        (&[0.9, 0.1], 1, [3.0, 2.0]),
        (&[1.0, 0.0], 1, [3.0, 2.0]),
        (&[1.0, 0.0], 1, [3.0, 2.0]),
        (&[1.1, 0.0], 1, [3.0, 2.0]),
        (&[1.0, 0.0], 1, [3.0, 2.0]),
        (&[1.0, 0.0], 1, [3.0, 2.0]),
        (&[0.9, 0.0], 1, [3.0, 2.0]),
        (&[1.0, 0.0], 1, [3.0, 2.0]),
    ];

    println!("=== Phase 1: Training (positive valence) ===");
    header();
    for (t, &(x, class, e_target)) in training.iter().enumerate() {
        let ri = identity.tick(x, Some(class), e_target, t as u64, DecayMode::Apply);
        let rc = clamp.tick(x, Some(class), e_target, t as u64, DecayMode::Apply);
        let rd = decay.tick(x, Some(class), e_target, t as u64, DecayMode::Apply);
        print_row(t, &ri, &rc, &rd);
    }

    println!("\nAttentive after training (theta_e=1.5, optimal=(0,0)):");
    println!("  Identity : {}", identity.is_attentive());
    println!("  Clamp    : {}", clamp.is_attentive());
    println!("  Decay    : {}", decay.is_attentive());

    // Phase 2 — Silence: out-of-distribution input, gaussian ≈ 0, activation sum ≈ 0.
    // kernel trained near [1.0, 0.0], sigma=2.0 → gaussian([10.0,0.0]) = exp(-81/8) ≈ 0.
    // Identity: no mechanism to drop — frozen at training-end value.
    // Clamp:    decay_factor=1.0, no activation → frozen at boundary [3.0, 2.0].
    // Decay:    E *= 0.5 each tick → halves toward neutral; enters attentive window in ~4 ticks.
    println!("\n=== Phase 2: Silence (OOD input, gaussian ≈ 0) ===");
    header();
    let base = training.len();
    for i in 0..8usize {
        let t = base + i;
        let ri = identity.tick(
            &[10.0, 0.0],
            Some(1),
            [0.0, 0.0],
            t as u64,
            DecayMode::Apply,
        );
        let rc = clamp.tick(
            &[10.0, 0.0],
            Some(1),
            [0.0, 0.0],
            t as u64,
            DecayMode::Apply,
        );
        let rd = decay.tick(
            &[10.0, 0.0],
            Some(1),
            [0.0, 0.0],
            t as u64,
            DecayMode::Apply,
        );
        print_row(t, &ri, &rc, &rd);
    }

    println!("\nAttentive after silence:");
    println!("  Identity : {}", identity.is_attentive());
    println!("  Clamp    : {}", clamp.is_attentive());
    println!("  Decay    : {}", decay.is_attentive());
}
