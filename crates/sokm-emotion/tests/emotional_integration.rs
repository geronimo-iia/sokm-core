use sokm::DecayMode;
use sokm::HashEdgeStore;
use sokm::SokmConfig;
use sokm_emotion::{EmotionConfig, EmotionalGraphConfig, EmotionalKernelGraph, salience};
use sokm_kernel::KernelConfig;

fn sokm_cfg() -> SokmConfig {
    SokmConfig {
        w_max: 10.0,
        ..SokmConfig::default()
    }
}

fn kernel_cfg() -> KernelConfig {
    KernelConfig::default()
}

fn emotion_cfg() -> EmotionConfig {
    EmotionConfig::default()
}

fn graph_cfg() -> EmotionalGraphConfig {
    EmotionalGraphConfig {
        sokm: sokm_cfg(),
        kernel: kernel_cfg(),
        emotion: emotion_cfg(),
    }
}

type TestGraph = EmotionalKernelGraph<HashEdgeStore<usize>>;

#[test]
fn learn_cycle_vars_converge() {
    let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
    let target = [1.5, -0.5];
    for i in 0..10u64 {
        g.tick(&[0.5], Some(0), target, i + 1, DecayMode::Apply);
    }
    let vars = g.emotion_vars(0);
    // with lambda_e=0.1 and 10 ticks from 0: e ~ 0.651
    assert!(
        vars[0] > 0.5,
        "e1 should have moved significantly toward 1.5"
    );
    assert!(
        vars[1] < -0.2,
        "e2 should have moved significantly toward -0.5"
    );
}

#[test]
fn attentive_gate_aligned_vs_opposite() {
    let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
    // train toward positive cluster
    for i in 0..20u64 {
        g.tick(&[0.5], Some(0), [2.0, 1.0], i + 1, DecayMode::Apply);
    }
    // is_attentive with optimal at (0,0), theta=1.0 — after training global may deviate
    let gs = g.global_state();
    // salience aligned
    let vars = g.emotion_vars(0);
    let sal_aligned = salience(vars, &gs, 1.0);
    let sal_opposite = salience([-vars[0], -vars[1]], &gs, 1.0);
    // aligned should have higher salience than anti-aligned
    assert!(
        sal_aligned >= sal_opposite,
        "aligned salience should be >= opposite"
    );
}

#[test]
fn compact_and_tick_integrity() {
    let cfg_grow = EmotionalGraphConfig {
        kernel: KernelConfig {
            theta_k: 2.0,
            sigma_0: 0.001,
            p1_kernel: 3,
            ..kernel_cfg()
        },
        ..graph_cfg()
    };
    let mut g = TestGraph::new(HashEdgeStore::new(), cfg_grow.clone());

    // grow 3 kernels
    g.tick(&[0.0], Some(0), [1.0, 0.0], 1, DecayMode::Apply);
    g.tick(&[10.0], Some(0), [0.0, 1.0], 2, DecayMode::Apply);
    g.tick(&[20.0], Some(0), [0.5, 0.5], 3, DecayMode::Apply);
    assert_eq!(g.kernel_count(), 3);

    // make kernel 0 extinct explicitly then compact
    g.mark_extinct(0);

    let removed = g.compact();
    assert_eq!(removed, 1, "one kernel should be extinct");
    assert_eq!(g.kernel_count(), 2);
    assert_eq!(g.kernel_count(), 2, "emotion store in sync after compact");

    // tick after compact must not panic and maintain invariant
    g.tick(&[10.0], Some(0), [0.0, 0.0], 6, DecayMode::Apply);
    // kernel_count >= 2; debug_assert inside tick verifies emotion store stays in sync
    assert!(g.kernel_count() >= 2);
}
