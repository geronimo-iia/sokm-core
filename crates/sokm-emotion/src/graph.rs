use sokm::{DecayMode, EdgeStore, HashEdgeStore, Reindex, SokmConfig};
use sokm_kernel::{
    KernelStore,
    graph::{KernelGraph, KernelTickReport},
    store::DefaultKernelStore,
};

use crate::{
    config::EmotionalGraphConfig,
    policy::{GlobalEmotionPolicy, IdentityPolicy},
    query::salience,
    state::{EmotionState, EmotionStore},
    update::{is_attentive, update_global_emotion, update_kernel_emotion_var},
};

/// Report from one emotional SOKM tick.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct EmotionalTickReport {
    /// Underlying KernelGraph tick report.
    pub kernel: KernelTickReport,
    /// Global emotion state after this tick.
    pub global: EmotionState,
    /// True if attentive condition holds after this tick. [Hoya Eq. 10.7]
    pub attentive: bool,
    /// Per-kernel salience scores (one per kernel). Empty when `alpha == 0.0`.
    pub salience_scores: Vec<f64>,
}

/// KernelGraph with per-kernel emotion variables and global emotion state.
/// [Hoya Eqs. 10.6, 10.7, 10.8; pp. 212–221] \[DIRECT\]
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "S: serde::Serialize, K: serde::Serialize, P: serde::Serialize",
        deserialize = "S: serde::de::DeserializeOwned, K: serde::de::DeserializeOwned, P: serde::de::DeserializeOwned",
    ))
)]
pub struct EmotionalKernelGraph<S, K = DefaultKernelStore, P = IdentityPolicy>
where
    S: EdgeStore<usize>,
    K: KernelStore,
    P: GlobalEmotionPolicy,
{
    pub(crate) graph: KernelGraph<S, K>,
    // Parallel Vec to KernelStore — one [e^1, e^2] entry per kernel, indexed by kernel position.
    //
    // INVARIANT: emotions.len() == graph.kernel_count() at all times.
    // emotions.push() is called exactly when report.grew is true in tick().
    // compact() reindexes emotions in lockstep with graph.compact_with_map().
    // Mismatch silently corrupts emotion_vars(i) reads and salience scores.
    // Guarded by debug_assert_eq! at the end of tick().
    pub(crate) emotions: EmotionStore,
    pub(crate) global: EmotionState,
    pub(crate) policy: P,
    pub(crate) cfg: EmotionalGraphConfig,
}

// Requires Default bounds — only for constructors.
impl<S, K, P> EmotionalKernelGraph<S, K, P>
where
    S: EdgeStore<usize>,
    K: KernelStore + Default,
    P: GlobalEmotionPolicy + Default,
{
    pub fn new(edges: S, cfg: EmotionalGraphConfig) -> Self {
        Self {
            graph: KernelGraph::new(edges, &cfg.kernel),
            emotions: EmotionStore::new(),
            global: EmotionState::default(),
            policy: P::default(),
            cfg,
        }
    }
}

// General methods — no Default required.
impl<S, K, P> EmotionalKernelGraph<S, K, P>
where
    S: EdgeStore<usize>,
    K: KernelStore,
    P: GlobalEmotionPolicy,
{
    pub fn with_store(edges: S, kernels: K, policy: P, cfg: EmotionalGraphConfig) -> Self {
        Self {
            graph: KernelGraph::with_store(edges, kernels, &cfg.kernel),
            emotions: EmotionStore::new(),
            global: EmotionState::default(),
            policy,
            cfg,
        }
    }

    pub fn kernel_count(&self) -> usize {
        self.graph.kernel_count()
    }
    pub fn stm_len(&self) -> usize {
        self.graph.stm_len()
    }
    pub fn stm_indices(&self) -> &[usize] {
        self.graph.stm_indices()
    }
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
    pub fn global_state(&self) -> EmotionState {
        self.global
    }
    /// Panics if `i >= kernel_count()`.
    pub fn emotion_vars(&self, i: usize) -> [f64; 2] {
        self.emotions.get(i)
    }
    pub fn is_attentive(&self) -> bool {
        crate::update::is_attentive(
            &self.global,
            &self.cfg.emotion.optimal,
            self.cfg.emotion.theta_e,
        )
    }
    pub fn cfg(&self) -> &EmotionalGraphConfig {
        &self.cfg
    }
    pub fn propagate_soft(&self, x: &[f64]) -> Vec<(usize, f64)> {
        self.graph.propagate_soft(x, &self.cfg.sokm)
    }
    /// Propagate using an explicit `SokmConfig`, ignoring the stored one.
    pub fn propagate_soft_with_cfg(&self, x: &[f64], cfg: &SokmConfig) -> Vec<(usize, f64)> {
        self.graph.propagate_soft(x, cfg)
    }
    pub fn max_activation(&self, x: &[f64]) -> f64 {
        sokm_kernel::max_activation(self.graph.kernels(), x)
    }
    pub fn mark_extinct(&mut self, pos: usize) {
        self.graph.kernels_mut().mark_extinct(pos);
    }

    /// One full emotional SOKM tick using the stored config.
    ///
    /// `e_target`: expected range e1 ∈ [-3, 3], e2 ∈ [-2, 2] (Hoya pp. 214–215).
    /// Winner-takes-all: only `activated_kernel` has vars trained; all kernels with score > 0
    /// contribute to global update.
    /// New kernel birth: new kernels receive partial first update on birth tick:
    /// `vars = [lambda_e * e_target[0], lambda_e * e_target[1]]`.
    ///
    /// Caller is responsible for validating `EmotionalGraphConfig` via `cfg.validate()`
    /// before first use. Invalid `lambda_e`/`theta_e`/`alpha` silently affect behaviour.
    ///
    /// Order:
    /// 1. Call graph.tick
    /// 2. If grew: push [0.0, 0.0] to EmotionStore
    /// 3. Update activated kernel vars toward e_target
    /// 4. Collect (K_i(x), [e_i^1, e_i^2]) for all kernels with score > 0
    /// 5. Update global state with decay
    /// 6. Apply policy
    /// 7. Check attentive condition
    /// 8. Compute salience scores (empty when alpha == 0.0)
    pub fn tick(
        &mut self,
        x: &[f64],
        class: Option<u32>,
        e_target: [f64; 2],
        current_tick: u64,
        decay: DecayMode,
    ) -> EmotionalTickReport {
        // Step 1
        let report = self.graph.tick(
            x,
            class,
            current_tick,
            &self.cfg.sokm,
            &self.cfg.kernel,
            decay,
        );

        // Step 2
        if report.grew {
            self.emotions.push();
        }

        // Step 3: update activated kernel vars only
        let idx = report.activated_kernel;
        let current = self.emotions.get(idx);
        let lambda_e = self.cfg.emotion.lambda_e;
        let updated = [
            update_kernel_emotion_var(current[0], e_target[0], lambda_e),
            update_kernel_emotion_var(current[1], e_target[1], lambda_e),
        ];
        self.emotions.set(idx, updated);

        // Step 4: collect activations before moving report.
        // INVARIANT: use report.scores here — never recompute gaussian scores on x.
        // KernelTickReport::scores is moved out of KernelGraph::tick for exactly this purpose.
        // A second gaussian scan would be O(N·D) redundant work and risks divergence.
        let activations: Vec<(f64, [f64; 2])> = report
            .scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s > 0.0)
            .map(|(i, &s)| (s, self.emotions.get(i)))
            .collect();

        // Step 5–6
        let emotion_decay = self.policy.decay_factor();
        let new_global = update_global_emotion(self.global, &activations, emotion_decay);
        self.global = self.policy.apply(new_global);

        // Step 7
        let attentive = is_attentive(
            &self.global,
            &self.cfg.emotion.optimal,
            self.cfg.emotion.theta_e,
        );

        // Step 8
        let alpha = self.cfg.emotion.alpha;
        let salience_scores = if alpha == 0.0 {
            Vec::new()
        } else {
            let global = self.global;
            (0..self.graph.kernel_count())
                .map(|i| salience(self.emotions.get(i), &global, alpha))
                .collect()
        };

        debug_assert_eq!(
            self.emotions.len(),
            self.graph.kernel_count(),
            "emotion store out of sync with kernel count"
        );

        EmotionalTickReport {
            kernel: report,
            global: self.global,
            attentive,
            salience_scores,
        }
    }

    /// Compact extinct kernels. Reindexes graph edges, STM, and EmotionStore in lockstep.
    pub fn compact(&mut self) -> usize
    where
        S: Reindex,
    {
        let map = self.graph.compact_with_map();
        let removed = map.iter().filter(|m| m.is_none()).count();
        if removed > 0 {
            self.emotions.compact_with_map(&map);
        }
        removed
    }

    pub fn compact_with_map(&mut self) -> Vec<Option<usize>>
    where
        S: Reindex,
    {
        let map = self.graph.compact_with_map();
        if map.iter().any(|m| m.is_none()) {
            self.emotions.compact_with_map(&map);
        }
        map
    }
}

/// Fully concrete: HashEdgeStore, AoS storage, IdentityPolicy (exact Hoya).
pub type DefaultEmotionalGraph =
    EmotionalKernelGraph<HashEdgeStore<usize>, DefaultKernelStore, IdentityPolicy>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EmotionConfig, EmotionalGraphConfig};
    use crate::policy::DecayPolicy;
    use sokm::HashEdgeStore;
    use sokm::SokmConfig;
    use sokm_kernel::KernelConfig;

    fn kernel_cfg() -> KernelConfig {
        KernelConfig::default()
    }

    fn sokm_cfg() -> SokmConfig {
        SokmConfig {
            w_max: 10.0,
            ..SokmConfig::default()
        }
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
    type DecayGraph = EmotionalKernelGraph<HashEdgeStore<usize>, DefaultKernelStore, DecayPolicy>;

    #[test]
    fn emotional_graph_starts_empty() {
        let g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        assert_eq!(g.kernel_count(), 0);
        assert_eq!(g.emotions.len(), 0);
    }

    #[test]
    fn emotional_tick_grows_emotion_store_in_lockstep() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        g.tick(&[1.0, 2.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);
        assert_eq!(g.emotions.len(), g.kernel_count());
    }

    #[test]
    fn emotional_tick_updates_activated_kernel_vars() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        let r = g.tick(&[1.0], Some(0), [1.0, 0.0], 1, DecayMode::Apply);
        let idx = r.kernel.activated_kernel;
        let vars = g.emotion_vars(idx);
        assert!(vars[0] > 0.0, "e1 should have moved toward 1.0");
    }

    #[test]
    fn emotional_tick_does_not_update_non_activated_kernels() {
        let cfg_grow = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 2.0,
                sigma_0: 0.01,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let mut g = TestGraph::new(HashEdgeStore::new(), cfg_grow.clone());
        g.tick(&[0.0], Some(0), [1.0, 0.0], 1, DecayMode::Apply);
        g.tick(&[100.0], Some(0), [1.0, 0.0], 2, DecayMode::Apply);
        g.emotions.set(0, [0.0, 0.0]);
        g.emotions.set(1, [0.0, 0.0]);
        // Switch to normal cfg — cfg is owned per graph, need new graph or override cfg
        // Use cfg_grow but mutate cfg in place via the cfg field for this test
        let normal_cfg = EmotionalGraphConfig {
            kernel: KernelConfig {
                sigma_0: 0.01,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        g.cfg = normal_cfg;
        let r = g.tick(&[0.0], Some(0), [1.0, 0.0], 3, DecayMode::Apply);
        let activated = r.kernel.activated_kernel;
        let other = 1 - activated;
        assert_eq!(
            g.emotion_vars(other),
            [0.0, 0.0],
            "non-activated kernel must not change"
        );
    }

    #[test]
    fn emotional_tick_global_state_updates() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        g.tick(&[1.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);
        g.emotions.set(0, [1.0, 0.5]);
        g.tick(&[1.0], Some(0), [0.0, 0.0], 2, DecayMode::Apply);
        assert!(
            g.global_state().e1 != 0.0 || g.global_state().e2 != 0.0,
            "global state should be non-zero"
        );
    }

    #[test]
    fn emotional_tick_attentive_when_near_optimal() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        g.tick(&[1.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);
        let r = g.tick(&[1.0], Some(0), [0.0, 0.0], 2, DecayMode::Apply);
        assert!(r.attentive);
    }

    #[test]
    fn emotional_tick_not_attentive_when_far_from_optimal() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        g.tick(&[1.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);
        g.global = EmotionState { e1: 5.0, e2: 5.0 };
        let r = g.tick(&[1.0], Some(0), [0.0, 0.0], 2, DecayMode::Apply);
        assert!(!r.attentive);
    }

    #[test]
    fn emotional_compact_reindexes_emotion_store() {
        let cfg_grow = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 2.0,
                sigma_0: 0.001,
                p1_kernel: 3,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let cfg_normal = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 0.1,
                sigma_0: 0.001,
                p1_kernel: 3,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let mut g = TestGraph::new(HashEdgeStore::new(), cfg_grow.clone());

        g.tick(&[0.0], Some(0), [1.0, 0.0], 1, DecayMode::Apply);
        g.tick(&[10.0], Some(0), [0.0, 1.0], 2, DecayMode::Apply);
        g.tick(&[20.0], Some(0), [0.5, 0.5], 3, DecayMode::Apply);
        assert_eq!(g.kernel_count(), 3);

        let vars0 = g.emotion_vars(0);
        let vars2 = g.emotion_vars(2);

        g.cfg = cfg_normal;
        g.tick(&[10.0], Some(0), [0.0, 0.0], 5, DecayMode::Apply);

        let removed = g.compact();
        assert_eq!(removed, 1);
        assert_eq!(g.emotions.len(), 2);
        assert_eq!(g.emotions.len(), g.kernel_count());
        let _ = (vars0, vars2);
    }

    #[test]
    fn emotional_compact_invariant_holds() {
        let cfg_grow = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 2.0,
                sigma_0: 0.001,
                p1_kernel: 3,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let cfg_normal = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 0.1,
                sigma_0: 0.001,
                p1_kernel: 3,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let mut g = TestGraph::new(HashEdgeStore::new(), cfg_grow.clone());

        for i in 0..3u64 {
            g.tick(
                &[i as f64 * 10.0],
                Some(0),
                [0.0, 0.0],
                i + 1,
                DecayMode::Apply,
            );
        }
        g.cfg = cfg_normal;
        g.tick(&[10.0], Some(0), [0.0, 0.0], 5, DecayMode::Apply);
        g.compact();
        assert_eq!(g.emotions.len(), g.kernel_count());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn emotional_graph_serde_roundtrip() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        for i in 0..5u64 {
            g.tick(
                &[i as f64 * 0.1],
                Some(0),
                [0.5, -0.3],
                i + 1,
                DecayMode::Apply,
            );
        }
        let n_kernels = g.kernel_count();
        let global = g.global_state();
        let json = serde_json::to_string(&g).unwrap();
        let back: TestGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kernel_count(), n_kernels);
        assert!((back.global_state().e1 - global.e1).abs() < 1e-10);
        assert!((back.global_state().e2 - global.e2).abs() < 1e-10);
        for i in 0..n_kernels {
            let a = back.emotion_vars(i);
            let b = g.emotion_vars(i);
            assert!((a[0] - b[0]).abs() < 1e-10, "kernel {i} e1 mismatch");
            assert!((a[1] - b[1]).abs() < 1e-10, "kernel {i} e2 mismatch");
        }
    }

    // --- new tests ---

    #[test]
    fn tick_class_none_emotion_vars_still_update() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        let r = g.tick(&[1.0], None, [1.0, 0.5], 1, DecayMode::Apply);
        let vars = g.emotion_vars(r.kernel.activated_kernel);
        assert!(vars[0] > 0.0);
        assert!(vars[1] > 0.0);
    }

    #[test]
    fn tick_e_target_zero_vars_decay_toward_zero() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        g.tick(&[1.0], Some(0), [1.0, 1.0], 1, DecayMode::Apply);
        // manually set vars to non-zero
        g.emotions.set(0, [0.5, 0.5]);
        let r = g.tick(&[1.0], Some(0), [0.0, 0.0], 2, DecayMode::Apply);
        let vars = g.emotion_vars(r.kernel.activated_kernel);
        assert!(vars[0] < 0.5, "e1 should move toward 0");
        assert!(vars[1] < 0.5, "e2 should move toward 0");
    }

    #[test]
    fn tick_successive_same_input_vars_converge() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        let target = [2.0, -1.0];
        for i in 0..50u64 {
            g.tick(&[0.5], Some(0), target, i + 1, DecayMode::Apply);
        }
        let vars = g.emotion_vars(0);
        assert!((vars[0] - 2.0).abs() < 0.5, "e1 should converge toward 2.0");
        assert!(
            (vars[1] - (-1.0)).abs() < 0.5,
            "e2 should converge toward -1.0"
        );
    }

    #[test]
    fn tick_with_decay_policy_global_is_clamped() {
        let cfg = EmotionalGraphConfig {
            sokm: sokm_cfg(),
            kernel: kernel_cfg(),
            emotion: EmotionConfig::default(),
        };
        let mut g: DecayGraph = EmotionalKernelGraph::with_store(
            HashEdgeStore::new(),
            DefaultKernelStore::default(),
            DecayPolicy::default(),
            cfg,
        );
        // drive with large e_target to saturate; set vars high after first kernel born
        for i in 0..20u64 {
            g.tick(&[0.5], Some(0), [3.0, 2.0], i + 1, DecayMode::Apply);
            if g.kernel_count() > 0 {
                g.emotions.set(0, [3.0, 2.0]);
            }
        }
        let gs = g.global_state();
        assert!(gs.e1 <= 3.0 && gs.e1 >= -3.0, "e1 must be clamped");
        assert!(gs.e2 <= 2.0 && gs.e2 >= -2.0, "e2 must be clamped");
    }

    #[test]
    fn compact_no_extinct_returns_zero() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        g.tick(&[1.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);
        let removed = g.compact();
        assert_eq!(removed, 0);
    }

    #[test]
    fn compact_empty_no_panic() {
        let mut g = TestGraph::new(HashEdgeStore::new(), graph_cfg());
        let removed = g.compact();
        assert_eq!(removed, 0);
    }

    #[test]
    fn compact_with_map_returns_correct_map() {
        let cfg_grow = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 2.0,
                sigma_0: 0.001,
                p1_kernel: 3,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let cfg_normal = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 0.1,
                sigma_0: 0.001,
                p1_kernel: 3,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let mut g = TestGraph::new(HashEdgeStore::new(), cfg_grow.clone());
        g.tick(&[0.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);
        g.tick(&[10.0], Some(0), [0.0, 0.0], 2, DecayMode::Apply);
        g.cfg = cfg_normal;
        g.tick(&[10.0], Some(0), [0.0, 0.0], 5, DecayMode::Apply);
        let map = g.compact_with_map();
        // at least one extinct
        assert!(map.iter().any(|m| m.is_none()));
        assert_eq!(g.emotions.len(), g.kernel_count());
    }

    #[test]
    fn post_compact_tick_works() {
        let cfg_grow = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 2.0,
                sigma_0: 0.001,
                p1_kernel: 3,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let cfg_normal = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 0.1,
                sigma_0: 0.001,
                p1_kernel: 3,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let mut g = TestGraph::new(HashEdgeStore::new(), cfg_grow.clone());
        g.tick(&[0.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);
        g.tick(&[10.0], Some(0), [0.0, 0.0], 2, DecayMode::Apply);
        g.cfg = cfg_normal;
        g.tick(&[10.0], Some(0), [0.0, 0.0], 5, DecayMode::Apply);
        g.compact();
        // tick after compact must not panic
        let r = g.tick(&[10.0], Some(0), [0.0, 0.0], 6, DecayMode::Apply);
        assert_eq!(g.emotions.len(), g.kernel_count());
        let _ = r;
    }

    #[test]
    fn tick_multiple_fire_only_activated_trains_vars() {
        let cfg_grow = EmotionalGraphConfig {
            kernel: KernelConfig {
                theta_k: 2.0,
                sigma_0: 0.01,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        let mut g = TestGraph::new(HashEdgeStore::new(), cfg_grow.clone());
        g.tick(&[0.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);
        g.tick(&[100.0], Some(0), [0.0, 0.0], 2, DecayMode::Apply);
        g.emotions.set(0, [0.0, 0.0]);
        g.emotions.set(1, [0.0, 0.0]);
        let normal = EmotionalGraphConfig {
            kernel: KernelConfig {
                sigma_0: 0.01,
                ..kernel_cfg()
            },
            ..graph_cfg()
        };
        g.cfg = normal;
        let r = g.tick(&[0.0], Some(0), [1.0, 0.5], 3, DecayMode::Apply);
        let activated = r.kernel.activated_kernel;
        let other = 1 - activated;
        // activated kernel should have non-zero vars
        assert!(g.emotion_vars(activated)[0] > 0.0);
        // other kernel unchanged
        assert_eq!(g.emotion_vars(other), [0.0, 0.0]);
    }

    #[test]
    fn with_store_vs_new_parity() {
        use sokm_kernel::store::DefaultKernelStore;
        let cfg = graph_cfg();
        let mut g1 = TestGraph::new(HashEdgeStore::new(), cfg.clone());
        let mut g2: EmotionalKernelGraph<HashEdgeStore<usize>> = EmotionalKernelGraph::with_store(
            HashEdgeStore::new(),
            DefaultKernelStore::default(),
            IdentityPolicy,
            cfg,
        );
        g1.tick(&[1.0], Some(0), [0.5, -0.3], 1, DecayMode::Apply);
        g2.tick(&[1.0], Some(0), [0.5, -0.3], 1, DecayMode::Apply);
        assert_eq!(g1.kernel_count(), g2.kernel_count());
        assert_eq!(g1.emotion_vars(0), g2.emotion_vars(0));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_with_decay_policy() {
        let cfg = EmotionalGraphConfig {
            sokm: sokm_cfg(),
            kernel: kernel_cfg(),
            emotion: EmotionConfig::default(),
        };
        let mut g: DecayGraph = EmotionalKernelGraph::with_store(
            HashEdgeStore::new(),
            DefaultKernelStore::default(),
            DecayPolicy::default(),
            cfg,
        );
        for i in 0..3u64 {
            g.tick(
                &[i as f64 * 0.1],
                Some(0),
                [0.5, -0.3],
                i + 1,
                DecayMode::Apply,
            );
        }
        let json = serde_json::to_string(&g).unwrap();
        let back: DecayGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kernel_count(), g.kernel_count());
        assert!((back.global_state().e1 - g.global_state().e1).abs() < 1e-10);
    }
}

#[cfg(test)]
mod configured_tests {
    use super::*;
    use crate::config::{EmotionConfig, EmotionalGraphConfig};
    use crate::policy::IdentityPolicy;
    use sokm::SparseEdgeStore;
    use sokm_kernel::store::DefaultKernelStore;

    type ConfiguredTestGraph =
        EmotionalKernelGraph<SparseEdgeStore, DefaultKernelStore, IdentityPolicy>;

    fn test_cfg() -> EmotionalGraphConfig {
        EmotionalGraphConfig {
            sokm: sokm::SokmConfig::default(),
            kernel: sokm_kernel::KernelConfig::default(),
            emotion: EmotionConfig::default(),
        }
    }

    #[test]
    fn tick_grows_on_first_call() {
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), test_cfg());
        assert_eq!(g.kernel_count(), 0);
        let report = g.tick(&[0.5, 0.5], None, [0.0, 0.0], 0, DecayMode::Apply);
        assert!(report.kernel.grew);
        assert_eq!(g.kernel_count(), 1);
    }

    #[test]
    fn tick_attentive_true_at_default_config() {
        // Default optimal = EmotionState::default() = [0,0], theta_e = 1.0.
        // After one tick with e_target = [0,0], global remains near [0,0].
        // Distance = 0.0 < theta_e = 1.0 → attentive = true.
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), test_cfg());
        let report = g.tick(&[0.5, 0.5], None, [0.0, 0.0], 0, DecayMode::Apply);
        assert!(
            report.attentive,
            "distance 0 from optimal must be attentive"
        );
    }

    #[test]
    fn tick_not_attentive_when_far_from_optimal() {
        // Use lambda_e = 1.0 so kernel vars jump fully to e_target on first tick.
        // Drive many ticks so global accumulates. theta_e very tight.
        let mut cfg = test_cfg();
        cfg.emotion.lambda_e = 1.0;
        cfg.emotion.theta_e = 0.001; // extremely tight band
        cfg.emotion.optimal = EmotionState::default(); // [0,0]
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), cfg);
        // Repeatedly tick with same input to let global accumulate
        let mut last_report = g.tick(&[0.5, 0.5], None, [3.0, 2.0], 0, DecayMode::Apply);
        for t in 1..20 {
            last_report = g.tick(&[0.5, 0.5], None, [3.0, 2.0], t, DecayMode::Apply);
        }
        // After many ticks with high e_target, global must be far from [0,0]
        assert!(
            !last_report.attentive,
            "global far from optimal after many ticks must not be attentive"
        );
    }

    #[test]
    fn cfg_accessor_returns_stored_config() {
        let cfg = test_cfg();
        let g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), cfg.clone());
        assert_eq!(g.cfg().emotion.lambda_e, cfg.emotion.lambda_e);
    }

    #[test]
    fn kernel_count_edge_count_stm_len_delegate() {
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), test_cfg());
        assert_eq!(g.kernel_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.stm_len(), 0);
        g.tick(&[0.5, 0.5], None, [0.0, 0.0], 0, DecayMode::Apply);
        assert_eq!(g.kernel_count(), 1);
    }

    #[test]
    fn compact_with_map_no_op_on_empty() {
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), test_cfg());
        let map = g.compact_with_map();
        assert!(map.is_empty());
    }

    #[test]
    fn compact_with_map_removes_extinct_kernel() {
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), test_cfg());
        g.tick(&[0.5, 0.5], None, [0.0, 0.0], 0, DecayMode::Apply);
        assert_eq!(g.kernel_count(), 1);

        g.mark_extinct(0);
        let map = g.compact_with_map();

        assert_eq!(map.len(), 1);
        assert_eq!(map[0], None);
        assert_eq!(g.kernel_count(), 0);
    }

    #[test]
    fn salience_scores_empty_when_alpha_zero() {
        // Default EmotionConfig has alpha = 0.0
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), test_cfg());
        let report = g.tick(&[0.5, 0.5], None, [0.0, 0.0], 0, DecayMode::Apply);
        assert!(
            report.salience_scores.is_empty(),
            "alpha=0.0 must produce empty salience_scores (no allocation)"
        );
    }

    #[test]
    fn salience_scores_populated_when_alpha_nonzero() {
        let mut cfg = test_cfg();
        cfg.emotion.alpha = 1.0;
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), cfg);

        let report = g.tick(&[0.5, 0.5], None, [0.0, 0.0], 0, DecayMode::Apply);

        assert_eq!(
            report.salience_scores.len(),
            g.kernel_count(),
            "salience_scores length must equal kernel_count()"
        );
        for &s in &report.salience_scores {
            assert!(s >= 0.0, "salience score must be non-negative, got {s}");
        }
    }

    #[test]
    fn salience_scores_len_matches_kernel_count_after_growth() {
        let mut cfg = test_cfg();
        cfg.emotion.alpha = 0.5;
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), cfg);
        g.tick(&[0.0, 0.0], Some(0), [0.0, 0.0], 0, DecayMode::Apply);
        let report = g.tick(&[3.0, 0.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);

        assert_eq!(g.kernel_count(), 2);
        assert_eq!(
            report.salience_scores.len(),
            2,
            "salience_scores must have one entry per kernel after growth"
        );
    }

    #[test]
    fn propagate_soft_with_cfg_uses_explicit_cfg_not_stored() {
        let stored_cfg = test_cfg();
        let mut g = ConfiguredTestGraph::new(SparseEdgeStore::new(0), stored_cfg.clone());

        g.tick(&[0.0, 0.0], Some(0), [0.0, 0.0], 0, DecayMode::Apply);
        g.tick(&[3.0, 0.0], Some(0), [0.0, 0.0], 1, DecayMode::Apply);
        assert_eq!(g.kernel_count(), 2);

        g.tick(&[1.5, 0.0], Some(0), [0.0, 0.0], 2, DecayMode::Apply);
        assert!(
            g.edge_count() > 0,
            "edges must exist for propagation to differ"
        );

        let x = &[1.5, 0.0f64];

        let stored_result = g.propagate_soft(x);
        assert!(
            !stored_result.is_empty(),
            "propagate_soft must return scores when edges exist"
        );

        let mut low_gamma_cfg = stored_cfg.sokm.clone();
        low_gamma_cfg.gamma = 0.001;
        let explicit_result = g.propagate_soft_with_cfg(x, &low_gamma_cfg);

        assert_ne!(
            stored_result, explicit_result,
            "propagate_soft_with_cfg must use the passed cfg, not stored cfg"
        );
    }
}
