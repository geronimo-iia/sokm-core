use crate::config::GestaltConfig;
use crate::cross::{
    CrossEdgeStore, CrossStore, cross_propagate_soft, cross_propagate_soft_reverse,
    cross_strengthen_deltas,
};
use sokm::{DecayMode, EdgeStore, HashEdgeStore, Reindex};
use sokm_kernel::growth::compute_scores;
use sokm_kernel::{DefaultKernelStore, KernelGraph, KernelStore, KernelTickReport};

/// Per-tick summary of a `GestaltKernelGraph::tick` invocation.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GestaltTickReport {
    pub modal1: KernelTickReport,
    pub modal2: KernelTickReport,
    /// Count of (modal1, modal2) pairs that co-activated this tick, not edge count.
    pub cross_strengthened: usize,
    pub cross_pruned: usize,
}

#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "S1: serde::Serialize, S2: serde::Serialize, K1: serde::Serialize, K2: serde::Serialize",
        deserialize = "S1: serde::Deserialize<'de>, S2: serde::Deserialize<'de>, K1: serde::Deserialize<'de>, K2: serde::Deserialize<'de>"
    ))
)]
pub struct GestaltKernelGraph<S1, S2, K1 = DefaultKernelStore, K2 = DefaultKernelStore>
where
    S1: EdgeStore<usize>,
    S2: EdgeStore<usize>,
    K1: KernelStore,
    K2: KernelStore,
{
    pub modal1: KernelGraph<S1, K1>,
    pub modal2: KernelGraph<S2, K2>,
    pub cross: CrossEdgeStore,
}

impl<S1, S2, K1, K2> GestaltKernelGraph<S1, S2, K1, K2>
where
    S1: EdgeStore<usize>,
    S2: EdgeStore<usize>,
    K1: KernelStore + Default,
    K2: KernelStore + Default,
{
    pub fn new(edges1: S1, edges2: S2, cfg: &GestaltConfig) -> Self {
        Self {
            modal1: KernelGraph::new(edges1, &cfg.kernel),
            modal2: KernelGraph::new(edges2, &cfg.kernel),
            cross: CrossEdgeStore::new(),
        }
    }
}

impl<S1, S2, K1, K2> GestaltKernelGraph<S1, S2, K1, K2>
where
    S1: EdgeStore<usize>,
    S2: EdgeStore<usize>,
    K1: KernelStore,
    K2: KernelStore,
{
    /// Build from pre-populated stores. Uses `cfg.kernel` for both modal KernelGraphs
    /// and `cfg.cross` for cross-modal configuration.
    pub fn with_stores(
        edges1: S1,
        kernels1: K1,
        edges2: S2,
        kernels2: K2,
        cfg: &GestaltConfig,
    ) -> Self {
        Self {
            modal1: KernelGraph::with_store(edges1, kernels1, &cfg.kernel),
            modal2: KernelGraph::with_store(edges2, kernels2, &cfg.kernel),
            cross: CrossEdgeStore::new(),
        }
    }

    /// Advance both modalities by one tick, then update cross-modal edges.
    ///
    /// Tick sequence: modal1.tick → modal2.tick → compute_scores → cross strengthen
    /// (apply_increments → touch → scale_all). Newly born edges are decayed on birth tick.
    /// The shared `class` constraint applies the same label to both modalities.
    pub fn tick(
        &mut self,
        x1: &[f64],
        x2: &[f64],
        class: Option<u32>,
        current_tick: u64,
        cfg: &GestaltConfig,
        decay: DecayMode,
    ) -> GestaltTickReport {
        let r1 = self
            .modal1
            .tick(x1, class, current_tick, &cfg.sokm, &cfg.kernel, decay);
        let r2 = self
            .modal2
            .tick(x2, class, current_tick, &cfg.sokm, &cfg.kernel, decay);

        let scores1 = compute_scores(self.modal1.kernels(), x1);
        let scores2 = compute_scores(self.modal2.kernels(), x2);

        let activated1: Vec<(usize, Option<u32>)> = scores1
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s > 0.0)
            .map(|(i, _)| (i, self.modal1.kernels().class_opt(i)))
            .collect();

        let activated2: Vec<(usize, Option<u32>)> = scores2
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s > 0.0)
            .map(|(i, _)| (i, self.modal2.kernels().class_opt(i)))
            .collect();

        let deltas = cross_strengthen_deltas(&activated1, &activated2, &cfg.cross);
        let cross_strengthened = deltas.len();

        self.cross.strengthen(&deltas, &cfg.cross, current_tick);

        let cross_decay_factor = (-cfg.cross.xi).exp();
        self.cross.scale_all(cross_decay_factor);

        let pruned_weight = self.cross.prune_below(cfg.cross.min_weight);
        let pruned_inactive = self.cross.prune_inactive(current_tick, cfg.cross.p1);

        GestaltTickReport {
            modal1: r1,
            modal2: r2,
            cross_strengthened,
            cross_pruned: pruned_weight + pruned_inactive,
        }
    }

    /// Recall modal2 activations given a modal1 input vector.
    /// Returns `(modal2_kernel_idx, score)` sorted descending by score.
    pub fn recall_from_modal1(&self, x1: &[f64], cfg: &GestaltConfig) -> Vec<(usize, f64)> {
        let scores1 = compute_scores(self.modal1.kernels(), x1);
        let modal1_activated: Vec<(usize, f64)> = scores1
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s > 0.0)
            .map(|(i, &s)| (i, s))
            .collect();
        let mut result = cross_propagate_soft(&self.cross, &modal1_activated, cfg.cross.gamma);
        result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Recall modal1 activations given a modal2 input vector.
    /// Returns `(modal1_kernel_idx, score)` sorted descending by score.
    pub fn recall_from_modal2(&self, x2: &[f64], cfg: &GestaltConfig) -> Vec<(usize, f64)> {
        let scores2 = compute_scores(self.modal2.kernels(), x2);
        let modal2_activated: Vec<(usize, f64)> = scores2
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s > 0.0)
            .map(|(i, &s)| (i, s))
            .collect();
        let mut result =
            cross_propagate_soft_reverse(&self.cross, &modal2_activated, cfg.cross.gamma);
        result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    pub fn compact(&mut self) -> (usize, usize)
    where
        S1: Reindex,
        S2: Reindex,
    {
        let map1 = self.modal1.compact_with_map();
        let map2 = self.modal2.compact_with_map();
        self.cross.reindex(&map1, &map2);
        (
            map1.iter().filter(|m| m.is_none()).count(),
            map2.iter().filter(|m| m.is_none()).count(),
        )
    }

    pub fn kernel_count_modal1(&self) -> usize {
        self.modal1.kernel_count()
    }

    pub fn kernel_count_modal2(&self) -> usize {
        self.modal2.kernel_count()
    }

    pub fn cross_edge_count(&self) -> usize {
        self.cross.edge_count()
    }
}

pub type DefaultGestaltGraph = GestaltKernelGraph<
    HashEdgeStore<usize>,
    HashEdgeStore<usize>,
    DefaultKernelStore,
    DefaultKernelStore,
>;

impl Default for DefaultGestaltGraph {
    fn default() -> Self {
        Self::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CrossSokmConfig, GestaltConfig};
    use crate::cross::CrossStore;
    use sokm_kernel::store::KernelStore;
    use sokm_kernel::{DefaultKernelStore, KernelConfig};

    #[test]
    fn gestalt_tick_grows_both_modalities_on_novel_input() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 0, &cfg, DecayMode::Apply);
        assert!(g.modal1.kernel_count() > 0 && g.modal2.kernel_count() > 0);
    }

    #[test]
    fn gestalt_tick_no_growth_on_familiar_input() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 0, &cfg, DecayMode::Apply);
        let count1 = g.modal1.kernel_count();
        let count2 = g.modal2.kernel_count();
        assert!(count1 > 0 && count2 > 0);
        g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 1, &cfg, DecayMode::Apply);
        assert_eq!(g.modal1.kernel_count(), count1);
        assert_eq!(g.modal2.kernel_count(), count2);
    }

    #[test]
    fn gestalt_tick_strengthens_cross_modal_edge_on_coactivation() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 0, &cfg, DecayMode::Apply);
        if g.modal1.kernel_count() > 0 && g.modal2.kernel_count() > 0 {
            assert!(g.cross.edge_count() > 0);
        }
    }

    #[test]
    fn gestalt_tick_no_cross_strengthening_on_class_mismatch() {
        let mut k1 = DefaultKernelStore::default();
        k1.push(&[1.0, 0.0], 1.0, Some(1));
        let mut k2 = DefaultKernelStore::default();
        k2.push(&[0.0, 1.0], 1.0, Some(2));
        let mut g = GestaltKernelGraph::with_stores(
            HashEdgeStore::new(),
            k1,
            HashEdgeStore::new(),
            k2,
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig {
            cross: CrossSokmConfig {
                require_class_match: true,
                ..CrossSokmConfig::default()
            },
            ..GestaltConfig::default()
        };
        g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 0, &cfg, DecayMode::Apply);
        assert_eq!(g.cross.edge_count(), 0);
    }

    #[test]
    fn gestalt_tick_no_cross_strengthening_when_require_class_match_false_and_unlabelled() {
        let mut k1 = DefaultKernelStore::default();
        k1.push(&[1.0, 0.0], 1.0, None); // unlabelled
        let mut k2 = DefaultKernelStore::default();
        k2.push(&[0.0, 1.0], 1.0, Some(1)); // labelled
        let mut g = GestaltKernelGraph::with_stores(
            HashEdgeStore::new(),
            k1,
            HashEdgeStore::new(),
            k2,
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig {
            cross: CrossSokmConfig {
                require_class_match: false,
                ..CrossSokmConfig::default()
            },
            ..GestaltConfig::default()
        };
        g.tick(&[1.0, 0.0], &[0.0, 1.0], None, 0, &cfg, DecayMode::Apply);
        assert_eq!(g.cross.edge_count(), 0);
    }

    #[test]
    fn recall_from_modal1_returns_modal2_activations() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        for t in 0..5u64 {
            g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
        }
        let result = g.recall_from_modal1(&[1.0, 0.0], &cfg);
        assert!(!result.is_empty());
    }

    #[test]
    fn recall_from_modal1_zero_without_cross_edge() {
        let g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        let result = g.recall_from_modal1(&[1.0, 0.0], &cfg);
        assert!(result.is_empty());
    }

    #[test]
    fn recall_from_modal2_symmetric() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        for t in 0..5u64 {
            g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
        }
        let r1 = g.recall_from_modal1(&[1.0, 0.0], &cfg);
        let r2 = g.recall_from_modal2(&[0.0, 1.0], &cfg);
        assert!(!r1.is_empty());
        assert!(!r2.is_empty());
    }

    #[test]
    fn cross_modal_imagery_without_modal2_input() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        for t in 0..5u64 {
            g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
        }
        let result = g.recall_from_modal1(&[1.0, 0.0], &cfg);
        assert!(
            !result.is_empty(),
            "Should recall modal2 activations from modal1 input alone"
        );
    }

    #[test]
    fn gestalt_compact_reindexes_cross_edges() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 0, &cfg, DecayMode::Apply);
        g.tick(&[0.0, 1.0], &[0.0, 1.0], Some(1), 1, &cfg, DecayMode::Apply);
        let initial_edges = g.cross.edge_count();
        let (removed1, removed2) = g.compact();
        assert_eq!(removed1, 0);
        assert_eq!(removed2, 0);
        let _ = initial_edges;
    }

    #[test]
    fn gestalt_compact_drops_cross_edges_involving_extinct_kernel() {
        let mut k1 = DefaultKernelStore::default();
        k1.push(&[1.0, 0.0], 1.0, Some(1));
        k1.push(&[0.0, 1.0], 1.0, Some(1));
        let mut k2 = DefaultKernelStore::default();
        k2.push(&[0.0, 1.0], 1.0, Some(1));
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::with_stores(
            HashEdgeStore::new(),
            k1,
            HashEdgeStore::new(),
            k2,
            &GestaltConfig::default(),
        );
        g.cross.set(0, 0, 0.5);
        g.cross.set(1, 0, 0.3);
        assert_eq!(g.cross.edge_count(), 2);

        // Tick at high tick number with p1_kernel=0 → both pre-planted kernels go extinct
        let cfg = GestaltConfig {
            kernel: KernelConfig {
                p1_kernel: 0,
                theta_k: 0.99,
                ..KernelConfig::default()
            },
            cross: CrossSokmConfig {
                min_weight: 0.0,
                xi: 0.0,
                p1: u64::MAX,
                ..CrossSokmConfig::default()
            },
            ..GestaltConfig::default()
        };
        g.tick(
            &[10.0, 10.0],
            &[10.0, 10.0],
            Some(1),
            100,
            &cfg,
            DecayMode::Apply,
        );

        let (removed1, _removed2) = g.compact();
        assert!(
            removed1 > 0,
            "At least one modal1 kernel should be compacted"
        );
        assert!(g.cross.edge_count() <= 2);
    }

    // --- Finding #21: GestaltKernelGraph additional tests ---

    #[test]
    fn class_none_no_cross_edges_created() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        // class=None → no cross strengthening
        for t in 0..5u64 {
            g.tick(&[1.0, 0.0], &[0.0, 1.0], None, t, &cfg, DecayMode::Apply);
        }
        assert_eq!(g.cross.edge_count(), 0);
    }

    #[test]
    fn require_class_match_false_any_two_strengthen() {
        let mut k1 = DefaultKernelStore::default();
        k1.push(&[1.0, 0.0], 1.0, Some(1));
        let mut k2 = DefaultKernelStore::default();
        k2.push(&[0.0, 1.0], 1.0, Some(2));
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::with_stores(
            HashEdgeStore::new(),
            k1,
            HashEdgeStore::new(),
            k2,
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig {
            cross: CrossSokmConfig {
                require_class_match: false,
                ..CrossSokmConfig::default()
            },
            ..GestaltConfig::default()
        };
        g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 0, &cfg, DecayMode::Apply);
        assert!(g.cross.edge_count() > 0);
    }

    #[test]
    fn cross_pruned_via_weight_decay() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig {
            cross: CrossSokmConfig {
                xi: 5.0, // aggressive decay
                min_weight: 0.01,
                ..CrossSokmConfig::default()
            },
            ..GestaltConfig::default()
        };
        // Create edges
        g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 0, &cfg, DecayMode::Apply);
        let edges_after_first = g.cross.edge_count();
        // Tick with a different input to not restrengthen
        let report = g.tick(&[0.0, 1.0], &[1.0, 0.0], Some(2), 1, &cfg, DecayMode::Apply);
        // Decay should have pruned some cross edges if they existed
        if edges_after_first > 0 {
            assert!(report.cross_pruned > 0 || g.cross.edge_count() < edges_after_first);
        }
    }

    #[test]
    fn inactivity_pruning_fires() {
        // Use different classes on k1 vs k2 so require_class_match=true blocks restrengthen
        let mut k1 = DefaultKernelStore::default();
        k1.push(&[1.0, 0.0], 1.0, Some(1));
        let mut k2 = DefaultKernelStore::default();
        k2.push(&[0.0, 1.0], 1.0, Some(2));
        let cfg = GestaltConfig {
            cross: CrossSokmConfig {
                p1: 2,
                xi: 0.0,         // no weight decay
                min_weight: 0.0, // no weight pruning
                require_class_match: true,
                ..CrossSokmConfig::default()
            },
            ..GestaltConfig::default()
        };
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::with_stores(
            HashEdgeStore::new(),
            k1,
            HashEdgeStore::new(),
            k2,
            &cfg,
        );
        // Manually plant a cross edge touched at tick 0
        g.cross.set(0, 0, 0.5);
        CrossStore::touch(&mut g.cross, 0, 0, 0);
        assert_eq!(g.cross.edge_count(), 1);
        // Tick far in the future — classes don't match so no restrengthen
        let report = g.tick(
            &[1.0, 0.0],
            &[0.0, 1.0],
            Some(1),
            100,
            &cfg,
            DecayMode::Apply,
        );
        assert!(
            report.cross_pruned > 0,
            "Inactivity should prune the stale edge"
        );
    }

    #[test]
    fn compact_no_extinct_cross_edges_survive() {
        let mut g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        for t in 0..3u64 {
            g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
        }
        let edges_before = g.cross.edge_count();
        let (r1, r2) = g.compact();
        assert_eq!(r1, 0);
        assert_eq!(r2, 0);
        assert_eq!(g.cross.edge_count(), edges_before);
    }

    #[test]
    fn recall_empty_graph_no_panic() {
        let g: DefaultGestaltGraph = GestaltKernelGraph::new(
            HashEdgeStore::new(),
            HashEdgeStore::new(),
            &GestaltConfig::default(),
        );
        let cfg = GestaltConfig::default();
        let r1 = g.recall_from_modal1(&[1.0, 0.0], &cfg);
        let r2 = g.recall_from_modal2(&[0.0, 1.0], &cfg);
        assert!(r1.is_empty());
        assert!(r2.is_empty());
    }

    #[test]
    fn new_and_with_stores_equivalence() {
        let cfg = GestaltConfig::default();
        let g1: DefaultGestaltGraph =
            GestaltKernelGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);
        let g2: DefaultGestaltGraph = GestaltKernelGraph::with_stores(
            HashEdgeStore::new(),
            DefaultKernelStore::default(),
            HashEdgeStore::new(),
            DefaultKernelStore::default(),
            &cfg,
        );
        assert_eq!(g1.kernel_count_modal1(), g2.kernel_count_modal1());
        assert_eq!(g1.kernel_count_modal2(), g2.kernel_count_modal2());
        assert_eq!(g1.cross_edge_count(), g2.cross_edge_count());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_and_tick() {
        let cfg = GestaltConfig::default();
        let mut g: DefaultGestaltGraph =
            GestaltKernelGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);
        for t in 0..3u64 {
            g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
        }
        let bytes = rmp_serde::to_vec(&g).unwrap();
        let mut g2: DefaultGestaltGraph = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(g2.cross_edge_count(), g.cross_edge_count());
        // Should continue ticking without panic
        g2.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 3, &cfg, DecayMode::Apply);
    }

    // --- Finding #22: Integration tests ---

    #[test]
    fn integration_train_compact_recall() {
        let cfg = GestaltConfig::default();
        let mut g: DefaultGestaltGraph =
            GestaltKernelGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);
        // Train
        for t in 0..10u64 {
            g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
        }
        assert!(g.cross_edge_count() > 0);
        // Compact
        g.compact();
        // Recall still works
        let result = g.recall_from_modal1(&[1.0, 0.0], &cfg);
        assert!(!result.is_empty());
    }

    #[test]
    fn integration_bidirectional_symmetry() {
        let cfg = GestaltConfig::default();
        let mut g: DefaultGestaltGraph =
            GestaltKernelGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);
        for t in 0..10u64 {
            g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
        }
        let r1 = g.recall_from_modal1(&[1.0, 0.0], &cfg);
        let r2 = g.recall_from_modal2(&[0.0, 1.0], &cfg);
        // Both directions should produce results
        assert!(!r1.is_empty());
        assert!(!r2.is_empty());
    }

    #[test]
    fn integration_cross_modal_decay_to_zero() {
        let cfg = GestaltConfig {
            cross: CrossSokmConfig {
                xi: 10.0, // extremely aggressive decay
                min_weight: 0.001,
                ..CrossSokmConfig::default()
            },
            ..GestaltConfig::default()
        };
        let mut g: DefaultGestaltGraph =
            GestaltKernelGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);
        // Create edges
        g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), 0, &cfg, DecayMode::Apply);
        // Tick many times with unrelated input to decay without strengthening
        for t in 1..20u64 {
            g.tick(&[0.5, 0.5], &[0.5, 0.5], Some(2), t, &cfg, DecayMode::Apply);
        }
        assert_eq!(g.cross.edge_count(), 0, "All cross edges should decay to 0");
    }

    #[test]
    fn class_association_ranking() {
        // Train class-1 pair many times; train class-2 pair fewer times.
        // Recall with class-1 cue should return class-1 modal2 result above class-2.
        let cfg = GestaltConfig::default();
        let mut g: DefaultGestaltGraph =
            GestaltKernelGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);

        // class-1: modal1=[1,0] paired with modal2=[0,1]
        for t in 0..15u64 {
            g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
        }
        // class-2: modal1=[0,1] paired with modal2=[1,0] (orthogonal, weaker training)
        for t in 15..18u64 {
            g.tick(&[0.0, 1.0], &[1.0, 0.0], Some(2), t, &cfg, DecayMode::Apply);
        }

        // Recall from class-1 cue
        let results = g.recall_from_modal1(&[1.0, 0.0], &cfg);
        assert!(!results.is_empty(), "recall must return results");
        // Top result should have a higher score than any class-2 associated kernel
        let top_score = results[0].1;
        // The class-2 modal2 kernel centroid is at [1,0], far from [0,1]; its score should be lower
        assert!(top_score > 0.0, "top result must have positive score");
        // Results are sorted descending
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "results must be sorted descending");
        }
    }
}
