use std::collections::HashMap;
use std::hash::Hash;

use crate::config::SokmConfig;
use crate::store::EdgeStore;

/// Outcome of one sokm tick cycle.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SokmReport {
    /// Total live edge count after decay.
    pub edges_alive: usize,
    pub strengthened: usize,
    pub pruned: usize,
}

/// Controls whether decay is applied during a `tick` cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecayMode {
    #[default]
    Apply,
    Skip,
}

/// Decay all edge weights by exp(-xi). [Hoya Eq 4.1]
pub fn decay<K: Copy + Eq + Hash + Ord>(store: &mut impl EdgeStore<K>, cfg: &SokmConfig) -> usize {
    let factor = (-cfg.xi).exp();
    store.scale_all(factor);
    store.edge_count()
}

/// Strengthen edges between co-activated nodes. [Hoya Eqs 4.6-4.7]
///
/// # Class constraint
/// Class-agnostic: callers MUST pre-filter `activated` to same-class pairs.
/// Passing cross-class pairs is not an error here — enforcement is the
/// caller's responsibility (e.g. `KernelGraph::tick` in `sokm-kernel`).
///
/// # INVARIANT: same-class filtering
/// This function never inspects kernel class labels — it has no access to them.
/// The caller (KernelGraph::tick) is solely responsible for ensuring that
/// `activated` contains only kernels of the same class as the current input.
/// Passing unlabelled kernels (class = None) or cross-class pairs produces
/// incorrect Hebbian strengthening with no error or warning.
pub fn strengthen<K: Copy + Eq + Hash + Ord>(
    store: &mut impl EdgeStore<K>,
    activated: &[(K, f64)],
    current_tick: u64,
    cfg: &SokmConfig,
) -> usize {
    if activated.len() < 2 {
        return 0;
    }
    let n = activated.len();
    let mut increments: Vec<(K, K, f64)> = Vec::with_capacity(n * (n - 1) / 2);
    let mut count = 0;

    for i in 0..activated.len() {
        for j in (i + 1)..activated.len() {
            let (a, score_a) = activated[i];
            let (b, score_b) = activated[j];
            let current = store.get_weight(a, b);
            let new_w = if current == 0.0 {
                cfg.w_init
            } else {
                (current + cfg.delta * score_a * score_b).min(cfg.w_max)
            };
            increments.push((a, b, new_w - current));
            store.touch(a, b, current_tick);
            count += 1;
        }
    }
    store.apply_increments(&increments);
    count
}

/// Prune edges in two phases:
/// 1. Weight-threshold: remove edges where `w < min_weight` (strict less-than).
/// 2. Inactivity extinction: remove edges where `(current_tick - last_active) > p1`
///    (strict greater-than — an edge active at exactly `last + p1` survives).
///
/// # INVARIANT: p1 boundary semantics
/// The inactivity condition is `> p1`, not `>= p1`. An edge touched at tick T
/// survives until `current_tick - T > p1`, i.e. it is alive for exactly p1
/// inactive ticks before being pruned on tick p1+1. This matches Hoya's
/// specification: extinction occurs *after* p1 ticks of inactivity.
pub fn prune<K: Copy + Eq + Hash + Ord>(
    store: &mut impl EdgeStore<K>,
    current_tick: u64,
    cfg: &SokmConfig,
) -> usize {
    let by_weight = store.prune_below(cfg.min_weight);
    let by_inactivity = store.prune_inactive(current_tick, cfg.p1);
    by_weight + by_inactivity
}

/// Spread activation through edges — binary form. [Hoya Eq 4.4]
///
/// Only kernels in `fired` contribute. Each sends a uniform signal: `gamma * w_ij`.
/// Use during construction (learning) — the same threshold that gates growth also
/// gates propagation. Caller is responsible for filtering to nodes >= theta_k.
pub fn propagate<K: Copy + Eq + Hash + Ord>(
    store: &impl EdgeStore<K>,
    fired: &[K],
    cfg: &SokmConfig,
) -> Vec<(K, f64)> {
    let mut spread: HashMap<K, f64> = HashMap::new();
    for &node in fired {
        for (neighbor, weight) in store.neighbors(node) {
            *spread.entry(neighbor).or_insert(0.0) += cfg.gamma * weight;
        }
    }
    spread.into_iter().collect()
}

/// Spread activation through edges — soft (graded) form. [Hoya Eq 4.3]
///
/// All nodes in `activated` contribute proportional to their score:
/// `gamma * w_ij * score_i`. Use during retrieval to obtain a graded similarity
/// landscape. Partial activations propagate — suited for query / recall passes.
pub fn propagate_soft<K: Copy + Eq + Hash + Ord>(
    store: &impl EdgeStore<K>,
    activated: &[(K, f64)],
    cfg: &SokmConfig,
) -> Vec<(K, f64)> {
    let mut spread: HashMap<K, f64> = HashMap::new();
    for &(node, score) in activated {
        for (neighbor, weight) in store.neighbors(node) {
            *spread.entry(neighbor).or_insert(0.0) += cfg.gamma * weight * score;
        }
    }
    spread.into_iter().collect()
}

/// One full SOKM cycle: decay -> strengthen -> prune. [Hoya Eqs 4.1, 4.6-4.7]
///
/// When `decay` is `DecayMode::Skip`, the decay step is skipped and `edges_alive` reflects
/// the current edge count without applying `exp(-xi)` attenuation.
/// `strengthen` and `prune` always run regardless of `decay`.
///
/// # Example
///
/// ```
/// use sokm::{HashEdgeStore, SokmConfig, DecayMode, tick};
/// use sokm::EdgeStore;
///
/// let mut store: HashEdgeStore<u32> = HashEdgeStore::new();
/// let cfg = SokmConfig::default();
/// let activated = vec![(0u32, 1.0), (1, 0.8)];
/// let report = tick(&mut store, &activated, 1, &cfg, DecayMode::Apply);
/// assert_eq!(report.strengthened, 1);
/// ```
pub fn tick<K: Copy + Eq + Hash + Ord>(
    store: &mut impl EdgeStore<K>,
    activated: &[(K, f64)],
    current_tick: u64,
    cfg: &SokmConfig,
    decay: DecayMode,
) -> SokmReport {
    let edges_alive = if decay == DecayMode::Skip {
        store.edge_count()
    } else {
        self::decay(store, cfg)
    };
    let strengthened = strengthen(store, activated, current_tick, cfg);
    let pruned = prune(store, current_tick, cfg);
    SokmReport {
        edges_alive,
        strengthened,
        pruned,
    }
}

// Utilities

/// Return top `n` nodes by activation score, sorted descending.
///
/// # Preconditions
/// All scores must be finite. NaN inputs violate this contract and produce
/// non-deterministic ordering (`partial_cmp` treats NaN as `Equal`).
pub fn top_n<K: Copy>(activated: &[(K, f64)], n: usize) -> Vec<(K, f64)> {
    debug_assert!(
        activated.iter().all(|(_, s)| s.is_finite()),
        "top_n: non-finite score in input"
    );
    let mut sorted = activated.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(n);
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SokmConfig;
    use crate::store::{EdgeStore, HashEdgeStore};

    fn cfg() -> SokmConfig {
        SokmConfig::default()
    }

    // --- decay ---

    #[test]
    fn decay_follows_eq_4_1() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        let cfg = cfg();
        decay(&mut s, &cfg);
        let expected = (-cfg.xi).exp();
        assert!((s.get_weight(0, 1) - expected).abs() < 1e-10);
    }

    #[test]
    fn decay_is_not_linear_subtraction() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        let cfg = SokmConfig {
            xi: 0.5,
            ..SokmConfig::default()
        };
        decay(&mut s, &cfg);
        let wrong = 1.0 - 0.5;
        let right = (-0.5f64).exp();
        let actual = s.get_weight(0, 1);
        assert!((actual - right).abs() < 1e-10);
        assert!((actual - wrong).abs() > 1e-6);
    }

    #[test]
    fn decay_two_successive_calls_compound() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        let cfg = cfg();
        decay(&mut s, &cfg);
        decay(&mut s, &cfg);
        let expected = (-2.0 * cfg.xi).exp();
        assert!((s.get_weight(0, 1) - expected).abs() < 1e-10);
    }

    #[test]
    fn decay_empty_store_no_panic() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        let count = decay(&mut s, &cfg());
        assert_eq!(count, 0);
    }

    // --- strengthen ---

    #[test]
    fn strengthen_initialises_new_link_at_w_init() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        let cfg = cfg();
        strengthen(&mut s, &[(0, 1.0), (1, 1.0)], 1, &cfg);
        assert!((s.get_weight(0, 1) - cfg.w_init).abs() < 1e-10);
    }

    #[test]
    fn strengthen_clamps_at_w_max() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.99);
        let cfg = SokmConfig {
            delta: 0.5,
            w_max: 1.0,
            ..SokmConfig::default()
        };
        strengthen(&mut s, &[(0, 1.0), (1, 1.0)], 1, &cfg);
        assert!((s.get_weight(0, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn strengthen_weighted_product_scales_with_scores() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.1);
        let cfg = SokmConfig {
            delta: 0.1,
            w_max: 10.0,
            ..SokmConfig::default()
        };
        strengthen(&mut s, &[(0, 0.5), (1, 0.8)], 1, &cfg);
        let expected = 0.1 + 0.1 * 0.5 * 0.8;
        assert!((s.get_weight(0, 1) - expected).abs() < 1e-10);
    }

    #[test]
    fn strengthen_updates_last_active_tick() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        let cfg = cfg();
        strengthen(&mut s, &[(0, 1.0), (1, 1.0)], 42, &cfg);
        assert_eq!(s.last_active(0, 1), 42);
    }

    #[test]
    fn strengthen_single_node_no_pairs() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        let count = strengthen(&mut s, &[(0, 1.0)], 1, &cfg());
        assert_eq!(count, 0);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn strengthen_score_zero_edge_unchanged() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        let cfg = cfg();
        // score_a * score_b = 0 -> delta contribution is 0
        strengthen(&mut s, &[(0, 0.0), (1, 1.0)], 1, &cfg);
        // new_w = (0.5 + delta * 0.0 * 1.0).min(w_max) = 0.5
        assert!((s.get_weight(0, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn strengthen_n4_creates_6_pairs() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        let cfg = cfg();
        let activated: Vec<(u32, f64)> = (0..4).map(|i| (i, 1.0)).collect();
        let count = strengthen(&mut s, &activated, 1, &cfg);
        assert_eq!(count, 6); // C(4,2) = 6
        assert_eq!(s.edge_count(), 6);
    }

    #[test]
    fn strengthen_duplicate_node_keys() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        let cfg = cfg();
        // Same key appears twice — forms pair with itself
        let count = strengthen(&mut s, &[(0, 1.0), (0, 1.0)], 1, &cfg);
        assert_eq!(count, 1);
        // Edge (0,0) is a self-loop
        assert!((s.get_weight(0, 0) - cfg.w_init).abs() < 1e-10);
    }

    // --- prune ---

    #[test]
    fn prune_removes_below_weight_threshold() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.set_weight(0, 2, 0.0005);
        let cfg = SokmConfig {
            min_weight: 0.001,
            ..SokmConfig::default()
        };
        let pruned = prune(&mut s, 1, &cfg);
        assert_eq!(pruned, 1);
        assert_eq!(s.edge_count(), 1);
    }

    #[test]
    fn prune_removes_inactive_edges_after_p1_ticks() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.set_weight(0, 2, 0.5);
        s.touch(0, 1, 150);
        let cfg = SokmConfig {
            p1: 100,
            ..SokmConfig::default()
        };
        let pruned = prune(&mut s, 200, &cfg);
        assert_eq!(pruned, 1);
        assert!(s.get_weight(0, 1) > 0.0);
        assert_eq!(s.get_weight(0, 2), 0.0);
    }

    #[test]
    fn prune_keeps_active_edge_above_p1() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.touch(0, 1, 150);
        let cfg = SokmConfig {
            p1: 100,
            ..SokmConfig::default()
        };
        let pruned = prune(&mut s, 200, &cfg);
        assert_eq!(pruned, 0);
        assert!(s.get_weight(0, 1) > 0.0);
    }

    #[test]
    fn prune_exact_min_weight_boundary_not_pruned() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.001); // exactly at min_weight
        s.touch(0, 1, 1); // keep active
        let cfg = SokmConfig {
            min_weight: 0.001,
            p1: 100,
            ..SokmConfig::default()
        };
        let pruned = prune(&mut s, 1, &cfg);
        assert_eq!(pruned, 0);
        assert!(s.get_weight(0, 1) > 0.0);
    }

    #[test]
    fn prune_exact_p1_boundary_not_pruned() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.touch(0, 1, 100); // current_tick - last = 200-100 = 100, p1=100 -> NOT > p1
        let cfg = SokmConfig {
            p1: 100,
            ..SokmConfig::default()
        };
        let pruned = prune(&mut s, 200, &cfg);
        assert_eq!(pruned, 0);
        assert!(s.get_weight(0, 1) > 0.0);
    }

    #[test]
    fn prune_both_conditions_simultaneously() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.0005); // below min_weight AND inactive
        let cfg = SokmConfig {
            min_weight: 0.001,
            p1: 10,
            ..SokmConfig::default()
        };
        let pruned = prune(&mut s, 100, &cfg);
        // Should be pruned by weight (first phase), not double-counted
        assert!(pruned >= 1);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn prune_empty_store_no_panic() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        let pruned = prune(&mut s, 100, &cfg());
        assert_eq!(pruned, 0);
    }

    // --- propagate (binary, Eq 4.4) ---

    #[test]
    fn propagate_attenuates_by_gamma() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        let cfg = SokmConfig {
            gamma: 0.5,
            ..SokmConfig::default()
        };
        let spread = propagate(&s, &[0u32], &cfg);
        let node1 = spread
            .iter()
            .find(|&&(k, _)| k == 1)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        assert!((node1 - cfg.gamma * 1.0).abs() < 1e-10);
    }

    #[test]
    fn propagate_follows_edge_weights() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.4);
        s.set_weight(0, 2, 0.8);
        let cfg = SokmConfig {
            gamma: 1.0,
            ..SokmConfig::default()
        };
        let spread = propagate(&s, &[0u32], &cfg);
        let v1 = spread
            .iter()
            .find(|&&(k, _)| k == 1)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        let v2 = spread
            .iter()
            .find(|&&(k, _)| k == 2)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        assert!((v1 - 0.4).abs() < 1e-10);
        assert!((v2 - 0.8).abs() < 1e-10);
    }

    #[test]
    fn propagate_binary_empty_fired_returns_empty() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        let spread = propagate(&s, &[] as &[u32], &cfg());
        assert!(spread.is_empty());
    }

    #[test]
    fn propagate_binary_ignores_subthreshold_nodes() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        let spread = propagate(&s, &[] as &[u32], &cfg());
        let node1 = spread
            .iter()
            .find(|&&(k, _)| k == 1)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        assert!((node1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn propagate_two_fired_nodes_same_target_scores_sum() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 2, 0.5);
        s.set_weight(1, 2, 0.3);
        let cfg = SokmConfig {
            gamma: 1.0,
            ..SokmConfig::default()
        };
        let spread = propagate(&s, &[0u32, 1], &cfg);
        let v2 = spread
            .iter()
            .find(|&&(k, _)| k == 2)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        // Should be sum: 0.5 + 0.3 = 0.8, not max
        assert!((v2 - 0.8).abs() < 1e-10);
    }

    // --- propagate_soft (graded, Eq 4.3) ---

    #[test]
    fn propagate_soft_attenuates_by_gamma_and_score() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        let cfg = SokmConfig {
            gamma: 0.5,
            ..SokmConfig::default()
        };
        let spread = propagate_soft(&s, &[(0u32, 0.5)], &cfg);
        let node1 = spread
            .iter()
            .find(|&&(k, _)| k == 1)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        assert!((node1 - cfg.gamma * 1.0 * 0.5).abs() < 1e-10);
    }

    #[test]
    fn propagate_soft_empty_activated_returns_empty() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        let spread = propagate_soft(&s, &[] as &[(u32, f64)], &cfg());
        assert!(spread.is_empty());
    }

    #[test]
    fn propagate_soft_partial_score_scales_spread() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        let cfg = SokmConfig {
            gamma: 1.0,
            ..SokmConfig::default()
        };
        let full = propagate_soft(&s, &[(0u32, 1.0)], &cfg);
        let half = propagate_soft(&s, &[(0u32, 0.5)], &cfg);
        let v_full = full
            .iter()
            .find(|&&(k, _)| k == 1)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        let v_half = half
            .iter()
            .find(|&&(k, _)| k == 1)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        assert!((v_half - v_full * 0.5).abs() < 1e-10);
    }

    #[test]
    fn propagate_binary_vs_soft_agree_at_full_activation() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.7);
        let cfg = SokmConfig {
            gamma: 0.8,
            ..SokmConfig::default()
        };
        let bin = propagate(&s, &[0u32], &cfg);
        let soft = propagate_soft(&s, &[(0u32, 1.0)], &cfg);
        let v_bin = bin
            .iter()
            .find(|&&(k, _)| k == 1)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        let v_soft = soft
            .iter()
            .find(|&&(k, _)| k == 1)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        assert!((v_bin - v_soft).abs() < 1e-10);
    }

    #[test]
    fn propagate_soft_two_fired_nodes_same_target_scores_sum() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 2, 0.5);
        s.set_weight(1, 2, 0.3);
        let cfg = SokmConfig {
            gamma: 1.0,
            ..SokmConfig::default()
        };
        let spread = propagate_soft(&s, &[(0u32, 1.0), (1, 1.0)], &cfg);
        let v2 = spread
            .iter()
            .find(|&&(k, _)| k == 2)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        assert!((v2 - 0.8).abs() < 1e-10);
    }

    // --- tick ---

    #[test]
    fn tick_order_is_decay_strengthen_prune() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        let cfg = cfg();
        let report = tick(&mut s, &[(0, 1.0), (1, 1.0)], 1, &cfg, DecayMode::Apply);
        assert_eq!(report.edges_alive, 1);
        assert_eq!(report.strengthened, 1);
        assert!(s.get_weight(0, 1) > 0.0);
    }

    #[test]
    fn tick_report_counts_are_correct() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.set_weight(2, 3, 0.0005);
        let cfg = cfg();
        let report = tick(&mut s, &[(0, 1.0), (1, 1.0)], 1, &cfg, DecayMode::Apply);
        assert!(report.pruned >= 1);
    }

    #[test]
    fn tick_empty_activated_no_panic() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        let cfg = cfg();
        let before = s.get_weight(0, 1);
        let report = tick(&mut s, &[], 1, &cfg, DecayMode::Apply);
        assert_eq!(report.strengthened, 0);
        // Weight decayed but not strengthened
        assert!(s.get_weight(0, 1) < before);
    }

    #[test]
    fn tick_10_convergence_with_fixed_activations() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        let cfg = cfg();
        let activated = vec![(0u32, 1.0), (1, 1.0)];
        for t in 1..=10 {
            tick(&mut s, &activated, t, &cfg, DecayMode::Apply);
        }
        // After 10 ticks with constant activation, edge should be strong
        let w = s.get_weight(0, 1);
        assert!(w > 0.0);
        assert!(w <= cfg.w_max);
    }

    #[test]
    fn tick_skip_decay_preserves_weights() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.touch(0, 1, 1);
        let cfg = SokmConfig::default();
        let report = tick(&mut s, &[], 2, &cfg, DecayMode::Skip);
        // weight must not have decayed
        assert!((s.get_weight(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(report.edges_alive, 1);
    }

    #[test]
    fn tick_no_skip_decay_applies_decay() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.touch(0, 1, 1);
        let cfg = SokmConfig::default();
        let _ = tick(&mut s, &[], 2, &cfg, DecayMode::Apply);
        let expected = 0.5 * (-cfg.xi).exp();
        assert!((s.get_weight(0, 1) - expected).abs() < 1e-10);
    }

    // --- top_n ---

    #[test]
    fn top_n_returns_highest_scores() {
        let activated = vec![(0u32, 0.1), (1, 0.9), (2, 0.5), (3, 0.7)];
        let top = top_n(&activated, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 3);
    }

    #[test]
    fn top_n_clamps_to_available() {
        let activated = vec![(0u32, 0.5)];
        let top = top_n(&activated, 10);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn top_n_zero_returns_empty() {
        let activated = vec![(0u32, 0.5), (1, 0.9)];
        let top = top_n(&activated, 0);
        assert!(top.is_empty());
    }

    #[test]
    fn top_n_empty_input_returns_empty() {
        let activated: Vec<(u32, f64)> = vec![];
        let top = top_n(&activated, 5);
        assert!(top.is_empty());
    }

    #[test]
    fn top_n_nan_input_no_panic() {
        // NaN violates the precondition. In debug builds the debug_assert fires.
        // This test only runs in release to confirm no UB (sort, truncate still terminate).
        #[cfg(not(debug_assertions))]
        {
            let activated = vec![(0u32, f64::NAN), (1, 0.5), (2, 0.9)];
            let top = top_n(&activated, 3);
            assert_eq!(top.len(), 3);
        }
    }
}
