use crate::config::CrossSokmConfig;

pub trait CrossStore {
    /// Weight of directed edge modal1\[i\] → modal2\[j\]. 0.0 if absent.
    fn get(&self, i: usize, j: usize) -> f64;
    /// Set edge weight. `w <= 0.0` removes the edge (both weight and tick entries).
    fn set(&mut self, i: usize, j: usize, w: f64);
    /// All modal2 targets reachable from modal1\[i\], with weights.
    fn targets(&self, i: usize) -> Vec<(usize, f64)>;
    /// All modal1 sources that reach modal2\[j\], with weights.
    /// O(1) lookup via reverse index.
    fn sources(&self, j: usize) -> Vec<(usize, f64)>;
    /// Mark edge (i, j) as active at `tick`. `0` is the sentinel meaning "never touched".
    /// Calling `touch` on an edge that has no weight entry is a no-op (does not create the edge).
    fn touch(&mut self, i: usize, j: usize, tick: u64);
    /// Last tick the edge was touched. Returns `0` (sentinel) if never touched or absent.
    /// `prune_inactive` treats `0` as "last_active at tick 0" — with `p1 = u64::MAX` this
    /// means the edge is never pruned by inactivity.
    fn last_active(&self, i: usize, j: usize) -> u64;
    fn edge_count(&self) -> usize;
    /// Multiply all weights by `factor`. Used for decay: `factor = exp(-xi)`.
    /// Pure multiply — no entries removed. All pruning via `prune_below`.
    fn scale_all(&mut self, factor: f64);
    /// Add delta increments to edge weights, clamped to [0, w_max].
    /// First co-activation creates edge at `w_init`; `delta` is NOT applied on first
    /// encounter — applied from second co-activation onward.
    fn apply_increments(&mut self, deltas: &[(usize, usize, f64)], cfg: &CrossSokmConfig);
    /// Apply increments and record the active tick atomically.
    /// Equivalent to calling `apply_increments` then `touch` for each delta.
    /// Prefer this over calling them separately.
    fn strengthen(&mut self, deltas: &[(usize, usize, f64)], cfg: &CrossSokmConfig, tick: u64) {
        self.apply_increments(deltas, cfg);
        for &(i, j, _) in deltas {
            self.touch(i, j, tick);
        }
    }
    /// Remove edges with weight < threshold. Returns count removed.
    /// Exact boundary (weight == threshold) is NOT pruned.
    fn prune_below(&mut self, threshold: f64) -> usize;
    /// Remove edges not touched in the last `p1` ticks. Returns count removed.
    /// Boundary: `current_tick - last_active == p1` is NOT pruned (uses strict `>`).
    fn prune_inactive(&mut self, current_tick: u64, p1: u64) -> usize;
    /// Reindex after compact on either modality.
    fn reindex(&mut self, map1: &[Option<usize>], map2: &[Option<usize>]);
}

/// Directed bipartite edge store (modal1 → modal2), HashMap-backed.
/// O(1) get/set, O(1) sources lookup via reverse index.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CrossEdgeStore {
    weights: std::collections::HashMap<(usize, usize), f64>,
    ticks: std::collections::HashMap<(usize, usize), u64>,
    reverse: std::collections::HashMap<usize, Vec<usize>>,
}

impl CrossEdgeStore {
    pub fn new() -> Self {
        Self {
            weights: std::collections::HashMap::new(),
            ticks: std::collections::HashMap::new(),
            reverse: std::collections::HashMap::new(),
        }
    }

    pub fn edge_count(&self) -> usize {
        <Self as CrossStore>::edge_count(self)
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        <Self as CrossStore>::get(self, i, j)
    }

    pub fn targets(&self, i: usize) -> Vec<(usize, f64)> {
        <Self as CrossStore>::targets(self, i)
    }

    pub fn sources(&self, j: usize) -> Vec<(usize, f64)> {
        <Self as CrossStore>::sources(self, j)
    }
}

impl Default for CrossEdgeStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossStore for CrossEdgeStore {
    fn get(&self, i: usize, j: usize) -> f64 {
        self.weights.get(&(i, j)).copied().unwrap_or(0.0)
    }

    fn set(&mut self, i: usize, j: usize, w: f64) {
        if w <= 0.0 {
            if self.weights.remove(&(i, j)).is_some()
                && let Some(sources) = self.reverse.get_mut(&j)
            {
                sources.retain(|&s| s != i);
                if sources.is_empty() {
                    self.reverse.remove(&j);
                }
            }
            self.ticks.remove(&(i, j));
        } else {
            if self.weights.insert((i, j), w).is_none() {
                // new edge — add to reverse index
                self.reverse.entry(j).or_default().push(i);
            }
        }
    }

    fn targets(&self, i: usize) -> Vec<(usize, f64)> {
        self.weights
            .iter()
            .filter(|(key, _)| key.0 == i)
            .map(|(key, &v)| (key.1, v))
            .collect()
    }

    fn sources(&self, j: usize) -> Vec<(usize, f64)> {
        self.reverse
            .get(&j)
            .map(|srcs| {
                srcs.iter()
                    .filter_map(|&i| self.weights.get(&(i, j)).map(|&w| (i, w)))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn touch(&mut self, i: usize, j: usize, tick: u64) {
        // Only touch if the edge actually exists in weights.
        if self.weights.contains_key(&(i, j)) {
            self.ticks.insert((i, j), tick);
        }
    }

    fn last_active(&self, i: usize, j: usize) -> u64 {
        self.ticks.get(&(i, j)).copied().unwrap_or(0)
    }

    fn edge_count(&self) -> usize {
        self.weights.len()
    }

    fn scale_all(&mut self, factor: f64) {
        for v in self.weights.values_mut() {
            *v *= factor;
        }
    }

    fn apply_increments(&mut self, deltas: &[(usize, usize, f64)], cfg: &CrossSokmConfig) {
        for &(i, j, _delta) in deltas {
            if let Some(w) = self.weights.get_mut(&(i, j)) {
                let new_w = (*w + _delta).min(cfg.w_max);
                *w = new_w;
            } else {
                self.weights.insert((i, j), cfg.w_init);
                self.reverse.entry(j).or_default().push(i);
            }
        }
    }

    fn prune_below(&mut self, threshold: f64) -> usize {
        let to_remove: Vec<(usize, usize)> = self
            .weights
            .iter()
            .filter(|&(_, &v)| v < threshold)
            .map(|(&k, _)| k)
            .collect();
        for &(i, j) in &to_remove {
            self.weights.remove(&(i, j));
            self.ticks.remove(&(i, j));
            if let Some(sources) = self.reverse.get_mut(&j) {
                sources.retain(|&s| s != i);
                if sources.is_empty() {
                    self.reverse.remove(&j);
                }
            }
        }
        to_remove.len()
    }

    fn prune_inactive(&mut self, current_tick: u64, p1: u64) -> usize {
        let to_remove: Vec<(usize, usize)> = self
            .weights
            .keys()
            .filter(|k| {
                let last = self.ticks.get(k).copied().unwrap_or(0);
                current_tick.saturating_sub(last) > p1
            })
            .copied()
            .collect();
        let count = to_remove.len();
        for &(i, j) in &to_remove {
            self.weights.remove(&(i, j));
            self.ticks.remove(&(i, j));
            if let Some(sources) = self.reverse.get_mut(&j) {
                sources.retain(|&s| s != i);
                if sources.is_empty() {
                    self.reverse.remove(&j);
                }
            }
        }
        count
    }

    fn reindex(&mut self, map1: &[Option<usize>], map2: &[Option<usize>]) {
        let old_weights: Vec<_> = self.weights.drain().collect();
        let old_ticks: Vec<_> = self.ticks.drain().collect();
        self.reverse.clear();

        for ((old_i, old_j), w) in old_weights {
            let new_i = map1.get(old_i).and_then(|x| *x);
            let new_j = map2.get(old_j).and_then(|x| *x);
            if let (Some(ni), Some(nj)) = (new_i, new_j) {
                self.weights.insert((ni, nj), w);
                self.reverse.entry(nj).or_default().push(ni);
            }
        }

        for ((old_i, old_j), t) in old_ticks {
            let new_i = map1.get(old_i).and_then(|x| *x);
            let new_j = map2.get(old_j).and_then(|x| *x);
            if let (Some(ni), Some(nj)) = (new_i, new_j) {
                self.ticks.insert((ni, nj), t);
            }
        }
    }
}

/// Propagate soft activation from modal1 into modal2 space via cross-modal edges.
/// All modal1 kernels with score > 0 contribute: modal2\[j\] += γ · w_ij · score_i.
/// \[Hoya Eq. 4.3 applied cross-modally\] \[INFERRED\]
pub fn cross_propagate_soft(
    cross: &impl CrossStore,
    modal1_scores: &[(usize, f64)],
    gamma: f64,
) -> Vec<(usize, f64)> {
    let mut acc: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for &(i, score) in modal1_scores {
        for (j, w) in cross.targets(i) {
            *acc.entry(j).or_insert(0.0) += gamma * w * score;
        }
    }
    acc.into_iter().collect()
}

/// Reverse propagation: modal2 activations → modal1 space via cross-modal edges.
/// modal1\[i\] += γ · w_ij · score_j for all (i, j) where j is active in modal2.
/// Symmetric counterpart to [`cross_propagate_soft`].
pub fn cross_propagate_soft_reverse(
    cross: &impl CrossStore,
    modal2_scores: &[(usize, f64)],
    gamma: f64,
) -> Vec<(usize, f64)> {
    let mut acc: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for &(j, score) in modal2_scores {
        for (i, w) in cross.sources(j) {
            *acc.entry(i).or_insert(0.0) += gamma * w * score;
        }
    }
    acc.into_iter().collect()
}

/// Compute cross-modal Hebbian strengthening deltas.
///
/// Filter logic:
/// - require_class_match=true:  matches!((c1,c2), (Some(a),Some(b)) if a==b)
/// - require_class_match=false: matches!((c1,c2), (Some(_),Some(_)))
/// - None kernels never participate regardless of flag
pub fn cross_strengthen_deltas(
    modal1_activated: &[(usize, Option<u32>)],
    modal2_activated: &[(usize, Option<u32>)],
    cfg: &CrossSokmConfig,
) -> Vec<(usize, usize, f64)> {
    let mut deltas = Vec::new();
    for &(i, class1) in modal1_activated {
        for &(j, class2) in modal2_activated {
            let fires = if cfg.require_class_match {
                matches!((class1, class2), (Some(a), Some(b)) if a == b)
            } else {
                matches!((class1, class2), (Some(_), Some(_)))
            };
            if fires {
                deltas.push((i, j, cfg.delta));
            }
        }
    }
    deltas
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CrossSokmConfig {
        CrossSokmConfig::default()
    }

    // --- CrossSokmConfig::validate tests (finding #18) ---

    #[test]
    fn validate_gamma_zero_invalid() {
        let c = CrossSokmConfig {
            gamma: 0.0,
            ..cfg()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_gamma_positive_valid() {
        let c = CrossSokmConfig {
            gamma: 1.0,
            ..cfg()
        };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_xi_zero_valid() {
        let c = CrossSokmConfig { xi: 0.0, ..cfg() };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_xi_negative_invalid() {
        let c = CrossSokmConfig { xi: -0.01, ..cfg() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_w_init_exceeds_w_max_invalid() {
        let c = CrossSokmConfig {
            w_init: 2.0,
            w_max: 1.0,
            ..cfg()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_w_init_equals_w_max_valid() {
        let c = CrossSokmConfig {
            w_init: 1.0,
            w_max: 1.0,
            ..cfg()
        };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_delta_zero_valid() {
        let c = CrossSokmConfig {
            delta: 0.0,
            ..cfg()
        };
        assert!(c.validate().is_ok());
    }

    // --- CrossEdgeStore tests (finding #19) ---

    #[test]
    fn set_zero_removes_edge() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 1, 1.0);
        CrossStore::touch(&mut s, 0, 1, 5);
        s.set(0, 1, 0.0);
        assert_eq!(s.edge_count(), 0);
        assert_eq!(CrossStore::last_active(&s, 0, 1), 0);
    }

    #[test]
    fn set_negative_removes_edge() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 1, 1.0);
        s.set(0, 1, -1.0);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn set_nonexistent_zero_noop() {
        let mut s = CrossEdgeStore::new();
        s.set(5, 5, 0.0);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn apply_increments_existing_edge_increments() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 0.5);
        let c = cfg();
        CrossStore::apply_increments(&mut s, &[(0, 0, 0.1)], &c);
        assert!((CrossStore::get(&s, 0, 0) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn apply_increments_new_edge_at_w_init() {
        let mut s = CrossEdgeStore::new();
        let c = cfg();
        CrossStore::apply_increments(&mut s, &[(0, 0, c.delta)], &c);
        assert!((CrossStore::get(&s, 0, 0) - c.w_init).abs() < 1e-9);
    }

    #[test]
    fn apply_increments_second_call_applies_delta() {
        // First call: edge created at w_init (delta ignored)
        // Second call: delta is applied
        let mut s = CrossEdgeStore::new();
        let c = cfg();
        CrossStore::apply_increments(&mut s, &[(0, 0, c.delta)], &c);
        let after_first = CrossStore::get(&s, 0, 0);
        assert!((after_first - c.w_init).abs() < 1e-9);
        CrossStore::apply_increments(&mut s, &[(0, 0, c.delta)], &c);
        let after_second = CrossStore::get(&s, 0, 0);
        assert!((after_second - (c.w_init + c.delta).min(c.w_max)).abs() < 1e-9);
    }

    #[test]
    fn apply_increments_clamps_to_w_max() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 0.95);
        let c = cfg();
        CrossStore::apply_increments(&mut s, &[(0, 0, 0.1)], &c);
        assert!((CrossStore::get(&s, 0, 0) - c.w_max).abs() < 1e-9);
    }

    #[test]
    fn scale_all_zero_preserves_edge_count() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 1.0);
        s.set(1, 0, 0.5);
        CrossStore::scale_all(&mut s, 0.0);
        assert_eq!(s.edge_count(), 2);
    }

    #[test]
    fn prune_below_exact_boundary_not_pruned() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 0.5);
        let removed = CrossStore::prune_below(&mut s, 0.5);
        assert_eq!(removed, 0);
        assert_eq!(s.edge_count(), 1);
    }

    #[test]
    fn prune_inactive_max_p1_never_prunes() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 1.0);
        // Never touched → last_active = 0, but p1 = u64::MAX means no prune
        let removed = CrossStore::prune_inactive(&mut s, 1000, u64::MAX);
        assert_eq!(removed, 0);
    }

    #[test]
    fn prune_inactive_untouched_edge_pruned() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 1.0);
        // last_active=0, current_tick=10, p1=5 → 10 - 0 = 10 > 5 → pruned
        let removed = CrossStore::prune_inactive(&mut s, 10, 5);
        assert_eq!(removed, 1);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn prune_inactive_boundary_not_pruned() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 1.0);
        CrossStore::touch(&mut s, 0, 0, 5);
        // current_tick=10, last_active=5, p1=5 → 10-5=5 == p1 → NOT pruned (strict >)
        let removed = CrossStore::prune_inactive(&mut s, 10, 5);
        assert_eq!(removed, 0);
    }

    #[test]
    fn touch_without_prior_set_no_panic() {
        let mut s = CrossEdgeStore::new();
        // No edge exists — touch should not create one and should not panic
        CrossStore::touch(&mut s, 99, 99, 1);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn reindex_preserves_tick_values() {
        let mut s = CrossEdgeStore::new();
        s.set(1, 1, 0.7);
        CrossStore::touch(&mut s, 1, 1, 42);
        s.reindex(&[None, Some(0)], &[None, Some(0)]);
        assert_eq!(s.edge_count(), 1);
        assert!((CrossStore::get(&s, 0, 0) - 0.7).abs() < 1e-9);
        assert_eq!(CrossStore::last_active(&s, 0, 0), 42);
    }

    #[test]
    fn reindex_empty_maps_no_panic() {
        let mut s = CrossEdgeStore::new();
        s.reindex(&[], &[]);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn reindex_both_modalities_compact() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 1.0);
        s.set(1, 1, 0.5);
        // Remove index 0 from both → only (1,1) survives as (0,0)
        s.reindex(&[None, Some(0)], &[None, Some(0)]);
        assert_eq!(s.edge_count(), 1);
        assert!((CrossStore::get(&s, 0, 0) - 0.5).abs() < 1e-9);
    }

    // --- CrossStore::strengthen tests ---

    #[test]
    fn strengthen_sets_weight_and_tick() {
        let mut store = CrossEdgeStore::new();
        let cfg = CrossSokmConfig::default();
        let deltas = vec![(0usize, 1usize, 0.1f64)];

        // First call creates edge at w_init
        store.strengthen(&deltas, &cfg, 5);
        assert!(store.get(0, 1) > 0.0);
        assert_eq!(CrossStore::last_active(&store, 0, 1), 5);
    }

    #[test]
    fn strengthen_touch_tick_updated_on_second_call() {
        let mut store = CrossEdgeStore::new();
        let cfg = CrossSokmConfig::default();
        let deltas = vec![(0usize, 1usize, 0.1f64)];

        store.strengthen(&deltas, &cfg, 5);
        store.strengthen(&deltas, &cfg, 10);
        assert_eq!(CrossStore::last_active(&store, 0, 1), 10);
    }

    #[test]
    fn strengthen_no_touch_for_empty_deltas() {
        let mut store = CrossEdgeStore::new();
        let cfg = CrossSokmConfig::default();
        store.strengthen(&[], &cfg, 99);
        assert_eq!(store.edge_count(), 0);
    }

    // --- Free function tests (finding #20) ---

    #[test]
    fn cross_propagate_soft_empty_scores_empty_result() {
        let s = CrossEdgeStore::new();
        let result = cross_propagate_soft(&s, &[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn cross_propagate_soft_zero_score_entry() {
        // If a score is 0.0, multiplication yields 0.0 but we still check behavior
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 1.0);
        let result = cross_propagate_soft(&s, &[(0, 0.0)], 1.0);
        // 0.0 * w * gamma = 0.0 — included in result as 0.0
        if !result.is_empty() {
            assert!(result.iter().all(|&(_, v)| v.abs() < 1e-15));
        }
    }

    #[test]
    fn cross_propagate_soft_reverse_symmetry() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 1, 0.8);
        // Forward: modal1[0] → modal2[1]
        let fwd = cross_propagate_soft(&s, &[(0, 1.0)], 1.0);
        // Reverse: modal2[1] → modal1[0]
        let rev = cross_propagate_soft_reverse(&s, &[(1, 1.0)], 1.0);
        assert_eq!(fwd.len(), 1);
        assert_eq!(rev.len(), 1);
        assert!((fwd[0].1 - rev[0].1).abs() < 1e-9);
    }

    #[test]
    fn cross_propagate_soft_reverse_empty_scores() {
        let s = CrossEdgeStore::new();
        let result = cross_propagate_soft_reverse(&s, &[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn cross_strengthen_deltas_empty_modal1_fired() {
        let c = cfg();
        let deltas = cross_strengthen_deltas(&[], &[(0, Some(1))], &c);
        assert!(deltas.is_empty());
    }

    #[test]
    fn cross_strengthen_deltas_empty_activations() {
        let c = cfg();
        let deltas = cross_strengthen_deltas(&[(0, Some(1))], &[], &c);
        assert!(deltas.is_empty());
    }

    #[test]
    fn cross_strengthen_deltas_no_class_match_allows_any() {
        let mut c = cfg();
        c.require_class_match = false;
        let deltas = cross_strengthen_deltas(&[(0, Some(1))], &[(0, Some(99))], &c);
        assert_eq!(deltas.len(), 1);
    }

    // --- Existing tests preserved ---

    #[test]
    fn cross_propagate_soft_transfers_activation() {
        let mut store = CrossEdgeStore::new();
        store.set(0, 1, 1.0);
        let result = cross_propagate_soft(&store, &[(0, 1.0)], 1.0);
        assert!(
            result
                .iter()
                .any(|&(j, v)| j == 1 && (v - 1.0).abs() < 1e-9)
        );
    }

    #[test]
    fn cross_propagate_soft_accumulates_from_multiple_sources() {
        let mut store = CrossEdgeStore::new();
        store.set(0, 2, 0.5);
        store.set(1, 2, 0.5);
        let result = cross_propagate_soft(&store, &[(0, 1.0), (1, 1.0)], 1.0);
        assert!(
            result
                .iter()
                .any(|&(j, v)| j == 2 && (v - 1.0).abs() < 1e-9)
        );
    }

    #[test]
    fn cross_propagate_soft_zero_without_edge() {
        let store = CrossEdgeStore::new();
        let result = cross_propagate_soft(&store, &[(0, 1.0)], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn cross_strengthen_deltas_fires_on_class_match() {
        let mut c = cfg();
        c.require_class_match = true;
        let deltas = cross_strengthen_deltas(&[(0, Some(1))], &[(0, Some(1))], &c);
        assert_eq!(deltas.len(), 1);
    }

    #[test]
    fn cross_strengthen_deltas_blocked_on_class_mismatch() {
        let mut c = cfg();
        c.require_class_match = true;
        let deltas = cross_strengthen_deltas(&[(0, Some(1))], &[(0, Some(2))], &c);
        assert_eq!(deltas.len(), 0);
    }

    #[test]
    fn cross_strengthen_deltas_blocked_on_unlabelled() {
        let mut c = cfg();
        c.require_class_match = false;
        let deltas = cross_strengthen_deltas(&[(0, None)], &[(0, Some(1))], &c);
        assert_eq!(deltas.len(), 0);
    }

    #[test]
    fn cross_strengthen_deltas_fires_without_class_match_when_disabled() {
        let mut c = cfg();
        c.require_class_match = false;
        let deltas = cross_strengthen_deltas(&[(0, Some(1))], &[(0, Some(2))], &c);
        assert_eq!(deltas.len(), 1);
        let deltas2 = cross_strengthen_deltas(&[(0, None)], &[(0, Some(2))], &c);
        assert_eq!(deltas2.len(), 0);
    }

    #[test]
    fn cross_store_reindex_drops_extinct_edges() {
        let mut store = CrossEdgeStore::new();
        store.set(0, 0, 1.0);
        store.reindex(&[None], &[Some(0)]);
        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn cross_store_reindex_remaps_survivors() {
        let mut store = CrossEdgeStore::new();
        store.set(1, 0, 0.5);
        store.reindex(&[None, Some(0)], &[Some(0)]);
        assert_eq!(store.edge_count(), 1);
        assert!((store.get(0, 0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn reverse_index_consistent_after_prune() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 1, 0.5);
        s.set(1, 1, 0.3);
        CrossStore::prune_below(&mut s, 0.4);
        let srcs = CrossStore::sources(&s, 1);
        assert_eq!(srcs.len(), 1);
        assert_eq!(srcs[0].0, 0);
    }

    #[test]
    fn reverse_index_consistent_after_set_zero() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 1, 0.8);
        s.set(0, 1, 0.0); // remove
        let srcs = CrossStore::sources(&s, 1);
        assert!(srcs.is_empty());
    }

    #[test]
    fn reverse_index_consistent_after_reindex() {
        let mut s = CrossEdgeStore::new();
        s.set(0, 0, 1.0);
        s.set(1, 0, 0.5);
        // remove index 0 from modal1
        s.reindex(&[None, Some(0)], &[Some(0)]);
        // only old i=1→j=0 survives as (0,0)
        let srcs = CrossStore::sources(&s, 0);
        assert_eq!(srcs.len(), 1);
        assert_eq!(srcs[0].0, 0);
    }
}
