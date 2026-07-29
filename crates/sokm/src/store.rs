use std::collections::HashMap;
use std::hash::Hash;

/// Abstract edge storage. Keys are canonicalized as (min(a,b), max(a,b)).
pub trait EdgeStore<K: Copy + Eq + Hash> {
    /// Get weight of edge (a, b). Returns 0.0 if edge does not exist.
    fn get_weight(&self, a: K, b: K) -> f64;
    /// Set weight of edge (a, b). Setting w <= 0.0 removes the edge entirely.
    fn set_weight(&mut self, a: K, b: K, w: f64);
    /// Multiply all edge weights by `factor`.
    fn scale_all(&mut self, factor: f64);
    /// Add deltas to existing edges (or insert from 0.0 if absent).
    fn apply_increments(&mut self, deltas: &[(K, K, f64)]);
    fn prune_below(&mut self, threshold: f64) -> usize;
    /// Remove edges inactive for more than p1 ticks. [Hoya p1 extinction]
    fn prune_inactive(&mut self, current_tick: u64, p1: u64) -> usize;
    /// Neighbors of node with their weights. Required for propagation [Hoya Eq 4.3].
    fn neighbors(&self, node: K) -> Vec<(K, f64)>;
    /// Record that edge (a,b) was activated at current_tick.
    fn touch(&mut self, a: K, b: K, tick: u64);
    /// Last tick edge (a,b) was activated. Returns 0 if never touched.
    fn last_active(&self, a: K, b: K) -> u64;
    /// Total number of live edges in the store.
    fn edge_count(&self) -> usize;
}

/// Marker trait for edge stores that support index remapping after compact_extinct.
pub trait Reindex {
    fn reindex_for_compact(&mut self, map: &[Option<usize>]);
}

pub(crate) fn canonical<K: Copy + Ord>(a: K, b: K) -> (K, K) {
    if a <= b { (a, b) } else { (b, a) }
}

/// HashMap-backed reference implementation; suitable for tests and small graphs.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "K: serde::Serialize + Eq + std::hash::Hash",
        deserialize = "K: serde::Deserialize<'de> + Eq + std::hash::Hash + Copy + Ord"
    ))
)]
pub struct HashEdgeStore<K> {
    weights: HashMap<(K, K), f64>,
    ticks: HashMap<(K, K), u64>,
}

impl<K: Copy + Eq + Hash + Ord> HashEdgeStore<K> {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            ticks: HashMap::new(),
        }
    }
}

impl<K: Copy + Eq + Hash + Ord> Default for HashEdgeStore<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Copy + Eq + Hash + Ord> EdgeStore<K> for HashEdgeStore<K> {
    fn get_weight(&self, a: K, b: K) -> f64 {
        *self.weights.get(&canonical(a, b)).unwrap_or(&0.0)
    }

    fn set_weight(&mut self, a: K, b: K, w: f64) {
        let key = canonical(a, b);
        if w <= 0.0 {
            self.weights.remove(&key);
            self.ticks.remove(&key);
        } else {
            self.weights.insert(key, w);
        }
    }

    fn scale_all(&mut self, factor: f64) {
        for w in self.weights.values_mut() {
            *w *= factor;
        }
    }

    fn apply_increments(&mut self, deltas: &[(K, K, f64)]) {
        for &(a, b, d) in deltas {
            let key = canonical(a, b);
            *self.weights.entry(key).or_insert(0.0) += d;
        }
    }

    fn prune_below(&mut self, threshold: f64) -> usize {
        let to_remove: Vec<(K, K)> = self
            .weights
            .iter()
            .filter(|&(_, &w)| w < threshold)
            .map(|(&k, _)| k)
            .collect();
        let count = to_remove.len();
        for k in to_remove {
            self.weights.remove(&k);
            self.ticks.remove(&k);
        }
        count
    }

    fn prune_inactive(&mut self, current_tick: u64, p1: u64) -> usize {
        let to_remove: Vec<(K, K)> = self
            .weights
            .keys()
            .filter(|k| {
                let last = *self.ticks.get(k).unwrap_or(&0);
                current_tick.saturating_sub(last) > p1 // > not >=: survives p1-th inactive tick [Hoya §4]
            })
            .copied()
            .collect();
        let count = to_remove.len();
        for k in to_remove {
            self.weights.remove(&k);
            self.ticks.remove(&k);
        }
        count
    }

    fn neighbors(&self, node: K) -> Vec<(K, f64)> {
        self.weights
            .iter()
            .filter_map(|(&(a, b), &w)| {
                if a == node {
                    Some((b, w))
                } else if b == node {
                    Some((a, w))
                } else {
                    None
                }
            })
            .collect()
    }

    fn touch(&mut self, a: K, b: K, tick: u64) {
        self.ticks.insert(canonical(a, b), tick);
    }

    fn last_active(&self, a: K, b: K) -> u64 {
        *self.ticks.get(&canonical(a, b)).unwrap_or(&0)
    }

    fn edge_count(&self) -> usize {
        self.weights.len()
    }
}

impl HashEdgeStore<usize> {
    pub(crate) fn reindex_usize(&mut self, map: &[Option<usize>]) {
        let mut new_weights = HashMap::new();
        let mut new_ticks = HashMap::new();
        for (&(a, b), &w) in &self.weights {
            let new_a = map.get(a).copied().flatten();
            let new_b = map.get(b).copied().flatten();
            if let (Some(na), Some(nb)) = (new_a, new_b) {
                let key = if na <= nb { (na, nb) } else { (nb, na) };
                new_weights.insert(key, w);
                if let Some(&t) = self.ticks.get(&(a, b)) {
                    new_ticks.insert(key, t);
                }
            }
        }
        self.weights = new_weights;
        self.ticks = new_ticks;
    }
}

impl Reindex for HashEdgeStore<usize> {
    fn reindex_for_compact(&mut self, map: &[Option<usize>]) {
        self.reindex_usize(map);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_store_symmetric_get_set() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(3, 1, 0.5);
        assert_eq!(s.get_weight(1, 3), 0.5);
        assert_eq!(s.get_weight(3, 1), 0.5);
    }

    #[test]
    fn hash_store_edge_count() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        assert_eq!(s.edge_count(), 0);
        s.set_weight(0, 1, 0.5);
        s.set_weight(0, 2, 0.3);
        assert_eq!(s.edge_count(), 2);
    }

    #[test]
    fn hash_store_scale_all() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 1.0);
        s.scale_all(0.5);
        assert!((s.get_weight(0, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn hash_store_prune_below() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.set_weight(0, 2, 0.001);
        let pruned = s.prune_below(0.01);
        assert_eq!(pruned, 1);
        assert_eq!(s.edge_count(), 1);
    }

    #[test]
    fn hash_store_neighbors() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.set_weight(0, 2, 0.3);
        let mut ns: Vec<u32> = s.neighbors(0).into_iter().map(|(k, _)| k).collect();
        ns.sort();
        assert_eq!(ns, vec![1, 2]);
    }

    #[test]
    fn hash_store_touch_and_last_active() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        assert_eq!(s.last_active(0, 1), 0);
        s.touch(0, 1, 42);
        assert_eq!(s.last_active(0, 1), 42);
        assert_eq!(s.last_active(1, 0), 42); // symmetric
    }

    #[test]
    fn hash_store_prune_inactive() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.set_weight(0, 2, 0.5);
        s.touch(0, 1, 50);
        // edge (0,2) last_active = 0, p1 = 10, current = 20 -> extinct
        let pruned = s.prune_inactive(20, 10);
        assert_eq!(pruned, 1);
        assert_eq!(s.get_weight(0, 1), 0.5); // untouched
        assert_eq!(s.get_weight(0, 2), 0.0); // pruned
    }

    #[test]
    fn hash_store_apply_increments() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.3);
        s.apply_increments(&[(0, 1, 0.2)]);
        assert!((s.get_weight(0, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn hash_edge_store_reindex_drops_extinct_edges() {
        let mut s: HashEdgeStore<usize> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.set_weight(1, 2, 0.3);
        // kernel 1 extinct -> both edges dropped
        let map = vec![Some(0), None, Some(1)];
        s.reindex_usize(&map);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn hash_edge_store_reindex_remaps_survivors() {
        let mut s: HashEdgeStore<usize> = HashEdgeStore::new();
        s.set_weight(0, 2, 0.7); // 0->0, 2->1 after compact
        let map = vec![Some(0), None, Some(1)];
        s.reindex_usize(&map);
        assert_eq!(s.edge_count(), 1);
        assert!((s.get_weight(0, 1) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn hash_edge_store_reindex_recanonicalize() {
        let mut s2: HashEdgeStore<usize> = HashEdgeStore::new();
        s2.set_weight(3, 5, 0.9);
        let map = vec![Some(0), Some(1), None, Some(2), Some(3), Some(4)];
        s2.reindex_usize(&map);
        // old 3->new 2, old 5->new 4 -> canonical (2,4)
        assert!((s2.get_weight(2, 4) - 0.9).abs() < 1e-10);
        assert_eq!(s2.edge_count(), 1);
    }

    // --- Finding 25: additional HashEdgeStore tests ---

    #[test]
    fn hash_store_self_loop() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(5, 5, 0.4);
        assert!((s.get_weight(5, 5) - 0.4).abs() < 1e-10);
        assert_eq!(s.edge_count(), 1);
    }

    #[test]
    fn hash_store_negative_weight_removes_edge() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        assert_eq!(s.edge_count(), 1);
        s.set_weight(0, 1, -0.1);
        assert_eq!(s.edge_count(), 0);
        assert_eq!(s.get_weight(0, 1), 0.0);
    }

    #[test]
    fn hash_store_scale_all_zero_preserves_edge_count() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.5);
        s.set_weight(1, 2, 0.3);
        s.scale_all(0.0);
        // scale_all does not prune — edges remain with weight 0.0
        assert_eq!(s.edge_count(), 2);
    }

    #[test]
    fn hash_store_prune_below_exact_boundary_not_pruned() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        s.set_weight(0, 1, 0.01); // exactly at threshold
        let pruned = s.prune_below(0.01);
        assert_eq!(pruned, 0);
        assert_eq!(s.edge_count(), 1);
    }

    #[test]
    fn hash_store_apply_increments_nonexistent_inserts_from_zero() {
        let mut s: HashEdgeStore<u32> = HashEdgeStore::new();
        assert_eq!(s.edge_count(), 0);
        s.apply_increments(&[(0, 1, 0.7)]);
        assert!((s.get_weight(0, 1) - 0.7).abs() < 1e-10);
        assert_eq!(s.edge_count(), 1);
    }

    #[test]
    fn hash_store_get_weight_nonexistent_returns_zero() {
        let s: HashEdgeStore<u32> = HashEdgeStore::new();
        assert_eq!(s.get_weight(99, 100), 0.0);
    }
}
