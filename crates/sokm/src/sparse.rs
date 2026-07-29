use crate::store::{EdgeStore, Reindex, canonical};
use std::collections::HashMap;

/// Production edge store. CSR matrix for fast iteration; pending HashMap
/// buffers new/updated edges; per-edge tick map for p1 inactivity tracking.
///
/// Use `from_triplets` to construct from known edges (panics on OOB indices;
/// duplicate pairs accumulate their weights). Use `grow` to expand capacity.
/// `reindex` remaps node indices and drops edges where either endpoint maps to None.
/// `pending_count` returns the number of edges buffered but not yet in CSR.
///
/// # INVARIANT: CSR and pending are mutually exclusive per edge
/// An edge key is present in at most one of `csr_index` or `pending` at any time.
/// `set_weight` writes to CSR if the key already exists there, otherwise to pending.
/// `apply_increments` follows the same rule.
/// `compact()` merges pending into CSR and clears pending — after compact, pending is empty.
/// Breaking this invariant causes `get_weight` to double-count (it sums both).
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseEdgeStore {
    num_nodes: usize,
    // CSR: upper-triangle canonical edges
    csr_rows: Vec<usize>, // row pointers (len = num_nodes + 1)
    csr_cols: Vec<usize>, // column indices
    csr_vals: Vec<f64>,   // edge weights (0.0 = dead/pruned)
    csr_index: HashMap<(usize, usize), usize>, // (row,col) -> csr offset
    col_index: HashMap<usize, Vec<(usize, usize)>>, // col -> [(row, csr_offset)]
    // Pending: new edges not yet in CSR
    pending: HashMap<(usize, usize), f64>,
    // Per-edge last-activation tick
    ticks: HashMap<(usize, usize), u64>,
    dead_count: usize, // count of zeroed CSR slots (for auto-compact trigger)
}

impl SparseEdgeStore {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            csr_rows: vec![0usize; num_nodes + 1],
            csr_cols: Vec::new(),
            csr_vals: Vec::new(),
            csr_index: HashMap::new(),
            col_index: HashMap::new(),
            pending: HashMap::new(),
            ticks: HashMap::new(),
            dead_count: 0,
        }
    }

    /// Construct from triplets (row, col, weight).
    ///
    /// Panics if any index >= `num_nodes`. Duplicate pairs accumulate weights.
    pub fn from_triplets(num_nodes: usize, edges: &[(usize, usize, f64)]) -> Self {
        for &(a, b, _) in edges {
            assert!(
                a < num_nodes && b < num_nodes,
                "index out of bounds: ({a}, {b}) for num_nodes={num_nodes}"
            );
        }
        let mut s = Self::new(num_nodes);
        for &(a, b, w) in edges {
            *s.pending.entry(canonical(a, b)).or_insert(0.0) += w;
        }
        s.compact();
        s
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Number of edges buffered in pending (not yet merged into CSR).
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Return all live edges as `(node_a, node_b, weight, last_active_tick)`.
    /// Includes edges from both CSR (non-zero weight) and pending buffer.
    /// Output order is unspecified; each edge is stored in canonical form (`node_a <= node_b`).
    pub fn all_edges(&self) -> Vec<(usize, usize, f64, u64)> {
        let mut out: std::collections::HashMap<(usize, usize), (f64, u64)> =
            std::collections::HashMap::new();
        // CSR and pending are disjoint; both loops needed to collect all live edges.
        for (&(a, b), &idx) in &self.csr_index {
            let w = self.csr_vals[idx];
            if w > 0.0 {
                let t = self.ticks.get(&(a, b)).copied().unwrap_or(0);
                out.insert((a, b), (w, t));
            }
        }
        for (&(a, b), &w) in &self.pending {
            if w > 0.0 {
                let t = self.ticks.get(&(a, b)).copied().unwrap_or(0);
                out.insert((a, b), (w, t));
            }
        }
        out.into_iter()
            .map(|((a, b), (w, t))| (a, b, w, t))
            .collect()
    }

    /// Expand capacity to `new_num_nodes`. Returns false if already large enough.
    pub fn grow(&mut self, new_num_nodes: usize) -> bool {
        if new_num_nodes <= self.num_nodes {
            return false;
        }
        self.num_nodes = new_num_nodes;
        self.csr_rows
            .resize(new_num_nodes + 1, *self.csr_rows.last().unwrap_or(&0));
        true
    }

    /// Merge pending into CSR, rebuild index, drop dead slots.
    pub fn compact(&mut self) {
        // Collect all live edges: CSR non-dead + pending
        let mut all: HashMap<(usize, usize), f64> = HashMap::new();
        for (&key, &idx) in &self.csr_index {
            let w = self.csr_vals[idx];
            if w > 0.0 {
                all.insert(key, w);
            }
        }
        for (&key, &w) in &self.pending {
            if w > 0.0 {
                all.entry(key).and_modify(|e| *e += w).or_insert(w);
            }
        }
        self.pending.clear();

        // Rebuild CSR row-major order
        let mut rows = vec![0usize; self.num_nodes + 1];
        for &(r, _) in all.keys() {
            rows[r + 1] += 1;
        }
        for i in 1..=self.num_nodes {
            rows[i] += rows[i - 1];
        }
        let nnz = all.len();
        let mut cols = vec![0usize; nnz];
        let mut vals = vec![0.0f64; nnz];
        let mut index = HashMap::with_capacity(nnz);
        let mut pos = rows[..self.num_nodes].to_vec();
        let mut sorted_edges: Vec<((usize, usize), f64)> = all.into_iter().collect();
        sorted_edges.sort_by_key(|&((r, c), _)| (r, c));
        for ((r, c), w) in sorted_edges {
            let idx = pos[r];
            cols[idx] = c;
            vals[idx] = w;
            index.insert((r, c), idx);
            pos[r] += 1;
        }

        self.csr_rows = rows;
        self.csr_cols = cols;
        self.csr_vals = vals;
        self.csr_index = index;
        // Rebuild reverse column index
        let mut col_idx: HashMap<usize, Vec<(usize, usize)>> = HashMap::with_capacity(nnz);
        for (&(r, c), &i) in &self.csr_index {
            col_idx.entry(c).or_default().push((r, i));
        }
        self.col_index = col_idx;
        self.dead_count = 0;
    }

    /// Remap node indices according to `map`. Drops edges where either endpoint is None.
    /// Preserves tick timestamps for surviving edges.
    pub fn reindex(&mut self, map: &[Option<usize>]) {
        let new_num_nodes = map
            .iter()
            .filter_map(|&m| m)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        // Collect all live edges from CSR + pending, merging duplicates
        let mut all: HashMap<(usize, usize), (f64, u64)> = HashMap::new();
        for (&(a, b), &idx) in &self.csr_index {
            let w = self.csr_vals[idx];
            if w > 0.0 {
                let t = self.ticks.get(&(a, b)).copied().unwrap_or(0);
                all.insert((a, b), (w, t));
            }
        }
        for (&(a, b), &w) in &self.pending {
            if w > 0.0 {
                let t = self.ticks.get(&(a, b)).copied().unwrap_or(0);
                all.entry((a, b))
                    .and_modify(|(ew, et)| {
                        *ew += w;
                        *et = (*et).max(t);
                    })
                    .or_insert((w, t));
            }
        }
        // Re-key survivors
        let mut new_pending: HashMap<(usize, usize), f64> = HashMap::new();
        let mut new_ticks: HashMap<(usize, usize), u64> = HashMap::new();
        for ((a, b), (w, t)) in all {
            let new_a = map.get(a).copied().flatten();
            let new_b = map.get(b).copied().flatten();
            if let (Some(na), Some(nb)) = (new_a, new_b) {
                let key = if na <= nb { (na, nb) } else { (nb, na) };
                new_pending.insert(key, w);
                new_ticks.insert(key, t);
            }
        }
        // Reset CSR and rebuild via compact
        self.num_nodes = new_num_nodes;
        self.csr_rows = vec![0; new_num_nodes + 1];
        self.csr_cols = Vec::new();
        self.csr_vals = Vec::new();
        self.csr_index = HashMap::new();
        self.col_index = HashMap::new();
        self.dead_count = 0;
        self.pending = new_pending;
        self.ticks = new_ticks;
        self.compact();
    }

    // compact when >25% of CSR slots are dead
    fn maybe_compact(&mut self) {
        let nnz = self.csr_vals.len();
        if nnz > 0 && self.dead_count * 4 > nnz {
            self.compact();
        }
    }

    /// CSR-only weight lookup (ignores pending buffer).
    fn csr_weight(&self, key: (usize, usize)) -> f64 {
        self.csr_index
            .get(&key)
            .map(|&i| self.csr_vals[i])
            .unwrap_or(0.0)
    }
}

impl EdgeStore<usize> for SparseEdgeStore {
    fn get_weight(&self, a: usize, b: usize) -> f64 {
        let key = canonical(a, b);
        let csr = self.csr_weight(key);
        let pending = self.pending.get(&key).copied().unwrap_or(0.0);
        // Invariant: a key is never non-zero in both CSR and pending simultaneously.
        // apply_increments routes to CSR when the key exists in csr_index, pending otherwise.
        // compact() merges pending into CSR and clears pending. set_weight clears pending for the key.
        debug_assert!(
            csr == 0.0 || pending == 0.0,
            "get_weight: key {key:?} non-zero in both CSR ({csr}) and pending ({pending})"
        );
        if pending > 0.0 { csr + pending } else { csr }
    }

    fn set_weight(&mut self, a: usize, b: usize, w: f64) {
        assert!(
            a < self.num_nodes && b < self.num_nodes,
            "set_weight: node index out of range (a={a}, b={b}, num_nodes={})",
            self.num_nodes
        );
        let key = canonical(a, b);
        if w <= 0.0 {
            if let Some(&idx) = self.csr_index.get(&key)
                && self.csr_vals[idx] > 0.0
            {
                self.csr_vals[idx] = 0.0;
                self.dead_count += 1;
            }
            self.pending.remove(&key);
            self.ticks.remove(&key);
        } else {
            if let Some(&idx) = self.csr_index.get(&key) {
                let was_dead = self.csr_vals[idx] == 0.0;
                self.csr_vals[idx] = w;
                if was_dead {
                    self.dead_count = self.dead_count.saturating_sub(1);
                }
            } else {
                self.pending.insert(key, w);
            }
        }
        self.maybe_compact();
    }

    fn scale_all(&mut self, factor: f64) {
        for v in &mut self.csr_vals {
            *v *= factor;
        }
        for v in self.pending.values_mut() {
            *v *= factor;
        }
    }

    fn apply_increments(&mut self, deltas: &[(usize, usize, f64)]) {
        for &(a, b, d) in deltas {
            let key = canonical(a, b);
            if let Some(&idx) = self.csr_index.get(&key) {
                self.csr_vals[idx] += d;
            } else {
                *self.pending.entry(key).or_insert(0.0) += d;
            }
        }
    }

    fn prune_below(&mut self, threshold: f64) -> usize {
        let mut count = 0;
        for v in &mut self.csr_vals {
            if *v > 0.0 && *v < threshold {
                *v = 0.0;
                count += 1;
                self.dead_count += 1;
            }
        }
        let before = self.pending.len();
        self.pending.retain(|_, v| *v >= threshold);
        count += before - self.pending.len();
        self.maybe_compact();
        count
    }

    fn prune_inactive(&mut self, current_tick: u64, p1: u64) -> usize {
        let dead: Vec<(usize, usize)> = self
            .csr_index
            .keys()
            .filter(|&&key| {
                self.csr_vals[self.csr_index[&key]] > 0.0
                    && current_tick.saturating_sub(*self.ticks.get(&key).unwrap_or(&0)) > p1 // > not >=: survives p1-th inactive tick [Hoya §4]
            })
            .copied()
            .collect();
        let mut count = dead.len();
        for key in dead {
            let idx = self.csr_index[&key];
            self.csr_vals[idx] = 0.0;
            self.dead_count += 1;
            self.ticks.remove(&key);
        }
        let pending_dead: Vec<(usize, usize)> = self
            .pending
            .keys()
            .filter(|k| current_tick.saturating_sub(*self.ticks.get(k).unwrap_or(&0)) > p1) // > not >=: survives p1-th inactive tick [Hoya §4]
            .copied()
            .collect();
        let pending_count = pending_dead.len();
        for k in pending_dead {
            self.pending.remove(&k);
        }
        count += pending_count;
        self.maybe_compact();
        count
    }

    fn neighbors(&self, node: usize) -> Vec<(usize, f64)> {
        let mut result = Vec::new();
        // CSR row for node (upper-triangle: node is row)
        if node < self.num_nodes {
            let start = self.csr_rows[node];
            let end = self.csr_rows[node + 1];
            for i in start..end {
                if self.csr_vals[i] > 0.0 {
                    result.push((self.csr_cols[i], self.csr_vals[i]));
                }
            }
        }
        // Reverse column index: O(degree) lookup instead of O(N) scan
        if let Some(entries) = self.col_index.get(&node) {
            for &(row, offset) in entries {
                let w = self.csr_vals[offset];
                if w > 0.0 {
                    result.push((row, w));
                }
            }
        }
        // Pending edges
        for (&(a, b), &w) in &self.pending {
            if w > 0.0 {
                if a == node {
                    result.push((b, w));
                } else if b == node {
                    result.push((a, w));
                }
            }
        }
        result
    }

    fn touch(&mut self, a: usize, b: usize, tick: u64) {
        self.ticks.insert(canonical(a, b), tick);
    }

    fn last_active(&self, a: usize, b: usize) -> u64 {
        *self.ticks.get(&canonical(a, b)).unwrap_or(&0)
    }

    fn edge_count(&self) -> usize {
        let csr_live = self.csr_vals.iter().filter(|&&v| v > 0.0).count();
        let pending_live = self.pending.values().filter(|&&v| v > 0.0).count();
        csr_live + pending_live
    }
}

impl Reindex for SparseEdgeStore {
    fn reindex_for_compact(&mut self, map: &[Option<usize>]) {
        self.reindex(map);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::EdgeStore;

    #[test]
    fn sparse_get_set_symmetric() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 3, 0.7);
        assert!((s.get_weight(0, 3) - 0.7).abs() < 1e-10);
        assert!((s.get_weight(3, 0) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn sparse_edge_count() {
        let mut s = SparseEdgeStore::new(4);
        assert_eq!(s.edge_count(), 0);
        s.set_weight(0, 1, 0.5);
        s.set_weight(1, 2, 0.3);
        assert_eq!(s.edge_count(), 2);
    }

    #[test]
    fn sparse_from_triplets_valid() {
        let s = SparseEdgeStore::from_triplets(3, &[(0, 1, 0.5), (1, 2, 0.3)]);
        assert!((s.get_weight(0, 1) - 0.5).abs() < 1e-10);
        assert!((s.get_weight(1, 2) - 0.3).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn sparse_from_triplets_panics_out_of_bounds() {
        SparseEdgeStore::from_triplets(2, &[(0, 5, 0.5)]);
    }

    #[test]
    fn sparse_grow_returns_true_on_upsize() {
        let mut s = SparseEdgeStore::new(4);
        assert!(s.grow(8));
        assert_eq!(s.num_nodes(), 8);
    }

    #[test]
    fn sparse_grow_returns_false_on_downsize() {
        let mut s = SparseEdgeStore::new(4);
        assert!(!s.grow(2));
        assert_eq!(s.num_nodes(), 4);
    }

    #[test]
    fn sparse_prune_below_threshold() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        s.set_weight(0, 2, 0.0005);
        let pruned = s.prune_below(0.001);
        assert_eq!(pruned, 1);
        assert_eq!(s.edge_count(), 1);
    }

    #[test]
    fn sparse_prune_inactive() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        s.set_weight(0, 2, 0.5);
        s.touch(0, 1, 50);
        // (0,2) last_active=0, current=20, p1=10 -> extinct
        let pruned = s.prune_inactive(20, 10);
        assert_eq!(pruned, 1);
        assert!(s.get_weight(0, 1) > 0.0);
        assert_eq!(s.get_weight(0, 2), 0.0);
    }

    #[test]
    fn sparse_touch_records_tick() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        assert_eq!(s.last_active(0, 1), 0);
        s.touch(0, 1, 77);
        assert_eq!(s.last_active(0, 1), 77);
        assert_eq!(s.last_active(1, 0), 77); // symmetric
    }

    #[test]
    fn sparse_autocompact_triggers_at_25pct_dead() {
        let mut s = SparseEdgeStore::new(20);
        for i in 0..10usize {
            s.set_weight(i, i + 10, 0.5);
        }
        // Force into CSR by compacting
        s.compact();
        assert_eq!(s.edge_count(), 10);
        // Kill 3 edges -- these are now in CSR as dead slots
        s.set_weight(0, 10, 0.0);
        s.set_weight(1, 11, 0.0);
        s.set_weight(2, 12, 0.0);
        // dead_count=3, nnz=10, 3*4=12 > 10 -> auto-compact fires
        // After compact, only 7 live edges remain
        assert_eq!(s.edge_count(), 7);
    }

    #[test]
    #[should_panic(expected = "node index out of range")]
    fn set_weight_panics_on_out_of_range_index() {
        let mut store = SparseEdgeStore::new(3);
        store.set_weight(0, 5, 1.0); // 5 >= num_nodes=3
    }

    #[test]
    fn set_weight_then_compact_valid() {
        let mut store = SparseEdgeStore::new(3);
        store.set_weight(0, 1, 0.5);
        store.set_weight(1, 2, 0.8);
        // force compact
        for _ in 0..100 {
            store.set_weight(0, 2, 0.3);
        }
        // no panic
    }

    #[test]
    fn sparse_neighbors_both_directions() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        s.set_weight(0, 2, 0.3);
        let mut ns: Vec<usize> = s.neighbors(0).into_iter().map(|(k, _)| k).collect();
        ns.sort();
        assert_eq!(ns, vec![1, 2]);
        let ns1: Vec<usize> = s.neighbors(1).into_iter().map(|(k, _)| k).collect();
        assert!(ns1.contains(&0));
    }

    #[test]
    fn neighbors_col_side_correct_after_multiple_compacts() {
        let mut s = SparseEdgeStore::new(100);
        for row in 0..10usize {
            s.set_weight(row, 50, 0.5 + row as f64 * 0.01);
        }
        s.compact();
        for row in 10..20usize {
            s.set_weight(row, 50, 0.3);
        }
        s.compact();
        let neighbors: Vec<(usize, f64)> = s.neighbors(50);
        assert_eq!(neighbors.len(), 20);
        let w0 = neighbors
            .iter()
            .find(|&&(k, _)| k == 0)
            .map(|&(_, w)| w)
            .unwrap();
        assert!((w0 - 0.5).abs() < 1e-10);
        let w10 = neighbors
            .iter()
            .find(|&&(k, _)| k == 10)
            .map(|&(_, w)| w)
            .unwrap();
        assert!((w10 - 0.3).abs() < 1e-10);
    }

    #[test]
    fn sparse_edge_store_reindex_drops_extinct_edges() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        s.set_weight(1, 2, 0.3);
        s.compact();
        // kernel 1 extinct -> both edges involving 1 dropped
        let map = vec![Some(0), None, Some(1), Some(2)];
        s.reindex(&map);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn sparse_edge_store_reindex_remaps_survivors() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 2, 0.7); // old 0->new 0, old 2->new 1
        s.compact();
        let map = vec![Some(0), None, Some(1), Some(2)];
        s.reindex(&map);
        assert_eq!(s.edge_count(), 1);
        assert!((s.get_weight(0, 1) - 0.7).abs() < 1e-10);
        assert_eq!(s.num_nodes(), 3);
    }

    #[test]
    fn neighbors_excludes_dead_col_side_edges() {
        let mut s = SparseEdgeStore::new(10);
        s.set_weight(0, 5, 0.5);
        s.set_weight(1, 5, 0.5);
        s.compact();
        s.set_weight(0, 5, 0.0);
        let ns: Vec<usize> = s.neighbors(5).into_iter().map(|(k, _)| k).collect();
        assert!(!ns.contains(&0));
        assert!(ns.contains(&1));
    }

    // --- Finding 26: additional SparseEdgeStore tests ---

    #[test]
    fn sparse_get_weight_pending_edge() {
        let mut s = SparseEdgeStore::new(4);
        // Insert directly into pending without compact
        s.set_weight(0, 1, 0.5);
        // edge is in pending (or CSR after maybe_compact, but get_weight covers both)
        assert!((s.get_weight(0, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn sparse_apply_increments_csr_path() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        s.compact(); // force into CSR
        s.apply_increments(&[(0, 1, 0.2)]);
        assert!((s.get_weight(0, 1) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn sparse_apply_increments_pending_path() {
        let mut s = SparseEdgeStore::new(4);
        // Don't compact — keep in pending
        s.apply_increments(&[(0, 1, 0.3)]);
        assert!((s.get_weight(0, 1) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn sparse_scale_all_applies_to_csr_and_pending() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 1.0);
        s.compact(); // CSR
        s.set_weight(2, 3, 1.0); // pending (or CSR after maybe_compact)
        // Force a state where we have both CSR and pending
        let mut s2 = SparseEdgeStore::new(10);
        s2.set_weight(0, 1, 1.0);
        s2.compact();
        // Now add a pending edge by inserting to a new key
        s2.set_weight(4, 5, 1.0);
        // Check pending_count > 0 or edge_count includes it
        s2.scale_all(0.5);
        assert!((s2.get_weight(0, 1) - 0.5).abs() < 1e-10);
        assert!((s2.get_weight(4, 5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn sparse_set_weight_twice_csr_path() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        s.compact();
        s.set_weight(0, 1, 0.8);
        assert!((s.get_weight(0, 1) - 0.8).abs() < 1e-10);
        s.set_weight(0, 1, 0.2);
        assert!((s.get_weight(0, 1) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn sparse_set_weight_twice_pending_path() {
        let mut s = SparseEdgeStore::new(10);
        // Use indices far apart so auto-compact doesn't trigger easily
        s.set_weight(0, 1, 0.5);
        s.set_weight(2, 3, 0.5);
        s.set_weight(4, 5, 0.5);
        s.set_weight(6, 7, 0.5);
        s.compact();
        // Now add a new pending edge
        s.set_weight(8, 9, 0.3);
        s.set_weight(8, 9, 0.7); // overwrite — goes to CSR since maybe_compact may fire
        assert!((s.get_weight(8, 9) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn sparse_from_triplets_duplicate_pair() {
        let s = SparseEdgeStore::from_triplets(3, &[(0, 1, 0.3), (0, 1, 0.4)]);
        // Duplicates accumulate
        assert!((s.get_weight(0, 1) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn sparse_prune_inactive_pending_only_edges() {
        let mut s = SparseEdgeStore::new(10);
        // Create edges that stay in pending
        s.set_weight(0, 1, 0.5);
        s.set_weight(2, 3, 0.5);
        s.set_weight(4, 5, 0.5);
        s.set_weight(6, 7, 0.5);
        s.compact();
        // Add a new pending edge without touch
        s.set_weight(8, 9, 0.5);
        // current_tick=100, p1=10 -> pending edge (last_active=0) is inactive
        let pruned = s.prune_inactive(100, 10);
        // All edges without touch are inactive (last_active=0, 100-0=100 > 10)
        assert!(pruned >= 1);
        assert_eq!(s.get_weight(8, 9), 0.0);
    }

    #[test]
    fn sparse_neighbors_includes_pending() {
        let mut s = SparseEdgeStore::new(10);
        s.set_weight(0, 1, 0.5);
        s.compact();
        // Add pending edge
        s.set_weight(0, 9, 0.3);
        let ns: Vec<usize> = s.neighbors(0).into_iter().map(|(k, _)| k).collect();
        assert!(ns.contains(&1));
        assert!(ns.contains(&9));
    }

    #[test]
    fn sparse_grow_then_set_weight_new_index() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        s.grow(8);
        s.set_weight(4, 7, 0.9);
        assert!((s.get_weight(4, 7) - 0.9).abs() < 1e-10);
    }

    #[test]
    fn get_weight_csr_and_pending_are_disjoint_per_key() {
        let mut s = SparseEdgeStore::new(4);

        // Edge (0,1): put into CSR via set_weight + compact
        s.set_weight(0, 1, 0.5);
        s.compact(); // flushes any pending state; (0,1) is now in csr_index

        // Edge (2,3): new key, not in csr_index — apply_increments routes to pending
        s.apply_increments(&[(2usize, 3, 0.3)]);
        assert_eq!(s.pending_count(), 1); // (2,3) is in pending

        // (0,1) also gets an increment — routes to CSR (key exists in csr_index)
        s.apply_increments(&[(0usize, 1, 0.1)]);
        assert_eq!(s.pending_count(), 1); // still only (2,3) in pending

        // get_weight reads from the correct path for each
        assert!((s.get_weight(0, 1) - 0.6).abs() < 1e-10); // CSR: 0.5 + 0.1
        assert!((s.get_weight(2, 3) - 0.3).abs() < 1e-10); // pending: 0.3
        // debug_assert in get_weight must not fire for either key
    }

    #[test]
    fn sparse_grow_same_size_is_noop() {
        let mut s = SparseEdgeStore::new(4);
        assert!(!s.grow(4));
        assert_eq!(s.num_nodes(), 4);
    }

    #[test]
    fn sparse_reindex_empty_no_panic() {
        let mut s = SparseEdgeStore::new(4);
        let map: Vec<Option<usize>> = vec![Some(0), Some(1), Some(2), Some(3)];
        s.reindex(&map);
        assert_eq!(s.edge_count(), 0);
    }

    #[test]
    fn sparse_reindex_preserves_tick_timestamps() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        s.touch(0, 1, 42);
        s.compact();
        let map = vec![Some(0), Some(1), Some(2), Some(3)];
        s.reindex(&map);
        assert_eq!(s.last_active(0, 1), 42);
    }

    #[test]
    fn sparse_edge_count_after_scale_all_zero() {
        let mut s = SparseEdgeStore::new(4);
        s.set_weight(0, 1, 0.5);
        s.set_weight(1, 2, 0.3);
        s.compact();
        s.scale_all(0.0);
        // scale_all zeroes weights but doesn't remove entries from CSR
        // edge_count counts only v > 0.0, so should be 0
        assert_eq!(s.edge_count(), 0);
    }
}
