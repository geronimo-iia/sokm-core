use crate::store::KernelStore;

/// Working memory bounded by excitation-based eviction. [Hoya p.164, Eq 10.5]
///
/// Holds indices into a kernel store. When at capacity, evicts
/// the kernel with the lowest excitation count (ε_i). [Hoya p.164]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Stm {
    capacity: usize,
    indices: Vec<usize>,
}

impl Stm {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "STM capacity must be > 0");
        Self {
            capacity,
            indices: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.indices.len() >= self.capacity
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Remap kernel indices after compact_extinct.
    /// Slots mapped to None (extinct kernel) are dropped.
    /// Capacity is unchanged — freed slots are available for future activations.
    /// Precondition: not called mid-tick. KernelGraph::compact is responsible for
    /// ensuring this — it must only be called between ticks.
    pub fn reindex(&mut self, map: &[Option<usize>]) {
        self.indices = self
            .indices
            .iter()
            .filter_map(|&idx| map.get(idx).copied().flatten())
            .collect();
    }

    /// Add kernel_idx to STM. If at capacity, evict the kernel with the
    /// lowest excitation count. [Hoya p.164]
    pub fn update(&mut self, kernel_idx: usize, store: &impl KernelStore) {
        if self.is_full() {
            let evict_pos = self
                .indices
                .iter()
                .enumerate()
                .min_by_key(|&(_, &idx)| store.excitation(idx))
                .map(|(pos, _)| pos)
                .unwrap();
            self.indices.remove(evict_pos);
        }
        self.indices.push(kernel_idx);
    }

    /// Blended STM output. [Hoya Eq 10.5]
    ///
    /// Averages active kernel centroids, then blends with current input x:
    /// o_STM = lambda * mean_centroid + (1 - lambda) * x
    pub fn blend_output(&self, x: &[f64], store: &impl KernelStore, lambda: f64) -> Vec<f64> {
        if self.indices.is_empty() || x.is_empty() {
            return x.to_vec();
        }
        let dim = x.len();
        let mut mean_centroid = vec![0.0f64; dim];
        for &idx in &self.indices {
            for (d, &v) in store.centroid(idx).iter().enumerate().take(dim) {
                mean_centroid[d] += v;
            }
        }
        let n = self.indices.len() as f64;
        mean_centroid.iter_mut().for_each(|v| *v /= n);

        mean_centroid
            .iter()
            .zip(x)
            .map(|(c, xi)| lambda * c + (1.0 - lambda) * xi)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::AosKernelStore;

    fn make_store(n: usize) -> AosKernelStore {
        let mut store = AosKernelStore::new();
        for i in 0..n {
            store.push(&[i as f64], 1.0, Some(0));
            // set excitation to i by incrementing i times
            for _ in 0..i {
                store.incr_excitation(i);
            }
        }
        store
    }

    #[test]
    fn stm_starts_empty() {
        let s = Stm::new(4);
        assert_eq!(s.len(), 0);
        assert!(!s.is_full());
    }

    #[test]
    fn stm_update_adds_kernel_index() {
        let mut s = Stm::new(4);
        let store = make_store(5);
        s.update(2, &store);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn stm_evicts_lowest_excitation_when_full() {
        let mut s = Stm::new(3);
        let store = make_store(10);
        // Add kernels 5, 6, 7 (excitations 5, 6, 7) — fills STM
        s.update(5, &store);
        s.update(6, &store);
        s.update(7, &store);
        assert!(s.is_full());
        // Add kernel 9 (excitation 9) — should evict kernel with lowest excitation (5)
        s.update(9, &store);
        assert_eq!(s.len(), 3);
        // kernel 5 (excitation=5) should be gone
        let indices = s.indices();
        assert!(!indices.contains(&5));
        assert!(indices.contains(&9));
    }

    #[test]
    fn stm_blend_eq_10_5() {
        // o_STM[i] = lambda * centroid[i] + (1 - lambda) * x[i]
        let mut s = Stm::new(4);
        let mut store = AosKernelStore::new();
        store.push(&[2.0], 1.0, Some(0));
        s.update(0, &store);
        let x = vec![0.0];
        let lambda = 0.7;
        let blended = s.blend_output(&x, &store, lambda);
        let expected = lambda * 2.0 + (1.0 - lambda) * 0.0;
        assert!((blended[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn stm_reindex_drops_extinct_slots() {
        let mut s = Stm::new(4);
        let store = make_store(5);
        s.update(0, &store);
        s.update(1, &store);
        s.update(2, &store);
        // kernel 1 is extinct → map[1]=None; old 0→new 0, old 2→new 1
        let map = vec![Some(0), None, Some(1), Some(2), Some(3)];
        s.reindex(&map);
        assert_eq!(s.len(), 2);
        // new indices are 0 (from old 0) and 1 (from old 2)
        let indices = s.indices();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    #[test]
    fn stm_reindex_remaps_survivors() {
        let mut s = Stm::new(4);
        let store = make_store(5);
        s.update(0, &store);
        s.update(2, &store);
        let map = vec![Some(0), None, Some(1), Some(2), Some(3)];
        s.reindex(&map);
        // old 0→new 0, old 2→new 1
        let indices = s.indices();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    #[test]
    fn stm_capacity_unchanged_after_reindex() {
        let mut s = Stm::new(4);
        let store = make_store(5);
        s.update(0, &store);
        s.update(1, &store);
        s.update(2, &store);
        let map = vec![Some(0), None, Some(1), Some(2)];
        s.reindex(&map);
        // capacity stays 4; only occupancy drops
        assert!(!s.is_full());
        let store2 = make_store(5);
        s.update(0, &store2);
        s.update(1, &store2);
        // can fill back up to capacity without panic
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn stm_blend_multiple_kernels_averages() {
        let mut s = Stm::new(4);
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0));
        store.push(&[4.0], 1.0, Some(0));
        s.update(0, &store);
        s.update(1, &store);
        let x = vec![2.0];
        let blended = s.blend_output(&x, &store, 0.5);
        // average centroid = 2.0; blend: 0.5*2.0 + 0.5*2.0 = 2.0
        assert!((blended[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn blend_output_empty_stm_returns_x() {
        let s = Stm::new(4);
        let store = AosKernelStore::new();
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(s.blend_output(&x, &store, 0.7), x);
    }

    #[test]
    fn blend_output_lambda_zero_returns_x() {
        let mut s = Stm::new(4);
        let mut store = AosKernelStore::new();
        store.push(&[10.0], 1.0, Some(0));
        s.update(0, &store);
        let x = vec![5.0];
        let out = s.blend_output(&x, &store, 0.0);
        assert!((out[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn blend_output_lambda_one_returns_centroid_mean() {
        let mut s = Stm::new(4);
        let mut store = AosKernelStore::new();
        store.push(&[10.0], 1.0, Some(0));
        s.update(0, &store);
        let x = vec![5.0];
        let out = s.blend_output(&x, &store, 1.0);
        assert!((out[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn capacity_one_second_insert_evicts_first() {
        let mut s = Stm::new(1);
        let store = make_store(5);
        s.update(3, &store); // excitation=3
        s.update(4, &store); // excitation=4 — evicts 3
        assert_eq!(s.len(), 1);
        assert_eq!(s.indices(), &[4]);
    }

    #[test]
    fn tie_break_evicts_fifo() {
        // Two kernels with equal excitation — first inserted is evicted (min_by_key is stable)
        let mut s = Stm::new(2);
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0)); // excitation=0
        store.push(&[1.0], 1.0, Some(0)); // excitation=0
        store.push(&[2.0], 1.0, Some(0)); // excitation=0
        s.update(0, &store);
        s.update(1, &store);
        // Full. Insert 2 — tie between 0 and 1, both excitation=0.
        // min_by_key returns first match (pos 0 = kernel 0) → evict kernel 0
        s.update(2, &store);
        assert!(
            !s.indices().contains(&0),
            "FIFO tie-break: first inserted evicted"
        );
        assert!(s.indices().contains(&1));
        assert!(s.indices().contains(&2));
    }

    #[test]
    fn reindex_empty_no_panic() {
        let mut s = Stm::new(4);
        let map = vec![Some(0), None, Some(1)];
        s.reindex(&map); // no panic
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn reindex_with_index_beyond_map_length() {
        let mut s = Stm::new(4);
        let store = make_store(10);
        s.update(5, &store);
        s.update(8, &store);
        // Map only covers indices 0..3 — indices 5 and 8 are beyond map length → filtered out
        let map = vec![Some(0), Some(1), Some(2)];
        s.reindex(&map);
        assert_eq!(s.len(), 0);
    }
}
