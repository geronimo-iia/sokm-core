use crate::unit::KernelUnit;

/// Abstract kernel storage backend.
pub trait KernelStore {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Centroid slice for kernel i: length d.
    fn centroid(&self, i: usize) -> &[f64];
    fn sigma(&self, i: usize) -> f64;
    /// Class label for kernel i. None if unlabelled (η = ∅). [Hoya §4.3]
    fn class_opt(&self, i: usize) -> Option<u32>;

    /// Assign class label to a previously unlabelled kernel. No-op if already labelled.
    fn set_class(&mut self, i: usize, class: u32);

    /// Returns the class label or zero if unset. Convenience for callers that
    /// treat unlabelled kernels as class 0.
    fn class_or_zero(&self, i: usize) -> u32 {
        self.class_opt(i).unwrap_or(0)
    }

    fn excitation(&self, i: usize) -> u64;
    fn incr_excitation(&mut self, i: usize);

    /// Logical tick of the last activation of kernel i.
    /// Returns 0 for a kernel that has never been activated.
    /// [Hoya pp. 80–99, Rule 3 inactivity check]
    fn last_activated(&self, i: usize) -> u64;

    /// Record that kernel i was activated at `tick`.
    /// Called by KernelGraph::tick immediately after incr_excitation.
    fn touch(&mut self, i: usize, tick: u64);

    /// Append new kernel.
    fn push(&mut self, centroid: &[f64], sigma: f64, class: Option<u32>);

    /// Index of kernel with globally lowest excitation. Returns None if empty.
    /// NOTE: scans ALL kernels — not the STM-local subset. Do NOT use inside
    /// Stm::update (which evicts the min-excitation kernel within STM indices only).
    fn min_excitation_idx(&self) -> Option<usize>;

    /// Mark kernel i as extinct. Skipped in all activation and growth computations.
    /// [Hoya pp. 80–99, Rule 3]
    fn mark_extinct(&mut self, i: usize);

    /// True if kernel i has been marked extinct.
    /// Returns false for out-of-bounds i (safe for serde compat with old snapshots).
    fn is_extinct(&self, i: usize) -> bool;

    /// Remove all extinct kernels and compact all parallel Vecs.
    /// Returns old_index → new_index mapping (None = kernel was extinct, removed).
    /// Invalidates all positional indices — caller must reindex edge store and STM.
    fn compact_extinct(&mut self) -> Vec<Option<usize>>;
}

/// AoS backend — wraps `Vec<KernelUnit>`. Identical behaviour to v0.1.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AosKernelStore {
    kernels: Vec<KernelUnit>,
}

impl AosKernelStore {
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
        }
    }
}

impl Default for AosKernelStore {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelStore for AosKernelStore {
    fn len(&self) -> usize {
        self.kernels.len()
    }
    fn centroid(&self, i: usize) -> &[f64] {
        &self.kernels[i].centroid
    }
    fn sigma(&self, i: usize) -> f64 {
        self.kernels[i].sigma
    }
    fn class_opt(&self, i: usize) -> Option<u32> {
        self.kernels[i].class
    }
    fn set_class(&mut self, i: usize, class: u32) {
        if self.kernels[i].class.is_none() {
            self.kernels[i].class = Some(class);
        }
    }
    fn excitation(&self, i: usize) -> u64 {
        self.kernels[i].excitation
    }
    fn incr_excitation(&mut self, i: usize) {
        self.kernels[i].increment_excitation()
    }
    fn push(&mut self, centroid: &[f64], sigma: f64, class: Option<u32>) {
        self.kernels
            .push(KernelUnit::new(centroid.to_vec(), sigma, class));
    }
    fn min_excitation_idx(&self) -> Option<usize> {
        self.kernels
            .iter()
            .enumerate()
            .filter(|(_, k)| !k.extinct)
            .min_by_key(|(_, k)| k.excitation)
            .map(|(i, _)| i)
    }
    fn last_activated(&self, i: usize) -> u64 {
        self.kernels[i].last_activated
    }
    fn touch(&mut self, i: usize, tick: u64) {
        self.kernels[i].last_activated = tick;
    }
    fn mark_extinct(&mut self, i: usize) {
        self.kernels[i].extinct = true;
    }
    fn is_extinct(&self, i: usize) -> bool {
        self.kernels.get(i).map(|k| k.extinct).unwrap_or(false)
    }
    fn compact_extinct(&mut self) -> Vec<Option<usize>> {
        let mut map = Vec::with_capacity(self.kernels.len());
        let mut new_idx = 0usize;
        let mut new_kernels = Vec::new();
        for i in 0..self.kernels.len() {
            if self.kernels[i].extinct {
                map.push(None);
            } else {
                map.push(Some(new_idx));
                new_kernels.push(self.kernels[i].clone());
                new_idx += 1;
            }
        }
        self.kernels = new_kernels;
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aos_store_push_and_len() {
        let mut store = AosKernelStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        store.push(&[1.0, 2.0], 0.5, Some(1));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        store.push(&[3.0, 4.0], 1.0, Some(0));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn aos_store_centroid_roundtrip() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0, 2.0, 3.0], 0.5, Some(7));
        assert_eq!(store.centroid(0), &[1.0, 2.0, 3.0]);
        assert_eq!(store.sigma(0), 0.5);
        assert_eq!(store.class_or_zero(0), 7);
        assert_eq!(store.excitation(0), 0);
        store.incr_excitation(0);
        assert_eq!(store.excitation(0), 1);
    }

    #[test]
    fn aos_push_none_class_creates_unlabelled() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, None);
        assert!(store.class_opt(0).is_none());
    }

    #[test]
    fn aos_class_opt_returns_none_for_unlabelled() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, None);
        assert_eq!(store.class_opt(0), None);
    }

    #[test]
    fn aos_set_class_assigns_label() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, None);
        store.set_class(0, 5);
        assert_eq!(store.class_opt(0), Some(5));
    }

    #[test]
    fn aos_set_class_noop_if_already_labelled() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(3));
        store.set_class(0, 7);
        assert_eq!(store.class_opt(0), Some(3));
    }

    #[test]
    fn aos_store_min_excitation_idx() {
        let mut store = AosKernelStore::new();
        assert_eq!(store.min_excitation_idx(), None);
        store.push(&[1.0], 1.0, Some(0));
        store.push(&[2.0], 1.0, Some(0));
        store.push(&[3.0], 1.0, Some(0));
        store.incr_excitation(0);
        store.incr_excitation(0);
        store.incr_excitation(1);
        // excitations: [2, 1, 0] — min at index 2
        assert_eq!(store.min_excitation_idx(), Some(2));
    }

    #[test]
    fn touch_records_tick() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0));
        store.touch(0, 42);
        assert_eq!(store.last_activated(0), 42);
    }

    #[test]
    fn last_activated_zero_on_new_kernel() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0));
        assert_eq!(store.last_activated(0), 0);
    }

    #[test]
    fn touch_updates_on_repeated_activation() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0));
        store.touch(0, 10);
        store.touch(0, 20);
        assert_eq!(store.last_activated(0), 20);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn aos_store_msgpack_roundtrip() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0, 2.0, 3.0], 0.5, Some(7));
        store.push(&[4.0, 5.0, 6.0], 1.0, Some(3));
        store.incr_excitation(0);
        store.touch(0, 42);

        let bytes = rmp_serde::to_vec(&store).unwrap();
        let back: AosKernelStore = rmp_serde::from_slice(&bytes).unwrap();

        assert_eq!(back.len(), 2);
        assert_eq!(back.centroid(0), &[1.0, 2.0, 3.0]);
        assert_eq!(back.sigma(0), 0.5);
        assert_eq!(back.class_opt(0), Some(7));
        assert_eq!(back.excitation(0), 1);
        assert_eq!(back.centroid(1), &[4.0, 5.0, 6.0]);
        assert_eq!(back.last_activated(0), 42);
        assert_eq!(back.last_activated(1), 0);
    }

    #[test]
    fn aos_is_extinct_false_on_new_kernel() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(0));
        assert!(!store.is_extinct(0));
    }

    #[test]
    fn aos_kernel_marked_extinct_after_mark_extinct() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(0));
        store.mark_extinct(0);
        assert!(store.is_extinct(0));
    }

    #[test]
    fn aos_compact_extinct_reduces_len() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(0));
        store.push(&[2.0], 1.0, Some(0));
        store.push(&[3.0], 1.0, Some(0));
        store.mark_extinct(1);
        store.compact_extinct();
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn aos_compact_extinct_returns_correct_mapping() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(0));
        store.push(&[2.0], 1.0, Some(0));
        store.push(&[3.0], 1.0, Some(0));
        store.mark_extinct(1);
        let map = store.compact_extinct();
        assert_eq!(map[0], Some(0));
        assert_eq!(map[1], None);
        assert_eq!(map[2], Some(1));
    }

    #[test]
    fn compact_extinct_on_empty_no_panic() {
        let mut store = AosKernelStore::new();
        let map = store.compact_extinct();
        assert!(map.is_empty());
    }

    #[test]
    fn compact_extinct_no_extinct_is_identity() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 0.5, Some(1));
        store.push(&[2.0], 1.0, Some(2));
        store.incr_excitation(0);
        store.touch(0, 42);
        let map = store.compact_extinct();
        assert_eq!(store.len(), 2);
        assert_eq!(map, vec![Some(0), Some(1)]);
        // Data preserved
        assert_eq!(store.centroid(0), &[1.0]);
        assert_eq!(store.sigma(0), 0.5);
        assert_eq!(store.excitation(0), 1);
        assert_eq!(store.last_activated(0), 42);
    }

    #[test]
    fn compact_extinct_preserves_fields_for_survivors() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 0.5, Some(1));
        store.push(&[2.0], 1.0, Some(2));
        store.push(&[3.0], 2.0, Some(3));
        store.incr_excitation(2);
        store.incr_excitation(2);
        store.touch(2, 99);
        store.mark_extinct(1);
        store.compact_extinct();
        // Old index 2 → new index 1
        assert_eq!(store.centroid(1), &[3.0]);
        assert_eq!(store.sigma(1), 2.0);
        assert_eq!(store.class_opt(1), Some(3));
        assert_eq!(store.excitation(1), 2);
        assert_eq!(store.last_activated(1), 99);
    }

    #[test]
    fn is_extinct_oob_returns_false() {
        let store = AosKernelStore::new();
        assert!(!store.is_extinct(0));
        assert!(!store.is_extinct(999));
    }

    #[test]
    fn min_excitation_idx_skips_extinct_kernel() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(0)); // index 0, excitation=0
        store.push(&[2.0], 1.0, Some(0)); // index 1, excitation=0
        store.incr_excitation(1); // excitation[1]=1
        store.mark_extinct(0);
        // Extinct kernel 0 has lower excitation but must be skipped.
        // Only live kernel 1 should be returned.
        assert_eq!(store.min_excitation_idx(), Some(1));
    }

    #[test]
    fn min_excitation_idx_all_extinct_returns_none() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(0));
        store.push(&[2.0], 1.0, Some(0));
        store.mark_extinct(0);
        store.mark_extinct(1);
        assert_eq!(store.min_excitation_idx(), None);
    }

    #[test]
    fn min_excitation_idx_live_kernel_with_higher_excitation_returned() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(0)); // index 0, excitation=0, extinct
        store.push(&[2.0], 1.0, Some(0)); // index 1, excitation=0
        store.push(&[3.0], 1.0, Some(0)); // index 2, excitation=0
        store.incr_excitation(1); // excitation[1]=1
        store.incr_excitation(2);
        store.incr_excitation(2); // excitation[2]=2
        store.mark_extinct(0);
        // Live minimum is index 1 (excitation=1)
        assert_eq!(store.min_excitation_idx(), Some(1));
    }

    #[test]
    #[should_panic]
    fn set_class_oob_panics() {
        let mut store = AosKernelStore::new();
        store.set_class(0, 1); // OOB — should panic
    }
}
