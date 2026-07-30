/// Global 2D emotion state. [Hoya pp. 214–215] \[DIRECT\]
/// E₁: ecstasy ↔ misery — 7 discrete levels ∈ [-3.0, 3.0]
/// E₂: rage ↔ relief    — 5 discrete levels ∈ [-2.0, 2.0]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmotionState {
    pub e1: f64,
    pub e2: f64,
}

impl Default for EmotionState {
    fn default() -> Self {
        Self { e1: 0.0, e2: 0.0 }
    }
}

/// Per-kernel emotional state variables. Parallel to KernelStore.
/// `vars[i]` = `[e_i^1, e_i^2]`: emotion variables for kernel i.
/// Initialised to [0.0, 0.0] on kernel creation.
/// [Hoya Eq. 10.8, p. 257] \[DIRECT\]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmotionStore {
    vars: Vec<[f64; 2]>,
}

impl Default for EmotionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionStore {
    pub fn new() -> Self {
        Self { vars: Vec::new() }
    }

    pub fn push(&mut self) {
        self.vars.push([0.0, 0.0]);
    }

    /// Panics if `i >= len()`.
    pub fn get(&self, i: usize) -> [f64; 2] {
        self.vars[i]
    }

    /// Panics if `i >= len()`.
    pub fn set(&mut self, i: usize, v: [f64; 2]) {
        self.vars[i] = v;
    }

    pub fn len(&self) -> usize {
        self.vars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Compact in lockstep with KernelStore::compact_extinct.
    /// `map`: old_idx → new_idx (None = extinct, dropped).
    pub fn compact_with_map(&mut self, map: &[Option<usize>]) {
        debug_assert_eq!(
            map.len(),
            self.vars.len(),
            "compact_with_map: map.len() must equal EmotionStore len"
        );
        let mut new_vars = Vec::new();
        for (i, &m) in map.iter().enumerate() {
            if m.is_some() {
                new_vars.push(self.vars[i]);
            }
        }
        self.vars = new_vars;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emotion_store_push_grows_len() {
        let mut s = EmotionStore::new();
        assert_eq!(s.len(), 0);
        s.push();
        assert_eq!(s.len(), 1);
        s.push();
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn emotion_store_get_set_roundtrip() {
        let mut s = EmotionStore::new();
        s.push();
        s.set(0, [1.0, 2.0]);
        assert_eq!(s.get(0), [1.0, 2.0]);
    }

    #[test]
    fn emotion_store_new_kernel_is_neutral() {
        let mut s = EmotionStore::new();
        s.push();
        assert_eq!(s.get(0), [0.0, 0.0]);
    }

    #[test]
    fn emotion_store_compact_drops_extinct() {
        let mut s = EmotionStore::new();
        s.push();
        s.set(0, [1.0, 2.0]);
        s.push();
        s.set(1, [3.0, 4.0]);
        s.push();
        s.set(2, [5.0, 6.0]);
        let map = vec![Some(0), None, Some(1)];
        s.compact_with_map(&map);
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0), [1.0, 2.0]);
        assert_eq!(s.get(1), [5.0, 6.0]);
    }

    #[test]
    fn compact_with_map_all_surviving_identity() {
        let mut s = EmotionStore::new();
        s.push();
        s.set(0, [1.0, 0.5]);
        s.push();
        s.set(1, [2.0, -1.0]);
        let map = vec![Some(0), Some(1)];
        s.compact_with_map(&map);
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0), [1.0, 0.5]);
        assert_eq!(s.get(1), [2.0, -1.0]);
    }

    #[test]
    fn compact_with_map_empty_no_panic() {
        let mut s = EmotionStore::new();
        let map: Vec<Option<usize>> = vec![];
        s.compact_with_map(&map);
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn compact_with_map_boundary_last_index() {
        let mut s = EmotionStore::new();
        s.push();
        s.set(0, [0.1, 0.2]);
        s.push();
        s.set(1, [0.3, 0.4]);
        s.push();
        s.set(2, [0.5, 0.6]);
        // keep only last
        let map = vec![None, None, Some(0)];
        s.compact_with_map(&map);
        assert_eq!(s.len(), 1);
        assert_eq!(s.get(0), [0.5, 0.6]);
    }
}
