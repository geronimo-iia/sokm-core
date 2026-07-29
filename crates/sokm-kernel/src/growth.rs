use crate::activation::gaussian;
use crate::config::KernelConfig;
use crate::store::KernelStore;

/// Maximum direct gaussian activation over all kernels for input x.
/// Returns 0.0 when kernel set is empty. Extinct kernels contribute 0.
pub fn max_activation(store: &impl KernelStore, x: &[f64]) -> f64 {
    compute_scores(store, x).iter().cloned().fold(0.0, f64::max)
}

/// Direct-activation-only growth check: max_i K_i(x) < theta_k. [Hoya Eq 3.8 gate]
///
/// Use when no edge store is available (e.g. ECS systems without KernelGraph).
/// KernelGraph::tick uses the full Hoya growth check (direct + propagated) internally.
pub fn should_grow_direct(store: &impl KernelStore, x: &[f64], cfg: &KernelConfig) -> bool {
    (0..store.len()).all(|i| {
        store.is_extinct(i) || gaussian(x, store.centroid(i), store.sigma(i)) < cfg.theta_k
    })
}

/// Gaussian activation scores for all kernels in `store` against input `x`.
/// Returns `Vec<f64>` of length `store.len()`, index i = K_i(x).
///
/// Hardcodes `gaussian` — update when KernelGraph exposes activation fn selection
/// (SIMD path, v0.4+).
pub fn compute_scores(store: &impl KernelStore, x: &[f64]) -> Vec<f64> {
    (0..store.len())
        .map(|i| {
            if store.is_extinct(i) {
                0.0
            } else {
                gaussian(x, store.centroid(i), store.sigma(i))
            }
        })
        .collect()
}

/// Add a new kernel centred at x with sigma_0 and the given class label.
pub fn grow(store: &mut impl KernelStore, x: &[f64], cfg: &KernelConfig, class: Option<u32>) {
    store.push(x, cfg.sigma_0, class);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KernelConfig;
    use crate::store::AosKernelStore;

    fn cfg() -> KernelConfig {
        KernelConfig::default() // theta_k=0.1, sigma_0=1.0
    }

    #[test]
    fn max_activation_empty_kernels_returns_zero() {
        let store = AosKernelStore::new();
        assert_eq!(max_activation(&store, &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn max_activation_at_centroid_returns_one() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0, 2.0], 1.0, Some(0));
        let score = max_activation(&store, &[1.0, 2.0]);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn max_activation_returns_highest_across_kernels() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0));
        store.push(&[5.0], 1.0, Some(0));
        let score = max_activation(&store, &[5.0]);
        assert!((score - 1.0).abs() < 1e-10); // exact match with kernel 1
    }

    #[test]
    fn growth_direct_blocked_when_kernel_explains_input() {
        // x == centroid → max_activation == 1.0 > theta_k=0.1
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(0));
        assert!(!should_grow_direct(&store, &[1.0], &cfg()));
    }

    #[test]
    fn growth_direct_fires_when_no_kernel_explains_input() {
        // All kernels far away → max_activation < theta_k
        let mut store = AosKernelStore::new();
        store.push(&[100.0], 1.0, Some(0));
        assert!(should_grow_direct(&store, &[0.0], &cfg()));
    }

    #[test]
    fn growth_fires_on_empty_kernels() {
        let store = AosKernelStore::new();
        assert!(should_grow_direct(&store, &[1.0], &cfg()));
    }

    #[test]
    fn grow_adds_kernel_with_correct_fields() {
        let mut store = AosKernelStore::new();
        let cfg = cfg();
        grow(&mut store, &[1.0, 2.0], &cfg, Some(3));
        assert_eq!(store.len(), 1);
        assert_eq!(store.centroid(0), &[1.0, 2.0]);
        assert_eq!(store.sigma(0), cfg.sigma_0);
        assert_eq!(store.class_opt(0), Some(3));
        assert_eq!(store.excitation(0), 0);
    }

    #[test]
    fn grow_with_none_class_creates_unlabelled_kernel() {
        let mut store = AosKernelStore::new();
        let cfg = cfg();
        grow(&mut store, &[1.0], &cfg, None);
        assert!(store.class_opt(0).is_none());
    }

    #[test]
    fn extinct_kernel_score_is_zero() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0)); // at centroid → score = 1.0 normally
        store.mark_extinct(0);
        let scores = compute_scores(&store, &[0.0]);
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn extinct_kernel_does_not_suppress_growth() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0));
        store.mark_extinct(0);
        // extinct kernel scores 0.0, so should_grow_direct should return true
        assert!(should_grow_direct(&store, &[0.0], &cfg()));
    }

    #[test]
    fn growth_sets_sigma_to_sigma_0() {
        let mut store = AosKernelStore::new();
        let cfg = KernelConfig {
            sigma_0: 2.5,
            ..KernelConfig::default()
        };
        grow(&mut store, &[0.0], &cfg, Some(0));
        assert!((store.sigma(0) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn max_activation_all_extinct_returns_zero() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0));
        store.push(&[1.0], 1.0, Some(0));
        store.mark_extinct(0);
        store.mark_extinct(1);
        assert_eq!(max_activation(&store, &[0.0]), 0.0);
    }

    #[test]
    fn should_grow_direct_with_mixed_extinct_alive() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0)); // at input — would block growth if alive
        store.push(&[100.0], 1.0, Some(0)); // far away
        store.mark_extinct(0);
        // Extinct kernel at input must not suppress growth
        assert!(should_grow_direct(&store, &[0.0], &cfg()));
    }

    #[test]
    fn compute_scores_returns_vec_of_store_len() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0));
        store.push(&[1.0], 1.0, Some(0));
        store.push(&[2.0], 1.0, Some(0));
        let scores = compute_scores(&store, &[0.0]);
        assert_eq!(scores.len(), 3);
    }
}
