use sokm::{EdgeStore, SokmConfig, propagate};

use crate::config::KernelConfig;
use crate::growth::compute_scores;
use crate::store::KernelStore;

/// Index and combined score (direct + propagated) of the best-matching kernel.
/// Returns None if store is empty.
/// [Hoya Testing Algorithm steps 1–3, pp. 80–99]
pub fn best_match<S: EdgeStore<usize>>(
    store: &impl KernelStore,
    edges: &S,
    x: &[f64],
    sokm_cfg: &SokmConfig,
    kernel_cfg: &KernelConfig,
) -> Option<(usize, f64)> {
    if store.is_empty() {
        return None;
    }
    let direct = compute_scores(store, x);
    let fired: Vec<usize> = direct
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s >= kernel_cfg.theta_k)
        .map(|(i, _)| i)
        .collect();
    let spread = propagate(edges, &fired, sokm_cfg);
    let mut prop = vec![0.0f64; store.len()];
    for (idx, score) in spread {
        debug_assert!(
            idx < prop.len(),
            "best_match: edge references kernel idx={idx} outside store (len={})",
            prop.len()
        );
        if idx < prop.len() {
            prop[idx] += score;
        }
    }
    direct
        .iter()
        .enumerate()
        .map(|(i, &d)| (i, d + prop[i]))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// Class label η̂ of the best-matching kernel.
/// Returns None if store is empty or if best match is unlabelled.
/// [Hoya Testing Algorithm step 4, pp. 80–99]
pub fn predict<S: EdgeStore<usize>>(
    store: &impl KernelStore,
    edges: &S,
    x: &[f64],
    sokm_cfg: &SokmConfig,
    kernel_cfg: &KernelConfig,
) -> Option<u32> {
    best_match(store, edges, x, sokm_cfg, kernel_cfg).and_then(|(idx, _)| store.class_opt(idx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KernelConfig;
    use crate::store::{AosKernelStore, KernelStore};
    use sokm::{HashEdgeStore, SokmConfig};

    fn sokm_cfg() -> SokmConfig {
        SokmConfig::default()
    }
    fn kernel_cfg() -> KernelConfig {
        KernelConfig::default()
    }

    #[test]
    fn predict_returns_none_on_empty_store() {
        let store = AosKernelStore::new();
        let edges: HashEdgeStore<usize> = HashEdgeStore::new();
        assert_eq!(
            predict(&store, &edges, &[1.0], &sokm_cfg(), &kernel_cfg()),
            None
        );
    }

    #[test]
    fn best_match_returns_none_on_empty_store() {
        let store = AosKernelStore::new();
        let edges: HashEdgeStore<usize> = HashEdgeStore::new();
        assert_eq!(
            best_match(&store, &edges, &[1.0], &sokm_cfg(), &kernel_cfg()),
            None
        );
    }

    #[test]
    fn predict_returns_class_of_nearest_kernel() {
        let mut store = AosKernelStore::new();
        store.push(&[1.0], 1.0, Some(7));
        let edges: HashEdgeStore<usize> = HashEdgeStore::new();
        assert_eq!(
            predict(&store, &edges, &[1.0], &sokm_cfg(), &kernel_cfg()),
            Some(7)
        );
    }

    #[test]
    fn predict_prefers_higher_activation_kernel() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(1)); // kernel 0, class 1
        store.push(&[1.0], 1.0, Some(2)); // kernel 1, class 2
        let edges: HashEdgeStore<usize> = HashEdgeStore::new();
        // x = [0.9] — closer to kernel 1
        assert_eq!(
            predict(&store, &edges, &[0.9], &sokm_cfg(), &kernel_cfg()),
            Some(2)
        );
    }

    #[test]
    fn best_match_score_is_direct_plus_propagated() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(1)); // kernel 0: fires at x=[0.0], direct=1.0
        store.push(&[5.0], 1.0, Some(2)); // kernel 1: far, direct≈0 at x=[0.0]

        let scfg = sokm_cfg();
        let cfg = kernel_cfg();

        let mut edges: HashEdgeStore<usize> = HashEdgeStore::new();
        edges.set_weight(0, 1, 1.0);

        let (idx_with, score_with) = best_match(&store, &edges, &[0.0], &scfg, &cfg).unwrap();
        let no_edges: HashEdgeStore<usize> = HashEdgeStore::new();
        let (idx_without, score_without) =
            best_match(&store, &no_edges, &[0.0], &scfg, &cfg).unwrap();

        assert_eq!(idx_with, 0);
        assert_eq!(idx_without, 0);
        // kernel 0 gets no propagation back from kernel 1 (kernel 1 doesn't fire)
        assert!((score_with - score_without).abs() < 1e-6);

        // Two kernels at identical centroid — same direct score; edge tips kernel 1 to win
        let mut store2 = AosKernelStore::new();
        store2.push(&[0.0], 1.0, Some(1)); // kernel 0
        store2.push(&[0.0], 1.0, Some(2)); // kernel 1 — identical centroid
        let mut edges2: HashEdgeStore<usize> = HashEdgeStore::new();
        edges2.set_weight(0, 1, 1.0); // kernel 0 fires, sends gamma*1.0 to kernel 1

        let (idx2, score2) = best_match(&store2, &edges2, &[0.0], &scfg, &cfg).unwrap();
        assert_eq!(
            idx2, 1,
            "propagated boost must make kernel 1 win over kernel 0"
        );
        assert!(
            score2 > 1.0,
            "kernel 1 score must exceed direct-only 1.0, got {score2}"
        );
        assert!(
            (score2 - (1.0 + scfg.gamma)).abs() < 1e-6,
            "kernel 1 score must equal direct(1.0) + gamma*weight(1.0) = {}, got {score2}",
            1.0 + scfg.gamma
        );
    }

    #[test]
    fn predict_uses_propagated_activation() {
        // gaussian = exp(-sq_dist/sigma²). At x=1.0, centroid=0: exp(-1)≈0.368 >= theta_k=0.1.
        // Kernel 1 (centroid=5) gets propagated gamma=0.9 → combined≈0.9 > 0.368 → kernel 1 wins.
        // Without edges kernel 0 wins (0.368 > exp(-16)) — proves propagation changed outcome.
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0)); // kernel 0, class 0
        store.push(&[5.0], 1.0, Some(1)); // kernel 1, class 1
        store.push(&[10.0], 1.0, Some(2)); // kernel 2, class 2

        let mut edges: HashEdgeStore<usize> = HashEdgeStore::new();
        edges.set_weight(0, 1, 1.0); // kernel 0 → kernel 1

        let cfg = kernel_cfg();
        let scfg = sokm_cfg();

        // direct_0 ≈ exp(-1) ≈ 0.368, propagated_1 = gamma=0.9 → kernel 1 combined ≈ 0.9 > 0.368
        let class_with_prop = predict(&store, &edges, &[1.0], &scfg, &cfg).unwrap();
        assert_eq!(
            class_with_prop, 1,
            "propagated activation must push kernel 1 (class 1) above kernel 0 and kernel 2"
        );

        // Without edges: kernel 0 (direct≈0.368) beats kernel 1 (direct≈exp(-16)≈0)
        let no_edges: HashEdgeStore<usize> = HashEdgeStore::new();
        let class_no_prop = predict(&store, &no_edges, &[1.0], &scfg, &cfg).unwrap();
        assert_eq!(class_no_prop, 0, "without propagation kernel 0 must win");
    }

    #[test]
    fn predict_unlabelled_kernel_returns_none() {
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, None); // unlabelled
        let edges: HashEdgeStore<usize> = HashEdgeStore::new();
        assert_eq!(
            predict(&store, &edges, &[0.0], &sokm_cfg(), &kernel_cfg()),
            None
        );
    }

    #[test]
    fn best_match_all_extinct_returns_some() {
        // All kernels extinct → all scores 0.0. best_match still returns Some
        // because store is non-empty. Lock: returns last index with max_by tie-break.
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0));
        store.push(&[1.0], 1.0, Some(1));
        store.mark_extinct(0);
        store.mark_extinct(1);
        let edges: HashEdgeStore<usize> = HashEdgeStore::new();
        let result = best_match(&store, &edges, &[0.0], &sokm_cfg(), &kernel_cfg());
        // All scores are 0.0 → max_by returns last equal element (index 1)
        let (idx, score) = result.unwrap();
        assert_eq!(idx, 1);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn extinct_kernel_wins_via_propagation() {
        // Lock behavior: an extinct kernel scores 0 directly but can receive propagation.
        // best_match adds propagated to direct, so extinct kernel can still win.
        let mut store = AosKernelStore::new();
        store.push(&[0.0], 1.0, Some(0)); // kernel 0: fires at x=0
        store.push(&[100.0], 1.0, Some(1)); // kernel 1: extinct, far away

        let mut edges: HashEdgeStore<usize> = HashEdgeStore::new();
        edges.set_weight(0, 1, 5.0); // strong edge 0→1

        let cfg = kernel_cfg();
        let scfg = SokmConfig {
            gamma: 1.0,
            ..SokmConfig::default()
        };

        // kernel 0 fires (score=1.0 at x=0), propagates gamma*5.0=5.0 to kernel 1.
        // kernel 1 direct=exp(-(100^2))≈0, propagated=5.0, combined≈5.0
        // kernel 0 combined=1.0 (no propagation back since kernel 1 doesn't fire).
        // kernel 1 wins with score ≈ 5.0
        let (idx, _score) = best_match(&store, &edges, &[0.0], &scfg, &cfg).unwrap();
        assert_eq!(
            idx, 1,
            "propagation can push any kernel (even extinct) to win"
        );
    }
}
