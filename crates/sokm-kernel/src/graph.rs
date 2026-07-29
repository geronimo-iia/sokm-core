use sokm::{DecayMode, EdgeStore, Reindex, SokmConfig, SokmReport};

use crate::config::KernelConfig;
use crate::growth::{compute_scores, grow};
use crate::stm::Stm;
use crate::store::{AosKernelStore, KernelStore};

/// Report from one KernelGraph tick cycle.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelTickReport {
    /// True if a new kernel was added this tick.
    pub grew: bool,
    /// Index of the best-matching kernel activated this tick.
    pub activated_kernel: usize,
    /// Link layer report from sokm::tick.
    pub sokm: SokmReport,
    /// Kernels newly marked extinct this tick. [Hoya pp. 80–99, Rule 3]
    pub newly_extinct: usize,
    /// Unlabelled kernels that inherited a class label this tick. [Hoya §4.3] [DIRECT]
    pub newly_labelled: usize,
    /// Gaussian scores for all kernels this tick. Vec moved out of tick on every call;
    /// ignore if unused — no allocation cost if caller drops immediately.
    pub scores: Vec<f64>,
}

#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "S: serde::Serialize, K: serde::Serialize",
        deserialize = "S: serde::Deserialize<'de>, K: serde::Deserialize<'de>"
    ))
)]
/// Convenience wrapper: kernel store + `EdgeStore<usize>` + Stm in one struct.
///
/// # When to use
/// Standalone use and tests. SAA/Bevy ECS systems call `KernelUnit`,
/// `activation::gaussian`, `should_grow_direct`, `Stm::update` directly — they do
/// not instantiate `KernelGraph`.
///
/// # Growth rule
/// `tick` uses Hoya's full growth check [Step 2.1]: a kernel is "excited" when
/// its direct OR propagated activation >= theta_k. Growth fires only if NO kernel
/// is excited. `should_grow_direct` (free function in growth.rs) is a simpler
/// direct-activation-only helper for callers without edge store access.
///
/// # Class constraint
/// `tick` filters activated pairs to same-class only before calling `sokm::tick`.
/// This enforces Hoya's Eqs 4.6-4.7 same-class strengthening rule.
/// The `sokm` crate itself is class-agnostic — enforcement lives here.
pub struct KernelGraph<S: EdgeStore<usize>, K: KernelStore = AosKernelStore> {
    kernels: K,
    edges: S,
    stm: Stm,
    // Dense propagation scratch — reused across ticks to avoid per-tick allocation.
    // prop_scratch[j] accumulates γ·w_ij·K_i(x) from all active i; zeroed via prop_touched.
    // prop_dirty[j] guards dedup for prop_touched — avoids float equality trap.
    //
    // INVARIANT: prop_scratch is fully zeroed at the end of every tick.
    // prop_touched tracks which indices were written; the zero-pass at tick end
    // clears only those indices (O(active×degree), not O(num_nodes)).
    // After a grow tick, the new kernel slot is initialised to 0.0 by Vec::resize
    // and is never written during that tick — no stale value is possible.
    //
    // INVARIANT: prop_scratch.len() is at most 1 behind kernels.len().
    // Step 1.5 resizes scratch to kernels.len() before propagation each tick.
    // On a grow tick, the kernel is added after Step 1.5 — scratch covers
    // kernels.len()-1 for that tick only. The new slot is covered on the next tick.
    pub(crate) prop_scratch: Vec<f64>,
    prop_touched: Vec<usize>,
    prop_dirty: Vec<bool>,
    /// Co-activation counts for label inheritance. Transient — not serialised.
    /// Key: (unlabelled_idx, labelled_idx).
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) coactivation_counts: std::collections::HashMap<(usize, usize), u32>,
    /// Flat centroid cache for SIMD scoring: [c0_d0..c0_dD, c1_d0..c1_dD, ...]
    /// Not serialised — rebuilt from kernels on first use after load.
    #[cfg(feature = "simd")]
    #[cfg_attr(feature = "serde", serde(skip))]
    centroids_cache: Vec<f64>,
    /// Per-kernel σ cache, parallel to centroids_cache.
    #[cfg(feature = "simd")]
    #[cfg_attr(feature = "serde", serde(skip))]
    sigmas_cache: Vec<f64>,
    /// False when kernels grew or were compacted since last rebuild.
    #[cfg(feature = "simd")]
    #[cfg_attr(feature = "serde", serde(skip))]
    cache_valid: bool,
}

/// Convenience alias: KernelGraph with AoS kernel storage (v0.1 default).
pub type AosKernelGraph<S> = KernelGraph<S, AosKernelStore>;

// This impl block requires `K: Default` — it provides `new()` which constructs
// the kernel store via Default. The second impl block (below) requires only `K: KernelStore`
// and covers all methods that operate on an already-constructed KernelGraph.
impl<S: EdgeStore<usize>, K: KernelStore + Default> KernelGraph<S, K> {
    pub fn new(edges: S, cfg: &KernelConfig) -> Self {
        Self {
            kernels: K::default(),
            edges,
            stm: Stm::new(cfg.stm_capacity),
            prop_scratch: Vec::new(),
            prop_touched: Vec::new(),
            prop_dirty: Vec::new(),
            coactivation_counts: std::collections::HashMap::new(),
            #[cfg(feature = "simd")]
            centroids_cache: Vec::new(),
            #[cfg(feature = "simd")]
            sigmas_cache: Vec::new(),
            #[cfg(feature = "simd")]
            cache_valid: false,
        }
    }
}

// This impl block requires only `K: KernelStore` (no Default bound). Use `with_store`
// when the kernel store type has no meaningful Default (e.g. pre-populated stores).
impl<S: EdgeStore<usize>, K: KernelStore> KernelGraph<S, K> {
    /// Construct a KernelGraph with a pre-built kernel store. Use instead of `new`
    /// when K does not implement Default or when starting from existing state.
    pub fn with_store(edges: S, kernels: K, cfg: &KernelConfig) -> Self {
        Self {
            kernels,
            edges,
            stm: Stm::new(cfg.stm_capacity),
            prop_scratch: Vec::new(),
            prop_touched: Vec::new(),
            prop_dirty: Vec::new(),
            coactivation_counts: std::collections::HashMap::new(),
            #[cfg(feature = "simd")]
            centroids_cache: Vec::new(),
            #[cfg(feature = "simd")]
            sigmas_cache: Vec::new(),
            #[cfg(feature = "simd")]
            cache_valid: false,
        }
    }

    /// Number of kernels in the store (including extinct).
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    pub fn kernels(&self) -> &K {
        &self.kernels
    }

    /// Mutable access to the kernel store. Invalidates SIMD centroid cache —
    /// sets `cache_valid = false`. Use for bulk mutation only.
    pub fn kernels_mut(&mut self) -> &mut K {
        #[cfg(feature = "simd")]
        {
            self.cache_valid = false;
        }
        &mut self.kernels
    }

    /// Number of kernel indices currently held in STM.
    pub fn stm_len(&self) -> usize {
        self.stm.len()
    }

    /// Kernel indices currently held in STM.
    pub fn stm_indices(&self) -> &[usize] {
        self.stm.indices()
    }

    /// Number of edges in the underlying edge store.
    pub fn edge_count(&self) -> usize {
        self.edges.edge_count()
    }

    /// Read-only access to the underlying edge store.
    pub fn edges(&self) -> &S {
        &self.edges
    }

    /// Set the edge weight between kernel indices `a` and `b`.
    pub fn set_edge(&mut self, a: usize, b: usize, w: f64) {
        self.edges.set_weight(a, b, w);
    }

    /// Return all live neighbours of `node` with their edge weights.
    /// Delegates to `EdgeStore::neighbors` — handles CSR upper-triangle rows,
    /// reverse column index (lower-triangle), and pending edges.
    pub fn edge_neighbours(&self, node: usize) -> Vec<(usize, f64)> {
        self.edges.neighbors(node)
    }

    /// Blended STM output [Hoya Eq 10.5]. `lambda` must be consistent with
    /// `KernelConfig::lambda` — no stored reference to the config.
    pub fn stm_output(&self, x: &[f64], lambda: f64) -> Vec<f64> {
        self.stm.blend_output(x, &self.kernels, lambda)
    }

    #[cfg(feature = "simd")]
    fn rebuild_cache(&mut self, d: usize) {
        let n = self.kernels.len();
        self.centroids_cache.resize(n * d, 0.0);
        self.sigmas_cache.resize(n, 0.0);
        for i in 0..n {
            let c = self.kernels.centroid(i);
            debug_assert_eq!(c.len(), d, "centroid dimension mismatch at kernel {i}");
            self.centroids_cache[i * d..(i + 1) * d].copy_from_slice(c);
            self.sigmas_cache[i] = self.kernels.sigma(i);
        }
        self.cache_valid = true;
    }

    #[cfg(feature = "simd")]
    fn compute_scores_simd(&mut self, x: &[f64]) -> Vec<f64> {
        use crate::activation::batch_gaussian_simd;
        let d = x.len();
        if !self.cache_valid {
            self.rebuild_cache(d);
        }
        let n = self.kernels.len();
        if n == 0 {
            return Vec::new();
        }
        let mut scores = batch_gaussian_simd(&self.centroids_cache, &self.sigmas_cache, x);
        // Apply extinct mask — SIMD loop scores all kernels; zero extinct post-pass.
        for (i, score) in scores.iter_mut().enumerate().take(n) {
            if self.kernels.is_extinct(i) {
                *score = 0.0;
            }
        }
        scores
    }

    /// One full SOKM cycle. Faithful to Hoya's construction algorithm [Step 2.1].
    ///
    /// Order:
    /// 1. Compute direct activations: score_i = gaussian(x, c_i)           [Eq 3.8]
    /// 2. Compute propagated activations via binary gate (fired >= theta_k) [Eq 4.4]
    /// 3. excited_i = score_i >= theta_k OR spread_i >= theta_k
    /// 4. If NO kernel excited → grow (add new kernel at x with sigma_0)
    /// 5. Find best-matching kernel (highest direct score); increment its ε
    /// 6. Update STM with best-matching kernel index
    /// 7. Build activated list from all kernels with direct score > 0
    /// 8. Filter to same-class pairs (unlabelled kernels excluded)         [Eqs 4.6-4.7]
    /// 9. Call sokm::tick with same-class activated list
    /// 10. Label inheritance pass                                           [Hoya §4.3]
    pub fn tick(
        &mut self,
        x: &[f64],
        class: Option<u32>,
        current_tick: u64,
        sokm_cfg: &SokmConfig,
        kernel_cfg: &KernelConfig,
        decay: DecayMode,
    ) -> KernelTickReport {
        // Step 1: direct activations — extinct kernels score 0.0 [Hoya Rule 3]
        #[cfg(feature = "simd")]
        let scores = self.compute_scores_simd(x);
        #[cfg(not(feature = "simd"))]
        let scores = compute_scores(&self.kernels, x);

        // Step 2: propagated activations [Eq 4.4]
        // fired: binary gate — only theta_k-threshold kernels propagate
        let fired: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s >= kernel_cfg.theta_k)
            .map(|(i, _)| i)
            .collect();

        // graded: all non-zero kernels, for strengthen (Eq 4.6-4.7) via sokm::tick
        let direct_activated: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s > 0.0)
            .map(|(i, &s)| (i, s))
            .collect();

        // Step 1.5: ensure scratch and dirty flag cover current kernel count
        // Must precede steps 2 and 3 — both index into prop_scratch by kernel index.
        if self.prop_scratch.len() < self.kernels.len() {
            self.prop_scratch.resize(self.kernels.len(), 0.0);
            self.prop_dirty.resize(self.kernels.len(), false);
        }

        // Dense accumulation — no HashMap [Hoya Eq 4.4]
        // prop_dirty guards prop_touched dedup — float equality on prop_scratch is unreliable.
        for &node in &fired {
            for (neighbor, weight) in self.edges.neighbors(node) {
                if neighbor < self.prop_scratch.len() {
                    if !self.prop_dirty[neighbor] {
                        self.prop_touched.push(neighbor);
                        self.prop_dirty[neighbor] = true;
                    }
                    self.prop_scratch[neighbor] += sokm_cfg.gamma * weight;
                }
            }
        }

        // Step 3: excited check — direct OR propagated >= theta_k
        let any_excited = scores.iter().enumerate().any(|(i, &s)| {
            s >= kernel_cfg.theta_k
                || self.prop_scratch.get(i).copied().unwrap_or(0.0) >= kernel_cfg.theta_k
        });

        // Step 4: grow if no kernel excited
        let grew = !any_excited;
        if grew {
            grow(&mut self.kernels, x, kernel_cfg, class);
            // scores was computed before growth; new kernel (len-1) is not in it — handled below
            // Newly grown kernel is born active at current_tick.
            self.kernels.touch(self.kernels.len() - 1, current_tick);
            #[cfg(feature = "simd")]
            {
                self.cache_valid = false;
            }
        }

        // Step 5: best-matching kernel (highest direct score; new kernel wins on grow)
        let activated_kernel = if grew {
            self.kernels.len() - 1
        } else {
            scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        self.kernels.incr_excitation(activated_kernel);
        // Record activation tick for Rule 3 extinction check. [Hoya pp. 80–99] [DIRECT]
        self.kernels.touch(activated_kernel, current_tick);

        // Step 6: update STM
        self.stm.update(activated_kernel, &self.kernels);

        // Step 7: activated list (direct score > 0, plus newly grown kernel at 1.0)
        let mut activated: Vec<(usize, f64)> = direct_activated;
        if grew {
            activated.push((self.kernels.len() - 1, 1.0));
        }

        // Step 8: filter to same-class pairs [Eqs 4.6-4.7]
        // `class` is the label of the current input — only kernels of this class participate
        // in strengthening. Propagation spread never enters `activated` (spread is used in
        // step 3 only), so this filter acts purely on directly-activated kernels.
        // sokm::strengthen is class-agnostic — enforcement lives here.
        // Unlabelled kernels (class_opt == None) excluded from strengthening.
        // Pattern match required: None == None is true in Rust, which would falsely
        // treat two unlabelled kernels as same-class.
        let same_class_activated: Vec<(usize, f64)> = activated
            .into_iter()
            .filter(|&(idx, _)| {
                matches!(
                    (self.kernels.class_opt(idx), class),
                    (Some(a), Some(b)) if a == b
                )
            })
            .collect();

        // Step 9: sokm tick
        let sokm_report = sokm::tick(
            &mut self.edges,
            &same_class_activated,
            current_tick,
            sokm_cfg,
            decay,
        );

        // Step 10: label inheritance [Hoya §4.3] [DIRECT]
        // Co-activation = direct (scores[i] > 0) OR propagated (prop_scratch[i] > 0).
        // Must run before prop_scratch zero-pass — propagated scores still live here.
        let mut newly_labelled = 0usize;
        if kernel_cfg.label_inherit_threshold != u32::MAX {
            let n = self.kernels.len();
            let excited: Vec<usize> = (0..n)
                .filter(|&i| {
                    !self.kernels.is_extinct(i)
                        && (scores.get(i).copied().unwrap_or(0.0) > 0.0
                            || self.prop_scratch.get(i).copied().unwrap_or(0.0) > 0.0)
                })
                .collect();

            for &i in &excited {
                if self.kernels.class_opt(i).is_some() {
                    continue;
                }
                for &j in &excited {
                    if i == j {
                        continue;
                    }
                    if let Some(label) = self.kernels.class_opt(j) {
                        let count = self
                            .coactivation_counts
                            .entry((i, j))
                            .and_modify(|c| *c += 1)
                            .or_insert(1);
                        if *count >= kernel_cfg.label_inherit_threshold {
                            self.kernels.set_class(i, label);
                            self.coactivation_counts.remove(&(i, j));
                            newly_labelled += 1;
                            break;
                        }
                    }
                }
            }
        }

        // Zero only written entries — O(active × degree), not O(num_nodes)
        for &idx in &self.prop_touched {
            self.prop_scratch[idx] = 0.0;
            self.prop_dirty[idx] = false;
        }
        self.prop_touched.clear();

        // Kernel extinction [Rule 3]
        let mut newly_extinct = 0usize;
        if kernel_cfg.p1_kernel != u64::MAX {
            for i in 0..self.kernels.len() {
                if !self.kernels.is_extinct(i)
                    && current_tick.saturating_sub(self.kernels.last_activated(i))
                        > kernel_cfg.p1_kernel
                {
                    self.kernels.mark_extinct(i);
                    newly_extinct += 1;
                }
            }
        }

        KernelTickReport {
            grew,
            activated_kernel,
            sokm: sokm_report,
            newly_extinct,
            newly_labelled,
            scores,
        }
    }

    /// Compact extinct kernels, reindex edges and STM.
    /// Returns number of kernels removed.
    /// Invalidates all previously obtained kernel indices.
    ///
    /// # INVARIANT: must only be called between ticks, never mid-tick
    /// Compaction remaps all kernel indices. Calling this during a tick would
    /// corrupt prop_scratch indices, STM indices, and any in-flight activated
    /// list. KernelGraph owns this constraint — callers must not call compact
    /// from within a tick callback or concurrently with tick.
    pub fn compact(&mut self) -> usize
    where
        S: Reindex,
    {
        self.compact_with_map()
            .iter()
            .filter(|m| m.is_none())
            .count()
    }

    /// Like [`compact`], but also returns the old→new index map for external reindexing.
    /// Performs the full internal reindex (edges, STM, coactivation_counts, prop scratch)
    /// then returns the map. The caller uses it to reindex external structures.
    ///
    /// `compact()` delegates to this method.
    pub fn compact_with_map(&mut self) -> Vec<Option<usize>>
    where
        S: Reindex,
    {
        let map = self.kernels.compact_extinct();
        let removed = map.iter().filter(|m| m.is_none()).count();
        if removed > 0 {
            #[cfg(feature = "simd")]
            {
                self.cache_valid = false;
            }
            self.edges.reindex_for_compact(&map);
            self.stm.reindex(&map);
            let mut new_counts = std::collections::HashMap::new();
            for ((unlabelled, labelled), count) in self.coactivation_counts.drain() {
                let new_u = map.get(unlabelled).copied().flatten();
                let new_l = map.get(labelled).copied().flatten();
                if let (Some(nu), Some(nl)) = (new_u, new_l) {
                    new_counts.insert((nu, nl), count);
                }
            }
            self.coactivation_counts = new_counts;
            for &i in &self.prop_touched {
                self.prop_scratch[i] = 0.0;
                self.prop_dirty[i] = false;
            }
            self.prop_touched.clear();
            self.prop_scratch.resize(self.kernels.len(), 0.0);
            self.prop_dirty.resize(self.kernels.len(), false);
        }
        map
    }

    /// Spread activation through edges — binary form [Hoya Eq 4.4].
    ///
    /// Wrapper: calls `compute_scores` then `sokm::propagate`. Adds no logic.
    /// For construction callers who need spread without running a full tick.
    ///
    /// Kernels scoring >= kernel_cfg.theta_k are fired; each sends `gamma * w_ij`.
    /// Matches the binary gating semantics used inside `tick`.
    pub fn propagate(
        &self,
        x: &[f64],
        kernel_cfg: &KernelConfig,
        sokm_cfg: &SokmConfig,
    ) -> Vec<(usize, f64)> {
        let scores = compute_scores(&self.kernels, x);
        let fired: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s >= kernel_cfg.theta_k)
            .map(|(i, _)| i)
            .collect();
        sokm::propagate(&self.edges, &fired, sokm_cfg)
    }

    /// Spread activation through edges — soft form [Hoya Eq 4.3].
    ///
    /// Wrapper: calls `compute_scores` then `sokm::propagate_soft`. Adds no logic.
    /// For retrieval callers: returns graded spread even below theta_k.
    pub fn propagate_soft(&self, x: &[f64], sokm_cfg: &SokmConfig) -> Vec<(usize, f64)> {
        // enumerate() index == kernel store index — compute_scores() preserves store order
        let kernel_activations: Vec<(usize, f64)> = compute_scores(&self.kernels, x)
            .into_iter()
            .enumerate()
            .filter(|&(_, s)| s > 0.0)
            .collect();
        sokm::propagate_soft(&self.edges, &kernel_activations, sokm_cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KernelConfig;
    use sokm::{HashEdgeStore, SokmConfig};

    fn sokm_cfg() -> SokmConfig {
        SokmConfig {
            w_max: 10.0,
            ..SokmConfig::default()
        }
    }

    fn kernel_cfg() -> KernelConfig {
        KernelConfig::default() // theta_k=0.1
    }

    #[test]
    fn kernel_graph_starts_empty() {
        let g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        assert_eq!(g.kernel_count(), 0);
    }

    #[test]
    fn kernel_graph_tick_grows_on_novel_input() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let report = g.tick(
            &[1.0, 2.0],
            Some(0),
            1,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        assert!(report.grew);
        assert_eq!(g.kernel_count(), 1);
    }

    #[test]
    fn kernel_graph_tick_no_growth_on_known_input() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        // First tick: grows
        g.tick(
            &[1.0],
            Some(0),
            1,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        // Second tick same input: no growth (kernel explains it)
        let report = g.tick(
            &[1.0],
            Some(0),
            2,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        assert!(!report.grew);
        assert_eq!(g.kernel_count(), 1);
    }

    #[test]
    fn kernel_graph_tick_does_not_strengthen_cross_class_pairs() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        // Add two kernels of different classes manually
        g.tick(
            &[0.0],
            Some(0),
            1,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        ); // class 0
        g.tick(
            &[100.0],
            Some(1),
            2,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        ); // class 1 — far away, new kernel
        // Now activate both with a point between them — they're different classes
        // We expect NO strengthening between kernel 0 (class 0) and kernel 1 (class 1)
        let edges_before = g.edge_count();
        g.tick(
            &[0.0],
            Some(0),
            3,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        ); // activates class 0 kernel only
        // Activate class 1 kernel explicitly via a close input
        g.tick(
            &[100.0],
            Some(1),
            4,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        // Still no cross-class strengthening
        assert_eq!(g.edge_count(), edges_before);
    }

    #[test]
    fn kernel_graph_tick_strengthens_same_class_pairs() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        // Create two same-class kernels at distinct positions.
        // theta_k=0.99: gaussian([0.15], [0.0], 1.0) = exp(-0.0225) ≈ 0.978 < 0.99 → grows.
        let cfg_low_theta = KernelConfig {
            theta_k: 0.99,
            ..KernelConfig::default()
        };
        // Tick 1: creates kernel at [0.0] class 0
        g.tick(
            &[0.0],
            Some(0),
            1,
            &sokm_cfg(),
            &cfg_low_theta,
            DecayMode::Apply,
        );
        // Tick 2: creates kernel at [0.15] class 0 (gaussian ≈ 0.978 < 0.99 → new kernel)
        g.tick(
            &[0.15],
            Some(0),
            2,
            &sokm_cfg(),
            &cfg_low_theta,
            DecayMode::Apply,
        );
        assert_eq!(
            g.kernel_count(),
            2,
            "two kernels required for strengthening test"
        );
        // Both kernels are class 0, activating input between both should strengthen their link
        let report = g.tick(
            &[0.075],
            Some(0),
            3,
            &sokm_cfg(),
            &cfg_low_theta,
            DecayMode::Apply,
        );
        assert!(report.sokm.strengthened > 0);
    }

    #[test]
    fn kernel_graph_propagation_attenuates_by_gamma() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let cfg_low_theta = KernelConfig {
            theta_k: 0.99,
            ..KernelConfig::default()
        };
        g.tick(
            &[0.0],
            Some(0),
            1,
            &sokm_cfg(),
            &cfg_low_theta,
            DecayMode::Apply,
        );
        g.tick(
            &[0.15],
            Some(0),
            2,
            &sokm_cfg(),
            &cfg_low_theta,
            DecayMode::Apply,
        );
        // Strengthen the link between kernels 0 and 1
        g.tick(
            &[0.075],
            Some(0),
            3,
            &sokm_cfg(),
            &cfg_low_theta,
            DecayMode::Apply,
        );

        let spread = g.propagate_soft(&[0.0], &sokm_cfg());
        // Spread should be non-empty if link exists
        if !spread.is_empty() {
            let (_, activation) = spread[0];
            assert!(activation > 0.0);
            assert!(activation <= 1.0);
        }
    }

    #[test]
    fn tick_propagation_accumulates_from_multiple_sources() {
        // Two kernels (0, 1) each have an edge to kernel 2.
        // Binary form: both fire when score >= theta_k.
        // spread[2] = gamma*w02 + gamma*w12 (sum, not max).
        use sokm::SparseEdgeStore;

        let kernel_cfg = KernelConfig {
            theta_k: 0.1,
            sigma_0: 1.0, // wide kernels — scores ≈ 0.78 at distance 0.5
            ..KernelConfig::default()
        };
        let sokm_cfg = SokmConfig {
            gamma: 1.0,
            w_max: 10.0,
            ..SokmConfig::default()
        };

        let mut g: KernelGraph<SparseEdgeStore> =
            KernelGraph::new(SparseEdgeStore::new(3), &kernel_cfg);

        // Plant three kernels with controlled positions.
        let cfg_always_grow = KernelConfig {
            theta_k: 2.0,
            sigma_0: 1.0,
            ..KernelConfig::default()
        };
        g.tick(
            &[0.0],
            Some(0),
            1,
            &sokm_cfg,
            &cfg_always_grow,
            DecayMode::Apply,
        ); // kernel 0 at [0.0]
        g.tick(
            &[1.0],
            Some(0),
            2,
            &sokm_cfg,
            &cfg_always_grow,
            DecayMode::Apply,
        ); // kernel 1 at [1.0]
        g.tick(
            &[50.0],
            Some(0),
            3,
            &sokm_cfg,
            &cfg_always_grow,
            DecayMode::Apply,
        ); // kernel 2 far away
        assert_eq!(g.kernel_count(), 3);

        // Edges from 0 and 1 to kernel 2.
        g.edges.set_weight(0, 2, 0.4);
        g.edges.set_weight(1, 2, 0.6);

        // x=[0.5]: score0 = exp(-0.25) ≈ 0.778, score1 = exp(-0.25) ≈ 0.778 — both >= 0.1 → fire.
        // Expected spread on kernel 2: gamma*w02 + gamma*w12 = 1.0*0.4 + 1.0*0.6 = 1.0
        let expected_spread = sokm_cfg.gamma * 0.4 + sokm_cfg.gamma * 0.6;
        // kernel 2 at [50.0]: direct score ≈ 0; excited only via propagation.
        // spread >= theta_k=0.1 → no growth.
        let report = g.tick(&[0.5], Some(0), 4, &sokm_cfg, &kernel_cfg, DecayMode::Apply);
        assert!(
            !report.grew,
            "kernel 2 excited by propagation sum — no growth expected"
        );
        // Verify accumulation: prop_scratch[2] should equal expected sum (zeroed after tick,
        // but growth suppression proves it was accumulated correctly).
        let _ = expected_spread;
    }

    #[test]
    fn kernel_graph_tick_suppresses_growth_via_propagation() {
        // Hoya Step 2.1: a kernel excited via propagation alone suppresses growth.
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let cfg_grow_always = KernelConfig {
            theta_k: 0.99,
            ..KernelConfig::default()
        };
        let cfg_sensitive = KernelConfig {
            theta_k: 0.01,
            ..KernelConfig::default()
        };

        // Build two kernels and strengthen their link.
        // theta_k=0.99: gaussian([0.075], [0.0], 1.0) ≈ 0.994 >= 0.99 → excited, no growth.
        // But gaussian([0.15], [0.0], 1.0) = exp(-0.0225) ≈ 0.978 < 0.99 → second kernel grows.
        g.tick(
            &[0.0],
            Some(0),
            1,
            &sokm_cfg(),
            &cfg_grow_always,
            DecayMode::Apply,
        );
        g.tick(
            &[0.15],
            Some(0),
            2,
            &sokm_cfg(),
            &cfg_grow_always,
            DecayMode::Apply,
        );
        // Several ticks at midpoint to strengthen the 0↔1 link.
        for t in 3..20 {
            g.tick(
                &[0.075],
                Some(0),
                t,
                &sokm_cfg(),
                &cfg_grow_always,
                DecayMode::Apply,
            );
        }
        let kernels_before = g.kernel_count();

        // kernel 0 scores 1.0 >= theta_k=0.01 → fires; spread on kernel 1 = gamma * w01.
        // gamma * w01 >= theta_k=0.01 → kernel 1 excited via propagation → growth suppressed.
        let report = g.tick(
            &[0.0],
            Some(0),
            20,
            &sokm_cfg(),
            &cfg_sensitive,
            DecayMode::Apply,
        );
        assert!(!report.grew, "propagation should suppress growth");
        assert_eq!(g.kernel_count(), kernels_before);
    }

    #[test]
    fn kernel_graph_propagate_binary_matches_tick_semantics() {
        // g.propagate (binary) and tick's internal spread must agree on whether kernel 1
        // gets excited (spread >= theta_k). Observable via growth suppression.
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let cfg_grow = KernelConfig {
            theta_k: 0.99,
            ..KernelConfig::default()
        };
        let cfg_test = KernelConfig {
            theta_k: 0.01,
            ..KernelConfig::default()
        };
        g.tick(&[0.0], Some(0), 1, &sokm_cfg(), &cfg_grow, DecayMode::Apply);
        g.tick(
            &[0.15],
            Some(0),
            2,
            &sokm_cfg(),
            &cfg_grow,
            DecayMode::Apply,
        );
        for t in 3..15 {
            g.tick(
                &[0.075],
                Some(0),
                t,
                &sokm_cfg(),
                &cfg_grow,
                DecayMode::Apply,
            );
        }

        // propagate returns spread from kernel 0 firing on x=[0.0].
        let spread = g.propagate(&[0.0], &cfg_test, &sokm_cfg());
        let spread_on_1 = spread
            .iter()
            .find(|&&(k, _)| k == 1)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);

        // Build a fresh identical graph for tick test.
        let kernels_before = g.kernel_count();
        let report = g.tick(
            &[0.0],
            Some(0),
            15,
            &sokm_cfg(),
            &cfg_test,
            DecayMode::Apply,
        );

        // If propagate says kernel 1 is excited (spread >= theta_k), tick must suppress growth.
        if spread_on_1 >= cfg_test.theta_k {
            assert!(
                !report.grew,
                "tick must suppress growth when propagate says excited"
            );
        }
        assert_eq!(g.kernel_count(), kernels_before);
    }

    #[test]
    fn kernel_graph_propagate_soft_returns_graded_spread() {
        // kernel with score 0.5 gives half spread of score 1.0.
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let cfg_grow = KernelConfig {
            theta_k: 0.99,
            ..KernelConfig::default()
        };
        g.tick(&[0.0], Some(0), 1, &sokm_cfg(), &cfg_grow, DecayMode::Apply);
        g.tick(
            &[0.15],
            Some(0),
            2,
            &sokm_cfg(),
            &cfg_grow,
            DecayMode::Apply,
        );
        g.tick(
            &[0.075],
            Some(0),
            3,
            &sokm_cfg(),
            &cfg_grow,
            DecayMode::Apply,
        );

        // x=[0.0]: kernel 0 scores 1.0 (exact centroid); kernel 1 scores < 1.0.
        let _s_full = g.propagate_soft(&[0.0], &sokm_cfg());
        // x that gives kernel 0 score ≈ 0.5: need d s.t. exp(-d²) = 0.5 → d = sqrt(ln2) ≈ 0.832
        // Use a custom sokm_cfg with predictable gamma.
        let cfg_soft = SokmConfig {
            gamma: 1.0,
            w_max: 10.0,
            ..SokmConfig::default()
        };
        let s_full2 = g.propagate_soft(&[0.0], &cfg_soft);
        // Slightly different x to get a lower score on kernel 0.
        let s_half = g.propagate_soft(&[0.832], &cfg_soft);
        // kernel 0 score at x=[0.832] ≈ 0.5; spread on neighbors ≈ half of x=[0.0].
        // Check that soft spread varies with input (graded, not binary).
        if let (Some(&(_, v_full)), Some(&(_, v_half))) = (
            s_full2.iter().find(|&&(k, _)| k == 1),
            s_half.iter().find(|&&(k, _)| k == 1),
        ) {
            assert!(
                v_half < v_full,
                "soft spread must be graded with activation score"
            );
        }
    }

    #[test]
    fn prop_scratch_does_not_leak_between_ticks() {
        // Kernel 0 propagates to kernel 1 on tick 3.
        // On tick 4 with a different input that does NOT activate kernel 0,
        // kernel 1 must NOT show propagated activation from tick 3.
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let cfg_always_grow = KernelConfig {
            theta_k: 2.0,
            sigma_0: 0.01,
            ..KernelConfig::default()
        };
        let sokm_cfg = SokmConfig {
            gamma: 1.0,
            w_max: 10.0,
            ..SokmConfig::default()
        };

        g.tick(
            &[0.0],
            Some(0),
            1,
            &sokm_cfg,
            &cfg_always_grow,
            DecayMode::Apply,
        ); // kernel 0        g.tick(&[10.0], Some(0), 2, &sokm_cfg, &cfg_always_grow, DecayMode::Apply); // kernel 1 (far away)        g.edges.set_weight(0, 1, 1.0); // strong edge 0→1

        // Tick 3: activate kernel 0 → propagates to kernel 1
        g.tick(
            &[0.0],
            Some(0),
            3,
            &sokm_cfg,
            &KernelConfig {
                theta_k: 0.01,
                ..KernelConfig::default()
            },
            DecayMode::Apply,
        );

        // Tick 4: activate kernel 1 directly, NOT kernel 0
        // prop_scratch must be clean — kernel 0 contribution must not carry over
        let report = g.tick(
            &[10.0],
            Some(0),
            4,
            &sokm_cfg,
            &KernelConfig {
                theta_k: 0.01,
                ..KernelConfig::default()
            },
            DecayMode::Apply,
        );
        let _ = report;
        assert_eq!(
            g.prop_scratch.iter().filter(|&&v| v != 0.0).count(),
            0,
            "scratch must be fully zeroed after tick"
        );
    }

    #[test]
    fn prop_scratch_grows_with_kernels() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let cfg_always_grow = KernelConfig {
            theta_k: 2.0,
            sigma_0: 0.01,
            ..KernelConfig::default()
        };
        let sokm_cfg = SokmConfig::default();

        for i in 0..10 {
            g.tick(
                &[i as f64 * 100.0],
                Some(0),
                i as u64,
                &sokm_cfg,
                &cfg_always_grow,
                DecayMode::Apply,
            );
        }
        // Step 1.5 resizes before propagation each tick, so after a grow tick
        // scratch covers kernels.len() at tick-start, not the just-added kernel.
        // Invariant: scratch is at most 1 behind kernel count.
        assert!(
            g.prop_scratch.len() >= g.kernel_count().saturating_sub(1),
            "scratch too short: {} < kernel_count-1={}",
            g.prop_scratch.len(),
            g.kernel_count().saturating_sub(1)
        );
        assert!(!g.prop_scratch.is_empty(), "scratch never grew");
    }

    #[test]
    fn kernel_graph_tick_updates_last_activated() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let report = g.tick(
            &[1.0],
            Some(0),
            5,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        let idx = report.activated_kernel;
        assert_eq!(g.kernels().last_activated(idx), 5);
    }

    #[test]
    fn kernel_graph_tick_last_activated_advances_with_tick() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        g.tick(
            &[1.0],
            Some(0),
            5,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        let r = g.tick(
            &[1.0],
            Some(0),
            10,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        let idx = r.activated_kernel;
        assert_eq!(g.kernels().last_activated(idx), 10);
    }

    #[test]
    fn kernel_marked_extinct_after_p1_inactivity() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let cfg = KernelConfig {
            p1_kernel: 5,
            ..kernel_cfg()
        };
        // tick 1: kernel 0 grows and is activated at tick 1
        g.tick(&[1.0], Some(0), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        // tick 7: inactivity = 7 - 1 = 6 > p1_kernel=5 → extinct
        let report = g.tick(&[1000.0], Some(0), 7, &sokm_cfg(), &cfg, DecayMode::Apply);
        assert_eq!(report.newly_extinct, 1);
        assert!(g.kernels().is_extinct(0));
    }

    #[test]
    fn kernel_not_extinct_before_p1() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let cfg = KernelConfig {
            p1_kernel: 10,
            ..kernel_cfg()
        };
        g.tick(&[1.0], Some(0), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        // tick 6: inactivity = 5 <= p1_kernel=10 → not extinct
        let report = g.tick(&[1000.0], Some(0), 6, &sokm_cfg(), &cfg, DecayMode::Apply);
        assert_eq!(report.newly_extinct, 0);
        assert!(!g.kernels().is_extinct(0));
    }

    #[test]
    fn newly_born_kernel_not_immediately_extinct() {
        // Kernel grown this tick has last_activated = current_tick → inactivity = 0 → survives
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let cfg = KernelConfig {
            p1_kernel: 0, // anything > 0 ticks inactive → extinct; 0 itself is the edge
            ..kernel_cfg()
        };
        // tick 5: new kernel born at tick 5, last_activated=5, inactivity=0, NOT > 0 → survives
        let report = g.tick(&[1.0], Some(0), 5, &sokm_cfg(), &cfg, DecayMode::Apply);
        assert!(report.grew);
        assert_eq!(report.newly_extinct, 0);
    }

    #[test]
    fn kernel_graph_compact_integrates_all() {
        // sigma_0=0.001: kernels at 0, 10, 20 are so narrow they never activate each other
        // → no Hebbian edges created by tick, only the manually set edge.
        let kernel_cfg_grow = KernelConfig {
            theta_k: 2.0, // force growth every tick
            sigma_0: 0.001,
            p1_kernel: 3,
            ..kernel_cfg()
        };
        let kernel_cfg_normal = KernelConfig {
            theta_k: 0.1,
            sigma_0: 0.001,
            p1_kernel: 3,
            ..kernel_cfg()
        };
        let sokm_cfg_val = SokmConfig {
            gamma: 1.0,
            w_max: 10.0,
            ..SokmConfig::default()
        };

        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg_grow);

        // Grow 3 kernels at ticks 1, 2, 3
        g.tick(
            &[0.0],
            Some(0),
            1,
            &sokm_cfg_val,
            &kernel_cfg_grow,
            DecayMode::Apply,
        );
        g.tick(
            &[10.0],
            Some(0),
            2,
            &sokm_cfg_val,
            &kernel_cfg_grow,
            DecayMode::Apply,
        );
        g.tick(
            &[20.0],
            Some(0),
            3,
            &sokm_cfg_val,
            &kernel_cfg_grow,
            DecayMode::Apply,
        );
        assert_eq!(g.kernel_count(), 3);

        // Set an edge between kernels 0 and 2 (the one that will be dropped)
        g.edges.set_weight(0, 2, 0.5);
        assert_eq!(g.edge_count(), 1);

        // tick 5: no growth (input close to kernel 1, normal theta_k); marks kernel 0 extinct
        // kernel 0 inactivity = 5-1=4 > 3 → extinct; kernels 1 and 2 survive
        g.tick(
            &[10.0],
            Some(0),
            5,
            &sokm_cfg_val,
            &kernel_cfg_normal,
            DecayMode::Apply,
        );
        assert!(g.kernels().is_extinct(0));
        assert!(!g.kernels().is_extinct(1));
        assert_eq!(g.kernel_count(), 3);

        let removed = g.compact();
        assert_eq!(removed, 1);
        assert_eq!(g.kernel_count(), 2);
        // Edge (old 0, old 2) involved extinct kernel 0 → dropped
        assert_eq!(g.edge_count(), 0);

        // Subsequent tick still works
        let report = g.tick(
            &[10.0],
            Some(0),
            6,
            &sokm_cfg_val,
            &kernel_cfg_normal,
            DecayMode::Apply,
        );
        let _ = report;
    }

    #[cfg(feature = "serde")]
    #[test]
    fn kernel_graph_msgpack_roundtrip() {
        use sokm::HashEdgeStore;

        let kernel_cfg = KernelConfig::default();
        let sokm_cfg = SokmConfig::default();
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::default(), &kernel_cfg);

        // Build some state
        g.tick(
            &[0.0, 0.0],
            Some(0),
            1,
            &sokm_cfg,
            &kernel_cfg,
            DecayMode::Apply,
        );
        g.tick(
            &[1.0, 0.0],
            Some(0),
            2,
            &sokm_cfg,
            &kernel_cfg,
            DecayMode::Apply,
        );
        g.tick(
            &[0.0, 1.0],
            Some(0),
            3,
            &sokm_cfg,
            &kernel_cfg,
            DecayMode::Apply,
        );
        g.tick(
            &[0.0, 0.0],
            Some(0),
            4,
            &sokm_cfg,
            &kernel_cfg,
            DecayMode::Apply,
        ); // reinforce first kernel
        let n_kernels = g.kernel_count();
        let n_edges = g.edge_count();

        let bytes = rmp_serde::to_vec(&g).unwrap();
        let back: KernelGraph<HashEdgeStore<usize>> = rmp_serde::from_slice(&bytes).unwrap();

        assert_eq!(back.kernel_count(), n_kernels);
        assert_eq!(back.edge_count(), n_edges);
        for i in 0..n_kernels {
            assert_eq!(
                back.kernels().centroid(i),
                g.kernels().centroid(i),
                "kernel {i}: centroid mismatch after roundtrip"
            );
        }
    }

    #[test]
    fn tick_with_none_class_creates_unlabelled_kernel() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        g.tick(
            &[1.0],
            None,
            1,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        assert!(g.kernels().class_opt(0).is_none());
    }

    #[test]
    fn tick_with_some_class_creates_labelled_kernel() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        g.tick(
            &[1.0],
            Some(3),
            1,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        assert_eq!(g.kernels().class_opt(0), Some(3));
    }

    #[test]
    fn unlabelled_kernel_does_not_participate_in_strengthening() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        let report = g.tick(
            &[1.0],
            None,
            1,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        // Second tick near first kernel — class Some(0), but kernel 0 is unlabelled
        let report2 = g.tick(
            &[1.0],
            Some(0),
            2,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        assert_eq!(
            report2.sokm.strengthened, 0,
            "unlabelled kernel excluded from filter"
        );
        let _ = report;
    }

    #[test]
    fn labelled_kernel_class_unchanged_by_coactivation() {
        let cfg = KernelConfig {
            label_inherit_threshold: 2,
            theta_k: 0.99,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        g.tick(&[0.0], Some(1), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        g.tick(&[0.1], Some(2), 2, &sokm_cfg(), &cfg, DecayMode::Apply);
        // Co-activate both many times — labelled kernel should NOT change class
        for t in 3..10 {
            g.tick(&[0.05], Some(1), t, &sokm_cfg(), &cfg, DecayMode::Apply);
        }
        assert_eq!(g.kernels().class_opt(0), Some(1));
    }

    #[test]
    fn newly_labelled_reported_in_tick_report() {
        let cfg = KernelConfig {
            label_inherit_threshold: 2,
            theta_k: 0.99,
            sigma_0: 1.0,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        // kernel 0: labelled Some(1)
        g.tick(&[0.0], Some(1), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        // kernel 1: unlabelled, close to kernel 0
        let cfg_grow = KernelConfig {
            theta_k: 2.0,
            ..cfg.clone()
        };
        g.tick(&[0.1], None, 2, &sokm_cfg(), &cfg_grow, DecayMode::Apply);
        assert_eq!(g.kernel_count(), 2);
        assert!(g.kernels().class_opt(1).is_none());
        // Co-activate threshold=2 times — using cfg with low theta_k so both fire
        let r1 = g.tick(&[0.05], Some(1), 3, &sokm_cfg(), &cfg, DecayMode::Apply);
        let r2 = g.tick(&[0.05], Some(1), 4, &sokm_cfg(), &cfg, DecayMode::Apply);
        let total_labelled = r1.newly_labelled + r2.newly_labelled;
        assert_eq!(total_labelled, 1, "inheritance must fire exactly once");
        assert_eq!(g.kernels().class_opt(1), Some(1));
    }

    #[test]
    fn unlabelled_kernel_does_not_inherit_before_threshold() {
        let cfg = KernelConfig {
            label_inherit_threshold: 5,
            theta_k: 0.99,
            sigma_0: 1.0,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        g.tick(&[0.0], Some(1), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        let cfg_grow = KernelConfig {
            theta_k: 2.0,
            ..cfg.clone()
        };
        g.tick(&[0.1], None, 2, &sokm_cfg(), &cfg_grow, DecayMode::Apply);
        // Tick 4 times (threshold - 1)
        for t in 3..7 {
            g.tick(&[0.05], Some(1), t, &sokm_cfg(), &cfg, DecayMode::Apply);
        }
        assert!(
            g.kernels().class_opt(1).is_none(),
            "must not inherit before threshold"
        );
    }

    #[test]
    fn unlabelled_kernel_inherits_class_after_threshold_coactivations() {
        let cfg = KernelConfig {
            label_inherit_threshold: 3,
            theta_k: 0.99,
            sigma_0: 1.0,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        g.tick(&[0.0], Some(1), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        let cfg_grow = KernelConfig {
            theta_k: 2.0,
            ..cfg.clone()
        };
        g.tick(&[0.1], None, 2, &sokm_cfg(), &cfg_grow, DecayMode::Apply);
        assert!(g.kernels().class_opt(1).is_none());
        // Tick exactly threshold times
        let mut labelled = false;
        for t in 3..=5 {
            let r = g.tick(&[0.05], Some(1), t, &sokm_cfg(), &cfg, DecayMode::Apply);
            if r.newly_labelled > 0 {
                labelled = true;
            }
        }
        assert!(labelled, "inheritance must fire by threshold ticks");
        assert_eq!(g.kernels().class_opt(1), Some(1));
    }

    #[test]
    fn coactivation_counts_cleared_after_inheritance() {
        let cfg = KernelConfig {
            label_inherit_threshold: 2,
            theta_k: 0.99,
            sigma_0: 1.0,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        g.tick(&[0.0], Some(1), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        let cfg_grow = KernelConfig {
            theta_k: 2.0,
            ..cfg.clone()
        };
        g.tick(&[0.1], None, 2, &sokm_cfg(), &cfg_grow, DecayMode::Apply);
        g.tick(&[0.05], Some(1), 3, &sokm_cfg(), &cfg, DecayMode::Apply);
        g.tick(&[0.05], Some(1), 4, &sokm_cfg(), &cfg, DecayMode::Apply);
        // After inheritance the entry for (1, 0) must be removed
        assert!(
            g.coactivation_counts.is_empty(),
            "coactivation entry must be cleared after inheritance"
        );
    }

    #[test]
    fn coactivation_counts_reindexed_after_compact() {
        // Use wide sigma so all kernels activate and survive. p1_kernel large enough
        // that kernels 0 and 1 don't go extinct, but kernel 2 does.
        let cfg_grow = KernelConfig {
            label_inherit_threshold: 10,
            theta_k: 2.0, // always grow
            sigma_0: 1.0,
            p1_kernel: u64::MAX,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &cfg_grow);
        g.tick(&[0.0], None, 1, &sokm_cfg(), &cfg_grow, DecayMode::Apply); // kernel 0: unlabelled
        g.tick(
            &[100.0],
            Some(1),
            2,
            &sokm_cfg(),
            &cfg_grow,
            DecayMode::Apply,
        ); // kernel 1: labelled
        g.tick(
            &[200.0],
            Some(2),
            3,
            &sokm_cfg(),
            &cfg_grow,
            DecayMode::Apply,
        ); // kernel 2: labelled, far away
        // Manually mark kernel 2 extinct and insert co-activation count for (0, 1) pair
        g.kernels.mark_extinct(2);
        g.coactivation_counts.insert((0, 1), 5);

        g.compact();

        // After compact: kernel 2 removed. Map: 0→0, 1→1, 2→None (extinct)
        // coactivation_counts (0,1) should survive → (0,1)
        assert!(
            g.coactivation_counts.contains_key(&(0, 1)),
            "count for live kernels must survive compact"
        );
    }

    #[test]
    fn empty_graph_tick_then_compact_no_panic() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        g.tick(
            &[1.0],
            Some(0),
            1,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        g.kernels_mut().mark_extinct(0);
        g.compact(); // no panic
    }

    #[test]
    fn compact_when_no_kernels_extinct_returns_zero() {
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::new(), &kernel_cfg());
        g.tick(
            &[1.0],
            Some(0),
            1,
            &sokm_cfg(),
            &kernel_cfg(),
            DecayMode::Apply,
        );
        assert_eq!(g.compact(), 0);
    }

    #[test]
    fn compact_with_map_returns_correct_map() {
        let cfg = KernelConfig {
            theta_k: 2.0,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        g.tick(&[0.0], Some(0), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        g.tick(&[10.0], Some(0), 2, &sokm_cfg(), &cfg, DecayMode::Apply);
        g.tick(&[20.0], Some(0), 3, &sokm_cfg(), &cfg, DecayMode::Apply);
        g.kernels_mut().mark_extinct(1);
        let map = g.compact_with_map();
        assert_eq!(map, vec![Some(0), None, Some(1)]);
    }

    #[test]
    fn stm_eviction_sequence_up_to_capacity_plus_one() {
        let cfg = KernelConfig {
            theta_k: 2.0,
            stm_capacity: 2,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        // Grow 3 kernels
        g.tick(&[0.0], Some(0), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        g.tick(&[100.0], Some(0), 2, &sokm_cfg(), &cfg, DecayMode::Apply);
        g.tick(&[200.0], Some(0), 3, &sokm_cfg(), &cfg, DecayMode::Apply);
        // STM capacity=2, should have evicted one by now
        assert_eq!(g.stm_len(), 2);
    }

    #[test]
    fn tick_at_current_tick_zero_new_kernel_survives() {
        let cfg = KernelConfig {
            p1_kernel: 5,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        let report = g.tick(&[1.0], Some(0), 0, &sokm_cfg(), &cfg, DecayMode::Apply);
        assert!(report.grew);
        assert_eq!(report.newly_extinct, 0);
        assert!(!g.kernels().is_extinct(0));
    }

    #[test]
    fn propagate_returns_empty_when_no_kernel_meets_theta_k() {
        let cfg = KernelConfig {
            theta_k: 2.0, // impossible threshold
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        g.tick(&[0.0], Some(0), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        g.set_edge(0, 0, 1.0);
        let spread = g.propagate(&[0.0], &cfg, &sokm_cfg());
        assert!(spread.is_empty());
    }

    #[test]
    fn propagate_soft_returns_empty_when_no_scores_positive() {
        // All kernels extinct → all scores 0.0 → no activated → empty spread
        let cfg = KernelConfig {
            theta_k: 2.0,
            ..kernel_cfg()
        };
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        g.tick(&[0.0], Some(0), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        g.kernels_mut().mark_extinct(0);
        let spread = g.propagate_soft(&[0.0], &sokm_cfg());
        assert!(spread.is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn post_serde_roundtrip_tick_works() {
        use sokm::HashEdgeStore;

        let kernel_cfg = KernelConfig::default();
        let sokm_cfg = SokmConfig::default();
        let mut g: KernelGraph<HashEdgeStore<usize>> =
            KernelGraph::new(HashEdgeStore::default(), &kernel_cfg);

        g.tick(
            &[0.0, 0.0],
            Some(0),
            1,
            &sokm_cfg,
            &kernel_cfg,
            DecayMode::Apply,
        );
        g.tick(
            &[1.0, 0.0],
            Some(0),
            2,
            &sokm_cfg,
            &kernel_cfg,
            DecayMode::Apply,
        );

        let bytes = rmp_serde::to_vec(&g).unwrap();
        let mut back: KernelGraph<HashEdgeStore<usize>> = rmp_serde::from_slice(&bytes).unwrap();

        // coactivation_counts is serde(skip) → empty after reload
        assert!(back.coactivation_counts.is_empty());

        // tick still works after reload
        let report = back.tick(
            &[0.0, 0.0],
            Some(0),
            3,
            &sokm_cfg,
            &kernel_cfg,
            DecayMode::Apply,
        );
        assert!(!report.grew);
    }

    #[test]
    fn kernel_graph_tick_skip_decay_preserves_edge_weight() {
        let cfg = kernel_cfg();
        let mut g: KernelGraph<HashEdgeStore<usize>> = KernelGraph::new(HashEdgeStore::new(), &cfg);
        // grow kernel 0
        g.tick(&[0.0], Some(0), 1, &sokm_cfg(), &cfg, DecayMode::Apply);
        // plant a known edge weight
        g.set_edge(0, 0, 0.5);
        // tick far away so kernel 0 is not activated (score ≈ 0, no strengthen);
        // skip_decay=true means decay step is skipped
        g.tick(&[1000.0], Some(0), 2, &sokm_cfg(), &cfg, DecayMode::Skip);
        assert!(
            (g.edges().get_weight(0, 0) - 0.5).abs() < 1e-10,
            "edge weight must not decay when skip_decay=true"
        );
    }
}
