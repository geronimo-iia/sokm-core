use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use sokm::{DecayMode, HashEdgeStore, SparseEdgeStore};
use sokm_multimodal::{
    CrossEdgeStore, CrossStore, DefaultGestaltGraph, GestaltConfig, GestaltKernelGraph,
};
use std::hint::black_box;

/// Build a trained gestalt graph with `n` kernels per modality and `d`-dimensional inputs.
/// Each modality sees distinct inputs so kernels are well-separated.
fn make_graph(n: usize, d: usize) -> DefaultGestaltGraph {
    let cfg = GestaltConfig::default();
    let mut g = DefaultGestaltGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);
    // Use large spacing so every tick grows a new kernel in both modalities.
    // Each centroid is far enough from all others that gaussian activation stays below theta_k.
    for i in 0..n {
        let x1: Vec<f64> = (0..d).map(|j| (i * d + j) as f64 * 10.0).collect();
        let x2: Vec<f64> = (0..d).map(|j| (i * d + j) as f64 * 20.0).collect();
        g.tick(&x1, &x2, Some(0), i as u64, &cfg, DecayMode::Apply);
    }
    g
}

fn make_graph_sparse(n: usize, d: usize) -> GestaltKernelGraph<SparseEdgeStore, SparseEdgeStore> {
    let cfg = GestaltConfig::default();
    let mut g = GestaltKernelGraph::new(
        SparseEdgeStore::new(n + 10),
        SparseEdgeStore::new(n + 10),
        &cfg,
    );
    for i in 0..n {
        let x1: Vec<f64> = (0..d).map(|j| (i * d + j) as f64 * 10.0).collect();
        let x2: Vec<f64> = (0..d).map(|j| (i * d + j) as f64 * 20.0).collect();
        g.tick(&x1, &x2, Some(0), i as u64, &cfg, DecayMode::Apply);
    }
    g
}

fn bench_gestalt_tick(c: &mut Criterion) {
    let cfg = GestaltConfig::default();
    let mut group = c.benchmark_group("gestalt_tick");
    let nd_pairs: &[(usize, usize)] =
        &[(100, 16), (500, 16), (1_000, 16), (500, 358), (1_000, 358)];
    for &(n, d) in nd_pairs {
        let x1: Vec<f64> = (0..d).map(|i| i as f64 * 0.005).collect();
        let x2: Vec<f64> = (0..d).map(|i| i as f64 * 0.003).collect();
        group.bench_function(BenchmarkId::new("NxD", format!("{n}x{d}")), |b| {
            b.iter_batched(
                || make_graph(n, d),
                |mut g| {
                    g.tick(
                        black_box(&x1),
                        black_box(&x2),
                        black_box(Some(0)),
                        black_box(9999),
                        black_box(&cfg),
                        DecayMode::Apply,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_gestalt_tick_sparse(c: &mut Criterion) {
    let cfg = GestaltConfig::default();
    let mut group = c.benchmark_group("gestalt_tick_sparse");
    let nd_pairs: &[(usize, usize)] =
        &[(100, 16), (500, 16), (1_000, 16), (500, 358), (1_000, 358)];
    for &(n, d) in nd_pairs {
        let x1: Vec<f64> = (0..d).map(|i| i as f64 * 0.005).collect();
        let x2: Vec<f64> = (0..d).map(|i| i as f64 * 0.003).collect();
        group.bench_function(BenchmarkId::new("NxD", format!("{n}x{d}")), |b| {
            b.iter_batched(
                || make_graph_sparse(n, d),
                |mut g| {
                    g.tick(
                        black_box(&x1),
                        black_box(&x2),
                        black_box(Some(0)),
                        black_box(9999),
                        black_box(&cfg),
                        DecayMode::Apply,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_recall_from_modal1(c: &mut Criterion) {
    let cfg = GestaltConfig::default();
    let mut group = c.benchmark_group("recall_from_modal1");
    let nd_pairs: &[(usize, usize)] =
        &[(100, 16), (500, 16), (1_000, 16), (500, 358), (1_000, 358)];
    for &(n, d) in nd_pairs {
        let g = make_graph(n, d);
        let x1: Vec<f64> = (0..d).map(|i| i as f64 * 0.005).collect();
        group.bench_function(BenchmarkId::new("NxD", format!("{n}x{d}")), |b| {
            b.iter(|| g.recall_from_modal1(black_box(&x1), black_box(&cfg)))
        });
    }
    group.finish();
}

/// require_class_match=false: all co-activations strengthen, no class filter.
/// Expect higher cross_edge growth than class-match variant — measures O(active²) edge churn.
fn bench_gestalt_tick_no_class_match(c: &mut Criterion) {
    let mut cfg = GestaltConfig::default();
    cfg.cross.require_class_match = false;
    let mut group = c.benchmark_group("gestalt_tick_no_class_match");
    let nd_pairs: &[(usize, usize)] =
        &[(100, 16), (500, 16), (1_000, 16), (500, 358), (1_000, 358)];
    for &(n, d) in nd_pairs {
        let x1: Vec<f64> = (0..d).map(|i| i as f64 * 0.005).collect();
        let x2: Vec<f64> = (0..d).map(|i| i as f64 * 0.003).collect();
        // Build graph with no-class-match config
        let make = || {
            let mut g = DefaultGestaltGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);
            for i in 0..n {
                let gx1: Vec<f64> = (0..d).map(|j| (i * d + j) as f64 * 10.0).collect();
                let gx2: Vec<f64> = (0..d).map(|j| (i * d + j) as f64 * 20.0).collect();
                g.tick(&gx1, &gx2, Some(0), i as u64, &cfg, DecayMode::Apply);
            }
            g
        };
        group.bench_function(BenchmarkId::new("NxD", format!("{n}x{d}")), |b| {
            b.iter_batched(
                make,
                |mut g| {
                    g.tick(
                        black_box(&x1),
                        black_box(&x2),
                        black_box(Some(0)),
                        black_box(9999),
                        black_box(&cfg),
                        DecayMode::Apply,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// compact() cost vs extinction fraction.
/// Setup: grow n kernels, mark k% extinct by pruning edges below a threshold that
/// eliminates exactly k% of cross edges. Measures CrossEdgeStore::reindex at different
/// sparsity levels — compact() delegates there.
fn bench_compact_reindex(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact_reindex");
    let n = 1_000usize;
    // Extinction fractions to test: 0%, 25%, 50%, 75%
    for &pct in &[0u32, 25, 50, 75] {
        group.bench_function(BenchmarkId::new("extinct_pct", pct), |b| {
            b.iter_batched(
                || {
                    // Build a CrossEdgeStore with n edges.
                    let mut store = CrossEdgeStore::new();
                    for i in 0..n {
                        store.set(i, i, 1.0);
                    }
                    // Build a reindex map that marks `pct`% of indices extinct (None).
                    let map: Vec<Option<usize>> = (0..n)
                        .map(|i| {
                            // mark extinct if i falls in the first pct% of indices
                            if (i * 100 / n) < pct as usize {
                                None
                            } else {
                                // remap survivors to contiguous indices
                                let survivors_before = i.saturating_sub(i * pct as usize / 100);
                                Some(survivors_before)
                            }
                        })
                        .collect();
                    (store, map)
                },
                |(mut store, map)| {
                    CrossStore::reindex(black_box(&mut store), black_box(&map), black_box(&map))
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// O(1) reverse index lookup — benchmarking to confirm scaling.
fn bench_recall_from_modal2(c: &mut Criterion) {
    let cfg = GestaltConfig::default();
    let mut group = c.benchmark_group("recall_from_modal2");
    let nd_pairs: &[(usize, usize)] =
        &[(100, 16), (500, 16), (1_000, 16), (500, 358), (1_000, 358)];
    for &(n, d) in nd_pairs {
        let g = make_graph(n, d);
        let x2: Vec<f64> = (0..d).map(|i| i as f64 * 0.003).collect();
        group.bench_function(BenchmarkId::new("NxD", format!("{n}x{d}")), |b| {
            b.iter(|| g.recall_from_modal2(black_box(&x2), black_box(&cfg)))
        });
    }
    group.finish();
}

#[cfg(feature = "simd")]
fn bench_recall_simd(c: &mut Criterion) {
    let cfg = GestaltConfig::default();
    let mut group = c.benchmark_group("recall_from_modal1_simd");
    let nd_pairs: &[(usize, usize)] = &[(500, 16), (1_000, 16), (500, 358), (1_000, 358)];
    for &(n, d) in nd_pairs {
        let g = make_graph(n, d);
        let x1: Vec<f64> = (0..d).map(|i| i as f64 * 0.005).collect();
        group.bench_function(BenchmarkId::new("NxD", format!("{n}x{d}")), |b| {
            b.iter(|| g.recall_from_modal1(black_box(&x1), black_box(&cfg)))
        });
    }
    group.finish();
}

#[cfg(not(feature = "simd"))]
criterion_group!(
    benches,
    bench_gestalt_tick,
    bench_gestalt_tick_sparse,
    bench_gestalt_tick_no_class_match,
    bench_compact_reindex,
    bench_recall_from_modal1,
    bench_recall_from_modal2,
);

#[cfg(feature = "simd")]
criterion_group!(
    benches,
    bench_gestalt_tick,
    bench_gestalt_tick_sparse,
    bench_gestalt_tick_no_class_match,
    bench_compact_reindex,
    bench_recall_from_modal1,
    bench_recall_from_modal2,
    bench_recall_simd,
);

criterion_main!(benches);
