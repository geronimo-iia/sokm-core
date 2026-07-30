use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use sokm::{DecayMode, HashEdgeStore};
use sokm_multimodal::{DefaultGestaltGraph, GestaltConfig};
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

fn bench_gestalt_tick(c: &mut Criterion) {
    let cfg = GestaltConfig::default();
    let mut group = c.benchmark_group("gestalt_tick");
    for &n in &[100usize, 500, 1_000] {
        let x1: Vec<f64> = (0..16).map(|i| i as f64 * 0.005).collect();
        let x2: Vec<f64> = (0..16).map(|i| i as f64 * 0.003).collect();
        group.bench_function(BenchmarkId::new("n_16d", n), |b| {
            b.iter_batched(
                || make_graph(n, 16),
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
    for &n in &[100usize, 500, 1_000, 2_000] {
        let g = make_graph(n, 16);
        let x1: Vec<f64> = (0..16).map(|i| i as f64 * 0.005).collect();
        group.bench_function(BenchmarkId::new("n_16d", n), |b| {
            b.iter(|| g.recall_from_modal1(black_box(&x1), black_box(&cfg)))
        });
    }
    group.finish();
}

/// O(E) reverse scan — this is the slow path flagged in .claude/note-sources-o-e-reverse-scan.md.
fn bench_recall_from_modal2(c: &mut Criterion) {
    let cfg = GestaltConfig::default();
    let mut group = c.benchmark_group("recall_from_modal2");
    for &n in &[100usize, 500, 1_000, 2_000] {
        let g = make_graph(n, 16);
        let x2: Vec<f64> = (0..16).map(|i| i as f64 * 0.003).collect();
        group.bench_function(BenchmarkId::new("n_16d", n), |b| {
            b.iter(|| g.recall_from_modal2(black_box(&x2), black_box(&cfg)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_gestalt_tick,
    bench_recall_from_modal1,
    bench_recall_from_modal2,
);
criterion_main!(benches);
