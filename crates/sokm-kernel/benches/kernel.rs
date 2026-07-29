use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use sokm::DecayMode;
use sokm::SokmConfig;
use sokm::SparseEdgeStore;
use sokm_kernel::KernelStore;
#[cfg(feature = "simd")]
use sokm_kernel::activation::batch_gaussian_simd;
use sokm_kernel::activation::{compact, gaussian};
use sokm_kernel::config::KernelConfig;
use sokm_kernel::graph::KernelGraph;
use sokm_kernel::growth::should_grow_direct;
use sokm_kernel::store::DefaultKernelStore;

fn make_graph(n: usize, d: usize, edges_per_node: usize) -> KernelGraph<SparseEdgeStore> {
    let sokm_cfg = SokmConfig::default();
    let kernel_cfg = KernelConfig::default();
    let mut g = KernelGraph::new(SparseEdgeStore::new(n + 10), &kernel_cfg);
    for i in 0..n {
        let x: Vec<f64> = (0..d).map(|j| (i * d + j) as f64 * 0.001).collect();
        g.tick(
            &x,
            Some(0),
            i as u64,
            &sokm_cfg,
            &kernel_cfg,
            DecayMode::Apply,
        );
    }
    for i in 0..n {
        for j in 1..=edges_per_node {
            let neighbor = (i + j) % n;
            g.set_edge(i, neighbor, 0.5);
        }
    }
    g
}

fn make_kernels(n: usize, d: usize) -> DefaultKernelStore {
    let mut store = DefaultKernelStore::new();
    for i in 0..n {
        let c: Vec<f64> = (0..d).map(|j| (i * d + j) as f64 * 0.001).collect();
        store.push(&c, 1.0, Some(0));
    }
    store
}

fn bench_gaussian(c: &mut Criterion) {
    let x: Vec<f64> = (0..64).map(|i| i as f64 * 0.01).collect();
    let center: Vec<f64> = (0..64).map(|i| i as f64 * 0.005).collect();
    c.bench_function("gaussian_64d", |b| b.iter(|| gaussian(&x, &center, 1.0)));
}

fn bench_compact(c: &mut Criterion) {
    let x: Vec<f64> = (0..64).map(|i| i as f64 * 0.01).collect();
    let center: Vec<f64> = (0..64).map(|i| i as f64 * 0.005).collect();
    c.bench_function("compact_64d", |b| {
        b.iter(|| compact(&x, &center, 1.0, 2.67))
    });
}

fn bench_should_grow_direct(c: &mut Criterion) {
    let cfg = KernelConfig::default();
    let store = make_kernels(1000, 16);
    let x: Vec<f64> = (0..16).map(|d| d as f64 * 0.005).collect();
    c.bench_function("should_grow_direct_1k", |b| {
        b.iter(|| should_grow_direct(black_box(&store), black_box(&x), black_box(&cfg)))
    });
}

fn bench_should_grow_direct_realistic(c: &mut Criterion) {
    let cfg = KernelConfig::default();
    let mut group = c.benchmark_group("should_grow_direct");
    for &n in &[1_000usize, 5_000, 10_000] {
        let store = make_kernels(n, 358);
        let x: Vec<f64> = (0..358).map(|i| i as f64 * 0.0005).collect();
        group.bench_function(format!("358d/{n}"), |b| {
            b.iter(|| should_grow_direct(black_box(&store), black_box(&x), black_box(&cfg)))
        });
    }
    group.finish();
}

fn bench_should_grow_direct_exit(c: &mut Criterion) {
    let cfg = KernelConfig::default();
    let mut group = c.benchmark_group("should_grow_direct_exit");
    for &n in &[1_000usize, 5_000, 10_000] {
        let store = make_kernels(n, 358);
        // x identical to first centroid → exits at i=0
        let x: Vec<f64> = store.centroid(0).to_vec();
        group.bench_function(format!("358d/{n}"), |b| {
            b.iter(|| should_grow_direct(black_box(&store), black_box(&x), black_box(&cfg)))
        });
    }
    group.finish();
}

fn bench_kernel_graph_tick_parametric(c: &mut Criterion) {
    let sokm_cfg = SokmConfig::default();
    let kernel_cfg = KernelConfig::default();
    let x: Vec<f64> = (0..16).map(|d| d as f64 * 0.005).collect();
    let mut group = c.benchmark_group("kernel_graph_tick");
    for &n in &[100usize, 500, 1000] {
        group.bench_function(BenchmarkId::new("n_16d", n), |b| {
            b.iter_batched(
                || make_graph(n, 16, 4),
                |mut g| g.tick(&x, Some(0), 9999, &sokm_cfg, &kernel_cfg, DecayMode::Apply),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_kernel_graph_tick_realistic(c: &mut Criterion) {
    let sokm_cfg = SokmConfig::default();
    let kernel_cfg = KernelConfig::default();
    let mut group = c.benchmark_group("kernel_graph_tick");
    for &n in &[1_000usize, 5_000, 10_000] {
        let x: Vec<f64> = (0..358).map(|i| i as f64 * 0.0005).collect();
        group.bench_function(format!("358d/{n}"), |b| {
            b.iter_batched(
                || make_graph(n, 358, 4),
                |mut g| g.tick(&x, Some(0), 99999, &sokm_cfg, &kernel_cfg, DecayMode::Apply),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

#[cfg(feature = "simd")]
fn bench_compute_scores_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_scores_simd_vs_scalar");
    for &n in &[1_000usize, 5_000, 10_000] {
        let d = 358usize;
        let centroids: Vec<f64> = (0..n * d).map(|i| (i as f64 * 0.001) % 2.0).collect();
        let sigmas = vec![1.0f64; n];
        let x: Vec<f64> = (0..d).map(|i| i as f64 * 0.0005).collect();

        // Scalar baseline: replicate compute_scores logic
        group.bench_function(format!("scalar/358d/{n}"), |b| {
            b.iter(|| {
                let _: Vec<f64> = (0..n)
                    .map(|k| {
                        let c = &centroids[k * d..(k + 1) * d];
                        let sq: f64 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
                        (-sq / (sigmas[k] * sigmas[k])).exp()
                    })
                    .collect();
            });
        });

        group.bench_function(format!("simd/358d/{n}"), |b| {
            b.iter(|| {
                black_box(batch_gaussian_simd(
                    black_box(&centroids),
                    black_box(&sigmas),
                    black_box(&x),
                ))
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "simd"))]
criterion_group!(
    benches,
    bench_gaussian,
    bench_compact,
    bench_should_grow_direct,
    bench_should_grow_direct_realistic,
    bench_should_grow_direct_exit,
    bench_kernel_graph_tick_parametric,
    bench_kernel_graph_tick_realistic,
);

#[cfg(feature = "simd")]
criterion_group!(
    benches,
    bench_gaussian,
    bench_compact,
    bench_should_grow_direct,
    bench_should_grow_direct_realistic,
    bench_should_grow_direct_exit,
    bench_kernel_graph_tick_parametric,
    bench_kernel_graph_tick_realistic,
    bench_compute_scores_simd_vs_scalar,
);

criterion_main!(benches);
