use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use sokm::{
    DecayMode, EdgeStore, SokmConfig, SparseEdgeStore, decay, propagate, propagate_soft, prune,
    strengthen, tick,
};

fn setup_store(nodes: usize, edges_per_node: usize) -> SparseEdgeStore {
    let mut s = SparseEdgeStore::new(nodes);
    for i in 0..nodes {
        for j in 1..=edges_per_node {
            let neighbor = (i + j) % nodes;
            s.set_weight(i, neighbor, 0.5);
        }
    }
    s
}

fn activated(n: usize) -> Vec<(usize, f64)> {
    (0..n).map(|i| (i, 0.5 + (i as f64 * 0.01) % 0.5)).collect()
}

fn bench_tick(c: &mut Criterion) {
    let cfg = SokmConfig::default();
    let active = activated(50);
    c.bench_function("tick_1k_nodes_50_active", |b| {
        b.iter_batched(
            || setup_store(1000, 5),
            |mut s| {
                tick(
                    black_box(&mut s),
                    black_box(&active),
                    black_box(1),
                    black_box(&cfg),
                    DecayMode::Apply,
                )
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_decay(c: &mut Criterion) {
    let cfg = SokmConfig::default();
    c.bench_function("decay_1k_nodes", |b| {
        b.iter_batched(
            || setup_store(1000, 5),
            |mut s| decay(black_box(&mut s), black_box(&cfg)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_strengthen(c: &mut Criterion) {
    let cfg = SokmConfig::default();
    let active = activated(50);
    c.bench_function("strengthen_50_active", |b| {
        b.iter_batched(
            || setup_store(1000, 5),
            |mut s| {
                strengthen(
                    black_box(&mut s),
                    black_box(&active),
                    black_box(1),
                    black_box(&cfg),
                )
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_prune(c: &mut Criterion) {
    let cfg = SokmConfig::default();
    c.bench_function("prune_1k_nodes", |b| {
        b.iter_batched(
            || setup_store(1000, 5),
            |mut s| prune(black_box(&mut s), black_box(1), black_box(&cfg)),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_propagate_soft(c: &mut Criterion) {
    let cfg = SokmConfig::default();
    let s = setup_store(1000, 5);
    let active = activated(50);
    c.bench_function("propagate_soft_50_active", |b| {
        b.iter(|| propagate_soft(black_box(&s), black_box(&active), black_box(&cfg)))
    });
}

fn bench_propagate_binary(c: &mut Criterion) {
    let cfg = SokmConfig::default();
    let s = setup_store(1000, 5);
    let fired: Vec<usize> = (0..50).collect();
    c.bench_function("propagate_binary_50_active", |b| {
        b.iter(|| propagate(black_box(&s), black_box(&fired), black_box(&cfg)))
    });
}

criterion_group!(
    benches,
    bench_tick,
    bench_decay,
    bench_strengthen,
    bench_prune,
    bench_propagate_soft,
    bench_propagate_binary
);
criterion_main!(benches);
