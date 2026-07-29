use sokm::{
    DecayMode, EdgeStore, HashEdgeStore, SokmConfig, SparseEdgeStore, decay, propagate, prune,
    strengthen, tick,
};

/// HashEdgeStore vs SparseEdgeStore parity: same ops produce same results.
#[test]
fn hash_vs_sparse_parity() {
    let cfg = SokmConfig::default();
    let activated: Vec<(usize, f64)> = vec![(0, 1.0), (1, 0.8), (2, 0.6)];

    let mut hash: HashEdgeStore<usize> = HashEdgeStore::new();
    let mut sparse = SparseEdgeStore::new(10);

    // Run several ticks
    for t in 1..=5u64 {
        tick(&mut hash, &activated, t, &cfg, DecayMode::Apply);
        tick(&mut sparse, &activated, t, &cfg, DecayMode::Apply);
    }

    // Compare edge counts
    assert_eq!(hash.edge_count(), sparse.edge_count());

    // Compare all pair weights
    for i in 0..3usize {
        for j in (i + 1)..3usize {
            let hw = hash.get_weight(i, j);
            let sw = sparse.get_weight(i, j);
            assert!(
                (hw - sw).abs() < 1e-10,
                "mismatch at ({i},{j}): hash={hw}, sparse={sw}"
            );
        }
    }

    // Compare propagation
    let hp = propagate(&hash, &[0usize], &cfg);
    let sp = propagate(&sparse, &[0usize], &cfg);
    for &(node, hv) in &hp {
        let sv = sp
            .iter()
            .find(|&&(k, _)| k == node)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        assert!(
            (hv - sv).abs() < 1e-10,
            "propagation mismatch at node {node}: hash={hv}, sparse={sv}"
        );
    }
}

/// Full learn cycle: strengthen x5 then propagate, verify spread.
#[test]
fn full_learn_cycle_strengthen_then_propagate() {
    let cfg = SokmConfig {
        gamma: 1.0,
        ..SokmConfig::default()
    };
    let mut store: HashEdgeStore<u32> = HashEdgeStore::new();
    let activated = vec![(0u32, 1.0), (1, 1.0), (2, 1.0)];

    for t in 1..=5u64 {
        strengthen(&mut store, &activated, t, &cfg);
    }

    // All pairs should have strong edges
    assert_eq!(store.edge_count(), 3);
    assert!(store.get_weight(0, 1) > cfg.w_init);
    assert!(store.get_weight(0, 2) > cfg.w_init);
    assert!(store.get_weight(1, 2) > cfg.w_init);

    // Propagate from node 0 should reach nodes 1 and 2
    let spread = propagate(&store, &[0u32], &cfg);
    let v1 = spread
        .iter()
        .find(|&&(k, _)| k == 1)
        .map(|&(_, v)| v)
        .unwrap_or(0.0);
    let v2 = spread
        .iter()
        .find(|&&(k, _)| k == 2)
        .map(|&(_, v)| v)
        .unwrap_or(0.0);
    assert!(v1 > 0.0);
    assert!(v2 > 0.0);
}

/// Prune convergence: decay-only until edge_count == 0.
#[test]
fn prune_convergence_decay_only_until_empty() {
    let cfg = SokmConfig {
        xi: 0.5, // aggressive decay
        min_weight: 0.001,
        p1: 1000, // don't trigger p1 prune
        ..SokmConfig::default()
    };
    let mut store: HashEdgeStore<u32> = HashEdgeStore::new();
    store.set_weight(0, 1, 1.0);
    store.set_weight(1, 2, 0.5);
    store.set_weight(2, 3, 0.8);

    let mut tick_count = 0;
    while store.edge_count() > 0 && tick_count < 100 {
        decay(&mut store, &cfg);
        prune(&mut store, tick_count, &cfg);
        tick_count += 1;
    }

    assert_eq!(store.edge_count(), 0);
    assert!(
        tick_count < 100,
        "should converge to 0 edges within 100 ticks"
    );
}
