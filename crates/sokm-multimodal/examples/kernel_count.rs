use sokm::{DecayMode, HashEdgeStore};
use sokm_multimodal::{DefaultGestaltGraph, GestaltConfig, GestaltKernelGraph};

/// Inspect how many kernels actually grow at different tick counts.
/// Useful for calibrating bench fixture size — growth saturates when inputs
/// are too close together (see make_graph spacing in benches/gestalt.rs).
fn main() {
    for &n in &[100usize, 500, 1_000, 2_000] {
        let cfg = GestaltConfig::default();
        let mut g = DefaultGestaltGraph::new(HashEdgeStore::new(), HashEdgeStore::new(), &cfg);
        for i in 0..n {
            let x1: Vec<f64> = (0..16).map(|j| (i * 16 + j) as f64 * 10.0).collect();
            let x2: Vec<f64> = (0..16).map(|j| (i * 16 + j) as f64 * 20.0).collect();
            g.tick(&x1, &x2, Some(0), i as u64, &cfg, DecayMode::Apply);
        }
        println!(
            "ticks={n:5}: modal1_kernels={:5} modal2_kernels={:5} cross_edges={:5}",
            g.modal1.kernel_count(),
            g.modal2.kernel_count(),
            g.cross_edge_count(),
        );
    }
}
