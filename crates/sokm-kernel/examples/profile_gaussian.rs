//! Profile target for Instruments CPU Profiler.
//! Build: cargo build --release -p sokm-kernel --example profile_gaussian
//! Run:   ./target/release/examples/profile_gaussian
//! Or:    instruments -t "CPU Profiler" target/release/examples/profile_gaussian

use sokm_kernel::config::KernelConfig;
use sokm_kernel::growth::should_grow_direct;
use sokm_kernel::store::{DefaultKernelStore, KernelStore};

fn main() {
    const N: usize = 10_000;
    const D: usize = 358;
    const ITERS: usize = 500;

    let cfg = KernelConfig::default();

    let mut store = DefaultKernelStore::new();
    for i in 0..N {
        let c: Vec<f64> = (0..D).map(|j| (i * D + j) as f64 * 0.001).collect();
        store.push(&c, 1.0, Some(0));
    }
    let x: Vec<f64> = (0..D).map(|i| i as f64 * 0.0005).collect();

    // Enough iterations for the profiler to collect meaningful samples.
    for _ in 0..ITERS {
        std::hint::black_box(should_grow_direct(&store, &x, &cfg));
    }
}
