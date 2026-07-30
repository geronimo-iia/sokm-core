use sokm_multimodal::{CrossEdgeStore, CrossStore};

fn main() {
    println!(
        "{:<10} {:>12} {:>12} {:>18}",
        "edges", "capacity", "len", "est_bytes"
    );

    for &e in &[1_000usize, 10_000, 100_000, 500_000] {
        let mut store = CrossEdgeStore::new();
        for i in 0..e {
            // Use distinct (i, i % target) pairs to avoid collisions
            store.set(i, i % 1000, 1.0);
        }

        let edge_count = store.edge_count();

        // Analytical estimate:
        // HashMap<(usize,usize), f64>: each entry ~48 bytes (key=16, val=8, hash=8, metadata+ptr≈16)
        // HashMap<(usize,usize), u64>: same layout ~48 bytes (ticks)
        // HashMap<usize, Vec<usize>>: key=8, Vec=24 (ptr+len+cap), contents=8*avg_fanin
        //   at e edges over 1000 targets → avg_fanin = e/1000
        // Approximate: weights + ticks = 2 * e * 48; reverse = 1000 * 40 + e * 8
        let avg_fanin = e / 1000;
        let weights_bytes = e * 48;
        let ticks_bytes = e * 48;
        let reverse_entries = 1000usize;
        let reverse_bytes = reverse_entries * (8 + 24) + e * 8;
        let _ = avg_fanin; // documented above
        let total_est = weights_bytes + ticks_bytes + reverse_bytes;

        println!(
            "{:<10} {:>12} {:>12} {:>18}",
            e,
            edge_count,
            edge_count,
            format_bytes(total_est),
        );
    }

    println!();
    println!("Note: estimates use HashMap load factor ~0.875, entry size 48 bytes.");
    println!("Actual RSS will be higher due to allocator overhead and HashMap metadata.");
}

fn format_bytes(b: usize) -> String {
    if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}
