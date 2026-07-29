# Contributing

## Prerequisites

- Rust stable (MSRV 1.95). No nightly required.
- `cargo` only — no build scripts, no external tools required for basic dev.
- Optional: `cargo-instruments` (macOS) for profiling. Install: `cargo install cargo-instruments`.

## Building

```bash
cargo build --workspace
cargo build --workspace --features sokm-memory/emotion
```

## Testing

Full test matrix — all combinations must pass before a PR:

```bash
cargo test --workspace
cargo test --workspace --features sokm-memory/emotion
cargo test --workspace --features sokm-kernel/simd
cargo test --workspace --features sokm-memory/emotion,sokm-kernel/simd
```

`sokm-emotion` tests run as part of `--workspace`. The `emotion` feature on `sokm-memory`
enables `EmotionalMemoryStore` — tested separately because it pulls in `sokm-emotion` as
a dep of `sokm-memory`.

## Formatting and linting

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features sokm-memory/emotion -- -D warnings
```

Both must be clean before committing. `cargo fmt --all` is run as a pre-commit step.

## Running benchmarks

```bash
# All benchmarks, default features
cargo bench --workspace

# With SIMD (sokm-kernel only)
cargo bench -p sokm-kernel --features simd

# Single benchmark by name filter
cargo bench -p sokm-kernel --features simd -- compute_scores
```

Benchmark results are not committed. Record significant results manually in `docs/benchmarks.md`
with date, machine, and feature flags.

## Crate map

| Crate | Responsibility |
|-------|---------------|
| `sokm` | Link layer — Hebbian mechanics (EdgeStore, decay, strengthen, prune, propagate, tick) |
| `sokm-kernel` | Kernel units, growth, STM, class inheritance |
| `sokm-multimodal` | Gestalt K³ cross-modal memory |
| `sokm-emotion` | Per-kernel emotion variables, global state |
| `sokm-memory` | Episodic store, ANN index, snapshot |

Full detail: [docs/architecture.md](docs/architecture.md).

## Feature flags

| Feature | Crate | Enables | Default |
|---------|-------|---------|---------|
| `simd` | `sokm-kernel` | `batch_gaussian_simd` via `wide::f64x4` — 2.35× scoring speedup at 358d/10k | off |
| `simd` | `sokm-emotion` | propagates `sokm-kernel/simd` | off |
| `simd` | `sokm-multimodal` | propagates `sokm-kernel/simd` | off |
| `simd` | `sokm-memory` | propagates `sokm-kernel/simd` | off |
| `emotion` | `sokm-memory` | `EmotionalMemoryStore`, `EmotionalRememberResult`, salience-weighted recall | off |
| `serde` | `sokm-emotion` | `Serialize`/`Deserialize` on result types | on (workspace dep) |

## Commit style

- Conventional commits: `feat(crate):`, `fix(crate):`, `test(crate):`, `docs:`, `style:`, `refactor(crate):`
- `crate` = short crate name without `sokm-` prefix where unambiguous (e.g. `feat(emotion):`, `fix(memory):`)
- One logical change per commit. Format before committing.

## Adding a new crate

1. Create under `crates/` with `version.workspace = true`, `edition.workspace = true`
2. Add to `[workspace] members` in root `Cargo.toml`
3. Add workspace dep entry
4. Add to crate map in `docs/architecture.md` and this file
5. All new public fallible functions in lower crates (`sokm-*` below `sokm-memory`) must use typed `thiserror` errors — see `docs/error-logging-strategy.md`
