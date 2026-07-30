# Contributing

## Prerequisites

- Rust stable (MSRV 1.95). No nightly required.
- `cargo` only — no build scripts, no external tools required for basic dev.
- Optional: `cargo-instruments` (macOS) for profiling. Install: `cargo install cargo-instruments`.
- Optional: `cargo-release` for releases. Install: `cargo install cargo-release`.

## Building

```bash
cargo build --workspace
cargo build --workspace --features sokm-kernel/simd
```

## Testing

Full test matrix — all combinations must pass before a PR:

```bash
cargo test --workspace
cargo test --workspace --features sokm-kernel/simd
```

## Formatting and linting

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

Both must be clean before committing.

## Running benchmarks

```bash
# All benchmarks, default features
cargo bench --workspace

# With SIMD (sokm-kernel only)
cargo bench -p sokm-kernel --features simd

# Single benchmark by name filter
cargo bench -p sokm-kernel --features simd -- compute_scores
```

Benchmark results are not committed. Record significant results manually in a PR comment with date, machine, and feature flags.

## Crate map

| Crate | Responsibility |
|-------|---------------|
| `sokm` | Link layer — Hebbian mechanics (EdgeStore, decay, strengthen, prune, propagate, tick) |
| `sokm-kernel` | Kernel units, growth, STM, class inheritance |
| `sokm-emotion` | Per-kernel emotion variables, global 2D state, attentive condition, policy |

## Feature flags

| Feature | Crate | Enables | Default |
|---------|-------|---------|---------|
| `simd` | `sokm-kernel` | `batch_gaussian_simd` via `wide::f64x4` — 2.35× scoring speedup at 358d/10k | off |
| `simd` | `sokm-emotion` | Delegates to `sokm-kernel/simd` | off |
| `serde` | `sokm` | `Serialize`/`Deserialize` on all public types | off |
| `serde` | `sokm-kernel` | `Serialize`/`Deserialize` on all public types | off |
| `serde` | `sokm-emotion` | `Serialize`/`Deserialize` on all public types | off |

## Commit style

Conventional commits, single line. Format: `type(scope): short description`.
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `ci`, `chore`.
Scope = short crate name without `sokm-` prefix where unambiguous (e.g. `fix(sparse):`, `feat(kernel):`).

## Pull requests

- One logical change per PR
- Tests must pass
- `cargo fmt` and `cargo clippy` must be clean

## Release process

### Branch strategy

`main` is always releasable — tagged commits only. Feature work lands on a
`release/vX.Y.Z` integration branch, not directly on `main`.

```
feat/xxx  ─┐
feat/yyy  ─┼─▶  release/vX.Y.Z  ─▶  main  (tag vX.Y.Z)
feat/zzz  ─┘
```

1. Open `release/vX.Y.Z` from `main` at the start of the milestone.
2. Each `feat/...` PR targets `release/vX.Y.Z`, not `main`.
3. Run the pre-release checklist as commits on `release/vX.Y.Z`.
4. One final PR merges `release/vX.Y.Z` → `main`; tag on the merge commit.

Hotfixes branch from the relevant tag and merge back to `main`.

### Pre-release checklist

- [ ] All tests pass: `cargo test --workspace`
- [ ] All tests pass (simd): `cargo test --workspace --features sokm-kernel/simd`
- [ ] Doc tests pass: `cargo test --workspace --doc`
- [ ] Formatted: `cargo fmt --all -- --check`
- [ ] No lint issues: `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Deny clean: `cargo deny check`
- [ ] Release build clean: `cargo build --workspace --release --locked`
- [ ] Examples compile: `cargo build --workspace --examples`
- [ ] Bench compiles: `cargo bench --workspace --all-features --no-run`
- [ ] Dry-run publish: `cargo release --dry-run` (publishes `sokm` → `sokm-kernel` → `sokm-emotion`)
- [ ] `CHANGELOG.md` section dated and complete
- [ ] Public types have `///` rustdoc; `cargo doc --workspace --no-deps --all-features` zero warnings
- [ ] Version bumped in workspace `Cargo.toml`, `Cargo.lock` updated

### Tagging and publishing

`release.toml` configures tag format (`vX.Y.Z`). Publish order follows the dependency graph: `sokm` → `sokm-kernel` → `sokm-emotion`.

```bash
# 1. Bump version in Cargo.toml, update CHANGELOG date
# Edit Cargo.toml: version = "X.Y.Z"
cargo update -p sokm -p sokm-kernel -p sokm-emotion

# 2. Commit on release branch, push, open PR
git commit -am "chore: release vX.Y.Z"
git push origin release/vX.Y.Z
gh pr create --title "chore: release vX.Y.Z" --base main

# 3. Wait for CI to pass, merge to main
git checkout main
git merge --no-ff release/vX.Y.Z

# 4. Tag and push — publishes sokm then sokm-kernel in order
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main
git push origin vX.Y.Z
```

Or use `cargo-release` directly (respects `release.toml` publish order):

```bash
cargo release X.Y.Z
```

Tags containing `-rc` (e.g. `v0.2.0-rc1`) follow the same steps but the
publish job is skipped — nothing sent to crates.io.

### Hotfix

```bash
git checkout -b hotfix/vX.Y.Z+1 vX.Y.Z
# apply fix, bump patch version in Cargo.toml
git commit -am "fix: description"
git tag -a vX.Y.Z+1 -m "Hotfix vX.Y.Z+1"
git push origin hotfix/vX.Y.Z+1 vX.Y.Z+1
git checkout main
git merge --no-ff hotfix/vX.Y.Z+1
git push origin main
```

### CHANGELOG format

Move `[Unreleased]` entries to a versioned section:

```markdown
## [0.2.0] — 2026-MM-DD

### Added
- …

### Fixed
- …

## [Unreleased]
```
