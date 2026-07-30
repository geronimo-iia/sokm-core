# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `sokm-multimodal` crate: Gestalt K³ cross-modal memory — two modalities coupled via directed bipartite cross-edge store
- `GestaltKernelGraph<S1, S2, K1, K2>`: generic cross-modal graph; `DefaultGestaltGraph` concrete alias
- `GestaltConfig`, `CrossSokmConfig`, `CrossConfigError`: cross-modal configuration and validation
- `CrossStore` trait, `CrossEdgeStore`: HashMap-backed directed bipartite edge store
- `cross_propagate_soft`, `cross_propagate_soft_reverse`, `cross_strengthen_deltas`: free-function primitives
- Cross-modal invariants (#8–#10), equations, and integration decision doc in `docs/`
- Criterion benchmarks: `gestalt_tick_sparse` (SparseEdgeStore), `recall_simd` (simd feature), all parametrized over `(n,d)` pairs including 358d
- `examples/convergence.rs`: 2-class cross-modal convergence validation (500 ticks)

### Changed
- Workspace version bumped to `0.3.0`
- `CrossEdgeStore::sources()` now O(1) via reverse index (was O(E) full scan)

## [0.2.0] - 2026-07-30

### Added
- `sokm-emotion` crate: per-kernel emotion variables, 2D global state `(E₁, E₂)`, attentive condition [Hoya Eqs. 10.6–10.8]
- `GlobalEmotionPolicy` trait: `IdentityPolicy` (exact Hoya), `ClampPolicy`, `DecayPolicy`
- `salience()` for emotion-weighted recall scoring
- Two examples: `emotional_learning`, `policy_comparison`
- Emotion invariants, equations, and decision doc in `docs/`
- `examples compile` step in pre-release checklist

### Changed
- Workspace version bumped to `0.2.0`

## [0.1.1] - 2026-07-30

### Changed
- Reordered declarations in `sokm-kernel/src/config.rs`, `sokm/src/config.rs`, and `sokm/src/ops.rs` to follow conventional Rust order: types before impls, enums before `impl Default`, private helper fns after impls

## [0.1.0] - 2026-07-30

### Added
- Initial release of `sokm` and `sokm-kernel`
