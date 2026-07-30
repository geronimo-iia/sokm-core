# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
