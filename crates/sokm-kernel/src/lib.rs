//! SOKM kernel unit layer.
//!
//! Implements kernel units, activation functions, one-pass growth, and STM
//! from Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*.
//!
//! # Primary API
//!
//! - [`KernelUnit`] — kernel unit struct (centroid, sigma, class, excitation)
//! - [`activation::gaussian`] / [`activation::compact`] — activation functions [Eq 3.8, 3.10]
//! - [`should_grow_direct`], [`grow`], [`max_activation`] — one-pass growth rule
//! - [`Stm`] — working memory [p.164, Eq 10.5]
//!
//! # Convenience wrapper
//!
//! [`KernelGraph`] bundles the above for standalone use and tests.

pub mod activation;
pub mod config;
pub mod graph;
pub mod growth;
pub mod query;
pub mod stm;
pub mod store;
pub mod unit;

// Primary surface — re-exported at crate root for direct ECS use
pub use activation::{compact, gaussian};
pub use config::{KernelConfig, KernelConfigError};
pub use growth::{compute_scores, grow, max_activation, should_grow_direct};
pub use query::{best_match, predict};
pub use stm::Stm;
pub use unit::KernelUnit;

// Store backends
pub use store::{AosKernelStore, KernelStore};

// Convenience wrapper — useful for standalone and tests
pub use graph::{AosKernelGraph, KernelGraph, KernelTickReport};
