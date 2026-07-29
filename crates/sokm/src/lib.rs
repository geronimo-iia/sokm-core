//! SOKM Hebbian link mechanics.
//!
//! Implements decay, strengthen, prune, and propagation from
//! Tetsuya Hoya (2005), *Artificial Mind System: Kernel Memory Approach*.
//!
//! The crate provides two edge-store backends (`HashEdgeStore` for tests/small
//! graphs and `SparseEdgeStore` for production CSR-backed workloads) plus the
//! core Hebbian operations that operate on any `EdgeStore` implementation.
//!
//! # Quick start
//!
//! ```
//! use sokm::{HashEdgeStore, SokmConfig, DecayMode, tick};
//! use sokm::EdgeStore;
//!
//! let mut store: HashEdgeStore<u32> = HashEdgeStore::new();
//! let cfg = SokmConfig::default();
//! let activated = vec![(0u32, 1.0), (1, 0.8)];
//! let report = tick(&mut store, &activated, 1, &cfg, DecayMode::Apply);
//! assert_eq!(report.strengthened, 1);
//! ```

pub(crate) mod config;
pub(crate) mod ops;
pub(crate) mod sparse;
pub(crate) mod store;

pub use config::{SokmConfig, SokmConfigError, SokmConfigWarning};
pub use ops::{
    DecayMode, SokmReport, decay, propagate, propagate_soft, prune, strengthen, tick, top_n,
};
pub use sparse::SparseEdgeStore;
pub use store::{EdgeStore, HashEdgeStore, Reindex};
