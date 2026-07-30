//! Cross-modal associative memory via Hebbian-linked kernel graphs (Gestalt K³).
//!
//! Two `KernelGraph` modalities are coupled through a directed bipartite cross-edge
//! store. Co-activation under a shared class label strengthens cross-modal edges;
//! decay and inactivity pruning keep the graph sparse.
//!
//! # Quick start
//!
//! ```
//! use sokm::DecayMode;
//! use sokm_multimodal::{DefaultGestaltGraph, GestaltConfig, GestaltKernelGraph};
//!
//! let cfg = GestaltConfig::default();
//! let mut g = DefaultGestaltGraph::default();
//!
//! // Train: pair auditory (modal1) with visual (modal2) under class 1.
//! for t in 0..5u64 {
//!     g.tick(&[1.0, 0.0], &[0.0, 1.0], Some(1), t, &cfg, DecayMode::Apply);
//! }
//!
//! // Recall: present modal1 alone → retrieve associated modal2 activations.
//! let modal2_activations = g.recall_from_modal1(&[1.0, 0.0], &cfg);
//! assert!(!modal2_activations.is_empty());
//! ```

pub(crate) mod config;
pub(crate) mod cross;
pub(crate) mod graph;

pub use config::{CrossConfigError, CrossSokmConfig, GestaltConfig};
pub use cross::{
    CrossEdgeStore, CrossStore, cross_propagate_soft, cross_propagate_soft_reverse,
    cross_strengthen_deltas,
};
pub use graph::{DefaultGestaltGraph, GestaltKernelGraph, GestaltTickReport};
