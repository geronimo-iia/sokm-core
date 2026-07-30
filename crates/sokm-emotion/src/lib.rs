pub(crate) mod config;
pub(crate) mod graph;
pub(crate) mod policy;
pub(crate) mod query;
pub(crate) mod state;
pub(crate) mod update;

pub use config::{EmotionConfig, EmotionConfigError, EmotionalGraphConfig};
pub use graph::{DefaultEmotionalGraph, EmotionalKernelGraph, EmotionalTickReport};
pub use policy::{ClampPolicy, DecayPolicy, GlobalEmotionPolicy, IdentityPolicy};
pub use query::salience;
pub use state::{EmotionState, EmotionStore};
pub use update::{is_attentive, update_global_emotion, update_kernel_emotion_var};
