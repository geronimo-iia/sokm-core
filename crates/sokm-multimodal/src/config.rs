/// Combined configuration for GestaltKernelGraph — bundles all three config layers.
/// Both modalities share the same SokmConfig and KernelConfig, matching Hoya's symmetric treatment.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GestaltConfig {
    pub sokm: sokm::SokmConfig,
    pub kernel: sokm_kernel::KernelConfig,
    pub cross: CrossSokmConfig,
}

impl GestaltConfig {
    /// Validate all sub-configs. Returns first error encountered.
    pub fn validate(&self) -> Result<(), CrossConfigError> {
        self.cross.validate()
    }
}

/// Hebbian parameters for cross-modal edges. Independent of intra-modal SokmConfig.
/// \[Hoya pp. 60–79; same three-rule Hebbian update as intra-modal\] \[INFERRED\]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CrossSokmConfig {
    /// Propagation attenuation factor γ ∈ (0,1]. [Hoya Eq. 4.3]
    pub gamma: f64,
    /// Strengthening increment δ per co-activation. [Hoya Eq. 4.7]
    pub delta: f64,
    /// Initial weight on first strengthen of a new cross-modal edge.
    pub w_init: f64,
    /// Maximum edge weight. [Hoya Eq. 4.7]
    pub w_max: f64,
    /// Minimum edge weight; edges below this are pruned. [Hoya Rule 2]
    pub min_weight: f64,
    /// Decay factor ξ per tick. [Hoya Eq. 4.1]
    pub xi: f64,
    /// Inactivity extinction period for cross-modal edges, in ticks. [Hoya Rule 3]
    /// Default: u64::MAX — disabled.
    pub p1: u64,
    /// Require matching class labels for cross-modal strengthening.
    /// Default: true — co-occurrence alone does not strengthen.
    /// When false: any two labelled kernels strengthen regardless of class.
    /// None (unlabelled) kernels never strengthen regardless of this flag.
    pub require_class_match: bool,
}

impl CrossSokmConfig {
    /// Validate configuration invariants.
    ///
    /// Rules:
    /// - `gamma > 0.0` (gamma=0.0 disables propagation entirely — invalid)
    /// - `xi >= 0.0` (negative decay is invalid)
    /// - `w_init <= w_max` (initial weight must not exceed maximum)
    /// - `delta >= 0.0` (negative strengthening is invalid)
    pub fn validate(&self) -> Result<(), CrossConfigError> {
        if self.gamma <= 0.0 {
            return Err(CrossConfigError::InvalidGamma(self.gamma));
        }
        if self.xi < 0.0 {
            return Err(CrossConfigError::InvalidXi(self.xi));
        }
        if self.w_init > self.w_max {
            return Err(CrossConfigError::WInitExceedsWMax {
                w_init: self.w_init,
                w_max: self.w_max,
            });
        }
        if self.delta < 0.0 {
            return Err(CrossConfigError::InvalidDelta(self.delta));
        }
        Ok(())
    }
}

/// Validation errors for [`CrossSokmConfig`].
#[derive(Debug, thiserror::Error)]
pub enum CrossConfigError {
    #[error("gamma must be > 0.0, got {0}")]
    InvalidGamma(f64),
    #[error("xi must be >= 0.0, got {0}")]
    InvalidXi(f64),
    #[error("w_init ({w_init}) must be <= w_max ({w_max})")]
    WInitExceedsWMax { w_init: f64, w_max: f64 },
    #[error("delta must be >= 0.0, got {0}")]
    InvalidDelta(f64),
}

impl Default for CrossSokmConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            delta: 0.05,
            w_init: 0.1,
            w_max: 1.0,
            min_weight: 0.001,
            xi: 0.01,
            p1: u64::MAX,
            require_class_match: true,
        }
    }
}
