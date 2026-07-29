#[cfg(feature = "serde")]
fn default_label_inherit_threshold() -> u32 {
    u32::MAX
}

/// Kernel layer configuration. Param names map to Hoya equation references.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelConfig {
    /// Growth threshold ∈ (0,1): add new kernel when max K_i(x) < theta_k. [Hoya Eq 3.8 gate]
    pub theta_k: f64,
    /// Initial bandwidth for new kernels. [try 1.0 for unit morphological space]
    pub sigma_0: f64,
    /// STM blend weight ∈ \[0,1\]: o_STM = lambda*c_k + (1-lambda)*x. [Hoya Eq 10.5]
    pub lambda: f64,
    /// Compact kernel cutoff ratio. Hoya's only concrete constant = 2.67. [Hoya Eq 3.10]
    pub q: f64,
    /// STM capacity N_{s,max}: max kernels in working memory. [Hoya p.164]
    pub stm_capacity: usize,
    /// Inactivity extinction period for kernels, in ticks. [Hoya pp. 80–99, Rule 3]
    /// Kernel inactive for > p1_kernel ticks since last_activated is marked extinct.
    /// Default: u64::MAX — extinction disabled unless caller opts in.
    /// See `KernelTickReport::newly_extinct` for the per-tick count.
    pub p1_kernel: u64,
    /// Co-activation count for label inheritance. [Hoya §4.3]
    /// u32::MAX = disabled (default).
    #[cfg_attr(feature = "serde", serde(default = "default_label_inherit_threshold"))]
    pub label_inherit_threshold: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            theta_k: 0.1,
            sigma_0: 1.0,
            lambda: 0.7,
            q: 2.67,
            stm_capacity: 16,
            p1_kernel: u64::MAX,
            label_inherit_threshold: u32::MAX,
        }
    }
}

/// Validation error returned by [`KernelConfig::validate`].
#[derive(Debug, thiserror::Error)]
pub enum KernelConfigError {
    #[error("theta_k must be in (0, 1), got {0}")]
    InvalidThetaK(f64),
    #[error("sigma_0 must be > 0, got {0}")]
    InvalidSigma0(f64),
    #[error("lambda must be in [0, 1], got {0}")]
    InvalidLambda(f64),
    #[error("q must be > 0, got {0}")]
    InvalidQ(f64),
    #[error("stm_capacity must be > 0")]
    InvalidStmCapacity,
}

impl KernelConfig {
    /// Validate all config fields. Returns `Err` on the first violated constraint.
    pub fn validate(&self) -> Result<(), KernelConfigError> {
        if self.theta_k <= 0.0 || self.theta_k >= 1.0 {
            return Err(KernelConfigError::InvalidThetaK(self.theta_k));
        }
        if self.sigma_0 <= 0.0 {
            return Err(KernelConfigError::InvalidSigma0(self.sigma_0));
        }
        if !(0.0..=1.0).contains(&self.lambda) {
            return Err(KernelConfigError::InvalidLambda(self.lambda));
        }
        if self.q <= 0.0 {
            return Err(KernelConfigError::InvalidQ(self.q));
        }
        if self.stm_capacity == 0 {
            return Err(KernelConfigError::InvalidStmCapacity);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_are_valid() {
        assert!(KernelConfig::default().validate().is_ok());
    }

    #[test]
    fn zero_theta_k_is_invalid() {
        let cfg = KernelConfig {
            theta_k: 0.0,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn theta_k_above_one_is_invalid() {
        let cfg = KernelConfig {
            theta_k: 1.1,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn theta_k_one_is_invalid() {
        // boundary: (0,1) is open — exactly 1.0 must be rejected
        let cfg = KernelConfig {
            theta_k: 1.0,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_sigma_0_is_invalid() {
        let cfg = KernelConfig {
            sigma_0: 0.0,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn lambda_out_of_range_is_invalid() {
        let cfg = KernelConfig {
            lambda: 1.5,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_stm_capacity_is_invalid() {
        let cfg = KernelConfig {
            stm_capacity: 0,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn q_must_be_positive() {
        let cfg = KernelConfig {
            q: 0.0,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn lambda_zero_is_valid() {
        let cfg = KernelConfig {
            lambda: 0.0,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn lambda_one_is_valid() {
        let cfg = KernelConfig {
            lambda: 1.0,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn negative_sigma_0_is_invalid() {
        let cfg = KernelConfig {
            sigma_0: -1.0,
            ..KernelConfig::default()
        };
        assert!(cfg.validate().is_err());
    }
}
