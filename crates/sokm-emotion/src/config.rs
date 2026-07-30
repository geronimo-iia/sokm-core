use crate::state::EmotionState;

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmotionConfig {
    /// Blend rate for per-kernel emotion variable update. [Hoya Eq. 10.8] \[DIRECT\]
    /// e_i^j(n+1) = e_i^j(n) + lambda_e × (e_target − e_i^j(n))
    pub lambda_e: f64,

    /// Attentive condition threshold. [Hoya Eq. 10.7] \[DIRECT\]
    /// Must be > 0.0.
    pub theta_e: f64,

    /// Optimal (target) emotion state E_i*. [Hoya Eq. 10.7] \[DIRECT\]
    pub optimal: EmotionState,

    /// Salience scaling factor α for recall weighting. \[INFERRED\]
    /// Set `alpha > 0.0` to enable emotion-weighted recall; `0.0` (default) disables salience entirely.
    pub alpha: f64,
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            lambda_e: 0.1,
            theta_e: 1.0,
            optimal: EmotionState::default(),
            alpha: 0.0,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmotionConfigError {
    #[error("lambda_e must be in [0, 1], got {0}")]
    InvalidLambdaE(f64),
    #[error("theta_e must be > 0, got {0}")]
    InvalidThetaE(f64),
    #[error("alpha must be >= 0, got {0}")]
    InvalidAlpha(f64),
}

impl EmotionConfig {
    pub fn validate(&self) -> Result<(), EmotionConfigError> {
        if !(0.0..=1.0).contains(&self.lambda_e) {
            return Err(EmotionConfigError::InvalidLambdaE(self.lambda_e));
        }
        if self.theta_e <= 0.0 {
            return Err(EmotionConfigError::InvalidThetaE(self.theta_e));
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(EmotionConfigError::InvalidAlpha(self.alpha));
        }
        Ok(())
    }
}

/// Bundled config for EmotionalKernelGraph::tick.
/// Combines the three configs previously passed as separate arguments.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmotionalGraphConfig {
    pub sokm: sokm::SokmConfig,
    pub kernel: sokm_kernel::KernelConfig,
    pub emotion: EmotionConfig,
}

impl EmotionalGraphConfig {
    pub fn validate(&self) -> Result<(), EmotionConfigError> {
        self.emotion.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_lambda_e_zero_valid() {
        let c = EmotionConfig {
            lambda_e: 0.0,
            ..EmotionConfig::default()
        };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_lambda_e_one_valid() {
        let c = EmotionConfig {
            lambda_e: 1.0,
            ..EmotionConfig::default()
        };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_lambda_e_above_one_invalid() {
        let c = EmotionConfig {
            lambda_e: 1.1,
            ..EmotionConfig::default()
        };
        assert!(matches!(
            c.validate(),
            Err(EmotionConfigError::InvalidLambdaE(_))
        ));
    }

    #[test]
    fn validate_lambda_e_negative_invalid() {
        let c = EmotionConfig {
            lambda_e: -0.1,
            ..EmotionConfig::default()
        };
        assert!(matches!(
            c.validate(),
            Err(EmotionConfigError::InvalidLambdaE(_))
        ));
    }

    #[test]
    fn validate_theta_e_zero_invalid() {
        let c = EmotionConfig {
            theta_e: 0.0,
            ..EmotionConfig::default()
        };
        assert!(matches!(
            c.validate(),
            Err(EmotionConfigError::InvalidThetaE(_))
        ));
    }

    #[test]
    fn validate_theta_e_negative_invalid() {
        let c = EmotionConfig {
            theta_e: -1.0,
            ..EmotionConfig::default()
        };
        assert!(matches!(
            c.validate(),
            Err(EmotionConfigError::InvalidThetaE(_))
        ));
    }

    #[test]
    fn validate_alpha_zero_valid() {
        let c = EmotionConfig {
            alpha: 0.0,
            ..EmotionConfig::default()
        };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_alpha_negative_invalid() {
        let c = EmotionConfig {
            alpha: -0.1,
            ..EmotionConfig::default()
        };
        assert!(matches!(
            c.validate(),
            Err(EmotionConfigError::InvalidAlpha(_))
        ));
    }

    #[test]
    fn validate_alpha_nan_invalid() {
        let c = EmotionConfig {
            alpha: f64::NAN,
            ..EmotionConfig::default()
        };
        assert!(matches!(
            c.validate(),
            Err(EmotionConfigError::InvalidAlpha(_))
        ));
    }

    #[test]
    fn validate_alpha_inf_invalid() {
        let c = EmotionConfig {
            alpha: f64::INFINITY,
            ..EmotionConfig::default()
        };
        assert!(matches!(
            c.validate(),
            Err(EmotionConfigError::InvalidAlpha(_))
        ));
    }
}
