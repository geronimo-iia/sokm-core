/// All parameters named after their Hoya equation references.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SokmConfig {
    /// Decay constant: w *= exp(-xi) per tick. [Hoya Eq 4.1]
    pub xi: f64,
    /// Strengthen increment per co-activation. [Hoya Eqs 4.6-4.7]
    pub delta: f64,
    /// Initial weight for newly created links. [Hoya Eq 4.6]
    pub w_init: f64,
    /// Saturation ceiling -- weights never exceed this. [Hoya Eq 4.7]
    pub w_max: f64,
    /// Weight-threshold prune floor -- edges below this are removed.
    pub min_weight: f64,
    /// Propagation attenuation factor in (0,1]. [Hoya Eq 4.3]
    pub gamma: f64,
    /// Inactivity extinction: edges not touched in p1 ticks are pruned.
    pub p1: u64,
}

/// Validation error for `SokmConfig` parameters.
#[derive(Debug, thiserror::Error)]
pub enum SokmConfigError {
    #[error("xi must be > 0, got {0}")]
    InvalidXi(f64),
    #[error("delta must be > 0, got {0}")]
    InvalidDelta(f64),
    #[error("w_init must be > 0, got {0}")]
    InvalidWInit(f64),
    #[error("w_max must be > 0, got {0}")]
    InvalidWMax(f64),
    #[error("w_init ({w_init}) must be <= w_max ({w_max})")]
    WInitExceedsWMax { w_init: f64, w_max: f64 },
    #[error("min_weight must be > 0, got {0}")]
    InvalidMinWeight(f64),
    #[error("gamma must be in (0, 1], got {0}")]
    InvalidGamma(f64),
    #[error("p1 must be > 0")]
    InvalidP1,
}

/// Non-fatal but suspicious configuration.
/// `SokmConfig::warnings()` returns these; callers should log them.
#[derive(Debug, Clone, PartialEq)]
pub enum SokmConfigWarning {
    /// `min_weight >= w_max` — pruning threshold equals or exceeds the maximum edge weight.
    /// All edges will be pruned immediately after formation. Network graph stays empty.
    MinWeightGteWMax { min_weight: f64, w_max: f64 },
}

impl std::fmt::Display for SokmConfigWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SokmConfigWarning::MinWeightGteWMax { min_weight, w_max } => write!(
                f,
                "min_weight ({min_weight}) >= w_max ({w_max}): all edges will be pruned immediately"
            ),
        }
    }
}

impl Default for SokmConfig {
    fn default() -> Self {
        Self {
            xi: 0.01,
            delta: 0.05,
            w_init: 0.1,
            w_max: 1.0,
            min_weight: 0.001,
            gamma: 0.9,
            p1: 100,
        }
    }
}

impl SokmConfig {
    /// Validate all parameters; returns `Err(SokmConfigError)` on any invalid combination.
    pub fn validate(&self) -> Result<(), SokmConfigError> {
        if self.xi <= 0.0 {
            return Err(SokmConfigError::InvalidXi(self.xi));
        }
        if self.delta <= 0.0 {
            return Err(SokmConfigError::InvalidDelta(self.delta));
        }
        if self.w_init <= 0.0 {
            return Err(SokmConfigError::InvalidWInit(self.w_init));
        }
        if self.w_max <= 0.0 {
            return Err(SokmConfigError::InvalidWMax(self.w_max));
        }
        if self.w_init > self.w_max {
            return Err(SokmConfigError::WInitExceedsWMax {
                w_init: self.w_init,
                w_max: self.w_max,
            });
        }
        if self.min_weight <= 0.0 {
            return Err(SokmConfigError::InvalidMinWeight(self.min_weight));
        }
        if self.gamma <= 0.0 || self.gamma > 1.0 {
            return Err(SokmConfigError::InvalidGamma(self.gamma));
        }
        if self.p1 == 0 {
            return Err(SokmConfigError::InvalidP1);
        }
        Ok(())
    }

    /// Return non-fatal but suspicious configuration conditions.
    /// Call after `validate()` succeeds. Log or surface these to the operator.
    pub fn warnings(&self) -> Vec<SokmConfigWarning> {
        let mut w = Vec::new();
        if self.min_weight >= self.w_max {
            w.push(SokmConfigWarning::MinWeightGteWMax {
                min_weight: self.min_weight,
                w_max: self.w_max,
            });
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_are_valid() {
        assert!(SokmConfig::default().validate().is_ok());
    }

    #[test]
    fn zero_xi_is_invalid() {
        let cfg = SokmConfig {
            xi: 0.0,
            ..SokmConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn w_init_above_w_max_is_invalid() {
        let cfg = SokmConfig {
            w_init: 2.0,
            w_max: 1.0,
            ..SokmConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_p1_is_invalid() {
        let cfg = SokmConfig {
            p1: 0,
            ..SokmConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn gamma_zero_is_invalid() {
        let cfg = SokmConfig {
            gamma: 0.0,
            ..SokmConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn w_init_equals_w_max_is_valid() {
        let cfg = SokmConfig {
            w_init: 1.0,
            w_max: 1.0,
            ..SokmConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn min_weight_lt_w_max_no_warning() {
        let cfg = SokmConfig {
            min_weight: 0.01,
            w_max: 1.0,
            ..SokmConfig::default()
        };
        assert!(cfg.validate().is_ok());
        assert!(cfg.warnings().is_empty());
    }

    #[test]
    fn min_weight_eq_w_max_warns() {
        let cfg = SokmConfig {
            min_weight: 1.0,
            w_max: 1.0,
            ..SokmConfig::default()
        };
        assert!(cfg.validate().is_ok(), "not an error, just a warning");
        let ws = cfg.warnings();
        assert_eq!(ws.len(), 1);
        assert!(matches!(
            ws[0],
            SokmConfigWarning::MinWeightGteWMax {
                min_weight: 1.0,
                w_max: 1.0
            }
        ));
    }

    #[test]
    fn min_weight_gt_w_max_warns() {
        let cfg = SokmConfig {
            min_weight: 1.5,
            w_max: 1.0,
            w_init: 0.5,
            ..SokmConfig::default()
        };
        assert!(cfg.validate().is_ok());
        let ws = cfg.warnings();
        assert_eq!(ws.len(), 1);
        assert!(matches!(ws[0], SokmConfigWarning::MinWeightGteWMax { .. }));
    }

    #[test]
    fn warning_display_contains_values() {
        let w = SokmConfigWarning::MinWeightGteWMax {
            min_weight: 0.5,
            w_max: 0.3,
        };
        let s = w.to_string();
        assert!(s.contains("0.5"));
        assert!(s.contains("0.3"));
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn config_serde_roundtrip() {
        let cfg = SokmConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: SokmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }
}
