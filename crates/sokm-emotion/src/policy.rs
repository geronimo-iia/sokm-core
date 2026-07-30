use crate::state::EmotionState;

pub trait GlobalEmotionPolicy {
    /// Pre-multiply factor applied to global state before adding activation sum.
    /// [Hoya Eq. 10.6 base: 1.0 — decay is an INFERRED extension for long-running agents]
    fn decay_factor(&self) -> f64;
    /// Bound the state after update (clamp, passthrough, etc.).
    /// Receives fully-updated post-activation-sum state; returns bounded or decayed result.
    fn apply(&self, state: EmotionState) -> EmotionState;
}

/// Exact Hoya base equation: no bounding, no decay. \[DIRECT\]
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IdentityPolicy;

impl GlobalEmotionPolicy for IdentityPolicy {
    fn decay_factor(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: EmotionState) -> EmotionState {
        state
    }
}

/// Clamp E₁ and E₂ to their defined ranges after each update. \[INFERRED\]
/// Matches Hoya's discrete level structure: E₁ ∈ [-3,3], E₂ ∈ [-2,2].
/// [Hoya pp. 214–215] [DIRECT for ranges; INFERRED for clamping as bounding rule]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClampPolicy {
    pub e1_min: f64,
    pub e1_max: f64,
    pub e2_min: f64,
    pub e2_max: f64,
}

impl Default for ClampPolicy {
    fn default() -> Self {
        Self {
            e1_min: -3.0,
            e1_max: 3.0,
            e2_min: -2.0,
            e2_max: 2.0,
        }
    }
}

impl GlobalEmotionPolicy for ClampPolicy {
    fn decay_factor(&self) -> f64 {
        1.0
    }

    fn apply(&self, state: EmotionState) -> EmotionState {
        EmotionState {
            e1: state.e1.clamp(self.e1_min, self.e1_max),
            e2: state.e2.clamp(self.e2_min, self.e2_max),
        }
    }
}

/// Decay E₁ and E₂ toward zero each tick, then clamp. \[INFERRED\]
/// Prevents runaway accumulation for long-running agents.
/// `decay` valid range `(0, 1]`. `0.0` resets global to activation sum only (prior state wiped).
/// `> 1.0` amplifies history (usually unintended).
/// decay ∈ (0, 1]: state *= decay before adding activation sum.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecayPolicy {
    pub decay: f64,
    pub clamp: ClampPolicy,
}

impl Default for DecayPolicy {
    fn default() -> Self {
        Self {
            decay: 0.99,
            clamp: ClampPolicy::default(),
        }
    }
}

impl GlobalEmotionPolicy for DecayPolicy {
    fn decay_factor(&self) -> f64 {
        self.decay
    }

    fn apply(&self, state: EmotionState) -> EmotionState {
        self.clamp.apply(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_policy_decay_factor_is_one() {
        assert_eq!(IdentityPolicy.decay_factor(), 1.0);
    }

    #[test]
    fn identity_policy_apply_is_passthrough() {
        let state = EmotionState { e1: 1.5, e2: -0.7 };
        assert_eq!(IdentityPolicy.apply(state), state);
    }

    #[test]
    fn clamp_policy_clamps_e1_above_max() {
        let s = ClampPolicy::default().apply(EmotionState { e1: 5.0, e2: 0.0 });
        assert_eq!(s.e1, 3.0);
    }

    #[test]
    fn clamp_policy_clamps_e1_below_min() {
        let s = ClampPolicy::default().apply(EmotionState { e1: -5.0, e2: 0.0 });
        assert_eq!(s.e1, -3.0);
    }

    #[test]
    fn clamp_policy_clamps_e2_above_max() {
        let s = ClampPolicy::default().apply(EmotionState { e1: 0.0, e2: 3.0 });
        assert_eq!(s.e2, 2.0);
    }

    #[test]
    fn clamp_policy_no_change_within_range() {
        let state = EmotionState { e1: 1.0, e2: -1.0 };
        assert_eq!(ClampPolicy::default().apply(state), state);
    }

    #[test]
    fn clamp_policy_decay_factor_is_one() {
        assert_eq!(ClampPolicy::default().decay_factor(), 1.0);
    }

    #[test]
    fn decay_policy_decay_factor_matches_config() {
        let p = DecayPolicy {
            decay: 0.95,
            clamp: ClampPolicy::default(),
        };
        assert_eq!(p.decay_factor(), 0.95);
    }

    #[test]
    fn decay_policy_still_clamps() {
        let s = DecayPolicy::default().apply(EmotionState { e1: 5.0, e2: 0.0 });
        assert_eq!(s.e1, 3.0);
    }

    // --- new tests ---

    #[test]
    #[should_panic]
    fn clamp_policy_inverted_range_panics() {
        // f64::clamp panics in debug when min > max
        let p = ClampPolicy {
            e1_min: 2.0,
            e1_max: 1.0,
            e2_min: -2.0,
            e2_max: 2.0,
        };
        let _ = p.apply(EmotionState { e1: 1.5, e2: 0.0 });
    }

    #[test]
    fn clamp_policy_e1_at_exact_max_unchanged() {
        let s = ClampPolicy::default().apply(EmotionState { e1: 3.0, e2: 0.0 });
        assert_eq!(s.e1, 3.0);
    }

    #[test]
    fn clamp_policy_e2_below_min() {
        let s = ClampPolicy::default().apply(EmotionState { e1: 0.0, e2: -5.0 });
        assert_eq!(s.e2, -2.0);
    }

    #[test]
    fn decay_policy_decay_zero_global_resets() {
        let p = DecayPolicy {
            decay: 0.0,
            clamp: ClampPolicy::default(),
        };
        // decay factor 0.0 — prior state wiped; apply receives only activation sum
        assert_eq!(p.decay_factor(), 0.0);
        // with no activations, global becomes 0
        use crate::update::update_global_emotion;
        let prior = EmotionState { e1: 2.0, e2: -1.0 };
        let new_state = update_global_emotion(prior, &[], p.decay_factor());
        let after = p.apply(new_state);
        assert_eq!(after.e1, 0.0);
        assert_eq!(after.e2, 0.0);
    }

    #[test]
    fn decay_policy_compound_then_clamp() {
        // After activation sum drives e1 out of range, clamp fires
        let p = DecayPolicy {
            decay: 1.0,
            clamp: ClampPolicy::default(),
        };
        let s = p.apply(EmotionState { e1: 10.0, e2: 0.0 });
        assert_eq!(s.e1, 3.0);
    }
}
