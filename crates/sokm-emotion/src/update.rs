use crate::state::EmotionState;

/// Update one per-kernel emotion variable toward target. [Hoya Eq. 10.8] \[DIRECT\]
/// e_i^j(n+1) = e_i^j(n) + lambda_e × (e_target - e_i^j(n))
pub fn update_kernel_emotion_var(current: f64, target: f64, lambda_e: f64) -> f64 {
    current + lambda_e * (target - current)
}

/// Update global emotion state from kernel activations. [Hoya Eq. 10.6]
/// Base formula \[DIRECT\]: E_i(n+1) = E_i(n) + Σ_j e_i^j · K_j(x)
/// With decay \[INFERRED\]: E_i(n+1) = decay × E_i(n) + Σ_j e_i^j · K_j(x)
///
/// `activations`: (K_j(x), [e_1^j, e_2^j]) for all kernels with score > 0.
pub fn update_global_emotion(
    current: EmotionState,
    activations: &[(f64, [f64; 2])],
    decay: f64,
) -> EmotionState {
    let e1 = current.e1 * decay + activations.iter().map(|(k, e)| k * e[0]).sum::<f64>();
    let e2 = current.e2 * decay + activations.iter().map(|(k, e)| k * e[1]).sum::<f64>();
    EmotionState { e1, e2 }
}

/// Check attentive condition. [Hoya Eq. 10.7] \[DIRECT\]
/// Returns true when Σ_i |E_i − E_i*| ≤ θ_E
pub fn is_attentive(current: &EmotionState, optimal: &EmotionState, theta_e: f64) -> bool {
    (current.e1 - optimal.e1).abs() + (current.e2 - optimal.e2).abs() <= theta_e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_emotion_var_update_moves_toward_target() {
        let result = update_kernel_emotion_var(0.0, 1.0, 0.5);
        assert_eq!(result, 0.5);
    }

    #[test]
    fn kernel_emotion_var_update_converges_to_target() {
        let mut v = 0.0;
        for _ in 0..100 {
            v = update_kernel_emotion_var(v, 1.0, 0.1);
        }
        assert!((v - 1.0).abs() < 1e-4);
    }

    #[test]
    fn kernel_emotion_var_update_no_change_at_target() {
        let result = update_kernel_emotion_var(1.0, 1.0, 0.1);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn global_emotion_update_weighted_by_activation() {
        let current = EmotionState::default();
        let a1 = vec![(1.0f64, [1.0f64, 0.0f64])];
        let a2 = vec![(0.5f64, [1.0f64, 0.0f64])];
        let s1 = update_global_emotion(current, &a1, 1.0);
        let s2 = update_global_emotion(current, &a2, 1.0);
        assert!(s1.e1 > s2.e1);
    }

    #[test]
    fn global_emotion_update_empty_activations_returns_decayed() {
        let current = EmotionState { e1: 1.0, e2: 0.5 };
        let s = update_global_emotion(current, &[], 0.9);
        assert!((s.e1 - 0.9).abs() < 1e-12);
        assert!((s.e2 - 0.45).abs() < 1e-12);
    }

    #[test]
    fn global_emotion_update_decay_one_no_change_without_activations() {
        let current = EmotionState { e1: 1.0, e2: -0.5 };
        let s = update_global_emotion(current, &[], 1.0);
        assert_eq!(s, current);
    }

    #[test]
    fn is_attentive_true_when_within_theta_e() {
        let current = EmotionState { e1: 0.3, e2: 0.3 };
        let optimal = EmotionState::default();
        assert!(is_attentive(&current, &optimal, 1.0));
    }

    #[test]
    fn is_attentive_false_when_outside_theta_e() {
        let current = EmotionState { e1: 2.0, e2: 0.0 };
        let optimal = EmotionState::default();
        assert!(!is_attentive(&current, &optimal, 1.0));
    }

    #[test]
    fn is_attentive_true_at_optimal() {
        let state = EmotionState { e1: 1.0, e2: -1.0 };
        assert!(is_attentive(&state, &state, 0.001));
    }

    // --- new tests ---

    #[test]
    fn update_kernel_emotion_var_lambda_zero_frozen() {
        let result = update_kernel_emotion_var(0.3, 1.0, 0.0);
        assert_eq!(result, 0.3);
    }

    #[test]
    fn update_kernel_emotion_var_lambda_one_instant() {
        let result = update_kernel_emotion_var(0.0, 1.0, 1.0);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn update_kernel_emotion_var_lambda_overshoot() {
        // lambda > 1.0 overshoots past target
        let result = update_kernel_emotion_var(0.0, 1.0, 2.0);
        assert!(
            result > 1.0,
            "lambda=2 should overshoot to 2.0, got {result}"
        );
    }

    #[test]
    fn update_global_emotion_decay_zero_wipes_prior() {
        let current = EmotionState { e1: 5.0, e2: -3.0 };
        let s = update_global_emotion(current, &[], 0.0);
        assert_eq!(s.e1, 0.0);
        assert_eq!(s.e2, 0.0);
    }

    #[test]
    fn is_attentive_at_exact_boundary() {
        // sum == theta_e → true
        let current = EmotionState { e1: 0.6, e2: 0.4 };
        let optimal = EmotionState::default();
        assert!(is_attentive(&current, &optimal, 1.0));
    }

    #[test]
    fn is_attentive_theta_e_zero_only_at_optimal() {
        let state = EmotionState { e1: 1.0, e2: -1.0 };
        assert!(is_attentive(&state, &state, 0.0));
        let other = EmotionState {
            e1: 1.001,
            e2: -1.0,
        };
        assert!(!is_attentive(&other, &state, 0.0));
    }
}
