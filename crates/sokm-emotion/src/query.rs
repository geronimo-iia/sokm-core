use crate::state::EmotionState;

/// Salience score for kernel i given current global emotion state. \[INFERRED\]
/// Post-tick recall weight modifier. `tick` does NOT apply this automatically.
/// Call manually with `EmotionalTickReport::global` and `emotion_vars(i)`.
/// Returns multiplier >= 0: 1.0 = neutral, > 1.0 = salient, < 1.0 = suppressed.
/// Clamped to 0.0 when `1 + alpha * alignment` goes negative (large alpha, strong misalignment).
pub fn salience(kernel_vars: [f64; 2], global: &EmotionState, alpha: f64) -> f64 {
    let alignment = kernel_vars[0] * global.e1 + kernel_vars[1] * global.e2;
    (1.0 + alpha * alignment).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salience_neutral_when_alpha_zero() {
        let global = EmotionState { e1: 1.0, e2: -1.0 };
        let s = salience([0.5, 0.5], &global, 0.0);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn salience_increases_with_alignment() {
        let global = EmotionState { e1: 1.0, e2: 0.0 };
        let s = salience([1.0, 0.0], &global, 1.0);
        assert!(s > 1.0);
    }

    #[test]
    fn salience_never_negative() {
        let global = EmotionState { e1: 1.0, e2: 0.0 };
        let s = salience([-100.0, 0.0], &global, 1.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn salience_zero_global_always_one() {
        let global = EmotionState { e1: 0.0, e2: 0.0 };
        // alignment is 0 regardless of vars or alpha
        let s = salience([10.0, -5.0], &global, 100.0);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn salience_zero_alignment_returns_one() {
        // orthogonal: kernel [1,0], global [0,1] → alignment = 0
        let global = EmotionState { e1: 0.0, e2: 1.0 };
        let s = salience([1.0, 0.0], &global, 5.0);
        assert_eq!(s, 1.0);
    }
}
