/// Hoya kernel unit KF = {c_i, σ_i, η_i, ε_i}. [Hoya §3]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelUnit {
    /// Centroid c_i ∈ ℝ^d.
    pub centroid: Vec<f64>,
    /// Bandwidth σ_i > 0.
    pub sigma: f64,
    /// Class label η_i. None = unlabelled (η = ∅). [Hoya §4.3] [DIRECT]
    pub class: Option<u32>,
    /// Excitation count ε_i — incremented each time this kernel activates.
    pub excitation: u64,
    /// Logical tick of last activation. 0 = never activated.
    pub last_activated: u64,
    /// True if marked extinct by Rule 3.
    pub extinct: bool,
}

impl KernelUnit {
    /// Create a new kernel unit. Initial values: `excitation=0`, `last_activated=0`, `extinct=false`.
    pub fn new(centroid: Vec<f64>, sigma: f64, class: Option<u32>) -> Self {
        Self {
            centroid,
            sigma,
            class,
            excitation: 0,
            last_activated: 0,
            extinct: false,
        }
    }

    pub fn dim(&self) -> usize {
        self.centroid.len()
    }

    pub fn increment_excitation(&mut self) {
        self.excitation = self.excitation.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_unit_new_sets_fields() {
        let k = KernelUnit::new(vec![1.0, 2.0, 3.0], 0.5, Some(1));
        assert_eq!(k.centroid, vec![1.0, 2.0, 3.0]);
        assert_eq!(k.sigma, 0.5);
        assert_eq!(k.class, Some(1));
        assert_eq!(k.excitation, 0);
    }

    #[test]
    fn kernel_unit_new_with_none_class_is_unlabelled() {
        let k = KernelUnit::new(vec![1.0], 1.0, None);
        assert!(k.class.is_none());
    }

    #[test]
    fn kernel_unit_increment_excitation() {
        let mut k = KernelUnit::new(vec![0.0], 1.0, Some(0));
        k.increment_excitation();
        assert_eq!(k.excitation, 1);
    }

    #[test]
    fn kernel_unit_dim() {
        let k = KernelUnit::new(vec![1.0, 2.0, 3.0, 4.0], 1.0, Some(0));
        assert_eq!(k.dim(), 4);
    }

    #[test]
    fn increment_excitation_at_max_no_panic() {
        let mut k = KernelUnit::new(vec![0.0], 1.0, Some(0));
        k.excitation = u64::MAX;
        k.increment_excitation(); // saturating_add — must not panic
        assert_eq!(k.excitation, u64::MAX);
    }

    #[test]
    fn new_sets_initial_values() {
        let k = KernelUnit::new(vec![1.0], 1.0, None);
        assert_eq!(k.excitation, 0);
        assert_eq!(k.last_activated, 0);
        assert!(!k.extinct);
    }
}
