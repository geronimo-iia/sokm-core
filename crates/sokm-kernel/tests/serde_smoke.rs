#[cfg(feature = "serde")]
#[test]
fn kernel_config_serde_roundtrip() {
    use sokm_kernel::KernelConfig;
    let cfg = KernelConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let back: KernelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg, back);
}

#[cfg(feature = "serde")]
#[test]
fn kernel_unit_serde_roundtrip() {
    use sokm_kernel::KernelUnit;
    let unit = KernelUnit::new(vec![1.0, 2.0, 3.0], 0.5, Some(42));
    let json = serde_json::to_string(&unit).unwrap();
    let back: KernelUnit = serde_json::from_str(&json).unwrap();
    assert_eq!(unit, back);
}
