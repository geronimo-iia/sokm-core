/// Gaussian kernel similarity ∈ (0,1]. [Hoya Eq 3.8]
///
/// K_i(x) = exp(−‖x−c_i‖²/σ_i²)
///
/// Returns 1.0 when x == centroid. Approaches 0.0 as distance grows.
/// `sigma` must be > 0; `sigma=0` produces NaN (division by zero in exponent).
pub fn gaussian(x: &[f64], centroid: &[f64], sigma: f64) -> f64 {
    debug_assert_eq!(x.len(), centroid.len(), "dimension mismatch");
    debug_assert!(sigma > 0.0, "sigma must be > 0");
    let sq_dist: f64 = x.iter().zip(centroid).map(|(a, b)| (a - b) * (a - b)).sum();
    (-sq_dist / (sigma * sigma)).exp()
}

/// Compact polynomial approximation of Gaussian kernel. [Hoya Eq 3.10]
///
/// K_i(x) = (1 − ‖x−c_i‖²/(q·σ²))²  if ‖x−c_i‖² < q·σ²
///         = 0.0                        otherwise
///
/// No `exp()` call — computationally cheaper than `gaussian`.
/// q = 2.67 is Hoya's only concrete numeric constant.
/// Returns 1.0 when x == centroid. Returns 0.0 outside the kernel's support radius.
/// `q` and `sigma` must both be > 0; either being 0 causes division by zero.
pub fn compact(x: &[f64], centroid: &[f64], sigma: f64, q: f64) -> f64 {
    debug_assert_eq!(x.len(), centroid.len(), "dimension mismatch");
    debug_assert!(sigma > 0.0, "sigma must be > 0");
    debug_assert!(q > 0.0, "q must be > 0");
    let sq_dist: f64 = x.iter().zip(centroid).map(|(a, b)| (a - b) * (a - b)).sum();
    let denom = q * sigma * sigma;
    if sq_dist >= denom {
        0.0
    } else {
        (1.0 - sq_dist / denom).powi(2)
    }
}

/// Batch gaussian scores for n kernels via f64x4 SIMD (4 kernels in parallel).
///
/// `centroids`: flat row-major [c0_d0..c0_d(D-1), c1_d0..c1_d(D-1), ...]
/// `sigmas`: per-kernel σ, length = n
/// `x`: query vector, length = D
///
/// Returns `Vec<f64>` of length n. Extinct mask NOT applied — caller zeroes extinct indices.
#[cfg(feature = "simd")]
pub fn batch_gaussian_simd(centroids: &[f64], sigmas: &[f64], x: &[f64]) -> Vec<f64> {
    use wide::f64x4;
    let d = x.len();
    let n = sigmas.len();
    debug_assert_eq!(centroids.len(), n * d);

    let mut scores = vec![0.0f64; n];
    let mut k = 0usize;

    // One lane per kernel — dimension loop accumulates sq_dist per lane independently.
    // No horizontal reduction needed; each lane stays separate throughout.
    while k + 4 <= n {
        let mut sq = f64x4::ZERO;
        for j in 0..d {
            let xj = f64x4::splat(x[j]);
            let cv = f64x4::from([
                centroids[(k) * d + j],
                centroids[(k + 1) * d + j],
                centroids[(k + 2) * d + j],
                centroids[(k + 3) * d + j],
            ]);
            let diff = xj - cv;
            sq += diff * diff;
        }
        let sigma_v = f64x4::from([sigmas[k], sigmas[k + 1], sigmas[k + 2], sigmas[k + 3]]);
        let sigma2 = sigma_v * sigma_v;
        let exp_v = (-(sq / sigma2)).exp();
        scores[k..k + 4].copy_from_slice(&exp_v.to_array());
        k += 4;
    }
    // Scalar tail: remaining kernels when n % 4 != 0
    while k < n {
        let c = &centroids[k * d..(k + 1) * d];
        let sq: f64 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
        scores[k] = (-sq / (sigmas[k] * sigmas[k])).exp();
        k += 1;
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_returns_one_at_centroid() {
        let x = vec![1.0, 2.0, 3.0];
        let c = vec![1.0, 2.0, 3.0];
        assert!((gaussian(&x, &c, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gaussian_decays_with_distance() {
        let x = vec![0.0];
        let c = vec![1.0];
        let v = gaussian(&x, &c, 1.0);
        // exp(-1) ≈ 0.3679
        assert!((v - 1.0_f64.exp().recip()).abs() < 1e-10);
    }

    #[test]
    fn gaussian_farther_means_lower_activation() {
        let c = vec![0.0];
        let near = gaussian(&[0.5], &c, 1.0);
        let far = gaussian(&[2.0], &c, 1.0);
        assert!(near > far);
    }

    #[test]
    fn gaussian_sigma_scales_response() {
        let x = vec![1.0];
        let c = vec![0.0];
        let narrow = gaussian(&x, &c, 0.5);
        let wide = gaussian(&x, &c, 2.0);
        assert!(wide > narrow); // wider sigma → higher activation at same distance
    }

    #[test]
    fn compact_returns_one_at_centroid() {
        let x = vec![1.0, 2.0];
        let c = vec![1.0, 2.0];
        assert!((compact(&x, &c, 1.0, 2.67) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn compact_zero_outside_support() {
        // dist² = 100, q*sigma² = 2.67 → outside support
        let x = vec![10.0];
        let c = vec![0.0];
        assert_eq!(compact(&x, &c, 1.0, 2.67), 0.0);
    }

    #[test]
    fn compact_positive_inside_support() {
        let x = vec![0.5];
        let c = vec![0.0];
        let v = compact(&x, &c, 1.0, 2.67);
        assert!(v > 0.0 && v <= 1.0);
    }

    #[test]
    fn compact_matches_hoya_eq_3_10() {
        // K(x) = (1 - ||x-c||² / (q*sigma²))²
        let x = vec![0.5];
        let c = vec![0.0];
        let sigma = 1.0;
        let q = 2.67;
        let sq_dist = 0.25_f64;
        let denom = q * sigma * sigma;
        let expected = (1.0 - sq_dist / denom).powi(2);
        assert!((compact(&x, &c, sigma, q) - expected).abs() < 1e-10);
    }

    #[test]
    fn compact_at_exact_boundary_returns_zero() {
        // sq_dist == q * sigma^2 → at boundary → returns 0.0
        let sigma: f64 = 2.0;
        let q: f64 = 2.67;
        // need distance = sqrt(q * sigma^2) = sqrt(2.67 * 4) = sqrt(10.68)
        let dist = (q * sigma * sigma).sqrt();
        let x = vec![dist];
        let c = vec![0.0];
        assert_eq!(compact(&x, &c, sigma, q), 0.0);
    }

    #[test]
    fn gaussian_and_compact_agree_at_centroid() {
        // Both must return 1.0 at centroid for any sigma > 0
        let sigma = 3.5;
        let x = vec![0.0, 0.0];
        let c = vec![0.0, 0.0];
        let g = gaussian(&x, &c, sigma);
        let k = compact(&x, &c, sigma, 2.67);
        assert!((g - 1.0).abs() < 1e-10);
        assert!((k - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "simd")]
    mod simd_tests {
        use super::*;

        fn make_random_centroids(n: usize, d: usize) -> Vec<f64> {
            (0..n * d).map(|i| (i as f64 * 0.003) % 3.0 - 1.5).collect()
        }

        fn scalar_scores(centroids: &[f64], sigmas: &[f64], x: &[f64]) -> Vec<f64> {
            let d = x.len();
            (0..sigmas.len())
                .map(|k| {
                    let c = &centroids[k * d..(k + 1) * d];
                    let sq: f64 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
                    (-sq / (sigmas[k] * sigmas[k])).exp()
                })
                .collect()
        }

        #[test]
        fn simd_scores_match_scalar_16d() {
            let (n, d) = (20, 16);
            let centroids = make_random_centroids(n, d);
            let sigmas = vec![1.0f64; n];
            let x: Vec<f64> = (0..d).map(|i| i as f64 * 0.05).collect();
            let simd = batch_gaussian_simd(&centroids, &sigmas, &x);
            let scalar = scalar_scores(&centroids, &sigmas, &x);
            for i in 0..n {
                assert!(
                    (simd[i] - scalar[i]).abs() < 1e-10,
                    "mismatch at kernel {i}: simd={} scalar={}",
                    simd[i],
                    scalar[i]
                );
            }
        }

        #[test]
        fn simd_scores_match_scalar_358d() {
            let (n, d) = (20, 358);
            let centroids = make_random_centroids(n, d);
            let sigmas = vec![1.0f64; n];
            let x: Vec<f64> = (0..d).map(|i| i as f64 * 0.001).collect();
            let simd = batch_gaussian_simd(&centroids, &sigmas, &x);
            let scalar = scalar_scores(&centroids, &sigmas, &x);
            for i in 0..n {
                assert!(
                    (simd[i] - scalar[i]).abs() < 1e-10,
                    "mismatch at kernel {i}: simd={} scalar={}",
                    simd[i],
                    scalar[i]
                );
            }
        }

        #[test]
        fn simd_scores_at_centroid_return_one() {
            let d = 8usize;
            let centroids: Vec<f64> = (0..4 * d).map(|i| i as f64 * 0.1).collect();
            let sigmas = vec![1.0f64; 4];
            // query = centroid of kernel 2
            let x = centroids[2 * d..3 * d].to_vec();
            let scores = batch_gaussian_simd(&centroids, &sigmas, &x);
            assert!((scores[2] - 1.0).abs() < 1e-10);
        }

        #[test]
        fn simd_tail_matches_scalar() {
            // n=5 → 4 SIMD + 1 scalar tail
            let (n, d) = (5, 16);
            let centroids = make_random_centroids(n, d);
            let sigmas = vec![1.0f64; n];
            let x: Vec<f64> = (0..d).map(|i| i as f64 * 0.05).collect();
            let simd = batch_gaussian_simd(&centroids, &sigmas, &x);
            let scalar = scalar_scores(&centroids, &sigmas, &x);
            assert!((simd[4] - scalar[4]).abs() < 1e-10);
        }

        #[test]
        fn simd_n_zero_empty() {
            let x = vec![1.0, 2.0];
            let scores = batch_gaussian_simd(&[], &[], &x);
            assert!(scores.is_empty());
        }

        #[test]
        fn simd_n_one_all_tail() {
            let d = 4;
            let centroids = vec![0.0; d];
            let sigmas = vec![1.0];
            let x = vec![0.0; d];
            let scores = batch_gaussian_simd(&centroids, &sigmas, &x);
            assert_eq!(scores.len(), 1);
            assert!((scores[0] - 1.0).abs() < 1e-10);
        }

        #[test]
        fn simd_n_four_exactly_one_lane() {
            let d = 2;
            let centroids = vec![0.0; 4 * d];
            let sigmas = vec![1.0; 4];
            let x = vec![0.0; d];
            let scores = batch_gaussian_simd(&centroids, &sigmas, &x);
            assert_eq!(scores.len(), 4);
            for s in &scores {
                assert!((s - 1.0).abs() < 1e-10);
            }
        }
    }
}
