import random
import numpy as np
import estimate_reg_bands as be
from plot_band import plot_regression_bands


random.seed(42)


n0 = 35
n = 100
h = 1
p0 = 0.95
k_low = 1.0
k_high = 3.5


def example_1() -> None:
    """Example 1: AR(1) with normal white noise"""

    phi = 0.5
    sigma_w = 1.0
    n_sim = 1_000_000

    ng = be.AR1NoiseGenerator(phi, sigma_w)
    cc = be.AR1CovarianceCalculator(phi, sigma_w)

    k = be.compute_k(n0, n, h, ng, cc, p0, n_sim, k_low, k_high)
    print(f"K for AR(1) with phi={phi}, sigma_w={sigma_w}: {k:.2f}")

    y = ng.generate(n0)
    x, y_est, lower_bound, upper_bound = be.estimate_confidence_band(n, h, y, k, cc)
    plot_regression_bands(x, y, y_est, lower_bound, upper_bound, "example1.png")


def example_2() -> None:
    """Example 2: AR(1) with uniform white noise"""

    phi = 0.5
    sigma_w = 1.0
    n_sim = 100_000

    ng = be.AR1NoiseGenerator(
        phi, sigma_w,
        white_noise_func=lambda size: be.uniform_white_noise(size, low=-np.sqrt(3), high=np.sqrt(3)))
    cc = be.AR1CovarianceCalculator(phi, sigma_w)

    k = be.compute_k(n0, n, h, ng, cc, p0, n_sim, k_low, k_high)
    print(f"K for AR(1) with uniform white noise, phi={phi}, sigma_w={sigma_w}: {k:.2f}")

    y = ng.generate(n0)
    x, y_est, lower_bound, upper_bound = be.estimate_confidence_band(n, h, y, k, cc)
    plot_regression_bands(x, y, y_est, lower_bound, upper_bound, "example2.png")


def example_3() -> None:
    """Example 3: AR(2) with normal white noise"""

    phi1, phi2 = 0.7, 0.1
    sigma_w = 1.0
    n_sim = 100_000

    ng = be.AR2NoiseGenerator(phi1, phi2, sigma_w)
    cc = be.AR2CovarianceCalculator(phi1, phi2, sigma_w)

    k = be.compute_k(n0, n, h, ng, cc, p0, n_sim, k_low, k_high )
    print(f"K for AR(2) with phi1={phi1}, phi2={phi2}, sigma_w={sigma_w}: {k:.2f}")

    y = ng.generate(n0)
    x, y_est, lower_bound, upper_bound = be.estimate_confidence_band(n, h, y, k, cc)
    plot_regression_bands(x, y, y_est, lower_bound, upper_bound, "example3.png")


def example_4() -> None:
    """Example 4: AR(2) with Laplace white noise"""

    phi1, phi2 = 0.7, 0.1
    sigma_w = 1.0
    p_tsp = 1.5   # symmetric TSP distribution power parameter
    n_sim = 10_000

    ng = be.AR2NoiseGenerator(
        phi1, phi2, sigma_w, white_noise_func=lambda size: be.tsp_white_noise(size, p_tsp))
    cc = be.AR2CovarianceCalculator(phi1, phi2, sigma_w)

    k = be.compute_k(n0, n, h, ng, cc, p0, n_sim, k_low, k_high)
    print(f"K for AR(2) with TSP white noise, phi1={phi1}, phi2={phi2}, sigma_w={sigma_w}, p={p_tsp}: {k:.2f}")

    y = ng.generate(n0)
    x, y_est, lower_bound, upper_bound = be.estimate_confidence_band(n, h, y, k, cc)
    plot_regression_bands(x, y, y_est, lower_bound, upper_bound, "example4.png")


def example_5() -> None:
    """Example 5: 1/f^a colored noise"""

    n_sim = 100_000
    a = 1.  # colored noise spectral slope

    ng = be.ColoredNoiseGenerator(
        a, scale=1., fs=1., target_std=1., f_min=0.01)
    cc = be.ColoredNoiseCovarianceCalculator(
        a, scale=1., fs=1., target_std=1., f_min=0.01, simulation_length=500_000)

    k = be.compute_k(n0, n, h, ng, cc, p0, n_sim, k_low, k_high)
    print(f"K for 1/f^{a} noise: {k:.2f}")

    y = ng.generate(n0)
    x, y_est, lower_bound, upper_bound = be.estimate_confidence_band(n, h, y, k, cc)
    plot_regression_bands(x, y, y_est, lower_bound, upper_bound, "example5.png")


def example_6() -> None:
    """Mixture of white and colored noise"""

    n_sim = 100_000
    white_weight = 0.3
    white_sigma = 1.0
    colored_a = 1.0  # spectral slope (pink noise)

    ng = be.MixedNoiseGenerator(white_weight, white_sigma, colored_a,
        colored_scale=1., colored_fs=1., colored_target_std=1., colored_f_min=0.01)
    cc = be.MixedNoiseCovarianceCalculator(white_weight, white_sigma, colored_a,
        colored_scale=1., colored_fs=1., colored_target_std=1., colored_f_min=0.01, simulation_length=100000)

    k = be.compute_k(n0, n, h, ng, cc, p0, n_sim, k_low, k_high)
    print(f"K for mixture of white and colored noise (w={white_weight}, a={colored_a}): {k}")

    y = ng.generate(n0)
    x, y_est, lower_bound, upper_bound = be.estimate_confidence_band(n, h, y, k, cc)
    plot_regression_bands(x, y, y_est, lower_bound, upper_bound, "example6.png")


if __name__ == "__main__":

    example_1()
    # example_6()
