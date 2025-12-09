# Author: a.stepanov

import numpy as np
import scipy.linalg as la
import bisect
from abc import ABC, abstractmethod
from typing import Callable, Optional


# ==================== INTERFACES ====================

class NoiseGenerator(ABC):
    """Abstract class for noise generators"""

    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """Generates noise of specified length"""
        pass


class CovarianceCalculator(ABC):
    """Abstract class for computing correlation matrices"""

    @abstractmethod
    def compute_correlation_matrix(self, n: int) -> np.ndarray:
        """Computes correlation matrix for n points"""
        pass

    @abstractmethod
    def get_variance(self) -> float:
        """Returns noise variance"""
        pass


# ==================== AR(1) IMPLEMENTATIONS ====================

class AR1NoiseGenerator(NoiseGenerator):
    """AR(1) noise generator with arbitrary white noise distribution"""

    def __init__(self, phi: float, sigma_w: float,
                 white_noise_func: Optional[Callable[[int], np.ndarray]] = None,
                 burn_in: int = 100):
        """
        Parameters:
        phi: float - autoregression parameter
        sigma_w: float - white noise standard deviation
        white_noise_func: white noise generation function returning array of specified size
        burn_in: int - number of initial points to discard
        """
        self.phi = phi
        self.sigma_w = sigma_w
        self.burn_in = burn_in

        if white_noise_func is None:
            self.white_noise_func = lambda size: np.random.normal(0, sigma_w, size)
        else:
            # Scale user function to required variance
            def scaled_white_noise(size):
                noise = white_noise_func(size)
                current_std = np.std(noise)
                if current_std > 0:
                    return noise * (sigma_w / current_std)
                return noise

            self.white_noise_func = scaled_white_noise

        self.sigma_eps = sigma_w / np.sqrt(1 - phi ** 2)

    def generate(self, n: int) -> np.ndarray:
        """Generate AR(1) noise considering burn_in"""
        total_n = n + self.burn_in
        eps = np.zeros(total_n)

        # Initial value from stationary distribution
        initial_std = np.std(self.white_noise_func(1000))
        eps[0] = self.white_noise_func(1)[0] * (self.sigma_eps / initial_std) if initial_std > 0 else 0

        # Generate remaining values
        for i in range(1, total_n):
            eps[i] = self.phi * eps[i - 1] + self.white_noise_func(1)[0]

        return eps[self.burn_in:]


class AR1CovarianceCalculator(CovarianceCalculator):
    """Correlation matrix computation for AR(1) process"""

    def __init__(self, phi: float, sigma_w: float):
        self.phi = phi
        self.sigma_w = sigma_w
        self.sigma_eps2 = sigma_w ** 2 / (1 - phi ** 2)

    def compute_correlation_matrix(self, n: int) -> np.ndarray:
        r = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                r[i, j] = self.phi ** abs(i - j)
        return r

    def get_variance(self) -> float:
        return self.sigma_eps2


# ==================== AR(2) IMPLEMENTATIONS ====================

class AR2NoiseGenerator(NoiseGenerator):
    """AR(2) noise generator with arbitrary white noise distribution"""

    def __init__(self, phi1: float, phi2: float, sigma_w: float,
                 white_noise_func: Optional[Callable[[int], np.ndarray]] = None,
                 burn_in: int = 100):
        """
        Parameters:
        phi1: float - first autoregression parameter
        phi2: float - second autoregression parameter
        sigma_w: float - white noise standard deviation
        white_noise_func: white noise generation function returning array of specified size
        burn_in: int - number of initial points to discard
        """
        self.phi1 = phi1
        self.phi2 = phi2
        self.sigma_w = sigma_w
        self.burn_in = burn_in

        if white_noise_func is None:
            self.white_noise_func = lambda size: np.random.normal(0, sigma_w, size)
        else:
            # Scale user function to required variance
            def scaled_white_noise(size):
                noise = white_noise_func(size)
                current_std = np.std(noise)
                if current_std > 0:
                    return noise * (sigma_w / current_std)
                return noise

            self.white_noise_func = scaled_white_noise

        # Check stationarity
        if not self.is_stationary():
            raise ValueError("AR(2) process parameters do not satisfy stationarity conditions")

    def is_stationary(self) -> bool:
        """Check stationarity conditions for AR(2) process"""
        return (self.phi2 + self.phi1 < 1 and
                self.phi2 - self.phi1 < 1 and
                abs(self.phi2) < 1)

    def generate(self, n: int) -> np.ndarray:
        """Generate AR(2) noise considering burn_in"""
        total_n = n + self.burn_in
        eps = np.zeros(total_n)

        # Initial values
        initial_std = np.std(self.white_noise_func(1000))
        eps[0] = self.white_noise_func(1)[0] if initial_std > 0 else 0
        eps[1] = self.white_noise_func(1)[0] if initial_std > 0 else 0

        # Generate remaining values
        for i in range(2, total_n):
            eps[i] = self.phi1 * eps[i - 1] + self.phi2 * eps[i - 2] + self.white_noise_func(1)[0]

        return eps[self.burn_in:]


class AR2CovarianceCalculator(CovarianceCalculator):
    """Correlation matrix computation for AR(2) process using Yule-Walker equations"""

    def __init__(self, phi1: float, phi2: float, sigma_w: float):
        self.phi1 = phi1
        self.phi2 = phi2
        self.sigma_w = sigma_w
        self.gamma0 = self._calculate_variance()
        self.rho = self._calculate_autocorrelation()

    def _calculate_variance(self) -> float:
        """Calculate AR(2) process variance using Yule-Walker equations"""
        a = np.array([
            [1, -self.phi1, -self.phi2],
            [-self.phi1, 1 - self.phi2, 0],
            [-self.phi2, -self.phi1, 1]
        ])

        b = np.array([self.sigma_w ** 2, 0, 0])

        try:
            gamma = np.linalg.solve(a, b)
            return gamma[0]
        except np.linalg.LinAlgError:
            return self.sigma_w ** 2 / (1 - self.phi1 ** 2 - self.phi2 ** 2)

    def _calculate_autocorrelation(self) -> np.ndarray:
        """Calculate autocorrelation function for AR(2) process"""
        rho = np.zeros(100)
        rho[0] = 1

        # Calculate ρ(1) from Yule-Walker equations
        rho[1] = self.phi1 / (1 - self.phi2)

        # Recurrence relation for k >= 2
        for k in range(2, len(rho)):
            rho[k] = self.phi1 * rho[k - 1] + self.phi2 * rho[k - 2]

        return rho

    def compute_correlation_matrix(self, n: int) -> np.ndarray:
        r = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                lag = abs(i - j)
                if lag < len(self.rho):
                    r[i, j] = self.rho[lag]
                else:
                    r[i, j] = self.phi1 * self.rho[lag - 1] + self.phi2 * self.rho[lag - 2]

        return r

    def get_variance(self) -> float:
        return self.gamma0


# ==================== COLORED NOISE IMPLEMENTATIONS ====================

class ColoredNoiseGenerator(NoiseGenerator):
    """Colored noise generator with 1/f^a spectral density law"""

    def __init__(self, a: float = 1.0,
                 scale: float = 1.0,
                 fs: float = 1.0,
                 target_std: float = 1.0,
                 f_min: float = 0.001):
        """
        Parameters:
        a: float - spectral slope parameter (1/f^a)
        scale: float - spectral density scale coefficient
        fs: float - sampling frequency
        target_std: float - target noise standard deviation
        f_min: float - minimum frequency for 1/f noise limitation
        """
        self.a = a
        self.scale = scale
        self.fs = fs
        self.target_std = target_std
        self.f_min = f_min

        # Create spectral density function based on parameters
        self.psd_func = self._create_psd_function()

    def _create_psd_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Create spectral density function based on a and scale parameters"""

        def psd_func(freqs):
            return self.scale / (np.maximum(np.abs(freqs), self.f_min) ** self.a)

        return psd_func

    def generate(self, n: int) -> np.ndarray:
        # Generate colored noise via inverse Fourier transform
        freqs = np.fft.fftfreq(n, 1 / self.fs)

        # Calculate spectral density
        psd = self.psd_func(freqs)

        # Generate random phases
        phases = 2 * np.pi * np.random.rand(n)

        # Create complex spectrum
        spectrum = np.sqrt(psd) * np.exp(1j * phases)

        # Inverse Fourier transform
        noise = np.real(np.fft.ifft(spectrum))

        # Normalize to target standard deviation
        current_std = np.std(noise)
        if current_std > 0:
            noise = noise * (self.target_std / current_std)

        return noise


class ColoredNoiseCovarianceCalculator(CovarianceCalculator):
    """Correlation matrix computation for colored noise with 1/f^a law"""

    def __init__(self, a: float = 1.0,
                 scale: float = 1.0,
                 fs: float = 1.0,
                 target_std: float = 1.0,
                 f_min: float = 0.001,
                 simulation_length: int = 10000):
        self.a = a
        self.scale = scale
        self.fs = fs
        self.target_std = target_std
        self.f_min = f_min
        self.simulation_length = simulation_length

        # Create spectral density function
        self.psd_func = self._create_psd_function()
        self.variance = self._estimate_variance()

    def _create_psd_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Create spectral density function based on a and scale parameters"""

        def psd_func(freqs):
            return self.scale / (np.maximum(np.abs(freqs), self.f_min) ** self.a)

        return psd_func

    def _estimate_variance(self) -> float:
        # Estimate variance by generating long realization
        generator = ColoredNoiseGenerator(
            a=self.a, scale=self.scale, fs=self.fs,
            target_std=self.target_std, f_min=self.f_min
        )
        long_noise = generator.generate(self.simulation_length)
        return np.var(long_noise)

    def compute_correlation_matrix(self, n: int) -> np.ndarray:
        # Generate long realization for autocorrelation function estimation
        generator = ColoredNoiseGenerator(
            a=self.a, scale=self.scale, fs=self.fs,
            target_std=1.0, f_min=self.f_min
        )
        long_noise = generator.generate(self.simulation_length)

        # Calculate autocorrelation function
        acf = np.correlate(long_noise, long_noise, mode='full')[self.simulation_length - 1:]
        acf = acf / acf[0]

        # Fill matrix
        r = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                lag = abs(i - j)
                if lag < len(acf):
                    r[i, j] = acf[lag]
                else:
                    r[i, j] = 0

        return r

    def get_variance(self) -> float:
        return self.variance


# ==================== WHITE NOISE GENERATION FUNCTIONS ====================

def uniform_white_noise(size: int, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """Generate uniform white noise"""
    return np.random.uniform(low, high, size)


def laplace_white_noise(size: int, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Generate white noise with Laplace distribution"""
    return np.random.laplace(loc, scale, size)


def student_t_white_noise(size: int, df: float = 3.0) -> np.ndarray:
    """Generate white noise with Student's t-distribution"""
    return np.random.standard_t(df, size)


def tsp_white_noise(size: int, p: float, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Generate white noise with symmetric TSP distribution"""
    r = scale * np.sqrt((p + 2) * (p + 1) / 2)

    u = np.random.uniform(0, 1, size)
    z = np.zeros_like(u)
    mask_left = u < 0.5
    z[mask_left] = -r * (1 - (1 - 2 * u[mask_left]) ** (1 / p))
    mask_right = u >= 0.5
    z[mask_right] = r * (1 - (2 * (1 - u[mask_right])) ** (1 / p))
    return loc + z


def mixed_white_noise(size: int, w1: float, scale1: float, scale2: float) -> np.ndarray:
    """Custom white noise generation function (mixture of two normal distributions), just for example"""
    if not 0 <= w1 <=1:
        raise ValueError(f"invalid weight: {w1}")
    result = np.zeros(size)
    for i in range(size):
        if np.random.rand() < w1:
            result[i] = np.random.normal(0, scale1)
        else:
            result[i] = np.random.normal(0, scale2)
    return result

# ==================== NOISE MIXTURE IMPLEMENTATIONS ====================


class MixedNoiseGenerator(NoiseGenerator):
    """Generator for mixture of white and colored noise"""

    def __init__(self, white_weight: float,
                 white_sigma: float,
                 colored_a: float = 1.0,
                 colored_scale: float = 1.0,
                 colored_fs: float = 1.0,
                 colored_target_std: float = 1.0,
                 colored_f_min: float = 0.001):
        """
        Parameters:
        white_weight: float - white noise weight in mixture (0-1)
        white_sigma: float - white noise standard deviation
        colored_a: float - colored noise spectral slope parameter (1/f^a)
        colored_scale: float - colored noise spectral density scale coefficient
        colored_fs: float - colored noise sampling frequency
        colored_target_std: float - colored noise target standard deviation
        colored_f_min: float - colored noise minimum frequency limitation
        """
        self.white_weight = white_weight
        self.white_sigma = white_sigma

        # Create colored noise generator
        self.colored_generator = ColoredNoiseGenerator(
            a=colored_a, scale=colored_scale, fs=colored_fs,
            target_std=colored_target_std, f_min=colored_f_min
        )

    def generate(self, n: int) -> np.ndarray:
        # Generate white noise
        white_noise = np.random.normal(0, self.white_sigma, n)

        # Generate colored noise
        colored_noise = self.colored_generator.generate(n)

        # Normalize colored noise to unit variance before mixing
        colored_std = np.std(colored_noise)
        if colored_std > 0:
            colored_noise = colored_noise / colored_std

        # Mix noises
        mixed_noise = (self.white_weight * white_noise +
                       (1 - self.white_weight) * colored_noise)

        return mixed_noise


class MixedNoiseCovarianceCalculator(CovarianceCalculator):
    """Correlation matrix computation for mixture of white and colored noise"""

    def __init__(self, white_weight: float,
                 white_sigma: float,
                 colored_a: float = 1.0,
                 colored_scale: float = 1.0,
                 colored_fs: float = 1.0,
                 colored_target_std: float = 1.0,
                 colored_f_min: float = 0.001,
                 simulation_length: int = 10000):
        """
        Parameters:
        white_weight: float - white noise weight in mixture (0-1)
        white_sigma: float - white noise standard deviation
        colored_a: float - colored noise spectral slope parameter (1/f^a)
        colored_scale: float - colored noise spectral density scale coefficient
        colored_fs: float - colored noise sampling frequency
        colored_target_std: float - colored noise target standard deviation
        colored_f_min: float - colored noise minimum frequency limitation
        simulation_length: int - simulation length for correlation matrix estimation
        """
        self.white_weight = white_weight
        self.white_sigma = white_sigma

        # Create calculator for colored noise
        self.colored_calculator = ColoredNoiseCovarianceCalculator(
            a=colored_a, scale=colored_scale, fs=colored_fs,
            target_std=colored_target_std, f_min=colored_f_min,
            simulation_length=simulation_length
        )
        self.simulation_length = simulation_length
        self.variance = self._estimate_variance()

    def _estimate_variance(self) -> float:
        # For noise mixture, variance equals sum of component variances
        white_variance = (self.white_weight * self.white_sigma) ** 2
        colored_variance = ((1 - self.white_weight) * np.sqrt(self.colored_calculator.get_variance())) ** 2
        return white_variance + colored_variance

    def compute_correlation_matrix(self, n: int) -> np.ndarray:
        # For mixture of white and colored noise, correlation matrix has form:
        # R = (w^2 * I + (1-w)^2 * R_color) / (w^2 + (1-w)^2)
        # where I - identity matrix, R_color - colored noise correlation matrix

        # Get colored noise correlation matrix
        r_color = self.colored_calculator.compute_correlation_matrix(n)

        # Create identity matrix for white noise
        i_mat = np.eye(n)

        # Calculate weights
        w2 = self.white_weight ** 2
        c2 = (1 - self.white_weight) ** 2

        # Calculate mixture correlation matrix
        r_mixed = (w2 * i_mat + c2 * r_color) / (w2 + c2)

        return r_mixed

    def get_variance(self) -> float:
        return self.variance


# ==================== GENERALIZED FUNCTION FOR K COMPUTATION ====================

def compute_uncertainties(n0, n, h, cov_calculator) -> tuple:
    """Computes standard uncertainties u(x) for generalized case."""
    # Design matrix
    x_points = np.arange(n0) * h
    x_mat = np.column_stack([np.ones(n0), x_points])

    # Correlation matrix
    r_mat = cov_calculator.compute_correlation_matrix(n0)
    r_inv = la.inv(r_mat)

    # Matrix A = (X^T R^{-1} X)^{-1}
    a_mat = la.inv(x_mat.T @ r_inv @ x_mat)
    variance = cov_calculator.get_variance()

    # Compute u(x) for each extrapolation point
    u_array = np.zeros(n)
    for j in range(n):
        x_val = j * h
        vec = np.array([1, x_val])
        u2 = variance * (vec @ a_mat @ vec.T)
        u_array[j] = np.sqrt(u2)

    return u_array, x_mat, r_inv, a_mat, variance


def compute_k(n0: int,
              n: int,
              h: float,
              noise_generator: NoiseGenerator,
              cov_calculator: CovarianceCalculator,
              p0: float,
              n_sim: int,
              k_low: float,
              k_high: float,
              k_tol=1e-4,
              print_u: bool = False) -> float:
    """Computes coverage factor K using Monte Carlo method and bisection"""
    # Precompute uncertainties and matrices
    u_array, x_mat, r_inv, a_mat, variance = compute_uncertainties(n0, n, h, cov_calculator)

    if print_u:
        print(u_array)

    m_list = []
    for _ in range(n_sim):
        # Generate noise
        eps = noise_generator.generate(n0)

        # Estimate regression coefficients
        beta_hat = a_mat @ (x_mat.T @ r_inv @ eps)

        # Compute M for this simulation
        max_ratio = 0
        for j in range(n):
            x_val = j * h
            y_hat_val = beta_hat[0] + beta_hat[1] * x_val
            ratio = np.abs(y_hat_val) / u_array[j]
            if ratio > max_ratio:
                max_ratio = ratio
        m_list.append(max_ratio)

    # Sort M list
    m_list.sort()

    # Check bounds
    idx_low = bisect.bisect_right(m_list, k_low)
    p_low = idx_low / n_sim
    idx_high = bisect.bisect_right(m_list, k_high)
    p_high = idx_high / n_sim

    if p_high < p0:
        # If k_high is too low, try to find suitable value
        m_max = max(m_list)
        if m_max > k_high:
            print(f"Warning: k_high={k_high} is too low, maximum M={m_max}. Increasing k_high.")
            k_high = m_max * 1.1
            idx_high = bisect.bisect_right(m_list, k_high)
            # p_high = idx_high / n_sim

    if p_low >= p0:
        print(f"Warning: k_low={k_low} is too high, p_low={p_low} >= p0. Decreasing k_low.")
        # Find minimum value where p >= p0
        for candidate in np.linspace(0, k_low, 100):
            idx_candidate = bisect.bisect_right(m_list, candidate)
            p_candidate = idx_candidate / n_sim
            if p_candidate >= p0:
                k_low = candidate
                # p_low = p_candidate
                break

    # Bisection
    while k_high - k_low > k_tol:
        k_mid = (k_low + k_high) / 2
        idx_mid = bisect.bisect_right(m_list, k_mid)
        p_mid = idx_mid / n_sim
        if p_mid < p0:
            k_low = k_mid
        else:
            k_high = k_mid

    return k_high


def estimate_confidence_band(n: int,
                             h: float,
                             y: np.ndarray,
                             k: float,
                             cov_calculator: CovarianceCalculator) -> tuple:
    """
    Estimate a regression confidence band using pre-calculated coverage factor K
    return a tuple of: x, y_est, lower and upper bounds of the band
    """
    n0 = len(y)
    if n0 > n:
        raise ValueError("n0 > n")
    u_array, x_mat, r_inv, a_mat, variance = compute_uncertainties(n0, n, h, cov_calculator)
    beta_hat = a_mat @ (x_mat.T @ r_inv @ y)
    print(f"est.: beta_0 = {beta_hat[0]:.3f}, beta_1 = {beta_hat[1]:.3f}")
    x = np.arange(0, n * h, h)
    y_hat = beta_hat[0] + beta_hat[1] * x

    # the band bounds
    lower_bound = y_hat - k * u_array
    upper_bound = y_hat + k * u_array
    return x, y_hat, lower_bound, upper_bound


# ==================== USAGE EXAMPLES ====================
#
# if __name__ == "__main__":
# 
#     # Parameters
#     n0 = 35
#     n = 100
#     h = 1
#     p0 = 0.95
#     n_sim = 50000
#     k_low = 1.0
#     k_high = 3.0
# 
#     # Example 1: AR(1) with normal white noise
#     print("Example 1: AR(1) with normal white noise")
#     phi = 0.5
#     sigma_w = 1.0
# 
#     noise_generator_ar1 = AR1NoiseGenerator(phi, sigma_w)
#     cov_calculator_ar1 = AR1CovarianceCalculator(phi, sigma_w)
# 
#     k_ar1 = compute_k(
#         n0, n, h, noise_generator_ar1, cov_calculator_ar1,
#         p0, n_sim, k_low, k_high
#     )
#     print(f"K for AR(1) with phi={phi}, sigma_w={sigma_w}: {k_ar1}")
# 
#     # Example 2: Colored noise
#     print("\nExample 5: Colored noise (pink noise, 1/f)")
#     colored_noise_gen = ColoredNoiseGenerator(
#         a=1., scale=1., fs=1., target_std=1., f_min=0.01)
#     colored_noise_cov = ColoredNoiseCovarianceCalculator(
#         a=1., scale=1., fs=1., target_std=1., f_min=0.01, simulation_length=500000)
# 
#     try:
#         k_colored = compute_k(
#             n0, n, h,
#             noise_generator=colored_noise_gen,
#             cov_calculator=colored_noise_cov,
#             p0=p0, n_sim=500000, k_low=k_low, k_high=4.
#         )
#         print(f"K for pink noise: {k_colored}")
#     except Exception as e:
#         print(f"Error computing K for colored noise: {e}")

