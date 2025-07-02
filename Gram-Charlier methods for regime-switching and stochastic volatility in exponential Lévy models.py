import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate, special
from scipy.linalg import expm
from scipy.special import factorial, kn  # For modified Bessel function K
import pandas as pd
import seaborn as sns
from numba import jit

# Set random seed for reproducibility
np.random.seed(42)

###########################################
# 1. Utilities for Moments and Cumulants  #
###########################################

def moments_from_cumulants(kappa, n_max):
    """
    Convert cumulants to moments using recursive formula (5) from the paper
    
    Parameters:
    kappa - list of cumulants starting from kappa_1
    n_max - maximum order of moments to compute
    
    Returns:
    moments - list of moments from E[X] to E[X^n_max]
    """
    moments = [0] * n_max
    moments[0] = kappa[0]  # First moment equals first cumulant
    
    for n in range(2, n_max + 1):
        sum_term = 0
        for k in range(1, n):
            sum_term += special.comb(n-1, k-1) * kappa[k-1] * moments[n-k-1]
        moments[n-1] = sum_term + kappa[n-1]
    
    return moments

def cumulants_from_moments(moments, n_max):
    """
    Convert moments to cumulants using recursive formula (6) from the paper
    
    Parameters:
    moments - list of moments starting from E[X]
    n_max - maximum order of cumulants to compute
    
    Returns:
    kappa - list of cumulants from kappa_1 to kappa_{n_max}
    """
    kappa = [0] * n_max
    kappa[0] = moments[0]  # First cumulant equals first moment
    
    for n in range(2, n_max + 1):
        sum_term = 0
        for k in range(1, n):
            sum_term += special.comb(n-1, k-1) * kappa[k-1] * moments[n-k-1]
        kappa[n-1] = moments[n-1] - sum_term
    
    return kappa

###########################################
# 2. Lévy Process Implementations        #
###########################################

class LevyProcess:
    def __init__(self):
        pass
    
    def levy_exponent(self, s):
        """Return the Lévy exponent κ(s)"""
        pass
    
    def cumulants(self, n, t=1):
        """Return the cumulants of order 1 to n at time t"""
        pass
    
    def moments(self, n, t=1):
        """Return the moments of order 1 to n at time t"""
        kappa = self.cumulants(n, t)
        return moments_from_cumulants(kappa, n)
    
    def simulate(self, t, n_paths=1):
        """Simulate the process at time t"""
        pass

class BrownianMotion(LevyProcess):
    def __init__(self, mu, sigma):
        """
        Brownian motion with drift mu and volatility sigma
        """
        self.mu = mu
        self.sigma = sigma
    
    def levy_exponent(self, s):
        """Lévy exponent: κ(s) = μs + σ²s²/2"""
        return self.mu * s + 0.5 * self.sigma**2 * s**2
    
    def cumulants(self, n, t=1):
        """
        Cumulants: κ₁ = μt, κ₂ = σ²t, κₙ = 0 for n > 2
        """
        kappa = [0] * max(n, 2)  # Ensure we have at least 2 elements
        kappa[0] = self.mu * t
        kappa[1] = self.sigma**2 * t
        return kappa
    
    def simulate(self, t, n_paths=1):
        """Simulate BM paths"""
        return self.mu * t + self.sigma * np.sqrt(t) * np.random.normal(0, 1, n_paths)

class JumpDiffusion(LevyProcess):
    def __init__(self, mu, sigma, lambda_, jump_dist):
        """
        Jump-diffusion process with:
        - drift mu
        - volatility sigma
        - jump intensity lambda_
        - jump_dist: a function that generates jump sizes
        """
        self.mu = mu
        self.sigma = sigma
        self.lambda_ = lambda_
        self.jump_dist = jump_dist
    
    def levy_exponent(self, s):
        """Lévy exponent depends on the jump distribution"""
        # This is a placeholder - actual implementation depends on jump_dist
        pass
    
    def cumulants(self, n, t=1):
        """
        Cumulants for jump diffusion (Example 1 in the paper):
        κ₁ = (a + λE[Y])t, κ₂ = (σ² + λE[Y²])t, κₙ = λE[Yⁿ]t for n > 2
        """
        # This implementation assumes we can compute the moments of jump_dist
        kappa = [0] * n
        # This is a placeholder - would need specific jump distribution moments
        return kappa
    
    def simulate(self, t, n_paths=1):
        """Simulate jump-diffusion paths"""
        # Brownian component
        bm_component = self.mu * t + self.sigma * np.sqrt(t) * np.random.normal(0, 1, n_paths)
        
        # Jump component
        jump_component = np.zeros(n_paths)
        for i in range(n_paths):
            n_jumps = np.random.poisson(self.lambda_ * t)
            if n_jumps > 0:
                jumps = self.jump_dist(n_jumps)
                jump_component[i] = np.sum(jumps)
        
        return bm_component + jump_component

class NIG(LevyProcess):
    def __init__(self, alpha, beta, delta, mu):
        """
        Normal Inverse Gaussian (NIG) process with parameters:
        - alpha: tail heaviness (steepness)
        - beta: skewness
        - delta: scale
        - mu: location
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu
        
        # Verify parameter constraints
        assert alpha > 0, "alpha must be positive"
        assert abs(beta) < alpha, "beta must satisfy |beta| < alpha"
        assert delta > 0, "delta must be positive"
    
    def levy_exponent(self, s):
        """
        Lévy exponent for NIG (Equation 12 in the paper):
        κ(s) = μs - δ(√(α² - (β+s)²) - √(α² - β²))
        """
        return self.mu * s - self.delta * (np.sqrt(self.alpha**2 - (self.beta + s)**2) - 
                                          np.sqrt(self.alpha**2 - self.beta**2))
    
    def cumulants(self, n, t=1):
        """
        Compute cumulants for NIG process.
        For n ≤ 4, we use the explicit formulas from the paper.
        For n > 4, we use numerical differentiation.
        """
        kappa = [0] * max(n, 4)  # Ensure we have at least 4 elements
        
        # First four cumulants
        gamma = np.sqrt(self.alpha**2 - self.beta**2)
        kappa[0] = t * (self.mu + self.delta * self.beta / gamma)
        kappa[1] = t * self.delta * self.alpha**2 / gamma**3
        kappa[2] = t * 3 * self.delta * self.beta * self.alpha**2 / gamma**5
        kappa[3] = t * 3 * self.delta * self.alpha**2 * (self.alpha**2 + 4*self.beta**2) / gamma**7
        
        # For higher order cumulants, we could use numerical differentiation
        # of the Lévy exponent as described in the paper's appendix
        
        return kappa
    
    def pdf(self, x, t=1):
        """
        Probability density function for NIG at time t (Equation 14 in the paper)
        """
        alpha_t = self.alpha
        beta_t = self.beta
        delta_t = self.delta * t
        mu_t = self.mu * t
        
        gamma = np.sqrt(alpha_t**2 - beta_t**2)
        
        # Compute the PDF using the formula from the paper
        factor = alpha_t * delta_t / np.pi
        exp_term = delta_t * gamma + beta_t * (x - mu_t)
        bessel_arg = alpha_t * np.sqrt(delta_t**2 + (x - mu_t)**2)
        
        return factor * np.exp(exp_term) * kn(1, bessel_arg) / np.sqrt(delta_t**2 + (x - mu_t)**2)
    
    def simulate(self, t, n_paths=1):
        """
        Simulate NIG process paths using the fact that NIG is a time-changed Brownian motion
        """
        # Simulate inverse Gaussian subordinator
        lambda_param = self.delta * np.sqrt(self.alpha**2 - self.beta**2)
        mu_param = self.delta / np.sqrt(self.alpha**2 - self.beta**2)
        
        # Generate inverse Gaussian random variables (time changes)
        # This is a simplified approximation
        time_changes = np.random.gamma(t/mu_param, mu_param**2, n_paths)
        
        # Generate Brownian motion with drift
        drift = self.mu + self.delta * self.beta / np.sqrt(self.alpha**2 - self.beta**2)
        volatility = self.delta / np.sqrt(self.alpha**2 - self.beta**2)
        
        # Combine to get NIG paths
        nig_paths = drift * t + self.beta * time_changes + volatility * np.sqrt(time_changes) * np.random.normal(0, 1, n_paths)
        
        return nig_paths

class CGMY(LevyProcess):
    def __init__(self, C_plus, C_minus, G, M, Y_plus, Y_minus):
        """
        CGMY (Carr-Geman-Madan-Yor) process with parameters:
        - C_plus, C_minus: intensity parameters
        - G, M: exponential decay parameters
        - Y_plus, Y_minus: tail decay parameters
        """
        self.C_plus = C_plus
        self.C_minus = C_minus
        self.G = G
        self.M = M
        self.Y_plus = Y_plus
        self.Y_minus = Y_minus
        
        # Verify parameter constraints
        assert C_plus >= 0 and C_minus >= 0, "C_plus and C_minus must be non-negative"
        assert G > 0 and M > 0, "G and M must be positive"
        assert 0 <= Y_plus < 2 and 0 <= Y_minus < 2, "Y_plus and Y_minus must be in [0,2)"
    
    def levy_exponent(self, s):
        """
        Lévy exponent for CGMY (Example 2 in the paper):
        κ(s) = C⁺(-Y⁺)[(M-s)^Y⁺ - M^Y⁺] + C⁻(-Y⁻)[(G+s)^Y⁻ - G^Y⁻]
        """
        term1 = self.C_plus * (-self.Y_plus) * ((self.M - s)**self.Y_plus - self.M**self.Y_plus)
        term2 = self.C_minus * (-self.Y_minus) * ((self.G + s)**self.Y_minus - self.G**self.Y_minus)
        return term1 + term2
    
    def cumulants(self, n, t=1):
        """
        Compute cumulants for CGMY process (Example 2 in the paper):
        κₙ = t[C⁺(-Y⁺)(-1)ⁿY⁺ⁿM^Y⁺⁻ⁿ + C⁻(-Y⁻)Y⁻ⁿG^Y⁻⁻ⁿ]
        """
        kappa = [0] * n
        
        for i in range(1, n+1):
            # Helper function for falling factorial
            def falling_factorial(y, n):
                result = 1
                for j in range(n):
                    result *= (y - j)
                return result
            
            term1 = self.C_plus * (-self.Y_plus) * ((-1)**i) * falling_factorial(self.Y_plus, i) * self.M**(self.Y_plus-i)
            term2 = self.C_minus * (-self.Y_minus) * falling_factorial(self.Y_minus, i) * self.G**(self.Y_minus-i)
            kappa[i-1] = t * (term1 + term2)
        
        return kappa
    
    def simulate(self, t, n_paths=1):
        """
        Simulate CGMY process paths - this is complex and beyond the scope of this demonstration
        For our examples, we'll use a simplified approximation
        """
        # Get moments for CGMY
        moments = self.moments(4, t)
        mean = moments[0]
        variance = moments[1] - mean**2
        skewness = moments[2] - 3*mean*variance - mean**3
        
        # Use normal approximation with matching first two moments
        # This is a very rough approximation just for demonstration
        return np.random.normal(mean, np.sqrt(variance), n_paths)

###########################################
# 3. Regime-Switching Implementation      #
###########################################

class RegimeSwitchingLevy:
    def __init__(self, levy_processes, generator_matrix, jump_dist=None):
        """
        Regime-switching Lévy process with:
        - levy_processes: list of Lévy processes for each state
        - generator_matrix: Q matrix for the Markov chain
        - jump_dist: dictionary of jump distributions when transitioning between states
                    Format: {(i,j): dist_function} where i,j are states
        """
        self.levy_processes = levy_processes
        self.Q = generator_matrix
        self.n_states = len(levy_processes)
        self.jump_dist = jump_dist if jump_dist else {}
        
        # Verify dimensions
        assert generator_matrix.shape == (self.n_states, self.n_states), "Generator matrix dimensions mismatch"
    
    def compute_moment_matrices(self, n_max, t):
        """
        Compute moment matrices m^(k)(t) using Theorem 1 from the paper
        
        Parameters:
        n_max - maximum order of moments to compute
        t - time horizon
        
        Returns:
        List of moment matrices m^(0)(t), m^(1)(t), ..., m^(n_max)(t)
        """
        # Construct matrices Q^(k) based on the cumulants of the Lévy processes
        Q_matrices = [np.zeros((self.n_states, self.n_states)) for _ in range(n_max+1)]
        
        # Q^(0) is the generator matrix
        Q_matrices[0] = self.Q.copy()
        
        # Compute Q^(k) for k ≥ 1
        for k in range(1, n_max+1):
            for i in range(self.n_states):
                # Diagonal elements: kth cumulant of the Lévy process in state i
                Q_matrices[k][i, i] = self.levy_processes[i].cumulants(k)[k-1]
                
                # Off-diagonal elements: kth moment of jump distribution
                for j in range(self.n_states):
                    if i != j and (i, j) in self.jump_dist:
                        # If there's a jump distribution for transition from i to j
                        # This assumes the jump_dist provides a method to compute kth moment
                        jump_moment = self.jump_dist[(i, j)].moment(k)
                        Q_matrices[k][i, j] = self.Q[i, j] * jump_moment
        
        # Construct the block-Toeplitz matrix G^(k) as in Theorem 1
        n_block = n_max + 1
        G_matrix = np.zeros((n_block * self.n_states, n_block * self.n_states))
        
        for i in range(n_block):
            for j in range(i, n_block):
                # Fill the (i,j) block with Q^(j-i) / (j-i)!
                block_idx_i = i * self.n_states
                block_idx_j = j * self.n_states
                G_matrix[block_idx_i:block_idx_i+self.n_states, 
                         block_idx_j:block_idx_j+self.n_states] = Q_matrices[j-i] / factorial(j-i)
        
        # Compute the matrix exponential e^{tG^(k)}
        H_matrix = expm(t * G_matrix)
        
        # Extract the moment matrices m^(k)(t)
        moment_matrices = []
        for k in range(n_max+1):
            block_idx = k * self.n_states
            moment_matrices.append(H_matrix[:self.n_states, block_idx:block_idx+self.n_states] * factorial(k))
        
        return moment_matrices
    
    def moments(self, n_max, t, initial_state=None):
        """
        Compute moments E[Z(t)^k] for k=1,...,n_max
        
        Parameters:
        n_max - maximum order of moments
        t - time horizon
        initial_state - initial state (if None, use stationary distribution)
        
        Returns:
        List of moments E[Z(t)], E[Z(t)^2], ..., E[Z(t)^n_max]
        """
        moment_matrices = self.compute_moment_matrices(n_max, t)
        
        # Determine initial state distribution
        if initial_state is None:
            # Use stationary distribution (solve π*Q = 0)
            # This is a simplified approximation for demonstration
            initial_dist = np.ones(self.n_states) / self.n_states
        else:
            initial_dist = np.zeros(self.n_states)
            initial_dist[initial_state] = 1.0
        
        # Compute moments as E[Z(t)^k] = initial_dist * m^(k)(t) * 1
        ones = np.ones(self.n_states)
        moments = []
        
        for k in range(1, n_max+1):
            moment_k = initial_dist @ moment_matrices[k] @ ones
            moments.append(moment_k)
        
        return moments
    
    def simulate(self, t, n_paths=1, initial_state=None):
        """
        Simulate paths of the regime-switching Lévy process
        
        Parameters:
        t - time horizon
        n_paths - number of paths to simulate
        initial_state - initial state (if None, sample from stationary distribution)
        
        Returns:
        Array of simulated values Z(t)
        """
        # Determine initial state
        if initial_state is None:
            # Sample from stationary distribution (approximate)
            initial_states = np.random.choice(self.n_states, size=n_paths)
        else:
            initial_states = np.full(n_paths, initial_state)
        
        # Simulate paths
        results = np.zeros(n_paths)
        
        for i in range(n_paths):
            # Simulate the continuous-time Markov chain
            current_state = initial_states[i]
            current_time = 0.0
            path_value = 0.0
            
            while current_time < t:
                # Time to next transition
                exit_rate = -self.Q[current_state, current_state]
                if exit_rate > 0:
                    time_to_next = np.random.exponential(1/exit_rate)
                else:
                    time_to_next = float('inf')
                
                # If transition occurs after t, just evolve until t
                if current_time + time_to_next > t:
                    # Evolve Lévy process in current state until t
                    remaining_time = t - current_time
                    path_value += self.levy_processes[current_state].simulate(remaining_time, 1)[0]
                    break
                
                # Evolve Lévy process in current state until transition
                path_value += self.levy_processes[current_state].simulate(time_to_next, 1)[0]
                current_time += time_to_next
                
                # Transition to new state
                transition_probs = self.Q[current_state, :].copy()
                transition_probs[current_state] = 0
                
                # Check if there are any valid transitions
                if np.sum(transition_probs) > 0:
                    transition_probs = transition_probs / np.sum(transition_probs)  # Normalize
                    new_state = np.random.choice(self.n_states, p=transition_probs)
                    
                    # Add jump if present
                    if (current_state, new_state) in self.jump_dist:
                        path_value += self.jump_dist[(current_state, new_state)].sample()
                    
                    current_state = new_state
                else:
                    # No valid transitions, stay in current state
                    continue
            
            results[i] = path_value
        
        return results

###########################################
# 4. Stochastic Volatility Implementation #
###########################################

class StochasticVolatility:
    def __init__(self, base_process, time_change_process):
        """
        Stochastic volatility model Z(t) = X(Y(t)) with:
        - base_process: the base Lévy process X
        - time_change_process: the process Y(t) driving the time change
        """
        self.base_process = base_process
        self.time_change_process = time_change_process
    
    def cumulants(self, n, t=1):
        """
        Compute cumulants using Proposition 1 from the paper:
        κ_Z(t)(s) = κ_Y(t)(κ_X(s))
        """
        # First, get cumulants of the time change process
        y_cumulants = self.time_change_process.cumulants(n, t)
        
        # Then, get cumulants of the base process
        x_cumulants = self.base_process.cumulants(n)
        
        # Compute cumulants of the time-changed process using Proposition 1
        # This is a simplified implementation and might need adjustment for specific cases
        z_cumulants = [0] * n
        
        for k in range(1, n+1):
            # Implement the Bell polynomial formula from Proposition 1
            # This is a placeholder - would need proper Bell polynomial implementation
            z_cumulants[k-1] = y_cumulants[0] * x_cumulants[k-1]  # Simplified version
        
        return z_cumulants
    
    def moments(self, n, t=1):
        """Compute moments from cumulants"""
        kappa = self.cumulants(n, t)
        return moments_from_cumulants(kappa, n)
    
    def simulate(self, t, n_paths=1):
        """
        Simulate the stochastic volatility process Z(t) = X(Y(t))
        """
        # First simulate the time change
        time_changes = self.time_change_process.simulate(t, n_paths)
        
        # Then simulate the base process at the random times
        results = np.zeros(n_paths)
        for i in range(n_paths):
            results[i] = self.base_process.simulate(time_changes[i], 1)[0]
        
        return results

###########################################
# 5. Gram-Charlier Expansion             #
###########################################

class NormalRefPDF:
    """Normal reference PDF for Gram-Charlier expansion"""
    
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def pdf(self, x):
        """Evaluate the normal PDF at x"""
        return stats.norm.pdf(x, self.mu, self.sigma)
    
    def orthonormal_polynomial(self, n, x):
        """Evaluate Hermite polynomial at x"""
        # Hermite polynomials for the normal distribution
        z = (x - self.mu) / self.sigma
        
        if n == 0:
            return 1.0
        elif n == 1:
            return z
        elif n == 2:
            return (z**2 - 1) / np.sqrt(2)
        elif n == 3:
            return (z**3 - 3*z) / np.sqrt(6)
        elif n == 4:
            return (z**4 - 6*z**2 + 3) / np.sqrt(24)
        else:
            # For higher orders, would need to implement recursion
            return 0

class GramCharlierExpansion:
    def __init__(self, target_process, reference_pdf, n_terms=4):
        """
        Gram-Charlier expansion for approximating a target density
        
        Parameters:
        target_process - the target stochastic process
        reference_pdf - reference probability density function and its orthonormal polynomials
        n_terms - number of terms in the expansion
        """
        self.target_process = target_process
        self.reference_pdf = reference_pdf
        self.n_terms = n_terms
        self.coefficients = None
    
    def compute_coefficients(self, t):
        """
        Compute the coefficients c_n = E[p_n(X)] for the expansion
        """
        # Compute moments of the target process
        moments = self.target_process.moments(self.n_terms, t)
        
        # For a normal reference, we can compute coefficients directly
        # This is a simplified implementation for normal reference
        coefficients = [1.0]  # c_0 = 1
        
        # If the reference is a normal with matching mean and variance,
        # then c_1 = c_2 = 0
        if isinstance(self.reference_pdf, NormalRefPDF):
            if abs(moments[0] - self.reference_pdf.mu) < 1e-10 and \
               abs(moments[1] - moments[0]**2 - self.reference_pdf.sigma**2) < 1e-10:
                coefficients.extend([0.0, 0.0])
                
                # For c_3, c_4, use skewness and kurtosis
                if self.n_terms >= 3:
                    # Skewness
                    skewness = (moments[2] - 3*moments[0]*moments[1] + 2*moments[0]**3) / \
                               (moments[1] - moments[0]**2)**(3/2)
                    coefficients.append(skewness / np.sqrt(6))
                
                if self.n_terms >= 4:
                    # Kurtosis
                    kurtosis = (moments[3] - 4*moments[0]*moments[2] + 6*moments[0]**2*moments[1] - 3*moments[0]**4) / \
                               (moments[1] - moments[0]**2)**2 - 3
                    coefficients.append(kurtosis / np.sqrt(24))
            else:
                # If not matching, would need to compute coefficients differently
                pass
        
        self.coefficients = coefficients
        return coefficients
    
    def evaluate(self, x, t):
        """
        Evaluate the Gram-Charlier approximation at point x
        """
        if self.coefficients is None:
            self.compute_coefficients(t)
        
        # Evaluate the reference PDF
        f_ref = self.reference_pdf.pdf(x)
        
        # Sum up the expansion terms
        expansion_sum = 0
        for n in range(min(len(self.coefficients), self.n_terms + 1)):
            # Evaluate the nth orthonormal polynomial
            p_n = self.reference_pdf.orthonormal_polynomial(n, x)
            expansion_sum += self.coefficients[n] * p_n
        
        return f_ref * expansion_sum
    
    def option_price(self, S0, K, r, t):
        """
        Compute European call option price using the Gram-Charlier expansion
        
        Parameters:
        S0 - initial asset price
        K - strike price
        r - risk-free rate
        t - time to maturity
        
        Returns:
        Approximated option price
        """
        # Compute the integral in equation (4) of the paper
        log_moneyness = np.log(K/S0)
        
        def integrand(x):
            return (S0 * np.exp(x) - K) * self.evaluate(x, t)
        
        # Numerical integration
        option_price, _ = integrate.quad(integrand, log_moneyness, np.inf)
        
        # Discount to present value
        return np.exp(-r*t) * option_price

###########################################
# 6. Example implementations and tests    #
###########################################

# Example 1: Black-Scholes model
def test_black_scholes():
    print("Testing Black-Scholes model...")
    # Parameters
    sigma = 0.25
    mu = 0.03 - 0.5 * sigma**2  # Risk-neutral drift
    
    # Create Brownian motion process
    bm = BrownianMotion(mu, sigma)
    
    # Compute moments for different time horizons
    t_values = [0.5, 1.0, 3.0]
    for t in t_values:
        moments = bm.moments(4, t)
        print(f"Moments at t={t}:")
        for i, mom in enumerate(moments, 1):
            print(f"E[X({t})^{i}] = {mom:.6f}")
    
    # Simulate paths
    n_paths = 10000
    t = 1.0
    simulations = bm.simulate(t, n_paths)
    
    # Plot histogram of simulations
    plt.figure(figsize=(10, 6))
    plt.hist(simulations, bins=50, density=True, alpha=0.7, label='Simulations')
    
    # Overlay theoretical normal density
    x = np.linspace(np.min(simulations) - 0.5, np.max(simulations) + 0.5, 1000)
    pdf = stats.norm.pdf(x, mu * t, sigma * np.sqrt(t))
    plt.plot(x, pdf, 'r-', lw=2, label='Theoretical PDF')
    
    plt.title(f'Black-Scholes Log-Returns at t={t}')
    plt.xlabel('Log-Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Compute Black-Scholes option prices
    S0 = 1.0
    r = 0.03
    strikes = [0.8, 1.0, 1.2]
    
    print("\nBlack-Scholes Option Prices:")
    print("Strike | T=0.5 | T=1.0 | T=3.0")
    print("------------------------------")
    
    for K in strikes:
        prices = []
        for t in t_values:
            d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
            d2 = d1 - sigma*np.sqrt(t)
            price = S0 * stats.norm.cdf(d1) - K * np.exp(-r*t) * stats.norm.cdf(d2)
            prices.append(price)
        
        print(f"{K:.1f}   | {prices[0]:.3f} | {prices[1]:.3f} | {prices[2]:.3f}")

# Example 2: Regime-switching Black-Scholes model
def test_regime_switching_bs():
    print("\nTesting Regime-Switching Black-Scholes model...")
    
    # Parameters
    sigma1 = 0.158  # Low volatility state
    sigma2 = 0.316  # High volatility state (4*sigma1)
    r = 0.03
    mu1 = r - 0.5 * sigma1**2  # Risk-neutral drift in state 1
    mu2 = r - 0.5 * sigma2**2  # Risk-neutral drift in state 2
    
    # Transition matrix
    q12 = 10/9  # Approx. once per year from low to high
    q21 = 10    # Spending 1:9 time in high:low volatility
    Q = np.array([
        [-q12, q12],
        [q21, -q21]
    ])
    
    # Create Brownian motion processes for each state
    bm1 = BrownianMotion(mu1, sigma1)
    bm2 = BrownianMotion(mu2, sigma2)
    
    # Create regime-switching process
    rs_bs = RegimeSwitchingLevy([bm1, bm2], Q)
    
    # Create regime-switching process with jumps
    # Jump distribution when entering high volatility state
    class JumpDist:
        def __init__(self, probs, values):
            self.probs = probs
            self.values = values
            
        def sample(self):
            return np.random.choice(self.values, p=self.probs)
            
        def moment(self, k):
            return np.sum(self.probs * np.power(self.values, k))
    
    # Jump is log(0.98) with prob 0.4, log(0.95) with prob 0.1, and 0 with prob 0.5
    jump_dist = JumpDist(
        [0.5, 0.4, 0.1],
        [0, np.log(0.98), np.log(0.95)]
    )
    
    # Adjust mu1 to maintain risk-neutrality
    mu1_adjusted = r - 0.5 * sigma1**2 - q12 * (0.5 + 0.4*np.exp(np.log(0.98)) + 0.1*np.exp(np.log(0.95)) - 1)
    bm1_adjusted = BrownianMotion(mu1_adjusted, sigma1)
    
    rs_bs_jump = RegimeSwitchingLevy(
        [bm1_adjusted, bm2], 
        Q,
        {(0, 1): jump_dist}  # Jump when transitioning from state 0 to 1
    )
    
    # Compute moments
    t_values = [0.5, 1.0, 3.0]
    for t in t_values:
        print(f"\nMoments at t={t}:")
        moments_rs = rs_bs.moments(4, t)
        moments_rs_jump = rs_bs_jump.moments(4, t)
        
        print("Order | RS-BS   | RS-BS+Jump")
        print("---------------------------")
        for i, (mom1, mom2) in enumerate(zip(moments_rs, moments_rs_jump), 1):
            print(f"{i}     | {mom1:.6f} | {mom2:.6f}")
    
    # Simulate paths
    n_paths = 10000
    t = 1.0
    
    sim_rs = rs_bs.simulate(t, n_paths)
    sim_rs_jump = rs_bs_jump.simulate(t, n_paths)
    
    # Plot histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(sim_rs, bins=50, density=True, alpha=0.7, label='RS-BS')
    # Overlay standard BS density for comparison
    bm_std = BrownianMotion(r - 0.5*0.25**2, 0.25)  # BS with sigma=0.25
    sim_bs = bm_std.simulate(t, n_paths)
    plt.hist(sim_bs, bins=50, density=True, alpha=0.4, label='Standard BS')
    plt.title(f'Regime-Switching BS at t={t}')
    plt.xlabel('Log-Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(sim_rs_jump, bins=50, density=True, alpha=0.7, label='RS-BS+Jump')
    plt.hist(sim_rs, bins=50, density=True, alpha=0.4, label='RS-BS')
    plt.title(f'RS-BS vs RS-BS+Jump at t={t}')
    plt.xlabel('Log-Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compute option prices
    S0 = 1.0
    strikes = [0.8, 1.0, 1.2]
    
    # Monte Carlo pricing for comparison
    def mc_option_price(simulations, S0, K, r, t):
        payoffs = np.maximum(S0 * np.exp(simulations) - K, 0)
        return np.exp(-r*t) * np.mean(payoffs)
    
    print("\nOption Prices (Monte Carlo):")
    print("Strike | T    | Standard BS | RS-BS    | RS-BS+Jump")
    print("---------------------------------------------------")
    
    for K in strikes:
        for t in t_values:
            sim_bs = bm_std.simulate(t, n_paths)
            sim_rs = rs_bs.simulate(t, n_paths)
            sim_rs_jump = rs_bs_jump.simulate(t, n_paths)
            
            price_bs = mc_option_price(sim_bs, S0, K, r, t)
            price_rs = mc_option_price(sim_rs, S0, K, r, t)
            price_rs_jump = mc_option_price(sim_rs_jump, S0, K, r, t)
            
            print(f"{K:.1f}   | {t:.1f} | {price_bs:.6f}    | {price_rs:.6f} | {price_rs_jump:.6f}")

# Example 3: NIG process and Gram-Charlier expansion
def test_nig_gc():
    print("\nTesting NIG process with Gram-Charlier expansion...")
    
    # Parameters from Carr et al. (2003) for S&P 500
    alpha = 29.5
    beta = 4.46  # Risk-neutral version
    delta = 0.177
    mu = 0.0
    
    # Create NIG process
    nig = NIG(alpha, beta, delta, mu)
    
    # Compute cumulants
    t = 1.0
    n_cumulants = 6
    cumulants = nig.cumulants(n_cumulants, t)
    
    print(f"NIG Cumulants at t={t}:")
    for i, kappa in enumerate(cumulants, 1):
        print(f"κ_{i} = {kappa:.6f}")
    
    # Compute moments
    moments = nig.moments(n_cumulants, t)
    
    print(f"\nNIG Moments at t={t}:")
    for i, mom in enumerate(moments, 1):
        print(f"E[X({t})^{i}] = {mom:.6f}")
    
    # Simulate NIG paths
    n_paths = 10000
    simulations = nig.simulate(t, n_paths)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(simulations, bins=50, density=True, alpha=0.7, label='NIG Simulations')
    
    # Create normal reference for Gram-Charlier expansion
    mean = moments[0]
    var = moments[1] - moments[0]**2
    normal_ref = NormalRefPDF(mean, np.sqrt(var))
    
    # Simple normal reference approximation
    x = np.linspace(np.min(simulations) - 0.5, np.max(simulations) + 0.5, 1000)
    plt.plot(x, normal_ref.pdf(x), 'r-', lw=2, label='Normal Approximation')
    
    # Create Gram-Charlier expansion
    gc = GramCharlierExpansion(nig, normal_ref, n_terms=4)
    coeffs = gc.compute_coefficients(t)
    
    # Plot GC approximation if coefficients were computed
    if len(coeffs) > 1:
        gc_values = [gc.evaluate(xi, t) for xi in x]
        plt.plot(x, gc_values, 'g-', lw=2, label='Gram-Charlier Approximation')
    
    plt.title(f'NIG Distribution at t={t}')
    plt.xlabel('Log-Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Option pricing
    S0 = 1.0
    r = 0.03
    strikes = [0.8, 1.0, 1.2]
    t_values = [0.5, 1.0, 3.0]
    
    # Monte Carlo pricing
    def mc_option_price(process, S0, K, r, t, n_paths=50000):
        simulations = process.simulate(t, n_paths)
        payoffs = np.maximum(S0 * np.exp(simulations) - K, 0)
        return np.exp(-r*t) * np.mean(payoffs)
    
    print("\nNIG Option Prices (Monte Carlo):")
    print("Strike | T=0.5   | T=1.0    | T=3.0")
    print("----------------------------------")
    
    for K in strikes:
        prices = []
        for t in t_values:
            price = mc_option_price(nig, S0, K, r, t)
            prices.append(price)
        
        print(f"{K:.1f}   | {prices[0]:.6f} | {prices[1]:.6f} | {prices[2]:.6f}")

# Example 4: Stochastic volatility model
def test_stochastic_volatility():
    print("\nTesting Stochastic Volatility model...")
    
    # Simplified CIR process for time change
    class SimplifiedCIR:
        def __init__(self, kappa, theta, sigma):
            self.kappa = kappa
            self.theta = theta
            self.sigma = sigma
        
        def cumulants(self, n, t=1):
            # Simplified implementation - only first cumulant
            # In reality, CIR cumulants are more complex
            kappa = [0] * max(n, 2)  # Ensure at least 2 cumulants
            kappa[0] = self.theta * t  # Mean of integrated CIR
            kappa[1] = self.theta * t * (1 + self.sigma**2 / (2 * self.kappa))  # Approx variance
            return kappa
        
        def simulate(self, t, n_paths=1):
            # Simplified simulation of integrated CIR process
            # This is a very rough approximation
            mean = self.theta * t
            variance = self.theta * t * (1 + self.sigma**2 / (2 * self.kappa))
            return np.random.gamma(mean**2/variance, variance/mean, n_paths)
    
    # Parameters from Carr et al. (2003)
    kappa = 1.63
    theta = 22.89
    sigma = 10.83
    
    # CGMY parameters
    C_plus = 37.66
    C_minus = 7.615
    G = 19.78
    M = 192.20
    Y_plus = 0.2893
    Y_minus = 0.3291
    
    # Create processes
    cir = SimplifiedCIR(kappa, theta, sigma)
    cgmy = CGMY(C_plus, C_minus, G, M, Y_plus, Y_minus)
    
    # Create stochastic volatility model
    sv_model = StochasticVolatility(cgmy, cir)
    
    # Compute moments
    t = 1.0
    n_moments = 4
    moments = sv_model.moments(n_moments, t)
    
    print(f"Stochastic Volatility Moments at t={t}:")
    for i, mom in enumerate(moments, 1):
        print(f"E[Z({t})^{i}] = {mom:.6f}")
    
    # Simulate paths
    n_paths = 10000
    simulations = sv_model.simulate(t, n_paths)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(simulations, bins=50, density=True, alpha=0.7, label='SV Simulations')
    
    # For comparison, simulate CGMY with constant time change = E[Y(t)]
    const_time = cir.cumulants(1, t)[0]
    deterministic_simulations = cgmy.simulate(const_time, n_paths)
    
    plt.hist(deterministic_simulations, bins=50, density=True, alpha=0.4, 
             label=f'CGMY with constant time {const_time:.2f}')
    
    plt.title(f'Stochastic Volatility vs Deterministic Time at t={t}')
    plt.xlabel('Log-Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Option pricing
    S0 = 1.0
    r = 0.03
    strikes = [0.8, 1.0, 1.2]
    
    # Monte Carlo pricing
    def mc_option_price(simulations, S0, K, r, t):
        payoffs = np.maximum(S0 * np.exp(simulations) - K, 0)
        return np.exp(-r*t) * np.mean(payoffs)
    
    print("\nOption Prices (Monte Carlo):")
    print("Strike | SV Model | Deterministic Time")
    print("-------------------------------------")
    
    for K in strikes:
        price_sv = mc_option_price(simulations, S0, K, r, t)
        price_det = mc_option_price(deterministic_simulations, S0, K, r, t)
        
        print(f"{K:.1f}   | {price_sv:.6f} | {price_det:.6f}")

if __name__ == "__main__":
    test_black_scholes()
    test_regime_switching_bs()
    test_nig_gc()
    test_stochastic_volatility()