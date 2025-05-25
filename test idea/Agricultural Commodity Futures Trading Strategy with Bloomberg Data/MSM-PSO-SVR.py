import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class MSM:
    """Markov Switching Multifractal Model"""
    
    def __init__(self, k_bar=5):
        """
        Initialize MSM model
        k_bar: number of volatility components (default=5)
        """
        self.k_bar = k_bar
        self.n_states = 2**k_bar
        self.params = None
        
    def _get_transition_probs(self, b, gamma1):
        """Calculate transition probabilities for each volatility component"""
        gamma_k = []
        for k in range(1, self.k_bar + 1):
            gamma_k.append(1 - (1 - gamma1)**(b**(k-1)))
        return np.array(gamma_k)
    
    def _get_transition_matrix(self, gamma_k, m0):
        """Construct transition matrix A"""
        n = self.n_states
        A = np.zeros((n, n))
        
        # Generate all possible states
        states = []
        for i in range(n):
            state = []
            for k in range(self.k_bar):
                if (i >> k) & 1:
                    state.append(2 - m0)
                else:
                    state.append(m0)
            states.append(state)
        
        # Fill transition matrix
        for i in range(n):
            for j in range(n):
                prob = 1.0
                for k in range(self.k_bar):
                    if states[i][k] == states[j][k]:
                        prob *= (1 - gamma_k[k])
                    else:
                        prob *= gamma_k[k] * 0.5
                A[i, j] = prob
                
        return A, states
    
    def _log_likelihood(self, params, returns):
        """Calculate log-likelihood for given parameters"""
        m0, sigma, b, gamma1 = params
        
        # Parameter constraints
        if m0 <= 0 or m0 > 2 or sigma <= 0 or b <= 1 or gamma1 <= 0 or gamma1 >= 1:
            return -np.inf
            
        gamma_k = self._get_transition_probs(b, gamma1)
        A, states = self._get_transition_matrix(gamma_k, m0)
        
        T = len(returns)
        ll = 0
        
        # Initial probability vector (uniform)
        pi = np.ones(self.n_states) / self.n_states
        
        for t in range(T):
            # Calculate likelihood for each state
            omega = np.zeros(self.n_states)
            for i in range(self.n_states):
                vol = sigma * np.prod(states[i])
                omega[i] = (1/(vol * np.sqrt(2*np.pi))) * np.exp(-0.5 * (returns[t]/vol)**2)
            
            # Update likelihood
            ll += np.log(np.dot(omega, pi))
            
            # Update probability vector
            pi = np.dot(omega * pi, A) / np.dot(omega, pi)
            
        return ll
    
    def fit(self, returns):
        """Estimate MSM parameters using MLE"""
        # Initial guess
        x0 = [1.4, np.std(returns), 5.0, 0.5]
        
        # Optimization bounds
        bounds = [(1e-6, 2), (1e-6, None), (1.001, 50), (1e-6, 0.999)]
        
        # Minimize negative log-likelihood
        result = minimize(lambda x: -self._log_likelihood(x, returns), 
                         x0, bounds=bounds, method='L-BFGS-B')
        
        self.params = result.x
        self.m0, self.sigma, self.b, self.gamma1 = self.params
        self.gamma_k = self._get_transition_probs(self.b, self.gamma1)
        self.A, self.states = self._get_transition_matrix(self.gamma_k, self.m0)
        
        return self
    
    def predict_volatility(self, returns, h=1):
        """Predict future volatility h steps ahead"""
        if self.params is None:
            raise ValueError("Model must be fitted first")
            
        T = len(returns)
        predictions = []
        
        # Initial probability vector
        pi = np.ones(self.n_states) / self.n_states
        
        for t in range(T):
            # Update probabilities based on observed return
            omega = np.zeros(self.n_states)
            for i in range(self.n_states):
                vol = self.sigma * np.prod(self.states[i])
                omega[i] = (1/(vol * np.sqrt(2*np.pi))) * np.exp(-0.5 * (returns[t]/vol)**2)
            
            pi = np.dot(omega * pi, self.A) / np.dot(omega, pi)
            
            # Predict h-step ahead volatility
            pi_h = pi
            for _ in range(h):
                pi_h = np.dot(pi_h, self.A)
            
            # Calculate expected volatility
            expected_vol2 = 0
            for i in range(self.n_states):
                vol2 = (self.sigma * np.prod(self.states[i]))**2
                expected_vol2 += vol2 * pi_h[i]
            
            predictions.append(np.sqrt(expected_vol2))
            
        return np.array(predictions)


class PSO:
    """Particle Swarm Optimization for SVR hyperparameter tuning"""
    
    def __init__(self, n_particles=20, max_iter=50, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        
    def optimize(self, objective_func, bounds, args=()):
        """
        Optimize objective function
        bounds: list of (min, max) tuples for each parameter
        """
        n_dims = len(bounds)
        
        # Initialize particles
        particles = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.n_particles, n_dims)
        )
        
        velocities = np.random.randn(self.n_particles, n_dims) * 0.1
        
        # Personal and global bests
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([objective_func(p, *args) for p in particles])
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # Optimization loop
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] + 
                                self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                self.c2 * r2 * (global_best_position - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                
                # Apply bounds
                for j in range(n_dims):
                    particles[i, j] = np.clip(particles[i, j], bounds[j][0], bounds[j][1])
                
                # Evaluate
                score = objective_func(particles[i], *args)
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i].copy()
        
        return global_best_position, global_best_score


class MSM_PSO_SVR:
    """MSM-PSO-SVR hybrid model"""
    
    def __init__(self, k_bar=5, kernel='rbf', lookback=15):
        self.msm = MSM(k_bar=k_bar)
        self.kernel = kernel
        self.lookback = lookback
        self.svr = None
        self.scaler = StandardScaler()
        self.pso = PSO()
        
    def _prepare_svr_data(self, returns, volatilities):
        """Prepare data for SVR training"""
        # Calculate absolute residuals
        residuals = np.abs(returns / volatilities)
        
        # Create feature matrix
        X, y = [], []
        for i in range(self.lookback, len(residuals)):
            X.append(residuals[i-self.lookback:i])
            y.append(residuals[i])
            
        return np.array(X), np.array(y)
    
    def _svr_objective(self, params, X_train, y_train, X_val, y_val):
        """Objective function for PSO optimization"""
        if self.kernel == 'fourier':
            # Custom Fourier kernel implementation
            q = params[0]
            svr = SVR(kernel='precomputed', C=params[1], epsilon=params[2])
            
            # Compute kernel matrices
            K_train = self._fourier_kernel(X_train, X_train, q)
            K_val = self._fourier_kernel(X_val, X_train, q)
            
            svr.fit(K_train, y_train)
            y_pred = svr.predict(K_val)
            
        elif self.kernel == 'wavelet':
            # Custom wavelet kernel implementation
            a = params[0]
            svr = SVR(kernel='precomputed', C=params[1], epsilon=params[2])
            
            # Compute kernel matrices
            K_train = self._wavelet_kernel(X_train, X_train, a)
            K_val = self._wavelet_kernel(X_val, X_train, a)
            
            svr.fit(K_train, y_train)
            y_pred = svr.predict(K_val)
            
        else:
            # Standard kernels
            if self.kernel == 'rbf':
                svr = SVR(kernel='rbf', gamma=params[0], C=params[1], epsilon=params[2])
            elif self.kernel == 'poly':
                svr = SVR(kernel='poly', degree=int(params[0]), C=params[1], epsilon=params[2])
            else:
                svr = SVR(kernel='linear', C=params[0], epsilon=params[1])
                
            svr.fit(X_train, y_train)
            y_pred = svr.predict(X_val)
        
        return mean_squared_error(y_val, y_pred)
    
    def _fourier_kernel(self, X, Y, q):
        """Fourier kernel implementation"""
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]
        K = np.zeros((n_samples_X, n_samples_Y))
        
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                k_sum = 0
                for d in range(X.shape[1]):
                    diff = X[i, d] - Y[j, d]
                    k_sum += (1 - q**2) / (2 * (1 - 2*q*np.cos(diff) + q**2))
                K[i, j] = k_sum
                
        return K
    
    def _wavelet_kernel(self, X, Y, a):
        """Morlet wavelet kernel implementation"""
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]
        K = np.zeros((n_samples_X, n_samples_Y))
        
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                k_prod = 1
                for d in range(X.shape[1]):
                    diff = X[i, d] - Y[j, d]
                    k_prod *= np.cos(1.75 * diff / a) * np.exp(-diff**2 / (2 * a**2))
                K[i, j] = k_prod
                
        return K
    
    def fit(self, returns, train_ratio=0.9):
        """Fit the MSM-PSO-SVR model"""
        n_train = int(len(returns) * train_ratio)
        train_returns = returns[:n_train]
        
        # Step 1: Fit MSM model
        print("Fitting MSM model...")
        self.msm.fit(train_returns)
        
        # Step 2: Get MSM volatility predictions
        msm_vols = self.msm.predict_volatility(train_returns)
        
        # Step 3: Prepare SVR data
        X, y = self._prepare_svr_data(train_returns, msm_vols)
        
        # Split for validation
        n_val = int(len(X) * 0.2)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        # Normalize data
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Step 4: Optimize SVR hyperparameters using PSO
        print(f"Optimizing {self.kernel} kernel SVR with PSO...")
        
        if self.kernel == 'fourier':
            bounds = [(0.1, 0.99), (0.1, 100), (0.001, 1)]
        elif self.kernel == 'wavelet':
            bounds = [(0.001, 1), (0.1, 100), (0.001, 1)]
        elif self.kernel == 'rbf':
            bounds = [(0.001, 10), (0.1, 100), (0.001, 1)]
        elif self.kernel == 'poly':
            bounds = [(2, 5), (0.1, 100), (0.001, 1)]
        else:  # linear
            bounds = [(0.1, 100), (0.001, 1)]
            
        best_params, _ = self.pso.optimize(
            self._svr_objective, 
            bounds, 
            args=(X_train, y_train, X_val, y_val)
        )
        
        # Step 5: Train final SVR with best parameters
        if self.kernel == 'fourier':
            self.svr = SVR(kernel='precomputed', C=best_params[1], epsilon=best_params[2])
            self.svr_params = {'q': best_params[0]}
            K_train = self._fourier_kernel(X_train, X_train, best_params[0])
            self.svr.fit(K_train, y_train)
            self.X_train = X_train  # Store for prediction
            
        elif self.kernel == 'wavelet':
            self.svr = SVR(kernel='precomputed', C=best_params[1], epsilon=best_params[2])
            self.svr_params = {'a': best_params[0]}
            K_train = self._wavelet_kernel(X_train, X_train, best_params[0])
            self.svr.fit(K_train, y_train)
            self.X_train = X_train  # Store for prediction
            
        else:
            if self.kernel == 'rbf':
                self.svr = SVR(kernel='rbf', gamma=best_params[0], 
                              C=best_params[1], epsilon=best_params[2])
            elif self.kernel == 'poly':
                self.svr = SVR(kernel='poly', degree=int(best_params[0]), 
                              C=best_params[1], epsilon=best_params[2])
            else:
                self.svr = SVR(kernel='linear', C=best_params[0], epsilon=best_params[1])
                
            self.svr.fit(X_train, y_train)
            
        print(f"Best parameters: {best_params}")
        
        return self
    
    def predict(self, returns):
        """Predict volatility"""
        # Get MSM volatility predictions
        msm_vols = self.msm.predict_volatility(returns)
        
        # Calculate residuals
        residuals = np.abs(returns / msm_vols)
        
        # Prepare features for SVR
        predictions = []
        
        for i in range(self.lookback, len(returns)):
            X_test = residuals[i-self.lookback:i].reshape(1, -1)
            X_test = self.scaler.transform(X_test)
            
            if self.kernel == 'fourier':
                K_test = self._fourier_kernel(X_test, self.X_train, self.svr_params['q'])
                y_pred = self.svr.predict(K_test)[0]
            elif self.kernel == 'wavelet':
                K_test = self._wavelet_kernel(X_test, self.X_train, self.svr_params['a'])
                y_pred = self.svr.predict(K_test)[0]
            else:
                y_pred = self.svr.predict(X_test)[0]
                
            # Final volatility prediction
            final_vol = msm_vols[i] * y_pred
            predictions.append(final_vol)
            
        return np.array(predictions)


def evaluate_models(returns, models, test_ratio=0.1):
    """Evaluate multiple models"""
    n_test = int(len(returns) * test_ratio)
    train_returns = returns[:-n_test]
    test_returns = returns[-n_test:]
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Fit model
        model.fit(train_returns)
        
        # Predict
        predictions = model.predict(test_returns)
        
        # Calculate actual volatility (squared returns as proxy)
        actual_vol = test_returns[model.lookback:]**2
        pred_vol = predictions**2
        
        # Evaluate
        mse = mean_squared_error(actual_vol, pred_vol)
        mae = mean_absolute_error(np.sqrt(actual_vol), np.sqrt(pred_vol))
        
        results[name] = {
            'MSE': mse * 100,  # Convert to percentage
            'MAE': mae * 100,
            'predictions': predictions
        }
        
        print(f"{name} - MSE: {mse*100:.2f}%, MAE: {mae*100:.2f}%")
        
    return results


# Example usage
if __name__ == "__main__":
    # Generate synthetic data (replace with real SPY 1-minute data)
    np.random.seed(42)
    n_samples = 10000
    
    # Simulate returns with volatility clustering
    volatility = np.zeros(n_samples)
    returns = np.zeros(n_samples)
    
    # GARCH-like volatility process
    omega = 0.00001
    alpha = 0.05
    beta = 0.94
    
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))
    returns[0] = np.random.normal(0, volatility[0])
    
    for t in range(1, n_samples):
        volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
        returns[t] = np.random.normal(0, volatility[t])
    
    # Create models
    models = {
        'MSM-PSO-FSVR': MSM_PSO_SVR(kernel='fourier'),
        'MSM-PSO-WSVR': MSM_PSO_SVR(kernel='wavelet'),
        'MSM-PSO-GSVR': MSM_PSO_SVR(kernel='rbf'),
        'MSM-PSO-PSVR': MSM_PSO_SVR(kernel='poly'),
        'MSM-PSO-LSVR': MSM_PSO_SVR(kernel='linear')
    }
    
    # Evaluate models
    results = evaluate_models(returns, models)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    for name, metrics in results.items():
        print(f"{name:15} - MSE: {metrics['MSE']:6.2f}%, MAE: {metrics['MAE']:6.2f}%")