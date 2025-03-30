"""
This module performs a maximum likelihood estimation on the data up to time T
by using the Score-Driven AR(1) model from 

[3] "Optimal formulations for nonlinear autoregressive processes" 
by Blasques,Koopman,Lucas.
"""

from scipy.optimize import minimize
import numpy as np

# Compute the log likelihood of the data given the parameters in the gaussian case
def neg_log_likelihood(params, y, d):
    epsilon = 1e-8
    omega, alpha, beta, f_0, sigma2 = params

    T = len(y)
    f_hat = np.zeros(T)
    f_hat[0] = f_0
    y_pred = np.zeros(T)
    log_likelihood = np.zeros(T)

    for t in range(1, T):
        f_hat[t] = update_function(y[t-1], y[t-2], f_hat[t-1], d, omega=omega, alpha=alpha, beta=beta, sigma2=sigma2)
        y_pred[t] = f_hat[t] * y[t-1]
        log_likelihood[t] = -0.5 * np.log(2 * np.pi * sigma2 + epsilon) - 0.5 * (y_pred[t] - y[t])**2 / (sigma2 + epsilon)
    
    neg_log_likelihood = -np.sum(log_likelihood)

    return neg_log_likelihood/T


#-------------------------------- UPDATE FUNCTIONS ------------------------------------

def update_function(y_t1, y_t2, f_t1, d, omega, alpha, beta, sigma2):    
    epsilon = 1e-8 
    
    if d == 0:
        f_t = omega + alpha * (y_t2 / (sigma2 + epsilon)) * (y_t1 - f_t1 * y_t2) + beta * f_t1
    if d == 1/2:
        f_t = omega + alpha * np.sign(y_t2) * (y_t1 - f_t1 * y_t2) / (np.sqrt(sigma2 )+ epsilon) + beta * f_t1

    return f_t

#-------------------------------------------------------------------------------------------

def optimized_params(y, d, initial_params = [0, 0.1, 0.9, 0.2, 1]):

    bounds = [(-np.inf, np.inf), (0.0001, 1), (0, 1), (-np.inf, np.inf), (0, np.inf)]  # Adjust bounds as needed
    
    result = minimize(lambda params: neg_log_likelihood(params, y, d), initial_params, bounds=bounds)

    
    return result.x

def model_estimation(data, d, parameters):
    data = np.array(data)
    omega, alpha, beta, f_0, sigma2 = optimized_params(data, d, parameters)
    T = data.shape[0]
    
    f_hat = np.zeros(T)
    for t in range(T):
        f_hat[t] = update_function(data[t-1], data[t-2], f_hat[t-1], d, omega, alpha, beta, sigma2)

    return f_hat[-1],[omega, alpha, beta, f_0,sigma2]

