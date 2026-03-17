import numpy, scipy
import cvxpy as cp
from pandas import Series
import datetime 
#import factor_optimiser
import numpy
import math

def compute_optimal_portfolio(excess_return_vector, covariance_matrix, tracking_error, lower_bounds = None, upper_bounds = None, set_upper_bounds_to_one = True, max_leverage = 1, allow_short = False,no_benchmark = False):
	dim = len(excess_return_vector)
	def historic_return(x):
		return -1*numpy.dot(excess_return_vector, x).item()

	x = cp.Variable(dim)
	constrains = []
	for i in range(0,dim):
		lower = 0
		upper = 0
		if (not lower_bounds is None) and (len(lower_bounds) == dim):
			lower= lower_bounds[i]
		if (not lower_bounds is None) and (len(upper_bounds) == dim):
			upper = upper_bounds[i]
		elif set_upper_bounds_to_one:
			upper = 1
		if(upper!=lower):
			e = [float(j==i) for j in range(0,dim)]
			constrains.append(e@x>=lower)
			constrains.append(e@x<=upper)

	es = [1 for j in range(0,dim)]
	if not no_benchmark:
		constrains.append(es@x<=1)
		constrains.append(cp.quad_form(x, covariance_matrix)<=tracking_error**2)
	else:
		constrains.append(es@x==1)

	opt = cp.Problem(cp.Minimize(-1*(excess_return_vector@x)), constrains)
	opt.solve()
	max_return = -1*opt.value


	constrains.append(x@excess_return_vector==max_return)
	opt2 = cp.Problem(cp.Minimize(cp.quad_form(x, covariance_matrix)), constrains)
	opt2.solve()
	return x.value
