import numpy as np
from scipy.optimize import minimize

# Define the objective function to be maximized
def objective(x):
    return np.sin(x) + 0.1 * x**2

# Define the majorization function
def majorization(x, x0):
    return objective(x0) + (x - x0) * np.cos(x0) + 0.1 * (x - x0) ** 2

# Define the maximization function.
def maximization(x0):
    return minimize(lambda x: -majorization(x, x0), x0, method='BFGS').x[0]

# Set the initial point for the optimization.
x0 = 0.0

# Set the maximum number of iterations
max_iter = 1000

# Set the convergence criterion
tolerance = 1e-6

# Run the minorize-maximization algorithm
for i in range(max_iter):
    x = maximization(x0)
    if np.abs(objective(x) - objective(x0)) < tolerance:
        print('Tolerance reached.')
        break
    x0 = x

# Print the result.
print('Optimal solution: ', x)