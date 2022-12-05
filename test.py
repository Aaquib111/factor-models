import numpy as np
from factor_portfolio import Portfolio

def test_factor_portfolio():
    

# 2 factors, 4 days
factor_changes = np.array([[5, 10, -3, 8], [9, -12, -3, -6]])
# print(factor_changes.shape)
# print(factor_changes.mean(axis=1))

# 3 stocks, 2 factors -> 2x3
betas = np.array([[5, 10, 5], [9, -12, -3]])
# 3 stocks, 1x3
weights = np.array([[1, 2, 3]])

alphas = np.array([[3, 4, 5]])

# print(np.dot(weights, np.transpose(betas)))
# print(np.dot(weights, np.transpose(alphas)))
# print(np.dot(weights, np.transpose(alphas)) + 4)