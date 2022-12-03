# first, do this with index model, only one beta
import numpy as np
from sklearn.linear_model import LinearRegression


# for one particular stock
# m days, k is num of factors
# factor_changes is kxm
# stock_changes is 1xm
# fit factor changes to stock_changes to get alpha, betas, episilon
# return alpha, betas, and epsilon variance (betas is kx1)
def get_factor_coefficients(factor_changes, stock_changes):
    reg = LinearRegression().fit(factor_changes, stock_changes)
    
    alpha = reg.intercept_
    betas = reg.coef_

    # get remaining variance of residuals
    pred_changes = reg.predict(factor_changes)
    new_variance = np.var(np.abs(stock_changes - pred_changes))

    return alpha, betas, new_variance

class Portfolio:

    # universe size is n
    # alphas is 1xn of alpha (risk premiums) from regressions
    # betas is kxn of beta coefficients
    # eps_vars is 1xn of variances of epsilon (firm specific risk)
    def __init__(self, alphas, betas, eps_vars):
        self.alphas = alphas
        self.betas = betas
        self.eps_vars = eps_vars
    
    # weights is 1xn of relative weights for each equity
    def update_weights(self, weights):
        self.weights = weights
    
    # get portfolio alpha
    def get_alpha(self):
        return np.dot(self.alphas, self.weights)
    
    # get portfolio betas, kx1 vector
    # add one for fully diversified market index, which has beta of 1
    def get_betas(self):
        return np.dot(self.betas, self.weights) + 1
    
    # get portfolio specific variance
    def get_eps_vars(self):
        return np.dot(np.square(self.weights), self.eps_vars)

    # calculate sharpe ratios
    # parameter: E(R_m), expected return of factor corresponding to Beta_m, kx1 vector
    def expected_return(self, expected_beta_returns):
        alpha = self.get_alpha
        betas = self.get_betas
        return alpha + np.dot(betas, expected_beta_returns)
    
    # assume all market factors are independent
    # factor_variances is kx1 vector
    def portfolio_variance(self, factor_variances):
        betas = self.get_betas
        var_term_1 = np.dot(np.square(betas), factor_variances)
        return var_term_1 + self.get_eps_vars
    
    def get_sharpe(self, expected_beta_returns, factor_variances):
        expected_return = self.expected_return(expected_beta_returns)
        stdev = np.sqrt(self.portfolio_variance(factor_variances))
        return expected_return/stdev



