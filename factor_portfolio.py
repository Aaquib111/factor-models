# first, do this with index model, only one beta
import numpy as np

class Portfolio:

    # universe size is n
    # alphas is nx1 of alpha (risk premiums) from regressions
    # betas is nxk of beta coefficients
    # eps_vars is nx1 of variances of epsilon (firm specific risk)
    def __init__(self, alphas, betas, eps_vars):
        self.alphas = alphas
        self.betas = betas
        self.eps_vars = eps_vars
    
    # weights is nx1 of relative weights for each equity
    def update_weights(self, weights):
        self.weights = weights
    
    # get portfolio alpha
    def get_alpha(self):
        return np.dot(self.alphas, self.weights)
    
    # get portfolio betas, 1xk vector
    # add one for fully diversified market index, which has beta of 1
    def get_betas(self):
        return np.dot(self.betas, self.weights) + 1
    
    # get portfolio specific variance
    def get_eps_vars(self):
        return np.dot(np.square(self.weights), self.eps_vars)

    # calculate sharpe ratios
    # parameter: E(R_m), expected return of factor corresponding to Beta_m, 1xk vector
    def expected_return(self, expected_beta_returns):
        alpha = self.get_alpha
        betas = self.get_betas
        return alpha + np.dot(betas, expected_beta_returns)
    
    # assume all market factors are independent
    # factor_variances is 1xk vector
    def portfolio_variance(self, factor_variances):
        betas = self.get_betas
        var_term_1 = np.dot(np.square(betas), factor_variances)
        return var_term_1 + self.get_eps_vars
    
    def get_sharpe(self, expected_beta_returns, factor_variances):
        expected_return = self.expected_return(expected_beta_returns)
        stdev = np.sqrt(self.portfolio_variance(factor_variances))
        return expected_return/stdev



