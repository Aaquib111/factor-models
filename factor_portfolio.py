# useful: http://www.mim.ac.mw/books/Bodie's%20Investments,%2010th%20Edition.pdf

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import optimize
from bayes_opt import BayesianOptimization

# for one particular stock
# m days, k is num of factors
# factor_changes is kxm
# stock_changes is 1xm
# regression from factor changes to stock_changes to get alpha, betas, episilon
# return alpha, betas, and epsilon variance (betas is kx1)
def get_factor_coefficients(factor_changes, stock_changes):
    reg = LinearRegression().fit(factor_changes, stock_changes)
    
    alpha = reg.intercept_
    betas = reg.coef_

    # get remaining variance of residuals
    pred_changes = reg.predict(factor_changes)
    new_variance = np.var(np.abs(stock_changes - pred_changes))

    return alpha, betas, new_variance


# A Class to keep track of a Portfolio that optimally weights its equities
# based on given factors. The factors are assumed to be independent.
class Portfolio:

    # universe size is n, m days
    # one equity in universe should be the full market, 
    # with alpha = 0, beta_m = 1, other betas = 0, episilon variance being 0
    # k factors

    # alphas is 1xn of alpha (risk premiums) from regressions
    # betas is kxn of beta coefficients
    # eps_vars is 1xn of variances of epsilon (firm specific risk)

    # factor changes is kxm matrix of factor data points for each of m days
    # factor changes is for regression 
    # factor_returns is kx1 of expected return for each factor, e.g. expected market return
    # factor variances is kx1 of expected variance for each factor
    def __init__(self, alphas, betas, eps_vars, 
    factor_changes, factor_returns = None, factor_variances = None):
        self.alphas = alphas
        self.betas = betas
        self.eps_vars = eps_vars
        
        self.factor_changes = factor_changes
        
        # calculate expected returns if factor_returns is not provided
        if factor_returns is None:
            # factor return is expected factor, row wise average
            self.factor_returns = factor_changes.mean(axis=1)
        else:
            self.factor_returns = factor_returns

        # calculate variance if factor_variances not provided
        if factor_variances is None:
            # factor variance is variance of each row, each factor
            self.factor_variances = factor_changes.var(axis=1)
        else:
            self.factor_variances = factor_variances

    
    # add a stock to the portfolio, calculate alpha, betas, epsilon
    # stock changes is 1xm for that particular stock, m days
    def add_stock(self, stock_changes):
        reg = LinearRegression().fit(self.factor_changes, stock_changes)
        
        alpha = reg.intercept_
        betas = reg.coef_

        # get remaining variance of residuals
        pred_changes = reg.predict(self.factor_changes)
        new_variance = np.var(np.abs(stock_changes - pred_changes))

        # add the new alpha to the alphas 1xn array
        self.alphas = np.hstack([self.alphas, alpha])

        # add the new betas kx1 to kxn of beta coefficients
        self.betas = np.hstack([self.betas, betas])

        self.eps_vars = np.hstack([self.eps_vars, new_variance])

    # weights is 1xn of relative weights for each equity, every weight positive
    # weights should sum up to 1
    def update_weights(self, weights=None):
        # if weights sum is 0 (all weights 0), then just make it balanced
        if weights is None or np.sum(weights) == 0:
            weights = np.ones_like(weights)
        self.weights = weights / np.sum(weights)
    
    # get portfolio alpha, scalar (maybe 1x1)
    def get_alpha(self):
        # weighted sum
        # return np.dot(self.alphas, self.weights)
        return self.weights @ np.transpose(self.alphas)
    
    # get portfolio betas, kx1 vector
    def get_betas(self):
        # return np.dot(self.betas, self.weights)
        return self.weights @ np.transpose(self.betas)
    
    # get portfolio specific variance, scalar (maybe 1x1)
    def get_eps_vars(self):
        # return np.dot(np.square(self.weights), self.eps_vars)
        return np.square(self.weights) @ np.transpose(self.eps_vars)

    # calculate sharpe ratios 
    def expected_return(self):
        # weighted portfolio alpha
        alpha = self.get_alpha

        # weighted portfolio betas, kx1 for k factors
        betas = self.get_betas
        # return alpha + np.dot(betas, self.factor_returns)
        return alpha + np.transpose(self.factor_returns) @ betas
    
    # assume all market factors are independent
    def portfolio_variance(self):
        # weighted portfolio betas, kx1 for k factors
        betas = self.get_betas
        # var_term_1 = np.dot(np.square(betas), self.factor_variances)
        var_term_1 = np.transpose(np.square(betas)) @ self.factor_variances
        return var_term_1 + self.get_eps_vars
    
    def get_sharpe(self):
        expected_return = self.expected_return()
        stdev = np.sqrt(self.portfolio_variance())
        return expected_return/stdev


    # get the explicit solution to optimal weights for Sharpe
    # for now, only exists for k=1 factor
    # def optimal_weights_explicit(self):



    # optimize the weights
    def optimize_weights_numerical(self):
        self.update_weights()
        optimal_weights = np.ones_like(self.weights)
        # optimize optimal_weights to maximize output of optimize_sharpe_weights
        

    # helper function to help optimize weights
        # weights is 1xn for every equity, n universe size
        def optimize_sharpe_weights(*args):
            weights = []
            for arg in args:
                weights.append(arg)
            # update weights, method automatically balances weights
            self.update_weights(np.array(weights))
            # cost function is sharpe, try to maximize sharpe
            return self.get_sharpe()

        # Bounded region of parameter space
        pbounds = {'x': (2, 4), 'y': (-3, 3)}

        optimizer = BayesianOptimization(
            f=self.optimize_sharpe_weights,
            pbounds=pbounds,
            random_state=1,
        )


