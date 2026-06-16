import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_4\datasets/TrainExer33.txt", sep=r"\s+")

# Data transformations
df["dlog_index"] = np.log(df["Index"]).diff()           # Calculate the change of log(Index)
df["BM_squared"] = df["BookMarket"]**2                  # Calculate the square of BookMarket ratio
df["Dummy_1980"] = np.where(df["Year"] >= 1980, 1, 0)   # Dummy that is 1 if year >= 1980 and 0 otherwise  
df = df.dropna()                                        # The first datapoint has no previous value to calculate the diff with, so we drop it from the dataset

# OLS estimator function
def ols_est(y, X):

    n, k = X.shape[0], X.shape[1]                               # Storing model's sample size and number of explanatory variables
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y                                    # OLS formula
    e = y - X @ beta                                            # Residual vector
    s_sq = (e.T @ e) / (n - k)
    s = np.sqrt(s_sq)                                           # Error variance estimator
    var_beta = s_sq * XtX_inv                                   # Variance-covariance matrix of beta_hat
    s_beta = np.sqrt(np.diag(var_beta))                         # Standard error of beta is the square root of the variance-covariance matrix diagonal
    t_values = beta / s_beta                                    # t-statistics vector
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=n-k))  # p-values vector

    return beta, t_values, p_values

# Define regressor matrix for question (b)
unit_vector = np.ones(len(df))                      # (86 x 1) unit vector
BMi = df["BookMarket"]                              # (86 x 1) book-market ratio observations vector
BMi_sq = df["BM_squared"]                           # (86 x 1) book-market ratio squared observations vector
X = np.column_stack([unit_vector, BMi, BMi_sq])     # Stack these vectors to form the (86 x 3) regressor matrix X

# Define dependent variable (change in log S&P500 index)
y = df["dlog_index"]

# Use ordinary least squares to compute beta
ols_beta = ols_est(y, X)

print(f"\nOLS estimator vector b = {ols_beta[0]}.\n\nThe t-statistic for coefficient b3 of the BMi squared is t_b3 = {ols_beta[1][2]:.3f}, with a p-value of p_3 = {ols_beta[2][2]:.3f}.\n\nSince p = 0.107 > 0.05 (and |t_b3| = 1.63 < 2), we fail to reject H0 (β3 = 0) at the 5% significance level.\n\nTherefore, there is no statistically significant evidence that the relationship between the index and BM is quadratic.")

# Define regressor matrix for question (c)
BMi_Di = df["BookMarket"] * df["Dummy_1980"]        # (86 x 1) interaction between the book-to-market ratio and the post-1980 dummy vector
X_2 = np.column_stack([unit_vector, BMi, BMi_Di])     # Stack these vectors to form the (86 x 3) regressor matrix X_2

# Use ordinary least squares to compute beta
ols_beta_2 = ols_est(y, X_2)

print(f"\nOLS estimator vector b_2 = {ols_beta_2[0]}.\n\nThe t-statistic for coefficient b3 of the BMi * Di interaction is t_b3 = {ols_beta_2[1][2]:.3f}, with a p-value of p_3 = {ols_beta_2[2][2]:.3f}.\n\nSince p = 0.575 >> 0.05 (and |t_b3| = 0.563 < 2), we fail to reject H0 (β3 = 0) at the 5% significance level.\n\nTherefore, there is no statistically significant evidence that the relationship between the index and BM changed over the pre- and post-1980 period.\n")