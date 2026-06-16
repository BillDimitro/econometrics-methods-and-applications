import pandas as pd
import numpy as np
import math
from scipy import stats
from scipy.stats import f

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_4\datasets/TrainExer35.txt", sep=r"\s+")

# OLS estimator function for multiple regression
def mult_regr(y, X):

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
    SSR = e.T @ e                                               # Sum of squared residuals
    SST = y.var() * (n - 1)                                     # Total sum of squares
    R_sq = 1 - (SSR / SST)                                      # R squared

    return beta, t_values, p_values, R_sq

def simple_regr(y, x):
    
    n = len(y)                                                  # Storing model's sample size
    b = x.cov(y) / x.var()                                      # Calculate a and b
    a = y.mean() - b * x.mean()
    ei = y - a - b * x                                          # Residuals
    SSR = (ei**2).sum() 
    R_sq = 1 - (SSR / (y.var() * (n-1)))                        # R squared 
    s = np.sqrt(SSR / (n-2))                                    # Standard error of regression
    sb = np.sqrt(s**2 / (x.var() * (n-1)))                      # Statistics for beta
    tb = b / sb
    
    return a, b, ei, R_sq, tb, SSR

# Define regressor matrix for model with all 5 explanatory variables
unit_vector = np.ones(len(df))                                  # (87 x 1) unit vector
BMi = df["BookMarket"]                                          # (87 x 1) book to market ratio observations vector
ISi = df["NTIS"]                                                # (87 x 1) issued stock observations vector
DPi = df["DivPrice"]                                            # (87 x 1) dividend to price ratio observations vector
EPi = df["EarnPrice"]                                           # (87 x 1) earnings to price ratio observations vector
INFLi = df["Inflation"]                                         # (87 x 1) inflations observations vector
X = np.column_stack([unit_vector, BMi, ISi, DPi, EPi, INFLi])   # Stack these vectors to form the (87 x 6) regressor matrix X

# Define dependent variable
y = df["LogEqPrem"]

OLS_full = mult_regr(y, X)

OLS_BM = simple_regr(y, BMi)

print(f"\na) Regressing the log equity premium on a constant and all five explanatory variables gives an R-squared of {OLS_full[3]:.3f}.")

print(f"\nRegressing the log equity premium on a constant and and only book-to-market gives an R-squared of {OLS_BM[3]:.3f}.")

print(f"\nBoth the full (unrestricted) model and the restricted model contain a constant term, so we can run an F-test under the null of \nH0: βis = βdp = βep = βinfl = 0, using each model's R-squared values.")

# F-test
R0 = OLS_BM[3]                                      # R-squared of restricted model
R1 = OLS_full[3]                                    # R-squared of unrestricted model
q = 4                                               # Number of restrictions
n = len(df)                                         # Sample size
Ku = 6                                              # Number of explanatory variables of the unrestricted model
F = ((R1 - R0) / q) / ((1 - R1) / (n - Ku))         # F-statistic
p_value = 1 - f.cdf(F, dfn=q, dfd=n-Ku)             # Calculate p-value of test

print(f"\nThe F-statistic is {F:.3f}, and the p-value {p_value:.3f} >> 5%, so H0: βis = βdp = βep = βinfl = 0 can not be rejected.\n\nTherefore, the four additional variables are not jointly statistically significant compared with the Book-to-Market-only model.\n\nThe simpler Book-to-Market model is preferred.")

# Define regressor matrix and b vector to compute fitted values of restricted model
X_fit = np.column_stack([unit_vector, BMi])         # Stack unit vector and B/M observations to form the (87 x 2) regressor matrix X_fit
b_fit = np.vstack([OLS_BM[0], OLS_BM[1]])           # Stack coefficients of constant and B/M ratio to form the (2x1) vector b_fit

# RESET test - compute square of fitted log equity premium values from regressing LogEqPrem with B/M only
y_fit = X_fit @ b_fit
y_fit_sq = y_fit**2

# Define regressor matrix for the RESET test with p=1
X_reset = np.column_stack([unit_vector, BMi, y_fit_sq])

OLS_reset = mult_regr(y, X_reset)

print(f"\nb) Because we only have p=1 restriction for the RESET F-test, we can perform a t-test. The test results are F = t-squared = {OLS_reset[1][2]**2:.3f}, with a p-value of {OLS_reset[2][2]:.3f}.\n\nSince 0.067 > 0.05, we do not reject H0: γ = 0. Therefore, the test provides no evidence against the linear specification of the Book-to-Market model,\nsuggesting that the model appears adequately specified at the 5% significance level.")

# Chow break test
df_pre_1980, df_post_1980 = df[df["Year"] < 1980], df[df["Year"] >= 1980]                # Split the observation dataframe in two subsamples          
BMi_pre_1980, BMi_post_1980 = df_pre_1980["BookMarket"], df_post_1980["BookMarket"]      # Book to market ratio observations for each subsample          
y_pre_1980, y_post_1980 = df_pre_1980["LogEqPrem"], df_post_1980["LogEqPrem"]            # Define dependent variable for each regression

# Run regressions of the log equity premium on a constant and the book-to-market ratio for each subsample
OLS_BM_pre_1980 = simple_regr(y_pre_1980, BMi_pre_1980)
OLS_BM_post_1980 = simple_regr(y_post_1980, BMi_post_1980)

# Get SSRs for each regression
S0 = OLS_BM[5]                          # Null hypothesis: β1 = β2 restricted model, SSRs of full sample
S1 = OLS_BM_pre_1980[5]                 # SSRs of first sample
S2 = OLS_BM_post_1980[5]                # SSRs of second sample

# F-test for Chow break
q_ch = 2                                                        # Number of restrictions equal to the number of explanatory variables
n = len(df)                                                     # Sample size
F_chow = ((S0 - S1 - S2) / q_ch) / ((S1 + S2) / (n - 2*q_ch))   # F-statistic
p_value_chow = 1 - f.cdf(F_chow, dfn=q_ch, dfd=n-2*q_ch)        # Calculate p-value of test

print(f"\nc) The F-statistic for the Chow break test is {F_chow:.3f}, and the p-value {p_value_chow:.3f} >> 5%, so H0: β1 = β2 can not be rejected.\n\nTherefore, the results suggest that the intercept and slope of the B/M model are not statistically different before and after 1980.")

# F-test for Chow forecast
n2 = len(df_post_1980)                                          # Number of restrictions equal to the number of observations of the second sample (γj)
n1 = len(df_pre_1980)                                           # First subsample size, will be needed for DOF of unrestricted model
k = 2                                                           # Number of explanatory variables
F_chow_f = ((S0 - S1) / n2) / (S1 / (n1 - k))                   # F-statistic
p_value_chow_f = 1 - f.cdf(F_chow_f, dfn=n2, dfd=n1-k)          # Calculate p-value of test

print(f"\nd) The F-statistic for the Chow forecast test is {F_chow_f:.3f}, and the p-value {p_value_chow_f:.3f} >> 5%, so H0: γ = 0 can not be rejected.\n\nTherefore, the results suggest that there is no evidence that the coefficients of the Book-to-Market model need to be adjusted when moving from the first\nsample to the second sample, suggesting parameter stability between the two periods.\n")