import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_5\datasets/TrainExer45.txt", sep=",")

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
    SSR = e.T @ e                                               # Sum of squared residuals
    SST = y.var() * (n - 1)                                     # Total sum of squares
    R_sq = 1 - (SSR / SST)                                      # R squared

    return beta, s_beta, t_values, R_sq

def two_sls(y, X, Z):                                           # Enter dependent variable, model regressor matrix, and instruments

    n, k = X.shape[0], X.shape[1] 
    Hz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T                       # Hz matrix to generate fitted X values
    X_hat = Hz @ X                                              # Obtain fitted values of endogenous variables
    b2sls = mult_regr(y, X_hat)[0]                              # Regress y on X_hat
    e2sls = y - X @ b2sls                                       # Residuals should be in terms of the real X variables
    s_sq = (e2sls.T @ e2sls) / (n - k)
    s = np.sqrt(s_sq)  
    XtHzX_inv = np.linalg.inv(X.T @ Hz @ X) 
    var_b2sls = s_sq * XtHzX_inv                                # Variance of b2sls is σ^2 * (X'HzZ)^-1                
    s_b2sls = np.sqrt(np.diag(var_b2sls))
    t_values = b2sls / s_b2sls                                  # t-statistics vector
    SSR = e2sls.T @ e2sls                                       # Sum of squared residuals
    SST = y.var() * (n - 1)                                     # Total sum of squares
    R_sq = 1 - (SSR / SST)                                      # Model fit

    return b2sls, s_b2sls, t_values, R_sq

def Sargan_test(y, X, Z, b_2sls):
    n, k, m = X.shape[0], X.shape[1], Z.shape[1]                # Get sample size, number of explanatory variables, and number of instruments
    e_2sls = y - X @ b_2sls                                     # Estimate the 2SLS residuals
    R_sq = mult_regr(e_2sls, Z)[1]                              # Get model fit of regressing the residuals to Z
    Sargan = n * R_sq                                           # Sargan statistic
    p = 1 - chi2.cdf(Sargan, df=m-k)
    return Sargan, p, R_sq

# Define regressor matrix for the general model with gender and participation as explanatory variables
unit_vector = np.ones(len(df))                                  # unit vector
Genderi = df["GENDER"]                                          # gender dummy vector
Participationi = df["PARTICIPATION"]                            # participation dummy vector
X = np.column_stack([unit_vector, Genderi, Participationi])     # Stack these vectors to form the regressor matrix X

# Define dependent variable 
y = df["GPA"]                                                   # Grade Point Average vector

OLS = mult_regr(y, X)

print(
    f"\nRegressing GPA on a constant, the gender dummy, and the participation dummy yields:\n"
    f"\nConstant ->\t\tb = {np.round(OLS[0][0], 3)}\tSE = {np.round(OLS[1][0], 3)}\tt-statistic = {np.round(OLS[2][0], 3)}\n"
    f"\nGender ->\t\tb = {np.round(OLS[0][1], 3)}\tSE = {np.round(OLS[1][1], 3)}\tt-statistic = {np.round(OLS[2][1], 3)}\n"
    f"\nParticipation ->\tb = {np.round(OLS[0][2], 3)}\tSE = {np.round(OLS[1][2], 3)}\tt-statistic = {np.round(OLS[2][2], 3)}\n"
    f"\nR-squared = {OLS[3]:.3f}\n"
    f"\nSame results as mentioned in the lecture were obtained\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Define first stage regressor matrix
unit_vector = np.ones(len(df))                                  # unit vector
Genderi = df["GENDER"]                                          # gender dummy vector
Emaili = df["EMAIL"]                                            # email dummy vector
X_stg1 = np.column_stack([unit_vector, Genderi, Emaili])        # Stack these vectors to form the regressor matrix X_stg1

# Define dependent variable for the first stage regression
y_stg1 = df["PARTICIPATION"]                                    # participation dummy vector

# Perform first stage regression
stg1_rgr = mult_regr(y_stg1, X_stg1)                                     

print(
    f"\nThe first stage regression, in which we explain Participation using all instruments, yields:\n"
    f"\nConstant ->\t\tb = {np.round(stg1_rgr[0][0], 3)}\tSE = {np.round(stg1_rgr[1][0], 3)}\tt-statistic = {np.round(stg1_rgr[2][0], 3)}\n"
    f"\nGender ->\t\tb = {np.round(stg1_rgr[0][1], 3)}\tSE = {np.round(stg1_rgr[1][1], 3)}\tt-statistic = {np.round(stg1_rgr[2][1], 3)}\n"
    f"\nParticipation ->\tb = {np.round(stg1_rgr[0][2], 3)}\tSE = {np.round(stg1_rgr[1][2], 3)}\tt-statistic = {np.round(stg1_rgr[2][2], 3)}\n"
    f"\nR-squared = {stg1_rgr[3]:.3f}\n"
    f"\nSame results as mentioned in the lecture were obtained\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Define second stage regressor matrix
unit_vector = np.ones(len(df))                                          # unit vector
Genderi = df["GENDER"]                                                  # gender dummy vector
γ = stg1_rgr[0]                                                         # obtain stage 1 coefficients
Participationi_hat = X_stg1 @ γ                                         # get predicted values of participation
X_stg2 = np.column_stack([unit_vector, Genderi, Participationi_hat])    # Stack these vectors to form the regressor matrix X for the second stage regression

# Perform second stage regression
stg2_rgr = mult_regr(y, X_stg2)

print(
    f"\nThe second stage regression, in which we explain GPA using the constant, the gender dummy, and the email dummy, yields:\n"
    f"\nConstant ->\t\tb = {np.round(stg2_rgr[0][0], 3)}\tSE = {np.round(stg2_rgr[1][0], 3)}\tt-statistic = {np.round(stg2_rgr[2][0], 3)}\n"
    f"\nGender ->\t\tb = {np.round(stg2_rgr[0][1], 3)}\tSE = {np.round(stg2_rgr[1][1], 3)}\tt-statistic = {np.round(stg2_rgr[2][1], 3)}\n"
    f"\nParticipation ->\tb = {np.round(stg2_rgr[0][2], 3)}\tSE = {np.round(stg2_rgr[1][2], 3)}\tt-statistic = {np.round(stg2_rgr[2][2], 3)}\n"
    f"\nR-squared = {stg2_rgr[3]:.3f}\n"
    f"\nWhile the parameters obtained are the same as reported in the lecture, there is a mismatch in the standard errors.\n"
    f"\nThis happens because the residuals should be in terms of the real X variables, not the variables used in the second stage regression.\n"
    f"\n-----------------------------------------------------------------------\n"
)