import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_5\datasets/TrainExer44.txt", sep=",")

# OLS estimator function for multiple regression
def mult_regr(y, X):

    n, k = X.shape[0], X.shape[1]                               # Storing model's sample size and number of explanatory variables
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y                                    # OLS formula
    e = y - X @ beta                                            # Residual vector
    SSR = e.T @ e                                               # Sum of squared residuals
    SST = y.var() * (n - 1)                                     # Total sum of squares
    R_sq = 1 - (SSR / SST)                                      # R squared

    return beta, R_sq, e

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
    
print(
    f"\nGasoline price may be endogenous because unobserved demand shocks (e.g. favorable weather or increased travel activity) affect gasoline consumption\n"
    f"and simultaneously influence market prices. These omitted factors are contained in the error term, implying Cov(PG, ε) != 0.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Define regressor matrix of model
unit_vector = np.ones(len(df))                                  # unit vector
PGi = df["PG"]                                                  # log of real gasoline price index vector
RIi = df["RI"]                                                  # log of real disposable income vector
X = np.column_stack([unit_vector, PGi, RIi])                    # Stack these vectors to form the regressor matrix X

# Define instrument matrix 
unit_vector = np.ones(len(df))                                  # unit vector
RIi = df["RI"]                                                  # log of real disposable income vector
RPTi = df["RPT"]                                                # log of real price index of public transport vector
RPNi = df["RPN"]                                                # log of real price index of new cars vector
RPUi = df["RPU"]                                                # log of real price index of used cars vector
Z = np.column_stack([unit_vector, RIi, RPTi, RPNi, RPUi])       # Stack these vectors to form the instrument matrix Z

# Define dependent variable
y = df["GC"]                                                    # log of real gasoline consumption

b_2sls = two_sls(y, X, Z)                                       # Estimate 2SLS 

print(
    f"\nAfter applying the 2SLS method, the estimated price elasticity is β2_hat = {np.round(b_2sls[1], 3)}\n"
    f"\nInterpretation: A 1% increase in gasoline price is associated with an estimated {np.round(b_2sls[1], 3)} % decrease in gasoline consumption.\n"
    f"\n-----------------------------------------------------------------------\n"
)

sargan = Sargan_test(y, X, Z, b_2sls)

print(
    f"\nAfter performing a Sargan test to test whether the instruments of Z are correlated with ε, we get:\n"
    f"\nR-sq = {np.round(sargan[2], 3)}, Sargan = n * R_sq = {np.round(sargan[0], 3)}, and a p-value = {np.round(sargan[1], 3)}.\n"
    f"\nSince {np.round(sargan[1], 3)} > 0.05, we do not reject the null hypothesis of no correlation between the instruments and ε.\n"
    f"\n-----------------------------------------------------------------------\n"
)