import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t, f, chi2

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_7\datasets/TrainExer64.txt", sep=r"\s+")

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
    p = 2 * (1 - t.cdf(abs(tb), df=n-2))
    
    return a, b, ei, R_sq, tb, p

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

    return beta, s_beta, t_values, R_sq, SSR

# F-test
def F_test(y, X0, X1, q):                                       # Enter dependent variable, regressor matrices of restricted and unrestricted model, and number of restrictions

    n = len(y)                                                  # Sample size
    k = X1.shape[1]                                             # Number of explanatory variables of the unrestricted model
    S0 = mult_regr(y, X0)[4]                                    # Get SSRs of restricted model
    S1 = mult_regr(y, X1)[4]                                    # Get SSRs of unrestricted model
    F = ((S0 - S1) / q) / (S1 / (n - k))                        # F-statistic
    p = 1 - f.cdf(F, dfn=q, dfd=n-k)                            # p-value

    return F, p

# Create lags for DX1 and DX2
df["DX1_L1"] = df["DX1"].shift(1)
df["DX1_L2"] = df["DX1"].shift(2)
df["DX2_L1"] = df["DX2"].shift(1)
df["DX2_L2"] = df["DX2"].shift(2)
df_a = df.copy()
df_a = df_a.dropna()                                            # We lose 2 observations to create the lagged values      

# Define regressor matrices for Granger causality tests
unit_vector = np.ones(len(df_a))
M1 = np.column_stack([unit_vector, df_a["DX1_L1"], df_a["DX1_L2"], df_a["DX2_L1"], df_a["DX2_L2"]])
M2 = np.column_stack([unit_vector, df_a["DX2_L1"], df_a["DX2_L2"], df_a["DX1_L1"], df_a["DX1_L2"]])

# Define dependent variable for each test
DX1 = df_a["DX1"]
DX2 = df_a["DX2"]

# Run regressions
OLS_ADL_1 = mult_regr(DX1, M1)
OLS_ADL_2 = mult_regr(DX2, M2)

# Define regressor matrices of restricted models for each test
M1r = np.column_stack([unit_vector, df_a["DX1_L1"], df_a["DX1_L2"]])
M2r = np.column_stack([unit_vector, df_a["DX2_L1"], df_a["DX2_L2"]])

# F-tests for Granger causality
FGr1 = F_test(DX1, M1r, M1, 2)
FGr2 = F_test(DX2, M2r, M2, 2)

print(
    f"\nThe F-test for the Granger causality of ΔX2 for ΔX1 yields:\n"
    f"\nF-statistic = {np.round(FGr1[0], 3)}, with a p-value = {np.round(FGr1[1], 3)}.\n"
    f"\nTherefore, we fail to reject the null hypothesis that γ = 0, meaning that ΔX2 does not help predict ΔX1.\n"
    f"\nThe F-test for the Granger causality of ΔX1 for ΔX2 yields:\n"
    f"\nF-statistic = {np.round(FGr2[0], 3)}, with a p-value = {np.round(FGr2[1], 3)}.\n"
    f"\nTherefore, we reject the null hypothesis that γ = 0, meaning that ΔX1 helps predict ΔX2.\n" 
    f"\nWe conclude that ΔX1 is Granger causal for ΔX2.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Create lags for X1 and X2
df_b = df.copy()
df_b = df_b.drop(columns=["DX1_L2", "DX2_L2"])
df_b["X1_L1"] = df["X1"].shift(1)
df_b["X2_L1"] = df["X2"].shift(1)
df_b = df_b.dropna() 
df_b["trend"] = np.arange(1, len(df_b) + 1)

# Define regressor matrices for ADF tests
unit_vector = np.ones(len(df_b))
M_adf1 = np.column_stack([unit_vector, df_b["trend"], df_b["X1_L1"], df_b["DX1_L1"]])
M_adf2 = np.column_stack([unit_vector, df_b["trend"], df_b["X2_L1"], df_b["DX2_L1"]])

# Define dependent variable for each ADF test
DX1_adf = df_b["DX1"]
DX2_adf = df_b["DX2"]

# Run regressions
OLS_ADF_1 = mult_regr(DX1_adf, M_adf1)
OLS_ADF_2 = mult_regr(DX2_adf, M_adf2)

print(
    f"\nThe ADF-test for X1 yields a coefficient ρ_hat = {np.round(OLS_ADF_1[0][2], 3)}, with an SE = {np.round(OLS_ADF_1[1][2], 3)} and a t-value = {np.round(OLS_ADF_1[2][2], 3)}.\n"
    f"\nTherefore, since tρ_hat > -3.5, we fail to reject the null of non-stationarity for X1.\n"
    f"\nThe ADF-test for X2 yields a coefficient ρ_hat = {np.round(OLS_ADF_2[0][2], 3)}, with an SE = {np.round(OLS_ADF_2[1][2], 3)} and a t-value = {np.round(OLS_ADF_2[2][2], 3)}.\n"
    f"\nTherefore, since tρ_hat > -3.5, we also fail to reject the null of non-stationarity for X2.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Run regression for the first step Engle-Granger test
OLS = simple_regr(df["X2"], df["X1"])

# Create difference and lags for residuals  
df_c = pd.DataFrame(OLS[2], columns=["EPS"])
df_c["EPS_L1"] = df_c["EPS"].shift(1)
df_c["DEPS"] = df_c["EPS"].diff()
df_c["DEPS_L1"] = df_c["DEPS"].shift(1)
df_c = df_c.dropna()

# Define regressor matrix for the 2nd-step Engle-Granger test
unit_vector = np.ones(len(df_c))
Xeg = np.column_stack([unit_vector, df_c["EPS_L1"], df_c["DEPS_L1"]])

# Run regression
OLS_eg = mult_regr(df_c["DEPS"], Xeg)

print(
    f"\nThe 2nd-step Engle-Granger test yields a coefficient ρ_hat = {np.round(OLS_eg[0][1], 3)}, with an SE = {np.round(OLS_eg[1][1], 3)} and a t-value = {np.round(OLS_eg[2][1], 3)}.\n"
    f"\nTherefore, since tρ_hat < -3.4, we reject the null of non-stationarity for the residuals, which means that X1 and X2 are cointegrated.\n"
    f"\nThe error correction term can therefore be written as X2_L1 - 0.92 * X1_L1.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Define ECT and regression matrices for ECMs
df_d = df_b.copy()
df_d["ECT_L1"] = df_d["X2_L1"] - 0.92 * df_d["X1_L1"]
unit_vector = np.ones(len(df_d))
M_ecm1 = np.column_stack([unit_vector, df_d["DX1_L1"], df_d["ECT_L1"]])
M_ecm2 = np.column_stack([unit_vector, df_d["DX2_L1"], df_d["ECT_L1"]])

# Run ECM regressions
OLS_ecm1 =  mult_regr(df_d["DX1"], M_ecm1)
OLS_ecm2 =  mult_regr(df_d["DX2"], M_ecm2)

print(
    f"\nThe ECM for X1 yields:\n"
    f"\nconstant ->\tβ = {np.round(OLS_ecm1[0][0], 3)}\tSE = {np.round(OLS_ecm1[1][0], 3)}\tt-value = {np.round(OLS_ecm1[2][0], 3)}.\n"
    f"\nΔX1 ->\t\tβ = {np.round(OLS_ecm1[0][1], 3)}\tSE = {np.round(OLS_ecm1[1][1], 3)}\tt-value = {np.round(OLS_ecm1[2][1], 3)}.\n"
    f"\nECT_L1 ->\tβ = {np.round(OLS_ecm1[0][2], 3)}\tSE = {np.round(OLS_ecm1[1][2], 3)}\tt-value = {np.round(OLS_ecm1[2][2], 3)}.\n"
    f"\nThe ECM for X2 yields:\n"
    f"\nconstant ->\tβ = {np.round(OLS_ecm2[0][0], 3)}\tSE = {np.round(OLS_ecm2[1][0], 3)}\tt-value = {np.round(OLS_ecm2[2][0], 3)}.\n"
    f"\nΔX1 ->\t\tβ = {np.round(OLS_ecm2[0][1], 3)}\tSE = {np.round(OLS_ecm2[1][1], 3)}\tt-value = {np.round(OLS_ecm2[2][1], 3)}.\n"
    f"\nECT_L1 ->\tβ = {np.round(OLS_ecm2[0][2], 3)}\tSE = {np.round(OLS_ecm2[1][2], 3)}\tt-value = {np.round(OLS_ecm2[2][2], 3)}.\n"
    f"\nInterpretation: The error correction term, ECT_L1 = X2_L1 - 0.92 * X1_L1, measures the deviation from the long-run equilibrium relationship between"
    f"\nthe two series. A positive value of the error correction term indicates that X2 is above its equilibrium value relative to X1. In the ECM for X1,"
    f"\nthe positive and statistically significant coefficient of the ECT (0.463) implies that X1 increases in the following period, reducing the disequilibrium."
    f"\nIn the ECM for X2, the negative and statistically significant coefficient (-0.443) implies that X2 decreases in the following period, again reducing the"
    f"\ndisequilibrium. Therefore, both variables participate in restoring the long-run equilibrium after temporary deviations. Furthermore, the lagged difference"
    f"\nof X1 is significant, indicating that the short-term dynamics of X1 also depend on its own past growth, whereas the lagged difference of X2 is insignificant,"
    f"\nsuggesting that the short-term adjustment of X2 is driven mainly by the error correction mechanism rather than by its own past growth."
    f"\n-----------------------------------------------------------------------\n"
)