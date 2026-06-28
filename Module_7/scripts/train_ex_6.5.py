import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t, f, chi2
import matplotlib.pyplot as plt

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

def error_eval(y, y_pred):
    n = len(y)
    rmse = np.sqrt(((y - y_pred)**2).sum() / n)
    mae = np.abs(y - y_pred).sum() / n
    err_sum = (y - y_pred).sum()
    return rmse, mae, err_sum

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_7\datasets/TrainExer65.txt", sep=r"\s+")

# Separate test and evaluation sample
df_t = df[df["YEAR"] < 2003]
df_e = df[df["YEAR"] >= 2003]

# Time series plots
time = np.arange(len(df_t)) 

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(time, df_t["LOGIP"], label="log(IP)")
axes[0].plot(time, df_t["LOGCLI"], label="log(CLI)")
axes[0].set_ylabel("Value")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time, df_t["GIP"], label="GIP")
axes[1].plot(time, df_t["GCLI"], label="GCLI")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Value")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
#plt.show()

print(
    f"\nConclusion from plots: log(IP) and log(CLI) both display a clear upward trend, so the level series do not appear stationary. Therefore, when testing these"
    f"\nvariables in levels, it is appropriate to allow for a deterministic trend. The two level series also seem to move together over time. However, the turning"
    f"\npoints of log(CLI) often appear slightly before the turning points of log(IP). This supports the idea that CLI may act as a leading indicator for IP, and"
    f"\ntherefore motivates testing whether past values of GCLI help predict GIP. In contrast, the first differences, GIP and GCLI, fluctuate around a relatively"
    f"\nstable mean and do not show a clear time trend. Thus, from visual inspection, the growth-rate series appear approximately stationary. This conclusion"
    f"\nshould be verified later using formal unit-root tests."
    f"\n-----------------------------------------------------------------------\n"
)

# Create required lags for the ADF test
df_t_b = df_t.copy()
df_t_b["LOGIP_L1"] = df_t_b["LOGIP"].shift(1)
df_t_b["LOGCLI_L1"] = df_t_b["LOGCLI"].shift(1)
df_t_b["GIP_L1"] = df_t_b["GIP"].shift(1)
df_t_b["GIP_L2"] = df_t_b["GIP"].shift(2)
df_t_b["GCLI_L1"] = df_t_b["GCLI"].shift(1)
df_t_b["GCLI_L2"] = df_t_b["GCLI"].shift(2)
df_t_b = df_t_b.dropna()

# Define regressor matrices for ADF tests
ub = np.ones(len(df_t_b))
trend_b = np.arange(1, len(df_t_b) + 1)
X1b = np.column_stack([ub, trend_b, df_t_b["LOGIP_L1"], df_t_b["GIP_L1"], df_t_b["GIP_L2"]])
X2b = np.column_stack([ub, trend_b, df_t_b["LOGCLI_L1"], df_t_b["GCLI_L1"], df_t_b["GCLI_L2"]])

# Run regressions
OLS_ADF_1 = mult_regr(df_t_b["GIP"], X1b)
OLS_ADF_2 = mult_regr(df_t_b["GCLI"], X2b)

print(
    f"\nThe ADF-test for log(IP) yields a coefficient ρ_hat = {np.round(OLS_ADF_1[0][2], 3)}, with an SE = {np.round(OLS_ADF_1[1][2], 3)} and a t-value = {np.round(OLS_ADF_1[2][2], 3)}.\n"
    f"\nTherefore, since tρ_hat > -3.5, we fail to reject the null of non-stationarity for log(IP).\n"
    f"\nThe ADF-test for log(CLI) yields a coefficient ρ_hat = {np.round(OLS_ADF_2[0][2], 3)}, with an SE = {np.round(OLS_ADF_2[1][2], 3)} and a t-value = {np.round(OLS_ADF_2[2][2], 3)}.\n"
    f"\nTherefore, since tρ_hat > -3.5, we also fail to reject the null of non-stationarity for log(CLI).\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Run regression for the first step Engle-Granger test
OLS = simple_regr(df_t["LOGIP"], df_t["LOGCLI"])

# Create difference and lags for residuals  
df_t_c = pd.DataFrame(OLS[2], columns=["EPS"])
df_t_c["EPS_L1"] = df_t_c["EPS"].shift(1)
df_t_c["DEPS"] = df_t_c["EPS"].diff()
df_t_c["DEPS_L1"] = df_t_c["DEPS"].shift(1)
df_t_c["DEPS_L2"] = df_t_c["DEPS"].shift(2)
df_t_c = df_t_c.dropna()

# Define regressor matrix for the 2nd-step Engle-Granger test
uc = np.ones(len(df_t_c))
trend_c = np.arange(1, len(df_t_c) + 1)
Xc = np.column_stack([uc, trend_c, df_t_c["EPS_L1"], df_t_c["DEPS_L1"], df_t_c["DEPS_L2"]])

# Run regression
OLS_eg = mult_regr(df_t_c["DEPS"], Xc)

print(
    f"\nThe 2nd-step Engle-Granger test yields a coefficient ρ_hat = {np.round(OLS_eg[0][1], 3)}, with an SE = {np.round(OLS_eg[1][1], 3)} and a t-value = {np.round(OLS_eg[2][1], 3)}.\n"
    f"\nTherefore, since tρ_hat > -3.8, we fail to reject the null of non-stationarity for the residuals, which means that log(IP) and log(CLI) are not cointegrated.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Granger causality tests
df_t_d = df_t_b.copy()

# Define regressor matrices for Granger causality tests
ud = np.ones(len(df_t_d))
Xd1 = np.column_stack([ud, df_t_d["GIP_L1"], df_t_d["GIP_L2"], df_t_d["GCLI_L1"], df_t_d["GCLI_L2"]])
Xd2 = np.column_stack([ud, df_t_d["GCLI_L1"], df_t_d["GCLI_L2"], df_t_d["GIP_L1"], df_t_d["GIP_L2"]])

# Run regressions
OLS_ADL_1 = mult_regr(df_t_d["GIP"], Xd1)
OLS_ADL_2 = mult_regr(df_t_d["GCLI"], Xd2)

# Define regressor matrices of restricted models for each test
Xd1r = np.column_stack([ud, df_t_d["GIP_L1"], df_t_d["GIP_L2"]])
Xd2r = np.column_stack([ud, df_t_d["GCLI_L1"], df_t_d["GCLI_L2"]])

# F-tests for Granger causality
FGr1 = F_test(df_t_d["GIP"], Xd1r, Xd1, 2)
FGr2 = F_test(df_t_d["GCLI"], Xd2r, Xd2, 2)

print(
    f"\nThe F-test for the Granger causality of GCLI for GIP yields:\n"
    f"\nF-statistic = {np.round(FGr1[0], 3)}, with a p-value = {np.round(FGr1[1], 3)}.\n"
    f"\nTherefore, we reject the null hypothesis that γ = 0, meaning that GCLI helps predict GIP.\n"
    f"\nThe F-test for the Granger causality of GIP for GCLI yields:\n"
    f"\nF-statistic = {np.round(FGr2[0], 3)}, with a p-value = {np.round(FGr2[1], 3)}.\n"
    f"\nSince 0.083 > 0.05, we fail to reject the null hypothesis that γ = 0, meaning that GIP does not help predict GCLI.\n" 
    f"\nWe conclude that GCLI is Granger causal for GIP.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# AR(2) model for GIP
df_t_e1 = df_t_b.copy()

# Define regressor matrix
ue = np.ones(len(df_t_e1))
Xe = np.column_stack([ue, df_t_e1["GIP_L1"], df_t_e1["GIP_L2"]])

# Run regression
OLS_AR2 = mult_regr(df_t_e1["GIP"], Xe)

print(
    f"\nThe AR(2) for GIP yields:\n"
    f"\nconstant ->\tβ = {np.round(OLS_AR2[0][0], 3)}\tSE = {np.round(OLS_AR2[1][0], 3)}\tt-value = {np.round(OLS_AR2[2][0], 3)}.\n"
    f"\nLag1 ->\t\tβ = {np.round(OLS_AR2[0][1], 3)}\tSE = {np.round(OLS_AR2[1][1], 3)}\tt-value = {np.round(OLS_AR2[2][1], 3)}.\n"
    f"\nLag2 ->\t\tβ = {np.round(OLS_AR2[0][2], 3)}\tSE = {np.round(OLS_AR2[1][2], 3)}\tt-value = {np.round(OLS_AR2[2][2], 3)}.\n"
    f"\nTherefore, since the absolute value of the t-statistic for both lags is smaller than 2, they are insignificant.\n"
)

# AR(1) model for GIP
df_t_e2 = df.copy()
df_t_e2["GIP_L1"] = df_t_e2["GIP"].shift(1)
df_t_e2 = df_t_e2.dropna()

# Run regression
OLS_AR1 = simple_regr(df_t_e2["GIP"], df_t_e2["GIP_L1"])

print(
    f"\nThe AR(1) for GIP yields:\n"
    f"\nLag1 ->\t\tβ = {np.round(OLS_AR1[1], 3)}\tt-value = {np.round(OLS_AR1[4], 3)}.\n"
    f"\nTherefore, since the absolute value of the t-statistic of the lag is smaller than 2, it is also insignificant.\n"
)

# Evaluation sample predictions with AR(1) model
α = OLS_AR1[0]
β = OLS_AR1[1]
prev_GIP = df_t["GIP"].iloc[-1]
AR1_forecasts = []
 
for year in df_e["YEAR"]:
    forecast = α + β * prev_GIP
    AR1_forecasts.append(forecast)
    prev_GIP = forecast

df_e["GIP_AR1"] = AR1_forecasts


# Evaluation sample predictions with sample mean model
df_e["GIP_mean"] = [df_t["GIP"].mean()] * len(df_e)

# ADL(2,2) model for GIP
df_t_f = df_t_b.copy()

# Define regressor matrix of unrestricted model
uf = np.ones(len(df_t_f))
Xf = np.column_stack([uf, df_t_f["GIP_L1"], df_t_f["GIP_L2"], df_t_f["GCLI_L1"], df_t_f["GCLI_L2"]])

# Define regressor matrix of restricted model
Xfr = np.column_stack([uf, df_t_f["GCLI_L1"]])

# F-test for β1=β2=γ2=0
F = F_test(df_t_f["GIP"], Xfr, Xf, 3)

print(
    f"\nThe F-test for the ADL(2,2) model for GIP yields:\n"
    f"\nF-statistic = {np.round(F[0], 3)}, with a p-value = {np.round(F[1], 3)}.\n"
    f"\nTherefore, we fail to reject the null hypothesis that β1 = β2 = γ2 = 0.\n"
)

# Run regression for ADL(0,1) model
OLS_ADL01 = simple_regr(df_t_f["GIP"], df_t_f["GCLI_L1"])

# Evaluation sample predictions with ADL(0,1) model
α = OLS_ADL01[0]
β = OLS_ADL01[1]
prev_GCLI = df_t["GCLI"].iloc[-1]
ADL01_forecasts = []
 
for year in df_e["YEAR"]:
    forecast = α + β * prev_GCLI
    ADL01_forecasts.append(forecast)
    prev_GCLI = forecast

df_e["GIP_ADL01"] = ADL01_forecasts

# Evaluation of predictions
error_AR1 = error_eval(df_e["GIP"], df_e["GIP_AR1"])
error_SM = error_eval(df_e["GIP"], df_e["GIP_mean"])
error_ADL01 = error_eval(df_e["GIP"], df_e["GIP_ADL01"])

print(
    f"\nModel evaluation table:\n"
    f"\n\t\tAR(1)\tSimple\tADL(0,1)\n"
    f"\nRMSE ->\t\t{np.round(error_AR1[0], 3)}\t{np.round(error_SM[0], 3)}\t{np.round(error_ADL01[0], 3)}\n"
    f"\nMAE ->\t\t{np.round(error_AR1[1], 3)}\t{np.round(error_SM[1], 3)}\t{np.round(error_ADL01[1], 3)}\n"
    f"\nSUM ->\t\t{np.round(error_AR1[2], 3)}\t{np.round(error_SM[2], 3)}\t{np.round(error_ADL01[2], 3)}\n"
)