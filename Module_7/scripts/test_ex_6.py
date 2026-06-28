import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t, f, chi2
from statsmodels.tsa.stattools import acf, pacf
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
    p = 2 * (1 - t.cdf(abs(t_values), df=n-2))
    SSR = e.T @ e                                               # Sum of squared residuals
    SST = y.var() * (n - 1)                                     # Total sum of squares
    R_sq = 1 - (SSR / SST)                                      # R squared

    return beta, s_beta, t_values, R_sq, SSR, p

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

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_7\datasets/TestExer6.txt", sep=r"\s+")

# Separate test and evaluation sample
df_t = df[df["YYYY-MM"] < "2011M01"]
df_e = df[df["YYYY-MM"] >= "2011M01"]

# Time series plots
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

axes[0].plot(df_t["YYYY-MM"], df_t["CPI_EUR"], label="CPI_EUR")
axes[0].plot(df_t["YYYY-MM"], df_t["CPI_USA"], label="CPI_USA")
axes[0].set_ylabel("Value")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df_t["YYYY-MM"], df_t["LOGPEUR"], label="log(CPI_EUR)")
axes[1].plot(df_t["YYYY-MM"], df_t["LOGPUSA"], label="log(CPI_USA)")
axes[1].set_ylabel("Value")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(df_t["YYYY-MM"], df_t["DPEUR"], label="Δlog(CPI_EUR)")
axes[2].plot(df_t["YYYY-MM"], df_t["DPUSA"], label="Δlog(CPI_USA)")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Value")
axes[2].set_xticks(df_t["YYYY-MM"][::12])
axes[2].set_xticklabels(df_t["YYYY-MM"].iloc[::12].str[:4], rotation=45)
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

print(
    f"\nConclusion from plots: CPI and log(CPI) series of both regions are trending upward and are therefore likely non-stationary, while the monthly inflation series"
    f"\nfluctuate around a roughly constant mean. The inflation series show some common movements, especially during volatile periods such as 2008-2009, but a clear"
    f"\nand systematic visual lead-lag pattern from USA inflation to Euro-area inflation is not obvious. Therefore, whether USA inflation has predictive power for"
    f"\nEuro-area inflation should be investigated formally.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Create required lags for the ADF test
df_t_b = df_t.copy()
df_t_b["LOGPEUR_L1"] = df_t_b["LOGPEUR"].shift(1)
df_t_b["LOGPUSA_L1"] = df_t_b["LOGPUSA"].shift(1)
df_t_b["DPEUR_L1"] = df_t_b["DPEUR"].shift(1)
df_t_b["DPEUR_L2"] = df_t_b["DPEUR"].shift(2)
df_t_b["DPEUR_L3"] = df_t_b["DPEUR"].shift(3)
df_t_b["DPUSA_L1"] = df_t_b["DPUSA"].shift(1)
df_t_b["DPUSA_L2"] = df_t_b["DPUSA"].shift(2)
df_t_b["DPUSA_L3"] = df_t_b["DPUSA"].shift(3)
df_t_b = df_t_b.dropna()

# Define regressor matrices for ADF tests
ub = np.ones(len(df_t_b))
X1b = np.column_stack([ub, df_t_b["TREND"], df_t_b["LOGPEUR_L1"], df_t_b["DPEUR_L1"], df_t_b["DPEUR_L2"], df_t_b["DPEUR_L3"]])
X2b = np.column_stack([ub, df_t_b["TREND"], df_t_b["LOGPUSA_L1"], df_t_b["DPUSA_L1"], df_t_b["DPUSA_L2"], df_t_b["DPUSA_L3"]])

# Run regressions
OLS_ADF_1 = mult_regr(df_t_b["DPEUR"], X1b)
OLS_ADF_2 = mult_regr(df_t_b["DPUSA"], X2b)

print(
    f"\nThe ADF-test for log(CPI_EUR) yields a coefficient ρ_hat = {np.round(OLS_ADF_1[0][2], 3)}, with an SE = {np.round(OLS_ADF_1[1][2], 3)} and a t-value = {np.round(OLS_ADF_1[2][2], 3)}.\n"
    f"\nTherefore, since tρ_hat > -3.5, we fail to reject the null of non-stationarity for log(CPI_EUR).\n"
    f"\nThe ADF-test for log(CPI_USA) yields a coefficient ρ_hat = {np.round(OLS_ADF_2[0][2], 3)}, with an SE = {np.round(OLS_ADF_2[1][2], 3)} and a t-value = {np.round(OLS_ADF_2[2][2], 3)}.\n"
    f"\nTherefore, since tρ_hat > -3.5, we also fail to reject the null of non-stationarity for log(CPI_USA).\n"
    f"\nBoth series are clearly non-stationary.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Estimate sample correlations and autocorrelations to motivate a certain model
df_t_c = df_t.copy()
df_t_c = df_t_c.dropna()

acf_vals = acf(df_t_c["DPEUR"], nlags=12)                       # Compute sample autocorrelations
pacf_vals = pacf(df_t_c["DPEUR"], nlags=12, method="ols")       # Compute sample partial autocorrelations

crit = 2 / np.sqrt(len(df_t_c))                                 # 5% critical value
lags = np.arange(1, 13)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ACF plot
axs[0].plot(lags, acf_vals[1:13], marker="o", label="ACF")
axs[0].axhline(crit, color="red", linestyle="dotted", label="5% critical value")
axs[0].axhline(-crit, color="red", linestyle="dotted", label="_nolegend_")
axs[0].axhline(0, color="black", linewidth=0.8)
axs[0].set_title("Sample ACF of DPEUR")
axs[0].set_ylabel("ACF")
axs[0].grid(True)
axs[0].legend()

# PACF plot
axs[1].plot(lags, pacf_vals[1:13], marker="o", label="PACF")
axs[1].axhline(crit, color="red", linestyle="dotted", label="5% critical value")
axs[1].axhline(-crit, color="red", linestyle="dotted", label="_nolegend_")
axs[1].axhline(0, color="black", linewidth=0.8)
axs[1].set_title("Sample PACF of DPEUR")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("PACF")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

print(
    f"\nThe ACF and PACF show pronounced and statistically significant spikes at lags 6 and 12, while most other lags are smaller or close to the significance bounds."
    f"\nSince the data are monthly, lag 6 corresponds to a half-year seasonal effect and lag 12 to an annual seasonal effect. This provides motivation for modeling"
    f"\nEuro-area inflation as an AR model depending on its 6th and 12th lag.\n"
)

# Create required lags for AR model
df_t_c2 = df_t.copy()
df_t_c2["DPEUR_L6"] = df_t_c2["DPEUR"].shift(6)
df_t_c2["DPEUR_L12"] = df_t_c2["DPEUR"].shift(12)
df_t_c2 = df_t_c2.dropna()

# Define regressor matrix
uc = np.ones(len(df_t_c2))
Xc = np.column_stack([uc, df_t_c2["DPEUR_L6"], df_t_c2["DPEUR_L12"]])

# Run regression
OLS_AR = mult_regr(df_t_c2["DPEUR"], Xc)

print(
    f"\nThe AR model with lags 6 and 12 for DPEUR yields:\n"
    f"\nconstant ->\tβ = {np.round(OLS_AR[0][0], 3)}\t\tSE = {np.round(OLS_AR[1][0], 3)}\tt-value = {np.round(OLS_AR[2][0], 3)}\t\tp-value = {np.round(OLS_AR[5][0], 3)}\n"
    f"\nLag6 ->\t\tβ = {np.round(OLS_AR[0][1], 3)}\tSE = {np.round(OLS_AR[1][1], 3)}\tt-value = {np.round(OLS_AR[2][1], 3)}\t\tp-value = {np.round(OLS_AR[5][1], 3)}\n"
    f"\nLag12 ->\tβ = {np.round(OLS_AR[0][2], 3)}\tSE = {np.round(OLS_AR[1][2], 3)}\tt-value = {np.round(OLS_AR[2][2], 3)}\t\tp-value = {np.round(OLS_AR[5][2], 3)}\n"
    f"\nTherefore, both lag 6 and lag 12 are statistically significant at the 5% level.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Create required lags for ADL model
df_t_d = df_t_c2.copy()
df_t_d["DPUSA_L1"] = df_t_d["DPUSA"].shift(1)
df_t_d["DPUSA_L6"] = df_t_d["DPUSA"].shift(6)
df_t_d["DPUSA_L12"] = df_t_d["DPUSA"].shift(12)
df_t_d = df_t_d.dropna()

# Define regressor matrix
ud = np.ones(len(df_t_d))
Xd = np.column_stack([ud, df_t_d["DPEUR_L6"], df_t_d["DPEUR_L12"], df_t_d["DPUSA_L1"], df_t_d["DPUSA_L6"], df_t_d["DPUSA_L12"]])

# Run regression
OLS_ADL = mult_regr(df_t_d["DPEUR"], Xd)

print(
    f"\nThe ADL model with lags 6 and 12 for DPEUR and lags 1, 6, and 12 for DPUSA yields:\n"
    f"\nconstant ->\tβ = {np.round(OLS_ADL[0][0], 3)}\t\tSE = {np.round(OLS_ADL[1][0], 3)}\tt-value = {np.round(OLS_ADL[2][0], 3)}\t\tp-value = {np.round(OLS_ADL[5][0], 3)}\n"
    f"\nEUR_Lag6 ->\tβ = {np.round(OLS_ADL[0][1], 3)}\tSE = {np.round(OLS_ADL[1][1], 3)}\tt-value = {np.round(OLS_ADL[2][1], 3)}\t\tp-value = {np.round(OLS_ADL[5][1], 3)}\n"
    f"\nEUR_Lag12 ->\tβ = {np.round(OLS_ADL[0][2], 3)}\tSE = {np.round(OLS_ADL[1][2], 3)}\tt-value = {np.round(OLS_ADL[2][2], 3)}\t\tp-value = {np.round(OLS_ADL[5][2], 3)}\n"
    f"\nUSA_Lag1 ->\tβ = {np.round(OLS_ADL[0][3], 3)}\tSE = {np.round(OLS_ADL[1][3], 3)}\tt-value = {np.round(OLS_ADL[2][3], 3)}\t\tp-value = {np.round(OLS_ADL[5][3], 3)}\n"
    f"\nUSA_Lag6 ->\tβ = {np.round(OLS_ADL[0][4], 3)}\tSE = {np.round(OLS_ADL[1][4], 3)}\tt-value = {np.round(OLS_ADL[2][4], 3)}\t\tp-value = {np.round(OLS_ADL[5][4], 3)}\n"
    f"\nUSA_Lag12 ->\tβ = {np.round(OLS_ADL[0][5], 3)}\tSE = {np.round(OLS_ADL[1][5], 3)}\tt-value = {np.round(OLS_ADL[2][5], 3)}\tp-value = {np.round(OLS_ADL[5][5], 3)}\n"
    f"\nTherefore, the coefficient of DPUSA at lag 6 is not statistically significant at the 5% level, we remove it from the model.\n"
)

# Drop DPUSA_L6 and run again
df_t_d2 = df_t.copy()
df_t_d2["DPEUR_L6"] = df_t_d2["DPEUR"].shift(6)
df_t_d2["DPEUR_L12"] = df_t_d2["DPEUR"].shift(12)
df_t_d2["DPUSA_L1"] = df_t_d2["DPUSA"].shift(1)
df_t_d2["DPUSA_L12"] = df_t_d2["DPUSA"].shift(12)
df_t_d2 = df_t_d2.dropna()

ud2 = np.ones(len(df_t_d2))
Xd2 = np.column_stack([ud2, df_t_d2["DPEUR_L6"], df_t_d2["DPEUR_L12"], df_t_d2["DPUSA_L1"], df_t_d2["DPUSA_L12"]])

# Run regression
OLS_ADL = mult_regr(df_t_d2["DPEUR"], Xd2)

print(
    f"\nThe ADL model with lags 6 and 12 for DPEUR and lags 1 and 12 for DPUSA yields:\n"
    f"\nconstant ->\tβ = {np.round(OLS_ADL[0][0], 3)}\t\tSE = {np.round(OLS_ADL[1][0], 3)}\tt-value = {np.round(OLS_ADL[2][0], 3)}\t\tp-value = {np.round(OLS_ADL[5][0], 3)}\n"
    f"\nEUR_Lag6 ->\tβ = {np.round(OLS_ADL[0][1], 3)}\tSE = {np.round(OLS_ADL[1][1], 3)}\tt-value = {np.round(OLS_ADL[2][1], 3)}\t\tp-value = {np.round(OLS_ADL[5][1], 3)}\n"
    f"\nEUR_Lag12 ->\tβ = {np.round(OLS_ADL[0][2], 3)}\tSE = {np.round(OLS_ADL[1][2], 3)}\tt-value = {np.round(OLS_ADL[2][2], 3)}\t\tp-value = {np.round(OLS_ADL[5][2], 3)}\n"
    f"\nUSA_Lag1 ->\tβ = {np.round(OLS_ADL[0][3], 3)}\tSE = {np.round(OLS_ADL[1][3], 3)}\tt-value = {np.round(OLS_ADL[2][3], 3)}\t\tp-value = {np.round(OLS_ADL[5][3], 3)}\n"
    f"\nUSA_Lag12 ->\tβ = {np.round(OLS_ADL[0][4], 3)}\tSE = {np.round(OLS_ADL[1][4], 3)}\tt-value = {np.round(OLS_ADL[2][4], 3)}\tp-value = {np.round(OLS_ADL[5][4], 3)}\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Create dataframe to access lags for out-of-sample AR predictions
df_AR = df.copy()
df_AR["DPEUR_L6"] = df_AR["DPEUR"].shift(6)
df_AR["DPEUR_L12"] = df_AR["DPEUR"].shift(12)
df_AR = df_AR.dropna()

# Evaluation sample predictions with AR model
α = OLS_AR[0][0]
β1 = OLS_AR[0][1]
β2 = OLS_AR[0][2]
AR_forecasts = []
 
for month in df_e["YYYY-MM"]:
    row = df_AR.loc[df_AR["YYYY-MM"] == month].squeeze()
    forecast = α + β1 * row["DPEUR_L6"] + β2 * row["DPEUR_L12"]
    AR_forecasts.append(forecast)

df_e["DPEUR_AR"] = AR_forecasts

# Create dataframe to access lags for out-of-sample ADL predictions
df_ADL = df.copy()
df_ADL["DPEUR_L6"] = df_ADL["DPEUR"].shift(6)
df_ADL["DPEUR_L12"] = df_ADL["DPEUR"].shift(12)
df_ADL["DPUSA_L1"] = df_ADL["DPUSA"].shift(1)
df_ADL["DPUSA_L12"] = df_ADL["DPUSA"].shift(12)
df_ADL = df_ADL.dropna()

# Evaluation sample predictions with ADL model
α = OLS_ADL[0][0]
β1 = OLS_ADL[0][1]
β2 = OLS_ADL[0][2]
γ1 = OLS_ADL[0][3]
γ2 = OLS_ADL[0][4]
ADL_forecasts = []
 
for month in df_e["YYYY-MM"]:
    row = df_ADL.loc[df_ADL["YYYY-MM"] == month].squeeze()
    forecast = α + β1 * row["DPEUR_L6"] + β2 * row["DPEUR_L12"] + γ1 * row["DPUSA_L1"] + γ2 * row["DPUSA_L12"]
    ADL_forecasts.append(forecast)

df_e["DPEUR_ADL"] = ADL_forecasts

print(
    f"\nThe monthly inflation forecasts for 2011 using the AR model are:\n"
    f"\n{df_e[["YYYY-MM", "DPEUR_AR"]].round(3).to_string(index=False)}"
)

print(
    f"\nThe monthly inflation forecasts for 2011 using the ADL model are:\n"
    f"\n{df_e[["YYYY-MM", "DPEUR_ADL"]].round(3).to_string(index=False)}"
)

# Evaluation of predictions
error_AR = error_eval(df_e["DPEUR"], df_e["DPEUR_AR"])
error_ADL = error_eval(df_e["DPEUR"], df_e["DPEUR_ADL"])

print(
    f"\nModel evaluation table:\n"
    f"\n\t\t\tAR\tADL\n"
    f"\nRMSE (x100) ->\t\t{np.round(error_AR[0] * 100, 3)}\t{np.round(error_ADL[0] * 100, 3)}\n"
    f"\nMAE (x100) ->\t\t{np.round(error_AR[1] * 100, 3)}\t{np.round(error_ADL[1] * 100, 3)}\n"
    f"\nSUM (x100) ->\t\t{np.round(error_AR[2] * 100, 3)}\t{np.round(error_ADL[2] * 100, 3)}\n"
)

print(
    f"\nThe forecast evaluation results indicate that the ADL model performs better than the pure AR model for forecasting Euro-area inflation in 2011.\n"
    f"The ADL model has a lower RMSE = 0.211 compared with 0.232 for the AR model, meaning that it produces smaller forecast errors on average when large\n"
    f"errors are penalized more heavily. It also has a lower MAE = 0.14 compared with 0.169. The sum of forecast errors is also much closer to zero for the\n"
    f"ADL model, meaning that it has almost no cumulative bias, because its positive and negative forecast errors nearly cancel out. These results suggest that\n"
    f"adding lagged USA inflation improves the out-of-sample forecasting performance for Euro-area inflation. Therefore, the evidence supports the idea that\n"
    f"USA inflation has predictive power for Euro-area inflation, at least in this 2011 forecast evaluation sample."
    f"\n-----------------------------------------------------------------------\n"
)