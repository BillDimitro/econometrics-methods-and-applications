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

# OLS_c estimator function for multiple regression
def mult_regr(y, X):

    n, k = X.shape[0], X.shape[1]                               # Storing model's sample size and number of explanatory variables
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y                                    # OLS_c formula
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

# RESET test function implementation
def RESET_test(y, X, p):                                        # Enter dependent variable, model regressor matrix, and the number of fitted values powers to be tested
    n, k = X.shape[0], X.shape[1]                               # Storing model's sample size and number of explanatory variables
    b, _, _, _, S0, _ = mult_regr(y, X)                         # Obtain model's OLS_c estimator to use for fitted values calculation and sum of squared residuals to be used later in the F statistic calculation
    y_fit = X @ b                                               # Obtain model's fitted values
    i = 0
    X_reset = X
    while (i < p):                                              # While loop to build RESET test regressor matrix according to defined p
        y_pow = y_fit ** (i + 2)
        X_reset = np.column_stack([X_reset, y_pow])
        i += 1
    S1 = mult_regr(y, X_reset)[4]                               # Obtain sum of squared residuals for the unrestricted model
    F = ((S0 - S1) / p) / (S1 / (n - k - p))                    # F-statistic for RESET test
    p = 1 - f.cdf(F, dfn=p, dfd=n-k-p)                          # p-value for RESET test
    return F, p

# General-to-specific function implementation, but only for intercation terms
def gen_to_spec(y, X, sig_lvl, var_names):                                                              # Enter dependent variable, general model regressor matrix, significance level, and variable names
    iter = 1                                                                                            # Capture the iteration number
    while True:
        model = mult_regr(y, X)
        p_values = model[5]
        p_max = p_values[12:].max()                                                                     # Only compare p-values of interaction terms
        if p_max <= sig_lvl:
            print(f"\nFinal model reached after {iter-1} variable eliminations.")
            print(f"\nRemaining variables: {var_names}.")
            return model
        idx_to_remove = p_values[12:].argmax() + 12                                                     # Get index of explanatory variable with maximum p-value, only for the interaction terms
        print(f"\nRegression {iter}:\nRemoving {var_names[idx_to_remove]} (p-value = {p_max:.3f}).")
        var_names.pop(idx_to_remove)                                                                    # Remove variable's name from remaining variable's list
        X = np.delete(X, idx_to_remove, axis=1)
        iter += 1

def error_eval(y, y_pred):
    n = len(y)
    rmse = np.sqrt(((y - y_pred)**2).sum() / n)
    mae = np.abs(y - y_pred).sum() / n
    err_sum = (y - y_pred).sum()
    return rmse, mae, err_sum

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_8\datasets/CaseProject.txt", sep=r"\s+")

# Define question (a) model regressor matrix to perform RESET test
ua = np.ones(len(df))
X_res = np.column_stack([ua, df["lot"], df["bdms"], df["fb"], df["sty"], df["drv"], df["rec"], df["ffin"], df["ghw"], df["ca"], df["gar"], df["reg"]])

# RESET test
reset_a = RESET_test(df["sell"], X_res, 1)

print(
    f"\nUsing p = 1, the RESET test for the model with sell as dependent variable yields an F-statistic = {np.round(reset_a[0], 3)}, with a p-value = {np.round(reset_a[1], 3)}.\n"
    f"\nSince p < 0.05, we reject the null of correct linear specification at the 5% level. The linear model with the sale price in levels appears to be misspecified.\n"
    f"\nThis suggests that the relationship between house price and the explanatory variables is not adequately captured by a simple linear model in the original\n"
    f"sale price.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# RESET test for log(sale) as dependent variable
df["logsell"] = np.log(df["sell"])

reset_b = RESET_test(df["logsell"], X_res, 1)

print(
    f"\nUsing p = 1, the RESET test for the model with log(sell) as dependent variable yields an F-statistic = {np.round(reset_b[0], 3)}, with a p-value = {np.round(reset_b[1], 3)}.\n"
    f"\nSince p > 0.05, we fail to reject the null of correct linear specification at the 5% level. Therefore, using log(sell) as the dependent variable improves\n"
    f"the functional form of the model.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Add log(lot) to model and check for significance
df["loglot"] = np.log(df["lot"])
X_c = np.column_stack([ua, df["lot"], df["loglot"], df["bdms"], df["fb"], df["sty"], df["drv"], df["rec"], df["ffin"], df["ghw"], df["ca"], df["gar"], df["reg"]])

# Run regression
OLS_c = mult_regr(df["logsell"], X_c)

print(
    f"\nRegressing logsell on a constant, loglot, and all other variables yields:\n"
    f"\nCons ->\t\tb = {np.round(OLS_c[0][0], 3)}\tSE = {np.round(OLS_c[1][0], 3)}\tt-statistic = {np.round(OLS_c[2][0], 3)}\tp-value = {np.round(OLS_c[5][0], 3)}\n"
    f"\nLot ->\t\tb = {np.round(OLS_c[0][1], 3)}\tSE = {np.round(OLS_c[1][1], 3)}\tt-statistic = {np.round(OLS_c[2][1], 3)}\tp-value = {np.round(OLS_c[5][1], 3)}\n"
    f"\nLogLot ->\tb = {np.round(OLS_c[0][2], 3)}\tSE = {np.round(OLS_c[1][2], 3)}\tt-statistic = {np.round(OLS_c[2][2], 3)}\tp-value = {np.round(OLS_c[5][2], 3)}\n"
    f"\nBdms ->\t\tb = {np.round(OLS_c[0][3], 3)}\tSE = {np.round(OLS_c[1][3], 3)}\tt-statistic = {np.round(OLS_c[2][3], 3)}\tp-value = {np.round(OLS_c[5][3], 3)}\n"
    f"\nFb ->\t\tb = {np.round(OLS_c[0][4], 3)}\tSE = {np.round(OLS_c[1][4], 3)}\tt-statistic = {np.round(OLS_c[2][4], 3)}\tp-value = {np.round(OLS_c[5][4], 3)}\n"
    f"\nSty ->\t\tb = {np.round(OLS_c[0][5], 3)}\tSE = {np.round(OLS_c[1][5], 3)}\tt-statistic = {np.round(OLS_c[2][5], 3)}\tp-value = {np.round(OLS_c[5][5], 3)}\n"
    f"\nDrv ->\t\tb = {np.round(OLS_c[0][6], 3)}\tSE = {np.round(OLS_c[1][6], 3)}\tt-statistic = {np.round(OLS_c[2][6], 3)}\tp-value = {np.round(OLS_c[5][6], 3)}\n"
    f"\nRec ->\t\tb = {np.round(OLS_c[0][7], 3)}\tSE = {np.round(OLS_c[1][7], 3)}\tt-statistic = {np.round(OLS_c[2][7], 3)}\tp-value = {np.round(OLS_c[5][7], 3)}\n"
    f"\nFfin ->\t\tb = {np.round(OLS_c[0][8], 3)}\tSE = {np.round(OLS_c[1][8], 3)}\tt-statistic = {np.round(OLS_c[2][8], 3)}\tp-value = {np.round(OLS_c[5][8], 3)}\n"
    f"\nGhw ->\t\tb = {np.round(OLS_c[0][9], 3)}\tSE = {np.round(OLS_c[1][9], 3)}\tt-statistic = {np.round(OLS_c[2][9], 3)}\tp-value = {np.round(OLS_c[5][9], 3)}\n"
    f"\nCa ->\t\tb = {np.round(OLS_c[0][10], 3)}\tSE = {np.round(OLS_c[1][10], 3)}\tt-statistic = {np.round(OLS_c[2][10], 3)}\tp-value = {np.round(OLS_c[5][10], 3)}\n"
    f"\nGar ->\t\tb = {np.round(OLS_c[0][11], 3)}\tSE = {np.round(OLS_c[1][11], 3)}\tt-statistic = {np.round(OLS_c[2][11], 3)}\tp-value = {np.round(OLS_c[5][11], 3)}\n"
    f"\nReg ->\t\tb = {np.round(OLS_c[0][12], 3)}\tSE = {np.round(OLS_c[1][12], 3)}\tt-statistic = {np.round(OLS_c[2][12], 3)}\tp-value = {np.round(OLS_c[5][12], 3)}\n"
    f"\nR-squared = {OLS_c[3]:.3f}\n"
    f"\nWhen both lot size and log lot size are included in the same regression, lot size itself is not statistically significant p = 0.359 > 0.05, while log(lot) is\n"
    f"strongly statistically significant. Therefore, the evidence supports using the logarithm of lot size rather than lot size in levels.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Add interaction terms of log(lot) with each variable and check for significance
df["loglot_bdms"] = df["loglot"] * df["bdms"]
df["loglot_fb"] = df["loglot"] * df["fb"]
df["loglot_sty"] = df["loglot"] * df["sty"]
df["loglot_drv"] = df["loglot"] * df["drv"]
df["loglot_rec"] = df["loglot"] * df["rec"]
df["loglot_ffin"] = df["loglot"] * df["ffin"]
df["loglot_ghw"] = df["loglot"] * df["ghw"]
df["loglot_ca"] = df["loglot"] * df["ca"]
df["loglot_gar"] = df["loglot"] * df["gar"]
df["loglot_reg"] = df["loglot"] * df["reg"]

X_d = np.column_stack([ua, df["loglot"], df["bdms"], df["fb"], df["sty"], df["drv"], df["rec"], df["ffin"], df["ghw"], df["ca"], df["gar"], df["reg"],
df["loglot_bdms"], df["loglot_fb"], df["loglot_sty"], df["loglot_drv"], df["loglot_rec"], df["loglot_ffin"], df["loglot_ghw"], df["loglot_ca"], df["loglot_gar"], df["loglot_reg"]])

# Run regression
OLS_d = mult_regr(df["logsell"], X_d)

print(
    f"\nMain effects:\n"
    f"\nCons ->\t\tb = {np.round(OLS_d[0][0], 3)}\tSE = {np.round(OLS_d[1][0], 3)}\tt-statistic = {np.round(OLS_d[2][0], 3)}\tp-value = {np.round(OLS_d[5][0], 3)}\n"
    f"\nLogLot ->\tb = {np.round(OLS_d[0][1], 3)}\tSE = {np.round(OLS_d[1][1], 3)}\tt-statistic = {np.round(OLS_d[2][1], 3)}\tp-value = {np.round(OLS_d[5][1], 3)}\n"
    f"\nBdms ->\t\tb = {np.round(OLS_d[0][2], 3)}\tSE = {np.round(OLS_d[1][2], 3)}\tt-statistic = {np.round(OLS_d[2][2], 3)}\tp-value = {np.round(OLS_d[5][2], 3)}\n"
    f"\nFb ->\t\tb = {np.round(OLS_d[0][3], 3)}\tSE = {np.round(OLS_d[1][3], 3)}\tt-statistic = {np.round(OLS_d[2][3], 3)}\tp-value = {np.round(OLS_d[5][3], 3)}\n"
    f"\nSty ->\t\tb = {np.round(OLS_d[0][4], 3)}\tSE = {np.round(OLS_d[1][4], 3)}\tt-statistic = {np.round(OLS_d[2][4], 3)}\tp-value = {np.round(OLS_d[5][4], 3)}\n"
    f"\nDrv ->\t\tb = {np.round(OLS_d[0][5], 3)}\tSE = {np.round(OLS_d[1][5], 3)}\tt-statistic = {np.round(OLS_d[2][5], 3)}\tp-value = {np.round(OLS_d[5][5], 3)}\n"
    f"\nRec ->\t\tb = {np.round(OLS_d[0][6], 3)}\tSE = {np.round(OLS_d[1][6], 3)}\tt-statistic = {np.round(OLS_d[2][6], 3)}\tp-value = {np.round(OLS_d[5][6], 3)}\n"
    f"\nFfin ->\t\tb = {np.round(OLS_d[0][7], 3)}\tSE = {np.round(OLS_d[1][7], 3)}\tt-statistic = {np.round(OLS_d[2][7], 3)}\tp-value = {np.round(OLS_d[5][7], 3)}\n"
    f"\nGhw ->\t\tb = {np.round(OLS_d[0][8], 3)}\tSE = {np.round(OLS_d[1][8], 3)}\tt-statistic = {np.round(OLS_d[2][8], 3)}\tp-value = {np.round(OLS_d[5][8], 3)}\n"
    f"\nCa ->\t\tb = {np.round(OLS_d[0][9], 3)}\tSE = {np.round(OLS_d[1][9], 3)}\tt-statistic = {np.round(OLS_d[2][9], 3)}\tp-value = {np.round(OLS_d[5][9], 3)}\n"
    f"\nGar ->\t\tb = {np.round(OLS_d[0][10], 3)}\tSE = {np.round(OLS_d[1][10], 3)}\tt-statistic = {np.round(OLS_d[2][10], 3)}\tp-value = {np.round(OLS_d[5][10], 3)}\n"
    f"\nReg ->\t\tb = {np.round(OLS_d[0][11], 3)}\tSE = {np.round(OLS_d[1][11], 3)}\tt-statistic = {np.round(OLS_d[2][11], 3)}\tp-value = {np.round(OLS_d[5][11], 3)}\n"
)

print(
    f"\nInteraction effects:\n"
    f"\nLoglot_Bdms ->\t\tb = {np.round(OLS_d[0][12], 3)}\tSE = {np.round(OLS_d[1][12], 3)}\tt-statistic = {np.round(OLS_d[2][12], 3)}\tp-value = {np.round(OLS_d[5][12], 3)}\n"
    f"\nLoglot_Fb ->\t\tb = {np.round(OLS_d[0][13], 3)}\tSE = {np.round(OLS_d[1][13], 3)}\tt-statistic = {np.round(OLS_d[2][13], 3)}\tp-value = {np.round(OLS_d[5][13], 3)}\n"
    f"\nLoglot_Sty ->\t\tb = {np.round(OLS_d[0][14], 3)}\tSE = {np.round(OLS_d[1][14], 3)}\tt-statistic = {np.round(OLS_d[2][14], 3)}\tp-value = {np.round(OLS_d[5][14], 3)}\n"
    f"\nLoglot_Drv ->\t\tb = {np.round(OLS_d[0][15], 3)}\tSE = {np.round(OLS_d[1][15], 3)}\tt-statistic = {np.round(OLS_d[2][15], 3)}\tp-value = {np.round(OLS_d[5][15], 3)}\n"
    f"\nLoglot_Rec ->\t\tb = {np.round(OLS_d[0][16], 3)}\tSE = {np.round(OLS_d[1][16], 3)}\tt-statistic = {np.round(OLS_d[2][16], 3)}\tp-value = {np.round(OLS_d[5][16], 3)}\n"
    f"\nLoglot_Ffin ->\t\tb = {np.round(OLS_d[0][17], 3)}\tSE = {np.round(OLS_d[1][17], 3)}\tt-statistic = {np.round(OLS_d[2][17], 3)}\tp-value = {np.round(OLS_d[5][17], 3)}\n"
    f"\nLoglot_Ghw ->\t\tb = {np.round(OLS_d[0][18], 3)}\tSE = {np.round(OLS_d[1][18], 3)}\tt-statistic = {np.round(OLS_d[2][18], 3)}\tp-value = {np.round(OLS_d[5][18], 3)}\n"
    f"\nLoglot_Ca ->\t\tb = {np.round(OLS_d[0][19], 3)}\tSE = {np.round(OLS_d[1][19], 3)}\tt-statistic = {np.round(OLS_d[2][19], 3)}\tp-value = {np.round(OLS_d[5][19], 3)}\n"
    f"\nLoglot_Gar ->\t\tb = {np.round(OLS_d[0][20], 3)}\tSE = {np.round(OLS_d[1][20], 3)}\tt-statistic = {np.round(OLS_d[2][20], 3)}\tp-value = {np.round(OLS_d[5][20], 3)}\n"
    f"\nLoglot_Reg ->\t\tb = {np.round(OLS_d[0][21], 3)}\tSE = {np.round(OLS_d[1][21], 3)}\tt-statistic = {np.round(OLS_d[2][21], 3)}\tp-value = {np.round(OLS_d[5][21], 3)}\n"
    f"\nR-squared = {np.round(OLS_d[3], 3)}\n"
    f"\nAmong the 10 interaction effects between log lot size and the other explanatory variables, 2 are individually significant at the 5% level:\n"
    f"\nthe interaction with driveway and the interaction with recreational room.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Define restricted model regressor matrix for F-test
X_e = np.column_stack([ua, df["loglot"], df["bdms"], df["fb"], df["sty"], df["drv"], df["rec"], df["ffin"], df["ghw"], df["ca"], df["gar"], df["reg"]])

# F-test for joint significance of interaction terms
F_joint_inter = F_test(df["logsell"], X_e, X_d, 10)

print(
    f"\nThe F-test for the joint significance of the interaction terms yields:\n"
    f"\nF-statistic = {np.round(F_joint_inter[0], 3)}, with a p-value = {np.round(F_joint_inter[1], 3)}.\n"
    f"\nTherefore, we fail to reject the null hypothesis that γj = 0, where j are the interaction terms indices.\n"
    f"\nAlthough 2 interaction effects were individually significant in question (d), the interaction effects are not jointly significant at the 5% level.\n"
    f"\nTherefore, as a group, the interactions between log lot size and the other explanatory variables do not add statistically significant explanatory power\n"
    f"to the model.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Run general-to-specific to eliminate the interaction effects
variable_names = ["const", "log_lot", "bdms", "fb", "sty", "drv", "rec", "ffin", "ghw", "ca", "gar", "reg", 
                "log_lot_bdms", "log_lot_fb", "log_lot_sty", "log_lot_drv", "log_lot_rec", "log_lot_ffin", "log_lot_ghw", "log_lot_ca", "log_lot_gar", "log_lot_reg"]

g_sp_model = gen_to_spec(df["logsell"], X_d, 0.05, variable_names)

print(
    f"\nAfter applying the general-to-specific approach, the final model only keeps the interaction with recreational room. The final model is now:\n"
    f"\nCons ->\t\tb = {np.round(g_sp_model[0][0], 3)}\tSE = {np.round(g_sp_model[1][0], 3)}\tt-statistic = {np.round(g_sp_model[2][0], 3)}\tp-value = {np.round(g_sp_model[5][0], 3)}\n"
    f"\nLogLot ->\tb = {np.round(g_sp_model[0][1], 3)}\tSE = {np.round(g_sp_model[1][1], 3)}\tt-statistic = {np.round(g_sp_model[2][1], 3)}\tp-value = {np.round(g_sp_model[5][1], 3)}\n"
    f"\nBdms ->\t\tb = {np.round(g_sp_model[0][2], 3)}\tSE = {np.round(g_sp_model[1][2], 3)}\tt-statistic = {np.round(g_sp_model[2][2], 3)}\tp-value = {np.round(g_sp_model[5][2], 3)}\n"
    f"\nFb ->\t\tb = {np.round(g_sp_model[0][3], 3)}\tSE = {np.round(g_sp_model[1][3], 3)}\tt-statistic = {np.round(g_sp_model[2][3], 3)}\tp-value = {np.round(g_sp_model[5][3], 3)}\n"
    f"\nSty ->\t\tb = {np.round(g_sp_model[0][4], 3)}\tSE = {np.round(g_sp_model[1][4], 3)}\tt-statistic = {np.round(g_sp_model[2][4], 3)}\tp-value = {np.round(g_sp_model[5][4], 3)}\n"
    f"\nDrv ->\t\tb = {np.round(g_sp_model[0][5], 3)}\tSE = {np.round(g_sp_model[1][5], 3)}\tt-statistic = {np.round(g_sp_model[2][5], 3)}\tp-value = {np.round(g_sp_model[5][5], 3)}\n"
    f"\nRec ->\t\tb = {np.round(g_sp_model[0][6], 3)}\tSE = {np.round(g_sp_model[1][6], 3)}\tt-statistic = {np.round(g_sp_model[2][6], 3)}\tp-value = {np.round(g_sp_model[5][6], 3)}\n"
    f"\nFfin ->\t\tb = {np.round(g_sp_model[0][7], 3)}\tSE = {np.round(g_sp_model[1][7], 3)}\tt-statistic = {np.round(g_sp_model[2][7], 3)}\tp-value = {np.round(g_sp_model[5][7], 3)}\n"
    f"\nGhw ->\t\tb = {np.round(g_sp_model[0][8], 3)}\tSE = {np.round(g_sp_model[1][8], 3)}\tt-statistic = {np.round(g_sp_model[2][8], 3)}\tp-value = {np.round(g_sp_model[5][8], 3)}\n"
    f"\nCa ->\t\tb = {np.round(g_sp_model[0][9], 3)}\tSE = {np.round(g_sp_model[1][9], 3)}\tt-statistic = {np.round(g_sp_model[2][9], 3)}\tp-value = {np.round(g_sp_model[5][9], 3)}\n"
    f"\nGar ->\t\tb = {np.round(g_sp_model[0][10], 3)}\tSE = {np.round(g_sp_model[1][10], 3)}\tt-statistic = {np.round(g_sp_model[2][10], 3)}\tp-value = {np.round(g_sp_model[5][10], 3)}\n"
    f"\nReg ->\t\tb = {np.round(g_sp_model[0][11], 3)}\tSE = {np.round(g_sp_model[1][11], 3)}\tt-statistic = {np.round(g_sp_model[2][11], 3)}\tp-value = {np.round(g_sp_model[5][11], 3)}\n"
    f"\nLoglot_Reg ->\tb = {np.round(g_sp_model[0][12], 3)}\tSE = {np.round(g_sp_model[1][12], 3)}\tt-statistic = {np.round(g_sp_model[2][12], 3)}\tp-value = {np.round(g_sp_model[5][12], 3)}\n"
    f"\nR-squared = {g_sp_model[3]:.3f}\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Separate test and evaluation sample
df_t = df[df["obs"] <= 400]
df_e = df[df["obs"] > 400]

# Estimate model parameters using the test sample
uf = np.ones(len(df_t))
X_f = np.column_stack([uf, df_t["loglot"], df_t["bdms"], df_t["fb"], df_t["sty"], df_t["drv"], df_t["rec"], df_t["ffin"], df_t["ghw"], df_t["ca"], df_t["gar"], df_t["reg"]])

est_param = mult_regr(df_t["logsell"], X_f)

print(
    f"\nModel parameters after estimation using the first 400 observations:\n"
    f"\nCons ->\t\tb = {np.round(est_param[0][0], 3)}\tSE = {np.round(est_param[1][0], 3)}\tt-statistic = {np.round(est_param[2][0], 3)}\tp-value = {np.round(est_param[5][0], 3)}\n"
    f"\nLogLot ->\tb = {np.round(est_param[0][1], 3)}\tSE = {np.round(est_param[1][1], 3)}\tt-statistic = {np.round(est_param[2][1], 3)}\tp-value = {np.round(est_param[5][1], 3)}\n"
    f"\nBdms ->\t\tb = {np.round(est_param[0][2], 3)}\tSE = {np.round(est_param[1][2], 3)}\tt-statistic = {np.round(est_param[2][2], 3)}\tp-value = {np.round(est_param[5][2], 3)}\n"
    f"\nFb ->\t\tb = {np.round(est_param[0][3], 3)}\tSE = {np.round(est_param[1][3], 3)}\tt-statistic = {np.round(est_param[2][3], 3)}\tp-value = {np.round(est_param[5][3], 3)}\n"
    f"\nSty ->\t\tb = {np.round(est_param[0][4], 3)}\tSE = {np.round(est_param[1][4], 3)}\tt-statistic = {np.round(est_param[2][4], 3)}\tp-value = {np.round(est_param[5][4], 3)}\n"
    f"\nDrv ->\t\tb = {np.round(est_param[0][5], 3)}\tSE = {np.round(est_param[1][5], 3)}\tt-statistic = {np.round(est_param[2][5], 3)}\tp-value = {np.round(est_param[5][5], 3)}\n"
    f"\nRec ->\t\tb = {np.round(est_param[0][6], 3)}\tSE = {np.round(est_param[1][6], 3)}\tt-statistic = {np.round(est_param[2][6], 3)}\tp-value = {np.round(est_param[5][6], 3)}\n"
    f"\nFfin ->\t\tb = {np.round(est_param[0][7], 3)}\tSE = {np.round(est_param[1][7], 3)}\tt-statistic = {np.round(est_param[2][7], 3)}\tp-value = {np.round(est_param[5][7], 3)}\n"
    f"\nGhw ->\t\tb = {np.round(est_param[0][8], 3)}\tSE = {np.round(est_param[1][8], 3)}\tt-statistic = {np.round(est_param[2][8], 3)}\tp-value = {np.round(est_param[5][8], 3)}\n"
    f"\nCa ->\t\tb = {np.round(est_param[0][9], 3)}\tSE = {np.round(est_param[1][9], 3)}\tt-statistic = {np.round(est_param[2][9], 3)}\tp-value = {np.round(est_param[5][9], 3)}\n"
    f"\nGar ->\t\tb = {np.round(est_param[0][10], 3)}\tSE = {np.round(est_param[1][10], 3)}\tt-statistic = {np.round(est_param[2][10], 3)}\tp-value = {np.round(est_param[5][10], 3)}\n"
    f"\nReg ->\t\tb = {np.round(est_param[0][11], 3)}\tSE = {np.round(est_param[1][11], 3)}\tt-statistic = {np.round(est_param[2][11], 3)}\tp-value = {np.round(est_param[5][11], 3)}\n"
    f"\nR-squared = {est_param[3]:.3f}\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Prediction of out-of-sample observations
α, β1, β2, β3, β4, β5, β6, β7, β8, β9, β10, β11 = est_param[0][0], est_param[0][1], est_param[0][2], est_param[0][3], est_param[0][4], est_param[0][5], est_param[0][6], est_param[0][7], est_param[0][8], est_param[0][9], est_param[0][10], est_param[0][11]
df_e["logsell_pred"] = α + β1 * df_e["loglot"] + β2 * df_e["bdms"] + β3 * df_e["fb"] + β4 * df_e["sty"] + β5 * df_e["drv"] + β6 * df_e["rec"] + β7 * df_e["ffin"] + β8 * df_e["ghw"] + β9 * df_e["ca"] + β10 * df_e["gar"] + β11 * df_e["reg"]

# Calculate MAE for the 146 observations
mean_abs_err = error_eval(df_e["logsell"], df_e["logsell_pred"])

print(
    f"\nThe model’s MAE in the evaluation sample is about {np.round(mean_abs_err[1], 3)} log points, which is roughly interpretable as an average error of about 12.8% in house prices.\n"
    f"\nThe standard deviation of the log of the price in the prediction sample is {np.round(df_e["logsell"].std(), 3)}\n"
    f"\nSo the MAE relative to the variability in the log of the price is {np.round(mean_abs_err[1], 3)} / {np.round(df_e["logsell"].std(), 3)} = 44.3%.\n"
    f"\nTherefore, the model has reasonably good predictive power, because its average prediction error is substantially smaller than the natural variation\n"
    f"of the log price.\n"
    f"\n-----------------------------------------------------------------------\n"
)