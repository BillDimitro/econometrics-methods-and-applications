import pandas as pd
import numpy as np
import math
from scipy import stats
from scipy.stats import f, chi2, skew, kurtosis

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_4\datasets/TestExer3.txt", sep=r"\s+")

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
    AIC = np.log(s**2) + 2 * (k / n)                            # Akaike information criterion
    BIC = np.log(s**2) + k * (np.log(n) / n)                    # Bayes information criterion

    return beta, t_values, p_values, R_sq, AIC, BIC, SSR, e

# General-to-specific function implementation
def gen_to_spec(y, X, sig_lvl, var_names):                                                              # Enter dependent variable, general model regressor matrix, significance level, and variable names
    iter = 1                                                                                            # Capture the iteration number
    while True:
        model = mult_regr(y, X)
        p_values = model[2]
        p_max = p_values[1:].max()                                                                      # We always keep the constant in the model to ensure the residuals have zero mean
        if p_max <= sig_lvl:
            print(f"\nFinal model reached after {iter-1} variable eliminations.")
            print(f"\nRemaining variables: {var_names}.")
            return model
        idx_to_remove = p_values[1:].argmax() + 1                                                       # Get index of explanatory variable with maximum p-value, always ignoring the constant
        print(f"\nRegression {iter}:\nRemoving {var_names[idx_to_remove]} (p-value = {p_max:.3f}).")
        var_names.pop(idx_to_remove)                                                                    # Remove variable's name from remaining variable's list
        X = np.delete(X, idx_to_remove, axis=1)
        iter += 1

# Specific-to-general function implementation
def spec_to_gen(y, X, sig_lvl, tot_vars):                                                               # Enter dependent variable, general model regressor matrix, significance level, and total variables list
    iter = 1
    X_init = X[:, 0]                                                                                    # Unit vector  
    fin_vars =[]                                                                                        # Variables of final model                                                                   
    while True:
        t_values = []
        p_values = []
        for idx in range(1, len(tot_vars)):                                                             # We always keep the constant in the model to ensure the residuals have zero mean
            X_temp = np.column_stack([X_init, X[:, idx]])                                               # Temp regressor matrix for each iteration
            model_temp = mult_regr(y, X_temp)
            t_values.append(model_temp[1][-1])
            p_values.append(model_temp[2][-1])
        p_min = min(p_values)                                                                  
        if p_min >= sig_lvl:                                                                            # Only proceed to select a variable if the significance level check is satisfied
            print(f"\nFinal model reached after {iter-1} variable additions.")
            print(f"\nModel variables: {fin_vars}.")
            return model
        idx_to_remove = np.argmax(np.abs(t_values)) + 1
        print(f"\nIteration {iter}: Selected variable is {tot_vars[idx_to_remove]}.")                   # Select variable with largest absolute t-statistic         
        fin_vars.append(tot_vars[idx_to_remove])                                                        # Add variable to final model variables list
        tot_vars.pop(idx_to_remove)
        model = model_temp
        X_init = np.column_stack([X_init, X[:, idx_to_remove]])                                         # Update model for next iteration with the selected variable
        X = np.delete(X, idx_to_remove, axis=1)                                                   
        iter += 1

# RESET test function implementation
def RESET_test(y, X, p):                                        # Enter dependent variable, model regressor matrix, and the number of fitted values powers to be tested
    n, k = X.shape[0], X.shape[1]                               # Storing model's sample size and number of explanatory variables
    b, _, _, _, _, _, S0, _ = mult_regr(y, X)                      # Obtain model's OLS estimator to use for fitted values calculation and sum of squared residuals to be used later in the F statistic calculation
    y_fit = X @ b                                               # Obtain model's fitted values
    i = 0
    X_reset = X
    while (i < p):                                              # While loop to build RESET test regressor matrix according to defined p
        y_pow = y_fit ** (i + 2)
        X_reset = np.column_stack([X_reset, y_pow])
        i += 1
    S1 = mult_regr(y, X_reset)[6]                               # Obtain sum of squared residuals for the unrestricted model
    F = ((S0 - S1) / p) / (S1 / (n - k - p))                    # F-statistic for RESET test
    p = 1 - f.cdf(F, dfn=p, dfd=n-k-p)                          # p-value for RESET test
    return F, p

# Chow break test function implementation
def Chow_break_test(y, X, df, break_point):                         # Enter dependent variable, model regressor matrix, dataset, and sample break point
    n, k = X.shape[0], X.shape[1]                                   # Storing model's sample size and number of explanatory variables
    break_idx = df[df["OBS"] == break_point].index.item()           # Get index of breaking point in dataset
    y_1, y_2 = y.iloc[:break_idx], y.iloc[break_idx:]               # Define the 2 samples
    X_1, X_2 = X[:break_idx], X[break_idx:]
    S0 = mult_regr(y, X)[6]                                         # Null hypothesis: β1 = β2 restricted model, SSRs of full sample
    S1 = mult_regr(y_1, X_1)[6]                                     # SSRs of first sample
    S2 = mult_regr(y_2, X_2)[6]                                     # SSRs of second sample
    F = ((S0 - S1 - S2) / k) / ((S1 + S2) / (n - 2 * k))            # F-statistic for Chow break test
    p = 1 - f.cdf(F, dfn=k, dfd=n-2*k)                              # p-value for Chow break test
    return F, p

# Chow forecast test function implementation
def Chow_forecast_test(y, X, df, break_point):                      # Enter dependent variable, model regressor matrix, dataset, and sample break point
    k = X.shape[1]                                                  # Storing model's number of explanatory variables
    break_idx = df[df["OBS"] == break_point].index.item()           # Get index of breaking point in dataset
    y_1, y_2 = y.iloc[:break_idx], y.iloc[break_idx:]               # Define the 2 samples
    X_1 = X[:break_idx]
    n1, n2 = len(y_1), len(y_2)                                     # Subsample size
    S0 = mult_regr(y, X)[6]                                         # Null hypothesis: γ = 0
    S1 = mult_regr(y_1, X_1)[6]                                     # SSRs of first sample (don't need to calculate for second sample because they are zero)
    F = ((S0 - S1) / n2) / (S1 / (n1 - k))                          # F-statistic for Chow forecast test
    p = 1 - f.cdf(F, dfn=n2, dfd=n1-k)                              # p-value for Chow forecast test
    return F, p

def Jarque_Bera(y, X):                                              # Enter dependent variable, and model regressor matrix
    n = X.shape[0]                                                  # Get sample size
    e = mult_regr(y, X)[7]                                          # Get residuals
    S = skew(e, bias=False)                                         # Get sample skewness
    K = kurtosis(e, fisher=False, bias=False)                       # Get sample kurtosis
    JB = (n / 6) * S**2 + (n / 24) * (K - 3)**2                     # Jarque-Bera statistic
    p = 1 - chi2.cdf(JB, df=2)                                      # It has χ-squared(2) distribution if residuals are normal
    return JB, p

# Define regressor matrix for the general model with all 7 explanatory variables
unit_vector = np.ones(len(df))                                  # (660 x 1) unit vector
INFLi = df["INFL"]                                              # (660 x 1) inflation observations vector
PRODi = df["PROD"]                                              # (660 x 1) production observations vector
UNEMPLi = df["UNEMPL"]                                          # (660 x 1) unemployment observations vector
COMMPRIi = df["COMMPRI"]                                        # (660 x 1) commodity prices observations vector
PCEi = df["PCE"]                                                # (660 x 1) personal consumption expenditure observations vector
PERSINCi = df["PERSINC"]                                        # (660 x 1) personal income observations vector
HOUSTi = df["HOUST"]                                            # (660 x 1) housing starts observations vector
X = np.column_stack([unit_vector, INFLi, PRODi, UNEMPLi,        # Stack these vectors to form the (660 x 8) regressor matrix X
                    COMMPRIi, PCEi, PERSINCi, HOUSTi])   
variable_names = ['const', 'INFL', 'PROD', 'UNEMPL', 'COMMPRI', # Store explanatory variable names for printing purposes
            'PCE', 'PERSINC', 'HOUST']

# Define dependent variable
y = df["INTRATE"]                                               # Federal funds interest rate

print(f"\nQuestion (a): General-to-specific approach")

# Question (a): Run general-to-specific to come to a model
g_sp_model = gen_to_spec(y, X, 0.05, variable_names)

print(
    f"\nAfter applying the general-to-specific approach, the final model gives:\n"
    f"\nOLS coefficient vector b = {np.round(g_sp_model[0], 3)}\n"
    f"\nt-statistic vector = {np.round(g_sp_model[1], 3)}\n"
    f"\np-value vector = {np.round(g_sp_model[2], 3)}\n"
    f"\nR-squared = {g_sp_model[3]:.3f}\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Define total variable list for the specific-to-general model
tot_vars = ['const', 'INFL', 'PROD', 'UNEMPL', 'COMMPRI', 
            'PCE', 'PERSINC', 'HOUST']

print(f"\nQuestion (b): Specific-to-general approach")

# Question (b): Run specific-to-general to come to a model
sp_g_model = spec_to_gen(y, X, 0.05, tot_vars)

print(
    f"\nAfter applying the specific-to-general approach, the final model gives:\n"
    f"\nOLS coefficient vector b = {np.round(sp_g_model[0], 3)}\n"
    f"\nt-statistic vector = {np.round(sp_g_model[1], 3)}\n"
    f"\np-value vector = {np.round(sp_g_model[2], 3)}\n"
    f"\nR-squared = {sp_g_model[3]:.3f}\n"
    f"\nWhile the selection paths differ because the significance of a variable depends on the other variables already included in the model,\n"
    f"the specific-to-general approach leads to the same final model as the general-to-specific approach.\n"
    f"\n-----------------------------------------------------------------------\n"
)

print(f"\nQuestion (c): General-to-specific model versus Taylor rule model")

# Define regressor matrix for Taylor rule model from equation (1)
X_taylor = np.column_stack([unit_vector, INFLi, PRODi])             # Stack inflation and production observation vectors to form the (660 x 3) regressor matrix X_taylor

# Regress the Taylor rule model
Taylor_rule = mult_regr(y, X_taylor)

print(
    f"\nFor the Taylor rule model we have:\n"
    f"\nR-squared = {Taylor_rule[3]:.3f}\n"
    f"\nAIC = {Taylor_rule[4]:.3f}\n"
    f"\nBIC = {Taylor_rule[5]:.3f}\n"
    f"\nFor the general-to-specific model we have:\n"
    f"\nR-squared = {g_sp_model[3]:.3f}\n"
    f"\nAIC = {g_sp_model[4]:.3f}\n"
    f"\nBIC = {g_sp_model[5]:.3f}\n"
    f"\nThe general-to-specific model is preferred over the simple Taylor rule specification.\n"
    f"\nIt provides a better fit to the data (R-squared = 0.637 versus 0.575) and achieves lower AIC and BIC values.\n\nTherefore, the improvement in explanatory power more than compensates for the increase in model complexity.\n"
    f"\n-----------------------------------------------------------------------\n"
)

print(f"\nQuestion (d): Taylor rule model evaluation")

# Run RESET test on Taylor rule for p = 1, 2, 3
Taylor_reset = RESET_test(y, X_taylor, 1)[1], RESET_test(y, X_taylor, 2)[1], RESET_test(y, X_taylor, 3)[1]

print(
    f"\nUsing p = 1, the RESET test gives a p-value = {np.round(Taylor_reset[0], 3)} > 0.05, so we do not reject the null of correct specification at the 5% level.\n"
    f"\nAs a robustness check, p = 2 gives a p-value = {np.round(Taylor_reset[1], 3)} > 0.05, so it also does not reject.\n"
    f"\nHowever, RESET test for p = 3 gives p-value = {np.round(Taylor_reset[2], 3)} < 0.05, suggesting possible higher-order nonlinearities.\n"
    f"\nSince the exercise does not specify the RESET order, I base the main conclusion on p = 1."
)

# Run Chow break test on Taylor rule for 1980
Chow_break = Chow_break_test(y, X_taylor, df, "1980:1")

print(
    f"\nThe Chow break test yields an F-statistic of {np.round(Chow_break[0], 3)} with a p-value that is effectively zero.\n"
    f"\nTherefore, we reject the null hypothesis of parameter stability across the two subsamples.\n"
    f"\nHence, there is strong evidence of a structural break in the Taylor rule at 1980."
)

# Run Chow forecast test on Taylor rule for 1980
Chow_forecast = Chow_forecast_test(y, X_taylor, df, "1980:1")

print(
    f"\nThe Chow forecast test yields an F-statistic of {np.round(Chow_forecast[0], 3)} with a p-value of {np.round(Chow_forecast[1], 3)}.\n"
    f"\nTherefore, we reject the null of γ = 0. This means that the Taylor rule estimated on the pre-1980 sample does not adequately describe\nthe post-1980 observations.\n"
    f"\nTogether with the Chow break test, this suggests that the Taylor rule relationship changed substantially after 1980."
)

# Test for normality of residuals with Jarque-Bera test
JB_Taylor = Jarque_Bera(y, X_taylor)

print(
    f"\nThe Jarque-Bera test gives JB = {np.round(JB_Taylor[0], 3)} with a p-value of {np.round(JB_Taylor[1], 3)} < 0.05.\n"
    f"\nTherefore, we reject the null hypothesis that the error terms are normally distributed.\n"
    f"\nCombined with the Chow tests, this suggests that the Taylor rule model suffers from structural instability and non-normal residuals.\n"
    f"\n-----------------------------------------------------------------------\n"
)