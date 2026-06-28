import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t, f
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_7\datasets/TrainExer61.txt", sep=r"\s+")

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

def F_test(y, X0, X1, q):                                       # Enter dependent variable, regressor matrices of restricted and unrestricted model, and number of restrictions

    n = len(y)                                                  # Sample size
    k = X1.shape[1]                                             # Number of explanatory variables of the unrestricted model
    S0 = mult_regr(y, X0)[4]                                    # Get SSRs of restricted model
    S1 = mult_regr(y, X1)[4]                                    # Get SSRs of unrestricted model
    F = ((S0 - S1) / q) / (S1 / (n - k))                        # F-statistic
    p = 1 - f.cdf(F, dfn=q, dfd=n-k)                            # p-value

    return F, p

# Parse xt, yt, εyt, and εxt
xt = df["X"]
yt = df["Y"]
εxt = df["EPSX"]
εyt = df["EPSY"]

# Plot xt and yt against time
time = np.arange(len(xt)) 

plt.figure(figsize=(10, 5))
plt.plot(time, xt, label='x_t')
plt.plot(time, yt, label='y_t')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series')
plt.legend()
plt.grid(True)

#plt.show()

# Scatter diagram of yt against xt
plt.figure(figsize=(10,5))
plt.scatter(xt, yt, edgecolor="black")

plt.title("Scatter Diagram of y_t vs x_t")
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.grid(alpha=0.3)

#plt.show()

print(
    f"\nThe time-series plots show that both xt and yt follow persistent trends typical of random walks. The scatter plot suggests a strong negative relationship between the two variables.\n"
    f"\nHowever, since xt and yt are generated from independent white-noise processes, they are in fact unrelated.\n" 
    f"\nTherefore, the apparent correlation is likely spurious and caused by the non-stationary trending behavior of the series.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Regression of εxt on a constant and εyt to prove they are not correlated
SR_e = simple_regr(εyt, εxt)

print(
    f"\nRegressing εyt on a constant and εxt yields a slope b = {SR_e[1]:.3f}, with a t-value = {SR_e[4]:.3f} and a p-value = {SR_e[5]:.3f}.\n"
    f"\nTherefore, we can not reject the null of β = 0, meaning that coefficient is not statistically significantly different from zero.\n"
    f"\nThere is insufficient evidence to conclude that εxt has an effect on εyt, thus they seem to be uncorrelated.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Create lagged values
df["EPSX_lag1"] = df["EPSX"].shift(1)
df["EPSX_lag2"] = df["EPSX"].shift(2)
df["EPSX_lag3"] = df["EPSX"].shift(3)
df["EPSY_lag1"] = df["EPSY"].shift(1)
df["EPSY_lag2"] = df["EPSY"].shift(2)
df["EPSY_lag3"] = df["EPSY"].shift(3)
df = df.dropna()                                # We lose 3 observations to create the lagged values of εxt and εyt

# Define regressor matrix of unrestricted model
unit_vector = np.ones(len(df))                  
epsx, epsx_lag1, epsx_lag2, epsx_lag3 = df["EPSX"], df["EPSX_lag1"], df["EPSX_lag2"], df["EPSX_lag3"]
epsy, epsy_lag1, epsy_lag2, epsy_lag3 = df["EPSY"], df["EPSY_lag1"], df["EPSY_lag2"], df["EPSY_lag3"]
X1 = np.column_stack([unit_vector, epsx, epsy_lag1, epsy_lag2, epsy_lag3, 
                    epsx_lag1, epsx_lag2, epsx_lag3])

# Define regressor matrix of restricted model
X0 = unit_vector

# Define dependent variable
y = epsy

# Regress εyt on a constant, εxt, 3 lagged values of εyt, and 3 lagged values of εxt
MR_e = mult_regr(y, X1)

# F-test
n = len(df)
k = X1.shape[1]                                             # Variables of unrestricted model
q = 7                                                       # Number of restrictions
S0 = np.sum((y - np.mean(y))**2)                            # SSRs for an intercept only model
S1 = MR_e[4]                                                # SSRs of unrestricted model
F = ((S0 - S1) / q) / (S1 / (n - k))                        # F-statistic
p = 1 - f.cdf(F, dfn=q, dfd=n-k)                            # p-value

print(
    f"\nThe F-test for for the joint insignificance of the seven parameters yields:\n"
    f"\nF(7, 239) = {np.round(F, 3)}, with a p-value = {np.round(p, 3)}.\n"
    f"\nSince the p-value is much larger than 0.05, we fail to reject the null hypothesis that all seven coefficients are jointly equal to zero.\n"
    f"\nmeaning that there is neither correlation bwtween εxt and εyt, nor between each series past values.\n"
    f"\nTherefore, there is no statistical evidence that current or lagged values of εxt or lagged values of εyt help explain εyt.\n"
    f"\nThis is consistent with the assumption that εxt and εyt are independent white-noise processes.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Regression of y on a constant and x
SR_y = simple_regr(yt, xt)

print(
    f"\nRegressing yt on a constant and xt yields a slope b = {SR_y[1]:.3f}, with a t-value = {SR_y[4]:.3f} and a p-value = {SR_y[5]:.3f}. Therefore, b is statistically significant.\n"
    f"\nWe would be tempted to conclude that the two variables have a strong negative relationship if we didn't know the data generating process.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Create lagged values of et from regressing y with x
df_d = pd.DataFrame(SR_y[2], columns=["et"])
df_d["et_lag1"] = df_d["et"].shift(1)
df_d = df_d.dropna()

# Regress et on a constant and et-1
SR_et = simple_regr(df_d["et"], df_d["et_lag1"])

print(
    f"\nRegressing et on a constant and et-1 yields a slope b = {SR_et[1]:.3f}, with a t-value = {SR_et[4]:.3f} and a p-value = {SR_et[5]:.3f}.\n"
    f"\nTherefore, we reject the null of β = 0, meaning that the coefficient of the lagged residual is statistically significant, indicating that the residuals are serially correlated.\n"
    f"\nThis violates the standard regression assumption A5, which states that E(eiej) = 0 for all i,j.\n"
    f"\n-----------------------------------------------------------------------\n"
)


