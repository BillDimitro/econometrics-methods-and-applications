import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f, chi2

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_5\datasets/TestExer4.txt", sep=",")

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
    R_sq = mult_regr(e_2sls, Z)[3]                              # Get model fit of regressing the residuals to Z
    Sargan = n * R_sq                                           # Sargan statistic
    p = 1 - chi2.cdf(Sargan, df=m-k)

    return Sargan, p, R_sq

print(f"\nQuestion (a): OLS in given model")

# Define regressor matrix for model explaining logWage with educ, exper, smsa, and south
unit_vector = np.ones(len(df))                                  # unit vector
educi = df["educ"]                                              # number of years of schooling vector
experi = df["exper"]                                            # working experience in years vector
exper2i = experi**2                                             # working experience in years squared vector
smsai = df["smsa"]                                              # lived in metropolitan area dummy vector
southi = df["south"]                                            # lived in the south vector
X = np.column_stack([unit_vector, educi, experi, exper2i,       # Stack these vectors to form the regressor matrix X
                    smsai, southi])   

# Define dependent variable
y = df["logw"]                                               # log wage

# Run regression on question (a) model
OLS = mult_regr(y, X)

print(
    f"\nRegressing logwage on a constant, education, experience, experience squared, and the metropolitan and south dummies yields:\n"
    f"\nConstant ->\t\tb = {np.round(OLS[0][0], 3)}\tSE = {np.round(OLS[1][0], 3)}\tt-statistic = {np.round(OLS[2][0], 3)}\n"
    f"\nEducation ->\t\tb = {np.round(OLS[0][1], 3)}\tSE = {np.round(OLS[1][1], 3)}\tt-statistic = {np.round(OLS[2][1], 3)}\n"
    f"\nExperience ->\t\tb = {np.round(OLS[0][2], 3)}\tSE = {np.round(OLS[1][2], 3)}\tt-statistic = {np.round(OLS[2][2], 3)}\n"
    f"\nExperience2 ->\t\tb = {np.round(OLS[0][3], 3)}\tSE = {np.round(OLS[1][3], 3)}\tt-statistic = {np.round(OLS[2][3], 3)}\n"
    f"\nMetropolitan ->\t\tb = {np.round(OLS[0][4], 3)}\tSE = {np.round(OLS[1][4], 3)}\tt-statistic = {np.round(OLS[2][4], 3)}\n"
    f"\nSouth ->\t\tb = {np.round(OLS[0][5], 3)}\tSE = {np.round(OLS[1][5], 3)}\tt-statistic = {np.round(OLS[2][5], 3)}\n"
    f"\nR-squared = {OLS[3]:.3f}\n"
    f"\nThe estimated β2 = 0.082 implies that, holding experience and location variables constant, an additional year of schooling is associated with an approximately 8.2% higher wage.\n"
    f"\n-----------------------------------------------------------------------\n"
)

print(f"\nQuestion (b): Possible endogeneity of education and experience")

print(
    f"\nEducation and experience may be endogenous because unobserved factors such as ability, motivation, and ambition can affect both education and work experience.\n"
    f"\nSpecifically, highly motivated people tend to pursue higher education, but also perform better in their job.\n"
    f"\nTherefore, omitting motivation as a variable may cause the OLS estimate to overshoot the effect of education on wage.\n"
    f"\nFor experience, a similar argument can be made: More motivated or able individuals may pursue internships or full-time jobs sooner, gaining more experience.\n"
    f"But they also perform better, which may also leads to increased wage.\n"
    f"\nTo conclude, both of these regressors may be correlated with the error term, where the omitted variable of motivation may appear, so the OLS estimators may be biased and inconsistent.\n"
    f"\nEven if the OLS estimates may be biased due to endogeneity, they are still useful because they provide an initial estimate of the relationship between education, experience, location, and wages.\n"
    f"\nComparing them with the results of the upcoming 2SLS method will help us assess the importance of a possible endogeneity problem.\n"
    f"\n-----------------------------------------------------------------------\n"
)


print(f"\nQuestion (c): Age and Age-squared as instruments for Experience and Experience-squared")

print(
    f"\nAge and age squared can be motivated as instruments for experience and experience squared because they are strongly related to work experience.\n"
    f"\nSpecifically, older individuals have had more time to accumulate work experience, and the nonlinear term age squared can help explain the nonlinear component experience squared.\n"
    f"\nTherefore, these instruments are likely to satisfy the correlation with the endogenous variable experience condition.\n"
    f"\nAssuming age and age² are uncorrelated with the error term, they can be used as instruments for experience and experience².\n"
    f"\n-----------------------------------------------------------------------\n"
)

print(f"\nQuestion (d): First stage regression for educ with age, age-squared, nearc, dadeduc, and momeduc as additional instruments.")

# Define first stage regressor matrix
unit_vector = np.ones(len(df))                                  # unit vector
agei = df["age"]                                                # age vector
age2i = agei**2                                                 # age squared vector
nearci = df["nearc"]                                            # lived near college dummy
dadeduci = df["daded"]                                          # education of the individual’s father
momeduci = df["momed"]                                          # education of the individual’s mother
X_stg1 = np.column_stack([unit_vector, smsai, southi, agei,     # Stack these vectors to form the regressor matrix X_stg1
                        age2i, nearci, dadeduci, momeduci])        

# Define dependent variable for the first stage regression
y_stg1 = df["educ"]                                             # number of years of schooling vector

# Perform first stage regression
stg1_rgr = mult_regr(y_stg1, X_stg1)  

print(
    f"\nThe first stage regression, in which we explain Education using all instruments, yields:\n"
    f"\nConstant ->\t\tb = {np.round(stg1_rgr[0][0], 3)}\tSE = {np.round(stg1_rgr[1][0], 3)}\tt-statistic = {np.round(stg1_rgr[2][0], 3)}\n"
    f"\nMetropolitan ->\t\tb = {np.round(stg1_rgr[0][1], 3)}\tSE = {np.round(stg1_rgr[1][1], 3)}\tt-statistic = {np.round(stg1_rgr[2][1], 3)}\n"
    f"\nSouth ->\t\tb = {np.round(stg1_rgr[0][2], 3)}\tSE = {np.round(stg1_rgr[1][2], 3)}\tt-statistic = {np.round(stg1_rgr[2][2], 3)}\n"
    f"\nAge ->\t\t\tb = {np.round(stg1_rgr[0][3], 3)}\tSE = {np.round(stg1_rgr[1][3], 3)}\tt-statistic = {np.round(stg1_rgr[2][3], 3)}\n"
    f"\nAge2 ->\t\t\tb = {np.round(stg1_rgr[0][4], 3)}\tSE = {np.round(stg1_rgr[1][4], 3)}\tt-statistic = {np.round(stg1_rgr[2][4], 3)}\n"
    f"\nNear College ->\t\tb = {np.round(stg1_rgr[0][5], 3)}\tSE = {np.round(stg1_rgr[1][5], 3)}\tt-statistic = {np.round(stg1_rgr[2][5], 3)}\n"
    f"\nDad Educ ->\t\tb = {np.round(stg1_rgr[0][6], 3)}\tSE = {np.round(stg1_rgr[1][6], 3)}\tt-statistic = {np.round(stg1_rgr[2][6], 3)}\n"
    f"\nMom Educ ->\t\tb = {np.round(stg1_rgr[0][7], 3)}\tSE = {np.round(stg1_rgr[1][7], 3)}\tt-statistic = {np.round(stg1_rgr[2][7], 3)}\n"
    f"\nR-squared = {stg1_rgr[3]:.3f}\n"
    f"\nWe will now perform an F-test to test the joint significance of the gamma coefficients for the instruments used."
)

# Define restricted model regressor matrix for the F-test
X0 = np.column_stack([unit_vector, smsai, southi]) 

# Perform F-test
f_test_γ = F_test(y_stg1, X0, X_stg1, 5)

print(
    f"\nThe F-test for sufficient correlation of the instruments with the endogenous variable Education yields:\n"
    f"\nF-statistic = {np.round(f_test_γ[0], 3)}, with a p-value = {np.round(f_test_γ[1], 3)}.\n"
    f"\nTherefore, we reject the null hypothesis that γ = 0, meaning that the proposed instruments seem to be jointly significant predictors of schooling.\n"
    f"\n-----------------------------------------------------------------------\n"
)

print(f"\nQuestion (e): Use 2SLS to correct for endogeneity of Education and Experience variables.")

# Define instrument matrix Z
Z = np.column_stack([unit_vector, smsai, southi, agei, age2i, nearci, dadeduci, momeduci])

# Perform 2SLS
twoSLS = two_sls(y, X, Z)

print(
    f"\nUsing 2SLS to correct for endogeneity of Education and Experience yields:\n"
    f"\nConstant ->\t\tb = {np.round(twoSLS[0][0], 3)}\tSE = {np.round(twoSLS[1][0], 3)}\tt-statistic = {np.round(twoSLS[2][0], 3)}\n"
    f"\nEducation ->\t\tb = {np.round(twoSLS[0][1], 3)}\t\tSE = {np.round(twoSLS[1][1], 3)}\tt-statistic = {np.round(twoSLS[2][1], 3)}\n"
    f"\nExperience ->\t\tb = {np.round(twoSLS[0][2], 3)}\tSE = {np.round(twoSLS[1][2], 3)}\tt-statistic = {np.round(twoSLS[2][2], 3)}\n"
    f"\nExperience2 ->\t\tb = {np.round(twoSLS[0][3], 3)}\tSE = {np.round(twoSLS[1][3], 3)}\tt-statistic = {np.round(twoSLS[2][3], 3)}\n"
    f"\nMetropolitan ->\t\tb = {np.round(twoSLS[0][4], 3)}\tSE = {np.round(twoSLS[1][4], 3)}\tt-statistic = {np.round(twoSLS[2][4], 3)}\n"
    f"\nSouth ->\t\tb = {np.round(twoSLS[0][5], 3)}\tSE = {np.round(twoSLS[1][5], 3)}\tt-statistic = {np.round(twoSLS[2][5], 3)}\n"
    f"\nR-squared = {twoSLS[3]:.3f}\n"  
    f"\nThe 2SLS estimate of the return to schooling increases from 8.2% to approximately 10%, suggesting a larger causal effect of education on wages\n"
    f"than implied by OLS.\n"
    f"\nThe 2SLS estimate of the return to experience decreases from 8.4% to 7.3%, indicating a somewhat smaller effect of market experience after\n"
    f"correcting for endogeneity.\n"  
    f"\n-----------------------------------------------------------------------\n"
)

print(f"\nQuestion (f): Perform the Sargan test for validity of the instruments.")

b_2sls = twoSLS[0]
sargan = Sargan_test(y, X, Z, b_2sls)

print(
    f"\nAfter performing the Sargan test to test whether the instruments of Z are correlated with ε, we get:\n"
    f"\nR-sq = {np.round(sargan[2], 3)}, Sargan = n * R_sq = {np.round(sargan[0], 3)}, and a p-value = {np.round(sargan[1], 3)}.\n"
    f"\nSince {np.round(sargan[1], 3)} > 0.05, we fail to reject the null hypothesis of no correlation between the instruments and ε.\n"
    f"\nIn this sense, the instruments used appear valid.\n"
    f"\n-----------------------------------------------------------------------\n"
)