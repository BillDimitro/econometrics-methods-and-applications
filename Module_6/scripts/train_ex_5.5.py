import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2

df = pd.read_csv(
    r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_6\datasets/TrainExer55.txt",
    sep=r"\s+",
    skiprows=1,                                                                                             # rep id is being parsed as 2 separate columns, needs mod
    names=["rep_id", "response", "male", "activity", "age"]
)

# Define regressor matrix for logit model explaining Response with Male, Activity, anf Age
unit_vector = np.ones(len(df))                                      # unit vector
malei = df["male"]                                                  # gender dummy vector
activei = df["activity"]                                            # activity indicator vector
agei = df["age"]                                                    # age vector
age2i = (agei / 10)**2                                              # age in decades squared vector
X = np.column_stack([unit_vector, malei, activei, agei, age2i])     # Stack these vectors to form the regressor matrix X   

# Define dependent variable
y = df["response"]                                                  # Binary variable: 1 -> yes 0 -> no

# Define logit model
model = sm.Logit(y, X)

# Estimate Maximum Likelihood
mle = model.fit(disp=0)

print(
    f"\nBelow lie the maximum likelihood estimates of our logit specification:\n"
    f"\n{mle.summary()}\n"
    f"\nThese match the results shown in Lecture 5.5.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Define the complement response variable
y_compl = 1 - y                                                     # Complement Binary variable: 0 -> yes 1 -> no

# Define logit model
model_compl = sm.Logit(y_compl, X)

# Estimate Maximum Likelihood
mle_compl = model_compl.fit(disp=0)

print(
    f"\nBelow lie the maximum likelihood estimates of the complement model specification:\n"
    f"\n{mle_compl.summary()}\n"
    f"\nAfter redefining the response variable so that the previous successes become failures and vice versa, all estimated coefficients change sign,\n" 
    f"while keeping the same magnitude.\n"
    f"\nThis can be interpreted as: Variables that originally increased the probability of a positive response now decrease it by the same amount, and vice versa.\n"
    f"\n-----------------------------------------------------------------------\n"
)

# Define regressor matrix for restricted logit model (β1=β2=0)
X0 = np.column_stack([unit_vector, agei, age2i])

# Define restricted logit model
model0 = sm.Logit(y, X0)

# Estimate Maximum Likelihood for restricted model
mle0 = model0.fit(disp=0)

# Get max log likelihoods for restricted and unrestricted model
llf0 = mle0.llf
llf1 = mle.llf

# Likelihood ratio test on H0: β1=β2=0
m = 2                                   # number of restrictions
LR = -2 * (llf0 - llf1)                 # likelihood ratio statistic
p = 1 - chi2.cdf(LR, df=m)              # p-value

print(
    f"\nThe LR-test for the null hypothesis of β1=β2=0 yields:\n"
    f"\nLR-statistic = {np.round(LR, 3)}, with a p-value = {np.round(p, 3)}.\n"
    f"\nSince {np.round(p, 3)} < 0.05, we reject the null hypothesis that β1 = β2 = 0, meaning that gender and activity are jointly significant and improve the explanatory power\n"
    f"of the model.\n"
    f"\n-----------------------------------------------------------------------\n"
)