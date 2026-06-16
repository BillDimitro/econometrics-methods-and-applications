import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_4\datasets/TrainExer31.txt", sep=r"\s+") # Split columns wherever there is one or more whitespace characters, spacing is weird in this dataset

df["dlog_index"] = np.log(df["Index"]).diff()   # Calculate the change of log(Index)
df = df.dropna()                                # The first datapoint has no previous value to calculate the diff with, so we drop it from the dataset

print(f"\nWe now have {len(df)} datapoints of index values, which is one less that the initial 87. Since we use the change of log(index), the first value can not be defined.\n")

def regression(y, x):
    
    n = len(y)
    b = x.cov(y) / x.var()
    a = y.mean() - b * x.mean()
    ei = y - a - b * x
    s = np.sqrt((ei**2).sum() / (n-2))
    sb = np.sqrt(s**2 / (x.var() * (n-1)))
    tb = b / sb

    return a, b, ei, tb

Bm = df["BookMarket"]
dlog_Idx = df["dlog_index"]

d_log_idx_Bm = regression(dlog_Idx, Bm)

print(f"\nRegressing the change in the log of the S&P500 index on a constant and the book-to market ratio, yields a constant = {d_log_idx_Bm[0]:.3f} and a coefficient b = {d_log_idx_Bm[1]:.3f}, with tb = {d_log_idx_Bm[3]:.3f}, validating the result presented in Lecture 3.1.\n")

Idx = df["Index"]
Idx_Bm = regression(Idx, Bm)

print(f"\nRegressing the S&P500 index on a constant and the book-to market ratio, yields a constant = {Idx_Bm[0]:.3f} and a coefficient b = {Idx_Bm[1]:.3f}, with tb = {Idx_Bm[3]:.3f}. Therefore, the Book-to-Market ratio is statistically significant in this specification too.\n")

e_a = d_log_idx_Bm[2]
e_b = Idx_Bm[2]
t = df["Year"]

# Plot residuals from (a) and (b)
plt.figure(figsize=(10,5))
plt.plot(t, e_a, label="Residuals of d_log_index model")
plt.plot(t, e_b, label="Residuals of raw index model")
plt.axhline(0, color='black', linestyle='--')
plt.legend()
plt.show()

print("The residuals of model (a), which uses the change in the logarithm of the S&P500 index, appear to be centered around zero and exhibit small variance. In contrast, the residuals of model (b), which uses the index level, display large fluctuations and persistent periods of positive and negative values. This suggests that the Book-to-Market ratio alone cannot adequately explain the level of the S&P500 index. The transformation used in part (a) produces a more stable specification and results in residuals that better satisfy the assumptions of the linear regression model.\n")