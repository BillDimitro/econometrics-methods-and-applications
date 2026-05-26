import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_2\datasets/TrainExer13.txt", sep="\t")

# Compute a and b
Gi = df["Game"]
Wi = df["Winning time men"]

b = Gi.cov(Wi) / Gi.var()
a = Wi.mean() - b * Gi.mean()

print(f"\nThe intercept and slope of the regression line is a = {a:.3f} and b = {b:.3f} respectively.\n")

# Compute R-squared
R_sq = b**2 * Gi.var() / Wi.var()

# Compute Error Variance
ei = Wi - a - b * Gi
n = len(ei)
s = np.sqrt((ei**2).sum() / (n-2))

print(f"The values of R squared and s are R_sq = {R_sq:.3f} and s = {s:.3f} respectively.\n")

# Prediction Function
def Winmen(game_nr):
    W = a + b * game_nr
    return W
    
# Winning Time Predictions for 2008, 2012, 2016
W_2008, W_2012, W_2016 = Winmen(16), Winmen(17), Winmen(18)

print(f"The model predicts the winning times of 2008, 2012, 2016 as {W_2008:.3f}, {W_2012:.3f}, and {W_2016:.3f} respectively.\n")