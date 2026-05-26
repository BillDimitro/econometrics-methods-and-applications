import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

# Choose between the original and the dataset without the outlier week
df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_2\datasets/TestExer1.txt", sep="\t")    # original
#df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_2\datasets/TestExer1b.txt", sep="\t")    # without outlier      

# Scatter Diagram Sales - Advertising
plt.figure(figsize=(8,5))
plt.scatter(df["Advert."], df["Sales"], edgecolor="black")

plt.title("Scatter Diagram of Sales vs Advertising")
plt.xlabel("Advertising")
plt.ylabel("Sales")
plt.grid(alpha=0.3)

plt.show()

# Compute a and b
Ai = df["Advert."]
Si = df["Sales"]

b = Ai.cov(Si) / Ai.var()
a = Si.mean() - b * Ai.mean()

print(f"\nThe intercept and slope of the regression line is a = {a:.3f} and b = {b:.3f} respectively.\n")

# Compute R-squared
R_sq = b**2 * Ai.var() / Si.var()

# Compute Error Variance
ei = Si - a - b * Ai
n = len(ei)
s = np.sqrt((ei**2).sum() / (n-2))

print(f"The values of R squared and s are R_sq = {R_sq:.3f} and s = {s:.3f} respectively.\n")

# Compute std_err & t-value of b
sb = np.sqrt(s**2 / (Ai.var() * (n-1)))  # Multiplying by (n-1) because .var() divides the sum of squared deviations by n-1 to compute sample variance
tb = b / sb

print(f"The standard error and t-value of b are sb = {sb:.3f} and tb = {tb:.3f} respectively.\n")

# Residuals
print(f"Vector of residuals ei:\n{np.array2string(ei.to_numpy(), precision=2, separator=',')}.\n")

# Residuals Histogram
plt.figure(figsize=(8,5))
plt.hist(ei, bins=30, align='left', edgecolor="black")

plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()