import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_2\datasets/TrainExer11.txt", sep="\t")

exp_uniq = df["Expenditures"].nunique()
exp_min = df["Expenditures"].min()
exp_max = df["Expenditures"].max()
binwidth = 1
#print(df["Expenditures"].value_counts()) # For cross-validation pursposes

# Expenditures Histogram
plt.figure(figsize=(8,5))
plt.hist(df["Expenditures"], bins=range(exp_min, exp_max + binwidth + 1, binwidth), align='left', edgecolor="black")

plt.title("Histogram of Expenditures")
plt.xlabel("Expenditures")
plt.xticks(range(exp_min, exp_max + 1, 1))
plt.ylabel("Frequency")
plt.show()

# Age Histogram
age_uniq = df["Age"].nunique()
age_min = df["Age"].min()
age_max = df["Age"].max()
binwidth = 1
#print(df["Age"].value_counts()) # For cross-validation purposes

# Ages Histogram
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=range(age_min, age_max + binwidth + 1, binwidth), align='left', edgecolor="black")

plt.title("Histogram of Ages")
plt.xlabel("Ages")
plt.xticks(range(age_min, age_max + 1, 1), rotation=90)
plt.ylabel("Frequency")
plt.show()

# Scatter Diagram Expenditures - Age
plt.figure(figsize=(8,5))
plt.scatter(df["Age"], df["Expenditures"], edgecolor="black")

plt.title("Scatter Diagram of Expenditures vs Age")
plt.xlabel("Age")
plt.ylabel("Expenditures")
plt.grid(alpha=0.3)

plt.show()

# Mean Expenditures
mean_exp = df["Expenditures"].mean()
print(f"\nThe sample mean of expenditures of all 26 clients is {mean_exp:.1f}.\n")

# Mean of clusters
df_40_or_more = df[df["Age"] >= 40]
mean_exp_40_or_more = df_40_or_more["Expenditures"].mean()
print(f"The sample mean of expenditures for clients of age >= 40 is {mean_exp_40_or_more:.1f}.\n")

df_40_minus = df[df["Age"] < 40]
mean_exp_40_minus = df_40_minus["Expenditures"].mean()
print(f"The sample mean of expenditures for clients of age < 40 is {mean_exp_40_minus:.1f}.\n")