import pandas as pd
import numpy as np
import math

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_3\datasets/TrainExer21.txt", sep="\t")

def log_wage_params(dataset):
    
    # Compute a and b
    Female = dataset["Female"]
    Wi = dataset["Wage"]

    b = Female.cov(np.log(Wi)) / Female.var()
    a = np.log(Wi).mean() - b * Female.mean()
    ei = np.log(Wi) - a - b * Female

    return a, b, ei

params = log_wage_params(df)

print(f"\nRegressing log-wage on a constant and the gender dummy ‘Female’ yields an intercept a = {params[0]:.2f} and a slope b = {params[1]:.2f}.\n")

def residual_regression(e_dataset, variable):
    
    # Compute a and b
    variabl = df[variable]
    
    b = variabl.cov(e_dataset) / variabl.var()
    a = e_dataset.mean() - b * variabl.mean()

    return a, b

residuals = params[2]
resid_edu = residual_regression(residuals, "Educ")
resid_job = residual_regression(residuals, "Parttime")

print(f"\nRegressing e on a constant and education yields an intercept a = {resid_edu[0]:.2f} and a slope b = {resid_edu[1]:.2f}.\n")
print(f"\nRegressing e on a constant and the part-time job dummy yields an intercept a = {resid_job[0]:.2f} and a slope b = {resid_job[1]:.2f}.\n")