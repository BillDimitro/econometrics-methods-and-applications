import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_5\datasets/TrainExer42.txt", sep=",")

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
    
    return a, b, ei, R_sq, tb, SSR

_, b0_0, _, R_sq0_0, _, _ = simple_regr(df["SALES0_0"], df["PRICE0"])
_, b0_1, _, R_sq0_1, _, _ = simple_regr(df["SALES0_1"], df["PRICE1"])
_, b0_5, _, R_sq0_5, _, _ = simple_regr(df["SALES0_5"], df["PRICE5"])
_, b0_10, _, R_sq0_10, _, _ = simple_regr(df["SALES0_10"], df["PRICE10"])

print(
    f"\nIn the case where α = 0, the price coefficients and the corresponding R-squared of each regression are:\n"
    f"\n(β = 0) -> b = {b0_0:.3f}, with R-squared = {R_sq0_0:.3f}\n"
    f"\n(β = 1) -> b = {b0_1:.3f}, with R-squared = {R_sq0_1:.3f}\n"
    f"\n(β = 5) -> b = {b0_5:.3f}, with R-squared = {R_sq0_5:.3f}\n"
    f"\n(β = 10) -> b = {b0_10:.3f}, with R-squared = {R_sq0_10:.3f}\n"
    f"\nAll coefficients are very close to the true price coefficient β_true = -1. The estimates do not indicate an endogeneity problem.\n"
    f"\nThis happens because Event affects only Price and not Sales, so Event is not part of the regression residuals. Therefore, cov(Price,e) = 0 -> OLS consistent.\n"
    f"\nR-squared pattern interpretation: As β increases, Event creates larger variation in Price. Since Sales depends directly on Price, Price explains a larger\n"
    f"share of Sales variation. Therefore, R-squared increases as β increases.\n"
    f"\n-----------------------------------------------------------------------\n"
)

_, b1_0, _, R_sq1_0, _, _ = simple_regr(df["SALES1_0"], df["PRICE0"])
_, b5_0, _, R_sq5_0, _, _ = simple_regr(df["SALES5_0"], df["PRICE0"])
_, b10_0, _, R_sq10_0, _, _ = simple_regr(df["SALES10_0"], df["PRICE0"])


print(
    f"\nIn the case where β = 0, the price coefficients and the corresponding R-squared of each regression are:\n"
    f"\n(α = 0) -> b = {b0_0:.3f}, with R-squared = {R_sq0_0:.3f}\n"
    f"\n(α = 1) -> b = {b1_0:.3f}, with R-squared = {R_sq1_0:.3f}\n"
    f"\n(α = 5) -> b = {b5_0:.3f}, with R-squared = {R_sq5_0:.3f}\n"
    f"\n(α = 10) -> b = {b10_0:.3f}, with R-squared = {R_sq10_0:.3f}\n"
    f"\nAll coefficients are very close to the true price coefficient β_true = -1. The estimates do not indicate an endogeneity problem.\n"
    f"\nIn this case, Price does not depend on Event. Meanwhile, Event enters Sales through α. The analyst omitts Event from the regression, so Event becomes part of\n"
    f"the regression residuals, but it is not correlated with Price. Therefore, cov(Price,e) = 0 -> OLS consistent.\n"
    f"\nR-squared pattern interpretation: As α increases, more and more variation in Sales comes from Event. Since Event is omitted, Price can not explain this\n"
    f"variation. Therefore, R-squared falls dramatically as α increases.\n"
    f"\n-----------------------------------------------------------------------\n"
)

_, b1_1, _, R_sq1_1, _, _ = simple_regr(df["SALES1_1"], df["PRICE1"])
_, b5_5, _, R_sq5_5, _, _ = simple_regr(df["SALES5_5"], df["PRICE5"])
_, b10_10, _, R_sq10_10, _, _ = simple_regr(df["SALES10_10"], df["PRICE10"])


print(
    f"\nIn the case where α = β = (0, 1, 5, 10), the price coefficients and the corresponding R-squared of each regression are:\n"
    f"\n(α = β = 0) -> b = {b0_0:.3f}, with R-squared = {R_sq0_0:.3f}\n"
    f"\n(α = β = 1) -> b = {b1_1:.3f}, with R-squared = {R_sq1_1:.3f}\n"
    f"\n(α = β = 5) -> b = {b5_5:.3f}, with R-squared = {R_sq5_5:.3f}\n"
    f"\n(α = β = 10) -> b = {b10_10:.3f}, with R-squared = {R_sq10_10:.3f}\n"
    f"\nAs α and β grow, the estimated Price coefficients move further away from the true price coefficient β_true = -1. This indicates an endogeneity problem.\n"
    f"\nThe analyst omitts Event from the regression, so it becomes part of the residuals. However, now Event is correleated with Price.\n"
    f"Therefore, cov(Price,e) != 0 -> OLS inconsistent.\n"
    f"\nR-squared pattern interpretation: Similar to the previous question. α increases -> Event creates more variation in Sales, but it's omitted -> R-sq drops.\n"
    f"\n-----------------------------------------------------------------------\n"
)