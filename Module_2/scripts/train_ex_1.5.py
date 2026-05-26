import pandas as pd
import numpy as np
import math

df = pd.read_csv(r"C:\Users\vassi\Desktop\Econometrics_Methods_and_Applications\Module_2\datasets/TrainExer15.txt", sep="\t")

def lin_trend(dataset, game_nr):
    
    # Compute a and b
    Gi = dataset["Game"]
    Wi_m = dataset["Winmen"]
    Wi_w = dataset["Winwomen"]

    b_m, b_w = Gi.cov(Wi_m) / Gi.var(), Gi.cov(Wi_w) / Gi.var()
    a_m, a_w = Wi_m.mean() - b_m * Gi.mean(), Wi_w.mean() - b_w * Gi.mean()

    # Prediction model
    W_m = a_m + b_m * game_nr
    W_w = a_w + b_w * game_nr

    return W_m, W_w, a_m, a_w, b_m, b_w

def nonlin_trend(dataset, game_nr):
    
    # Compute a and b
    Gi = dataset["Game"]
    Wi_m = dataset["Winmen"]
    Wi_w = dataset["Winwomen"]

    b_m, b_w = Gi.cov(np.log(Wi_m)) / Gi.var(), Gi.cov(np.log(Wi_w)) / Gi.var()
    a_m, a_w = np.log(Wi_m).mean() - b_m * Gi.mean(), np.log(Wi_w).mean() - b_w * Gi.mean()

    # Prediction model
    W_m = math.exp(a_m + b_m * game_nr)
    W_w = math.exp(a_w + b_w * game_nr)

    return W_m, W_w, a_m, a_w, b_m, b_w

def calc_intersection(dataset, model):  # Choose "lin_trend" for linear and "nonlin_trend" for logarithmic regression model
    i = 16
    while True:
        W_m, W_w = model(dataset, i)[0], model(dataset, i)[1]
        m, w = round(W_m, 2), round(W_w, 2)
        
        if m == w:
            year = 1944 + 4 * i
            return year, m
        i += 1

intersection_linear = calc_intersection(df, lin_trend)
intersection_nonlinear = calc_intersection(df, nonlin_trend)

print(f"\nThe linear trend model predicts equal winning times at year {intersection_linear[0]}.\n")

print(f"The nonlinear trend model predicts equal winning times at year {intersection_nonlinear[0]}.\n")

print(f"The linear trend model predicts equal winning times of approximately {intersection_linear[1]} seconds.\n")

print("We can conclude that both models produce similar results short-term, but considerably different results long-term.\n")