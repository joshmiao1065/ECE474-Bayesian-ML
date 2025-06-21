import numpy as np
import matplotlib
import argparse
import random

#function to generate our dataset X with an expected of .5
def generate_data(N):
    #100 terms with indices from 0 - 99
    for (i < N):
        X[i] = random.randint(0, 1)
    return X

#function to compute_mean_known
def compute_mean(X):
    mean_known = sum(X) / len(X)
    return mean_known

#funciton to compute var_known
def compute_var(X):
    var_known = np.var(X)
    return var_known

def known_variance(num_obs, mean_prior, var_prior, var_known, X):
    #N is both our current number of observations as well as our indexing variable
    #im choosing to start N = 1 because N = 0 yields the mean prior and otherwise I'd have issues with N being both num_obs and index term
    N = 1
    sum_ML = 0
    mean_mean = mean_prior
    var_prior = var_known
    precission = 0
    for (N <= num_obs):       
        weight_prior = (var_known ** 2) / (N * (var_prior ** 2) + (var_known ** 2))
        weight_observed = (N * (var_prior ** 2)) / (N * (var_prior ** 2) + (var_known ** 2))
        sum_ML += X[N]
        mean_ML = sum_ML / N
        mean_prior = mean_mean
        var_prior = num
        precission = (1 / (var_prior ** 2)) + (N / (var_known ** 2))
        var_prior = precission ** -1
        mean_mean = (weight_prior * mean_prior) + (weight_observed * mean_ML)


def main():
    num_obs = 100
    X = generate_data(num_obs)
    mean_known = compute_mean(X)
    var_known = compute_var(X)
    #I expect 50 1s and 50 0s
    #guess that our true mean is 0.5
    mean_prior = 0.5
    #guess that our true variance is 0.5
    var_prior = 0.5


if __name__ == "__main__":
    main()

    ##does var new become var prior