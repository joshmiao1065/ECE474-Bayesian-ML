import numpy as np
import matplotlib.pyplot as plt

# Generate N Bernoulli samples with p = 0.5
def generate_data(N):
    return np.random.randint(0, 2, size=N)

# Compute ML mean
def compute_mean(X):
    return np.mean(X)

# For known variance case, compute Bayesian mean estimate with conjugate prior
def bayesian_mean_estimates(X, mean_prior, var_prior, var_known):
    mean_posteriors = []
    mean_MLs = []
    mse_map = []
    mse_ml = []

    sum_ML = 0
    mean_mean = mean_prior
    var_post = var_prior

    for N in range(1, len(X) + 1):
        x_n = X[N-1]
        sum_ML += x_n
        mean_ML = sum_ML / N

        # Update posterior
        weight_prior = var_known / (N * var_prior + var_known)
        weight_obs = (N * var_prior) / (N * var_prior + var_known)

        mean_post = weight_prior * mean_prior + weight_obs * mean_ML
        var_post = 1 / (1 / var_prior + N / var_known)

        # Store estimates
        mean_posteriors.append(mean_post)
        mean_MLs.append(mean_ML)

        # Assume true mean is 0.5
        true_mean = 0.5
        mse_map.append((mean_post - true_mean)**2)
        mse_ml.append((mean_ML - true_mean)**2)

    return mean_posteriors, mean_MLs, mse_map, mse_ml

def main():
    num_obs = 100
    X = generate_data(num_obs)

    mean_prior = 0.3  # Try others like 0.5, 0.8
    var_prior = 0.1   # Controls confidence in prior
    var_known = 0.25  # Known variance of observations (assume some value)

    posteriors, mls, mse_map, mse_ml = bayesian_mean_estimates(X, mean_prior, var_prior, var_known)

    # Plot MSE
    plt.plot(mse_ml, label='ML MSE')
    plt.plot(mse_map, label='Bayesian MSE')
    plt.title('MSE of ML vs Bayesian Estimates (Known Variance)')
    plt.xlabel('# Observations')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional: plot posterior over time (e.g., every 10 steps)
    import scipy.stats as stats
    xs = np.linspace(0, 1, 200)
    plt.figure()
    for i in [1, 5, 20, 100]:
        mean = posteriors[i-1]
        std = np.sqrt(var_prior * var_known / (i * var_prior + var_known))
        plt.plot(xs, stats.norm.pdf(xs, mean, std), label=f'After {i} obs')
    plt.title('Posterior PDF Evolution')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
