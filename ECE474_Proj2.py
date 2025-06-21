import random
import numpy as np
import matplotlib.pyplot as plt

def generate_data(N, a_0, a_1):
    x = np.sort(np.random.uniform(-1, 1, N))
    y_true = a_0 + (a_1 * x)
    noise = np.random.normal(-.1, .1, N)
    y = y_true + noise
    return x, y, y_true

def data_space():

def likelihood():

def prior():
 

def main():
    N = 20
    a_0 = -0.3
    a_1 = 0.5
    x, y, y_true = generate_data(N, a_0, a_1)
    plt.scatter(x, y, label="Noisy data")
    plt.plot(x, y_true, color='red', label="True line: y = -0.3 + 0.5x")
    plt.xlabel("x")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.ylabel("y")
    plt.title("Synthetic Linear Regression Data")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()