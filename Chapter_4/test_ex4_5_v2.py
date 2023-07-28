import ex4_5_v2
import numpy as np
from scipy.stats import poisson

def run():
    pass

if __name__ == "__main__":
    s_prime = np.random.random((2, 3, 3))
    print(s_prime)
    terminals = s_prime[0] - s_prime[1]
    print(terminals)
    terminals = np.where(terminals < 0, 1, 0)
    print(terminals)




