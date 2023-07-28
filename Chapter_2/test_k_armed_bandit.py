from Chapter_2.k_armed_bandit import *
import numpy as np

if __name__ == "__main__":
    Q = [0] * 10
    Q[1] += 1
    print(Q)
