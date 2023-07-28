"""Design and conduct an experiment to demonstrate the difficulties that
sample-average methods have for nonstationary problems. Use a modified version of the 10-armed
testbed in which all the q∗(a) start out equal and then take independent random walks (say by adding
a normally distributed increment with mean zero and standard deviation 0.01 to all the q∗(a) on each
step). Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally
computed, and another action-value method using a constant step-size parameter, α = 0.1. Use ε = 0.1
and longer runs, say of 10,000 steps."""

import numpy as np
import random
import matplotlib.pyplot as plt

k = 10 # number of bandit arms
n = 10000 # number of time steps

def reward(R):
    """Changes reward distribution of each action slowly over time"""
    samples = np.random.normal(0, 0.01, k) # mean 0, std 0.01, sample size 10
    R = [R[i] + samples[i] for i in range(len(R))]
    return R

def policy(Q, eps):
    """Chooses an exploratory (random) action with probability epsilon and otherwise chooses the action with the highest
    action value"""
    if random.random() <= eps:
        return random.randint(0, k-1) # return random action
    else:
        return Q.index(max(Q)) # return action with highest action-value
        # should ideally break ties randomly but not implemented here.

def Q_update(Q, R, a, t, alpha, type=0):
    """Updates Q values. Type 0 uses sample-average method, type 1 uses constant step-size method"""
    if type==0:
        Q[a] += (1/t)*(R[a]-Q[a]) # sample-average method
    else:
        Q[a] += alpha*(R[a]-Q[a]) # constant step-size method
    return Q[a]

def run(type=0):
    """If type == 0, use sample-average method for learning. If type == 1, use constant step-size"""
    R = [1]*k # start with all rewards as 1
    Q = [0]*k # action values
    eps = 0.1
    alpha = 0.1
    opt_a = np.array([0]*n) # to store whether action at time t was optimal
    for t in range(1, n+1):
        R = reward(R)
        a = policy(Q, eps)
        if type==0:
            Q[a] = Q_update(Q, R, a, t, alpha, type=0)
        else:
            Q[a] = Q_update(Q, R, a, t, alpha, type=1)

        if R.index(max(R)) == a: # optimal action selected
            opt_a[t-1] = 1 # t starts from 1 to need to do t-1
    return Q, R, opt_a


if __name__ == "__main__":
    # sample-average optimal action percentage
    opt_a_tot_0 = np.array([0]*n)
    for i in range(2000):
        _, _, opt_a = run(type=0)
        opt_a_tot_0 += opt_a
    opt_a_avg_0 = opt_a_tot_0/2000
    
    # constant step-size optimal action percentage
    opt_a_tot_1 = np.array([0]*n)
    for i in range(2000):
        _, _, opt_a = run(type=1)
        opt_a_tot_1 += opt_a
    opt_a_avg_1 = opt_a_tot_1/2000

    t = np.array([i for i in range(1, n+1)])
    plt.plot(t, opt_a_avg_0, label='sample-average')
    plt.plot(t, opt_a_avg_1, label='constant step-size')
    plt.ylabel('Fraction optimal action')
    plt.xlabel('Time step')
    plt.legend()
    plt.title('Graph showing percentage optimal action against time step for a non stationary problem')
    plt.show()


