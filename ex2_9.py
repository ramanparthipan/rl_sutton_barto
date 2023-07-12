"""Make a figure analogous to Figure 2.6 for the non-stationary case outlined
in Exercise 2.5. Include the constant-step-size ε-greedy algorithm with α = 0.1. Use runs of 200,000
steps and, as a performance measure for each algorithm and parameter setting, use the average reward
over the last 100,000 steps."""

import numpy as np
import random
import matplotlib.pyplot as plt

k = 10
n = 10000


def reward(R):
    """Changes reward distribution of each action slowly over time. R is a numpy array"""
    samples = np.array(np.random.normal(0, 0.01, k)) # mean 0, std 0.01, sample size 10
    R += samples
    return R

def policy(H):
    """Returns the chosen action given the action preferences H"""
    pi = softmax(H)
    x = random.random() # random number between 0 and 1
    y = 0
    for i in range(len(pi)):
        y += pi[i]
        if x <= y:
            return i

def policy_update(H, A_t, R_t, R_tbar, alpha):
    """Updates H(a) for all actions 'a', where A_t was the action taken that lead to reward R_t, R_tbar is
     the average of all previous rewards and alpha is the learning rate"""
    H += alpha*(R_t - R_tbar)*(-softmax(H))
    H[A_t] += alpha*(R_t - R_tbar) # unique update for action A_t, i.e. alpha*(R_t - R_tbar)*(1-softmax(H))
    return H

def softmax(H):
    """Applies the softmax function to an input array H"""
    # I could use scipy.special.softmax() but might as well implement myself
    e_H = np.exp(H)
    pi = e_H/sum(e_H)
    return pi

def run(alpha=0.1):
    R = np.array([0.0]*k) # float32 array
    H = softmax(np.random.random(10)) # initialise random action preferences
    R_tbar = 0
    opt_a = np.array([0]*n) # to store whether action at time t was optimal
    for t in range(1, n+1):
        a = policy(H)
        H = policy_update(H, a, R[a], R_tbar, alpha)
        R = reward(R) # changes reward distribution (for a non-stationary problem)
        R_tbar += (1/t)*(R[a] - R_tbar) # iterative method of calculating average
        
        if R.tolist().index(max(R)) == a: # optimal action selected
            opt_a[t-1] = 1 # t starts from 1 to need to do t-1
    return opt_a



if __name__ == "__main__":
    opt_a_tot = np.array([0.01]*n)
    for i in range(2000):
        opt_a = run(0)
        opt_a_tot += opt_a
    opt_a_avg = opt_a_tot/2000
    
    t = np.array([i for i in range(1, n+1)])
    plt.plot(t, opt_a_avg, label='Gradient Bandit')
    plt.ylabel('Fraction optimal action')
    plt.xlabel('Time step')
    plt.legend()
    plt.title('Graph showing percentage optimal action against time step for a non stationary problem')
    plt.show()


# optimal action selection decreases, probably because of the mean reward affecting the update size as the problem changes
# not going to do the exact question.