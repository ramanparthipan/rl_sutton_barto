"""k-armed bandit for a stationary problem, using sample-average update rule"""

import numpy as np
import random

# For k actions, reward_means and rewards() simulate the environment (gaussian distributions, as in 10-armed testbed)
# policy() chooses the actions using epsilon-greedy, and run() executes the repetitions and updating the Q values.

k = 10 # number of bandit arms (actions)

def reward(mean):
    """Returns the reward for an action given its mean"""
    return np.random.normal(mean, 1)

def policy(Q, eps):
    """Chooses an exploratory (random) action with probability epsilon and otherwise chooses the action with the highest
    action value"""
    if random.random() <= eps:
        return random.randint(0, k-1) # return random action
    else:
        return Q.index(max(Q)) # return action with highest action-value
        # should ideally break ties randomly but not implemented here.

def run():
    mean = 0
    std_dev = 1 
    sample_size = k
    n = 10000 # number of iterations
    eps = 0.1
    reward_means = np.random.normal(mean, std_dev, sample_size) # represents the means of the reward distributions for each action

    Q = [0] * k
    for i in range(1, n+1):
        action = policy(Q, eps)
        r = reward(reward_means[action])
        # the action has returned a reward. Now need to update Q value
        Q[action] += (1/i)*(r-Q[action]) # update rule (derivation from pg 24)

    return Q, reward_means # Q should be close to reward_means 



if __name__ == "__main__":
    Q, reward_means = run()
    print(f'Q: {Q}')
    print(f'reward_means: {reward_means}')