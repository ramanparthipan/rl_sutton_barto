"""Jack's (original) car rental problem"""

import numpy as np
    
n = 10 # try with max 10 cars, then try 20 if computer is fast enough

def env(s, a):
    """Return s_prime (the number of cars in each location, or whether the business is lost), and the reward given s, a.
    a can be [-5, 5], positive being transferring cars from location 1 to 2, and vice versa for negative. Make sure
    the action doesn't transfer more cars than there are at a particular location"""
    
    s[0] -= a # transfers cars between locations
    s[1] += a
    reward = -2*a # -2 for each car moved

    rented = [np.random.poisson(3), np.random.poisson(4)]
    returned = [np.random.poisson(3), np.random.poisson(2)]
    lost = 0 # whether the business is lost
    if s[0] > rented[0]: # there are enough cars in each location to be rented out
        if s[1] >= rented[1]: 
            reward = 10*(sum(rented))
            s[0] -= rented[0]
            s[1] -= rented[1]
        else: # not enough cars in location 2
            reward = 10*(rented[0] + s[1]) 
            lost = 1
            s[0] -= rented[0]
            s[1] = 0
    else: # not enough cars in location 1
        reward = 10*(s[0] + min(s[1], rented[1])) 
        lost = 1
        s[0] = 0
        s[1] = max(s[1] - rented[1], 0)

    s = [s[i] + returned[i] for i in range(len(s))] # add returned cars to existing cars at each location
    for i in range(len(s)):
        if s[i] > n:
            s[i] = n # can't be more than n cars at each location
    
    s_prime = s
    
    return s_prime, lost, reward

def policy(pi, s):
    """Chooses actions using policy pi"""
    return pi[s]

def policy_update(pi, s, ):
    """Updates policy pi using policy iteration"""
    pass

def run():

    pass

if __name__ == "__main__":
    pass
