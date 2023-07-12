"""Jack's (original) car rental problem, attempt 2."""

import numpy as np
import gymnasium as gym
from scipy.stats import poisson

class JacksCarRental():
    """MDP Environment for Jack's Car Rental"""
    
    def __init__(self, max_poisson=10) -> None:
        """max_poisson is the max feasible value to be sampled from the poisson random variables"""
        self.max_poisson = max_poisson
        pass

    def prob_rent_return(self, lmbda):
        """Returns the probabilities of the permutations of rent requests or returns on a single day, given lmbda, 
        a list of the poisson parameters for location 1 and 2 respectively"""
        
        permutation = np.indices([self.max_poisson]*2) # represents a grid of x, y coordinates representing
        # all the combinations of rent requests or returns for location 1 (x) and 2 (y)

        prob_1 = poisson(lmbda[0]).pmf(permutation[0]) # probabilities of rent requests or returns for location 1
        prob_2 = poisson(lmbda[1]).pmf(permutation[1])

        return permutation, prob_1*prob_2 # the permutation of rents or returns and their respective probabilities
    
    def step(self, s, a):
        """provides an array of s_prime, r, p (their probabilities of occurring), and which of s_prime are terminating.
        a is an integer from -5 to 5."""
        if a >= 0: # transferring cars from location 1 to 2
            s[0] -= min(s[0], a)
            s[1] += min(s[0], a)
            r = -2*min(s[0], a) # -2 for each car transferred
        else:
            s[1] -= min(s[1], a)
            s[0] += min(s[1], a)
            r = -2*min(s[1], a)
        # could instead make the action space MultiDiscrete, so when locations don't have enough cars, corresponding
        # actions aren't allowed, but it's easier to keep it the action choice uniform and do the above instead.

        # now need to return the possible states from the random distrubution of rented requests and returns, together
        # with their probabilities, rewards, and whether they lead to termination.

        permutation, rent_prob = self.prob_rent_return([3, 4])
        _, return_prob = self.prob_rent_return([3, 2]) # possible permutations for returns are the same as for rents

        s_prime = np.array([s[0] - permutation[0], s[1] - permutation[1]]) # after taking out cars for rent requests
        # s_prime[0] represents cars in location 1 (not yet including the returned cars), and s_prime[1] for location 2.
        terminals = np.minimum(s_prime[0], s_prime[1])
        terminals = np.where(terminals < 0, 1, 0) # if any of the locations have negative cars, the business is lost so terminate.

        r += 10*(np.minimum(s[0], permutation[0]) + np.minimum(s[1], permutation[1]))

        # calculating the s_prime after returns is going to be finicky, because for each probability of rent requests,
        # there will be another tree of probability for the returns.



# after finishing the environment, policy iteration can be done. In particular, the state value functions for a policy, then the
# action value functions, can be evaluated. Then a new policy can be chosen by choosing q(s, a) greedily.
# With this problem, there are so many possible stochastic s_primes for an action, each with a different probability
# of occurring, so I won't try implementing this further.  


def run():
    pass

if __name__ == "__main__":
    pass