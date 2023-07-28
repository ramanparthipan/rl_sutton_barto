"""ex 4.9"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class GamblerEnv(gym.Env):
    """An environment for the Gambler's problem"""
    observation_space = gym.spaces.Discrete(101) # current capital, including 0 and 100
    # action_space = gym.spaces.Discrete(50) # different number of allowable stakes for each state
    # the allowable stakes will change depending on the state, but it is easier to consider it as 50 and restrict the agent's choices of actions accordingly

    
    def __init__(self, p_h=0.4) -> None:
        """p_h is the probability of flipping heads"""
        self.p_h = p_h
        pass

    def step(self, s, a):
        """Returns a matrix of possible s_prime, and a vector of r, p and which of s_prime are terminal states"""
        s_prime = np.array([s+a, s-a]) # if flips heads, capital increases by stake. If tails, capital decreases by stake 
        r = np.where((s_prime == 100), 1, 0) # +1 reward for reaching 100
        p = np.array([self.p_h, 1-self.p_h])
        continuation = np.where((s_prime == 100)|(s_prime == 0), 0, 1) # 0 if terminal state

        return s_prime, r, p, continuation

class Agent():
    def __init__(self, theta=1e-3, gamma=1) -> None:
        self.theta = theta
        self.gamma = gamma
    
    def value_iteration(self, env):
        """Samples from the environment to learn a value function for each state"""
        v = np.random.random(env.observation_space.n) # randomly initialise value functions.
        v[0] = 0 # set v of terminal states to 0
        v[env.observation_space.n - 1] = 0
        pi = np.zeros(env.observation_space.n)
        delta = self.theta + 1 # any value above theta 
        while delta > self.theta:
            delta = 0
            for s in range(1, v.shape[0] - 1): # s from 1 to 99
                v_old = v[s]
                q_s_max = 0
                for a in range(1, min(s+1, 100-s+1)): # takes into account valid actions
                    s_prime, r, p, continuation = env.step(s, a)
                    q_s = np.sum(p*(r + self.gamma*v[s_prime]*continuation))
                    q_s_max = max(q_s_max, q_s)
                v[s] = q_s_max
                delta = max(delta, abs(v_old - v[s]))
        
        # v_optimal achieved. Now compute policy pi
        for s in range(1, pi.shape[0] - 1): # s from 1 to 99
            a_max = 0
            q_s_max = 0
            for a in range(1, min(s+1, 100-s+1)):
                s_prime, r, p, continuation = env.step(s, a)
                q_s = np.sum(p*(r + self.gamma*v[s_prime]*continuation))
                if q_s > q_s_max:
                    a_max = a
                    q_s_max = q_s
            pi[s] = a_max

        return pi, v



def run():
    env = GamblerEnv(p_h=0.7)
    agent = Agent(gamma=1, theta=1e-5)
    pi, v = agent.value_iteration(env)
    print(f'pi: {pi}', "\n")
    print(f'v: {v}')

    fig, axs = plt.subplots(2)
    axs[0].bar(np.arange(len(pi)), pi)
    axs[0].set_xlabel("Capital")
    axs[0].set_ylabel("Stake")
    axs[0].set_title("pi")

    axs[1].bar(np.arange(len(v)), v)
    axs[1].set_title("v")
    axs[1].set_xlabel("Capital")
    axs[1].set_ylabel("Value function")
    plt.show()

if __name__ == "__main__":
    run()

# with p_h = 0.55, the policy is to stake 1 each time, which makes sense because p_h > 0.5, so maximising the number
# of flips maximises the chance of winning.
# for p_h = 0.25, the results aren't stable until about theta = 1e-30, probably because of the stochasticity of value iteration
# The policy seems to be of a similar family to the one in the book, but with some differences.

# Some p_h take longer to simulate than others, especially around p_h = 0.5
# this probably could be because it takes longer for the game to terminate by either winning or losing, so value iteration takes longer to converge

