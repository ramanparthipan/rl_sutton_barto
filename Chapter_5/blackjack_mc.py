"""Solving Blackjack using on-policy every-visit Monte Carlo (MC) control with e-greedy, as well as exploring starts"""
import numpy as np
import gymnasium as gym
import random

class BlackJackEnv(gym.Env):
    """Environment for Blackjack"""
    def step(self, s, a):
        """from the state np.array([player_sum, usable_ace, dealer_showing_card, dealer_ace]) and the action hit (a=1), sample an s_prime, an r(=0) and whether or not the player is bust
        If action is stick (0), sample the dealer's final sum and return the reward r.
        Assume an infinite deck of cards."""
        s_prime = s
        r = 0
        if a == 1: # hit
            x = random.randint(2, 11)
            s_prime[0] += x
            # print("s_prime updated")
            if x == 11:
                s_prime[1] += 1 # increment usable ace
            if s_prime[0] > 9: # 21 is encoded as 9 due to indexing
                if s_prime[1] > 0:
                    s_prime[0] -= 10 # turn an ace from value 11 to 1
                    s_prime[1] -= 1
            if s_prime[0] > 9: # player goes bust
                return s_prime, -1, 1 # reward of -1 and game terminates
            return s_prime, 0, 0 # 0 reward and game doesn't terminate
        else: # stick
            dealer_hand = [s[2], s[3]]
            while dealer_hand[0] < 15: # dealer hits (for dealer, 2 is encoded as 0 so 17 is encoded as 15) 
                x = random.randint(2, 11)
                dealer_hand[0] += x
                if x == 11:
                    dealer_hand[1] += 1 # increment dealer ace
                if dealer_hand[0] > 19: # corresponds to 21
                    if dealer_hand[1] > 0:
                        dealer_hand[0] -= 10 # turn an ace from value 11 to 1
                        dealer_hand[1] -= 1
        if dealer_hand[0] > 19: # dealer goes bust
            r = 1 
        elif s[0] > dealer_hand[0]: # player_sum closer to 21 than dealer (both player and dealer sum are confirmed to be <= 21)
            r = 1
        elif s[0] < dealer_hand[0]:
            r = -1
        # reward defaults to 0 so no need to consider the draw case
        return s_prime, r, 1
        
        

class Agent():
    
    def __init__(self, eps=0.1, n=1000) -> None:
        self.eps = eps
        self.n = n

    def mc(self, env):
        shape = (10, 2, 10, 2, 2) # player can only have at most one usable ace at a time
        pi = np.full(shape, self.eps/shape[-1]) # shape[-1] represents the number of actions (2). pi contains the probabilities of choosing all s, a pairs
        pi[:, :, :, :, 1] += 1 - self.eps # arbitrarily set sticking as the greedy action 
        q = np.random.random(shape)
        num_visits = np.zeros(shape) # records the number of visits to an (s, a) pair, to be used to average the return across mutliple episodes 
        
        for i in range(self.n): # generate episodes
            s = np.random.randint((10, 2, 10, 2)) # initial s (exploration starts).
            # s[0] += 12 # turn into player points
            # s[2] += 2 # dealer points
            if s[3] == 1: # if dealer has ace, dealer points must be 11
                 s[2] = 9 # because of indexing, 2 is coded as 0 and 11 is coded as 9
            print(f'initial s: {s}')

            episode = []
            terminate = 0
            while terminate == 0:
                index = tuple(np.append(s, 0))
                if random.random() < pi[index]: # choose action 0 (stick)
                    a = 0
                else:
                    a = 1
                s_prime, r, terminate = env.step(s, a)
                print(f's: {s}')
                episode.append((s.tolist(), a, r)) # generates episodes = [(s, a, r), (s, a, r), ...]
                s = s_prime
            #  implementation of every-visit MC. For Blackjack, states are only visited once anyway so makes no difference
            # print(episode)
            G = 0
            print(episode)
            for t in range(len(episode)-1, -1, -1): # decrement from len(episode)-1 to 0
                s_t = episode[t][0]
                print(s_t)
                a_t = episode[t][1]
                print(a_t)
                G += episode[t][2]
                num = num_visits[tuple(np.append(s_t, a_t))] # the number of times the (s, a) pair has been visited
                q[tuple(np.append(s_t, a_t))] = (num*q[tuple(np.append(s_t, a_t))] + G)/(num + 1) # compute new average of return from (s, a) pair from all episodes
                num_visits[tuple(np.append(s_t, a_t))] += 1
                a_greedy = np.argmax(q[tuple(s_t)])
                pi[tuple(s_t)] = self.eps/shape[-1] # prefixed exploration for each action
                pi[tuple(np.append(s_t, a_greedy))] += 1 - self.eps # choose new greedy action

        return pi, q
                
            
            
def run():
    env = BlackJackEnv()
    agent = Agent()
    pi, q = agent.mc(env)
    # print(f'final pi: {pi}')


if __name__ == "__main__":
    run()
