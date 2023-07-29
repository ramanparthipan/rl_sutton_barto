"""ex 5.10"""
import numpy as np
import random
import matplotlib.pyplot as plt
import gymnasium as gym


class RaceTrackEnv(gym.Env):
    """Environment for the racetrack problem"""
    min_velocity = 0
    max_velocity = 4
    
    
    def __init__(self, track_file='tracks/test_1.txt', start_s=None) -> None:
        self.racetrack = np.fliplr(np.genfromtxt(track_file, delimiter=',', dtype=int).T) # loads track into a numpy array and transforms it accordingly so the x and y indices of the array match a cartesian coordinate system
        self.reset(start_s) # initialises the state
        self.observation_space = gym.spaces.MultiDiscrete(np.concatenate((self.racetrack.shape, np.array([self.max_velocity-self.min_velocity+1]*2))))
        self.action_space = gym.spaces.MultiDiscrete([3, 3])

    def step(self, a): # state is an attribute of the environment that can only be changed through actions, just like in a normal episode you won't be able to jump to any state you choose
        """Returns the reward and whether the episode has terminated (the state can be obtained by self.s). 's' is np.array([s_x, s_y, v_x, v_y]) and 'a' is np.array([a_x, a_y])"""
        position, velocity = self.s[:2], self.s[2:]
        
        # update the position
        if velocity[0] >= velocity[1]:
            path_index = zip(np.arange(position[0], position[0]+velocity[0]+1), np.linspace(position[1], position[1]+velocity[1], len(np.arange(position[0], position[0]+velocity[0]+1)), dtype=int)) # use linspace for the y positions of path so the length of the arrays are the same. Truncate the array to integers using int.
        else:
            path_index = zip(np.linspace(position[0], position[0]+velocity[0], len(np.arange(position[1], position[1]+velocity[1]+1)), dtype=int), np.arange(position[1], position[1]+velocity[1]+1)) # use linspace for the x positions
        path_index = list(path_index) # need to convert generator zip object to list because we require it more than once. After one use it will be used up
        try:
            path = np.array([self.racetrack[index] for index in path_index])
        except IndexError: # path goes off the canvas
            self.reset()
            return self.s, -1, False, {}
        if (path == 0).any(): # path goes off track
            self.reset()
            return self.s, -1, False, {}
        elif (path == 3).any(): # path reaches finish line legitimately
            idx = np.where(path == 3)[0][0]
            position = np.array(list(path_index)[idx])
            self.s = np.concatenate((position, np.array([0, 0]))) # the final state before termination (useful for plotting)
            return self.s, -1, True, {} # new state is not necessary because of termination
        else: # agent moves to new position
            position += velocity
        
        velocity = np.clip(velocity+a, self.min_velocity, self.max_velocity) # acceleration is added to velocity, with values clipped so no velocity component is less than 0 or not less than 5
        if (velocity == 0).all():
            velocity = np.array([0, 1]) # if both velocity components happen to be zero, arbitrarily set the velocity to be 1 upwards.
        self.s = np.concatenate((position, velocity))
        return self.s, -1, False, {}
    

    def reset(self, start_s=None):
        """Resets the environment, at the start of an episode or when the agent goes off the track"""
        if start_s is not None:
            self.s = start_s
        else:
            x_idx, y_idx = np.where(self.racetrack == 2) # find the start positions on the track
            idx = random.randint(0, len(x_idx)-1)
            position = np.array([x_idx[idx], y_idx[idx]])
            self.s = np.concatenate((position, np.array([0, 0])))

    def render(self):
        """Renders the episode. Requires the attribute self.episode to be created for the RaceTrackEnv class before running"""
        print(self.episode)
        track_plot = np.fliplr(self.racetrack).T
        plt.imshow(track_plot, cmap='cividis')
        plt.colorbar()
        plt.title('Agent track path for an episode')

        pos_list = [np.unravel_index(rsa[1], self.observation_space.nvec)[:2]  for rsa in self.episode]
        x_coords = [pos[0] for pos in pos_list]
        y_coords = [self.racetrack.shape[1] - 1 - pos[1] for pos in pos_list] # because points are plotted with origin at top left instead of bottom left, transform the y coordinates so they show up correctly
        plt.plot(x_coords, y_coords, color='red', marker='o')
        plt.plot(x_coords, y_coords, linestyle='-', linewidth=2)
        plt.show()
        #By gym conventions, env.render() only takes self as a parameter, so make episode an attribute that is retrievable.
        # The more conventional way is to render after each env.step(), but it is tricky to draw lines between the current and previous
        # state because the previous state is not stored. Also, by convention, should add a 'render_modes' parameter to env.step() to determine
        # how the episode should be rendered, e.g 'human', 'ansi', etc. See gymnasium.Env.render docs.
        
        


class Agent():
    """Agent for the racetrack problem"""
    
    def mc_egreedy(self, env, n=100000, eps=0.1):
        pi = np.zeros((env.observation_space.nvec.prod(), env.action_space.nvec.prod()))
        N = np.zeros_like(pi) # number of times each state-action pair is visited
        Q = np.zeros_like(pi) # action value function
        pi += 1/env.action_space.nvec.prod() # random policy
        for i in range(n):
            episode = self.generate_episode(env, pi)
            G = episode[-1][0] # reward at termination
            for t in range(len(episode)-2, 0, -1):
                s_rav = episode[t][1]
                a_rav = episode[t][2]
                Q[s_rav][a_rav] = ((N[s_rav][a_rav]*Q[s_rav][a_rav]) + G)/(N[s_rav][a_rav] + 1)
                N[s_rav][a_rav] += 1
                a_greedy = np.argmax(Q[s_rav])
                pi[s_rav] = eps/env.action_space.nvec.prod() # exploration evenly divided between probability eps
                pi[s_rav][a_greedy] += 1 - eps # greedy action
                assert np.isclose(np.sum(pi[s_rav]), 1.0) == True
                G += episode[t][0]
            if (i % 10000) == 0:
                print(f'iteration: {i}')
        print('Finished')
        return pi, Q


    def generate_episode(self, env, pi, render=False):
        """Generates an episode given the environment env and policy pi"""
        env.reset()
        a_choices = np.arange(0, env.action_space.nvec.prod()) # array of 0 to 8, representing each action
        terminated = False
        r = 0
        episode = []
        s = env.s
        while terminated == False:
            s_rav = np.ravel_multi_index(s, env.observation_space.nvec)
            a_rav = np.random.choice(a_choices, p=pi[s_rav]) # ravel_multi_index takes s and returns its index for the ravelled 1d array, which is what is used for pi
            episode.append((r, s_rav, a_rav))
            a = np.array(np.unravel_index(a_rav, env.action_space.nvec)) - 1 # unravels back to the indices of a 3x3 matrix. Subtract one to achieve the desired acceleration values
            s, r, terminated, info = env.step(a) # this is the convention for what env.step() returns for a gym environment. info contains diagnostic data but this is empty in this case.
            # env.step should also return 'truncated' as well, representing whether the episode stopped due to the agent leaving the state space, but this is not done here.
        episode.append((r, np.ravel_multi_index(s, env.observation_space.nvec))) # final reward and state at termination
        
        if render == True:
            setattr(RaceTrackEnv, 'episode', episode) # set episode attribute of RaceTrackEnv to episode. ie env.episode = episode
            env.render()
        
        return episode


gym.register(id='RaceTrackEnv-v0', 
             entry_point='racetrack:RaceTrackEnv',
             kwargs={'track_file':'tracks/test_1.txt', 'start_s':None},
             )
 