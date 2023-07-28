from racetrack import *
import numpy as np
import gymnasium as gym

def test_racetrack():
    env1 = RaceTrackEnv(track_file='tracks/test.txt', start_s=np.array([1, 0, 0, 0]))
    r, terminated = env1.step(np.array([1, 1]))
    assert (env1.s == np.array([1, 0, 1, 1])).all()
    assert r == -1
    assert terminated == False

    assert type(env1.observation_space) == gym.spaces.MultiDiscrete
    assert env1.action_space == gym.spaces.MultiDiscrete([3, 3])


    env2 = RaceTrackEnv(track_file='tracks/test.txt', start_s=np.array([4, 0, 3, 0]))
    r, terminated = env2.step(np.array([0, 0])) # path should go off canvas, so state should reset
    assert r == -1
    assert terminated == False

    env3 = RaceTrackEnv(track_file='tracks/test.txt')
    agent = Agent()
    pi, Q = agent.mc_egreedy(env3, n=100000)
    # print(f'pi: {pi}')
    # print(f'Q: {Q}')
    # print(np.sum(np.where((pi > 0.5), 1, 0), axis=0))
    
    # pi = np.zeros((env3.observation_space.nvec.prod(), env3.action_space.nvec.prod()))
    # pi += 1/env3.action_space.nvec.prod() # random policy
    for i in range(5):
        agent.generate_episode(env3, pi, render=True)
    


if __name__ == "__main__":
    test_racetrack()