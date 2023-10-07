from racetrack import *
import numpy as np
import gymnasium as gym

def test_racetrack():
    env1 = RaceTrackEnv(track_file='tracks/test_1.txt', start_s=np.array([1, 0, 0, 0]))
    s, r, terminated, info = env1.step(np.array([1, 1]))
    assert (env1.s == np.array([1, 0, 1, 1])).all()
    assert r == -1
    assert terminated == False

    assert type(env1.observation_space) == gym.spaces.MultiDiscrete
    assert env1.action_space == gym.spaces.MultiDiscrete([3, 3])


    env2 = RaceTrackEnv(track_file='tracks/test_1.txt', start_s=np.array([4, 0, 3, 0]))
    s, r, terminated, info = env2.step(np.array([0, 0])) # path should go off canvas, so state should reset
    assert r == -1
    assert terminated == False

    env3 = RaceTrackEnv(track_file='tracks/test_1.txt')
    agent = Agent()
    pi, Q = agent.mc_egreedy(env3, n=100000)
    for i in range(5):
        agent.generate_episode(env3, pi, render=True)
    


if __name__ == "__main__":
    test_racetrack()