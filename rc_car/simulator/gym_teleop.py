#!/usr/bin/env python
# manual
import pyglet
import numpy as np

from pyglet.window import key
from rc_car.simulator.sim_interface import GymDonkeyInterface


window = pyglet.window.Window()
keys = key.KeyStateHandler()
# Register a keyboard handler
key_handler = key.KeyStateHandler()
window.push_handlers(key_handler)

class TeleopClient:
    def __init__(self, n_episodes, env):
        self.i = 0

        self.actions = []
        self.observations = []
        self.rewards = []
        self.episode_returns = np.zeros((n_episodes,))
        self.episode_starts  = []
        self.episode_reward = 0
        self.episodes_played = 0
        self.max_episodes = n_episodes
        self.env = env
        self.step_count = 0
                
    def update(self, dt):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """

        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action = np.array([0.0, 0.5])
        if key_handler[key.DOWN]:
            action = np.array([0.0, -0.5])
        if key_handler[key.LEFT]:
            action = np.array([-0.5, +0.0])
        if key_handler[key.RIGHT]:
            action = np.array([0.5, 0.0])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])

        # Speed boost
        if key_handler[key.LSHIFT]:
            action *= 1.5

        obs, reward, done, info = self.env.step(action)
        print(action)
        self.episode_reward += reward
        print('step_count = %s, reward=%.3f' % (self.step_count, reward))



        self.actions.append(action)
        self.observations.append(obs)
        self.rewards.append(reward)
        self.episode_starts.append(done)

        if done:
            print('done!')
            print(f"Episode reward {self.episode_reward}")
            self.step_count = 0
            self.env.reset()
            self.env.render()
        
        if self.episodes_played == self.max_episodes:
            print("Terminating because max_episodes reached")
            #time.sleep(4)
            self.rewards        = np.array(self.rewards)
            self.episode_starts = np.array(self.episode_starts[:-1])
            #self.actions        = np.concatenate(self.actions).reshape((-1,) + env.action_space.shape)
            #self.observations   = np.concatenate(self.observations).reshape((-1,) + env.observation_space.shape)
            self.observations   = np.array(self.observations)
            self.observations   = self.observations.reshape((self.observations.shape[0], self.observations.shape[-1]))
            self.actions        = np.array(self.actions)
            print(self.observations.shape)
            print(self.observations[0].shape)
            assert len(self.observations) == len(self.actions)

            self.env.close()
            sys.exit(0)

        self.env.render()
        
    
if __name__ == "__main__":
    
    #Make an environment test our trained policy
    sim = GymDonkeyInterface("donkey-mountain-track-v0", "/home/robot/DonkeySimLinux/donkey_sim.x86_64" ,9091, './logs')
            
    uw = TeleopClient(50, sim)
    pyglet.clock.schedule_interval(uw.update, 1.0 / 120.0)


    # Enter main event loop
    pyglet.app.run()

    env.close()