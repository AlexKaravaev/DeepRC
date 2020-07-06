import os
import argparse
import gym
import gym_donkeycar
import numpy as np

from stable_baselines.common.policies import MlpPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines import TD3
from stable_baselines.td3.policies import LnCnnPolicy


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env.reset()
        return env
    set_global_seeds(seed)
    return _init


""" Sometimes simulation is not stopped, when car is behind the wall. Quick fix to deal with that"""
class CrashWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(CrashWrapper, self).__init__(env)
        self.prev_action = [0,0]
        self.action_ = [0,0]

    def action(self, action):

        self.prev_action = self.action_
        self.action_ = [action[0] * 0.8, action[1] * 0.8]
        self.action_ = action
        print(f"Prev action: {self.prev_action}\nAction: {self.action_}\nSpeed: {self.viewer.handler.speed}")

        return self.action_

if __name__ == "__main__":

	# Initialize the donkey environment
    # where env_name one of:    
    env_list = [
       "donkey-warehouse-v0",
       "donkey-generated-roads-v0",
       "donkey-avc-sparkfun-v0",
       "donkey-generated-track-v0",
       "donkey-mountain-track-v0"
    ]
	
    parser = argparse.ArgumentParser(description='ppo_train')
    parser.add_argument('--sim', type=str, default="sim_path", help='path to unity simulator. maybe be left at manual if you would like to start the sim on your own.')
    parser.add_argument('--port', type=int, default=9091, help='port to use for tcp')
    parser.add_argument('--test', action="store_true", help='load the trained model and play')
    parser.add_argument('--multi', action="store_true", help='start multiple sims at once')
    parser.add_argument('--env_name', type=str, default='donkey-generated-track-v0', help='name of donkey sim environment', choices=env_list)
    
    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)
    
    env_id = args.env_name

    sim_path = "/home/robot/dev/DonkeySimLinux/donkey_sim.x86_64"
    if args.test:

        #Make an environment test our trained policy
        env = gym.make(env_id, exe_path=sim_path, port=args.port)
        env = DummyVecEnv([lambda: env])

        model = TD3.load("td3_donkey")
    
        obs = env.reset()
        for i in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

        print("done testing")
        
    else:
    
        if args.multi:

            #setup random offset of network ports
            os.environ['DONKEY_SIM_MULTI'] = '1'

            # Number of processes to use
            num_cpu = 4

            # Create the vectorized environment
            env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

        else:

            #make gym env
            env = gym.make(env_id, exe_path=sim_path,port=args.port)




        # The noise objects for TD3
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


        #create recurrent policy
        model = TD3(LnCnnPolicy, env,
                          action_noise=action_noise, verbose=1,
                    n_cpu_tf_sess=4,gradient_steps=1)


        #set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=10000)

        obs = env.reset()
        
        for i in range(1000):
            
            action, _states = model.predict(obs)
            
            obs, rewards, dones, info = env.step(action)

            try:
                env.render()
            except Exception as e:
                print(e)
                print("failure in render, continuing...")
                
            if i % 100 == 0:
                print('saving...')
                model.save("td3_donkey")

        # Save the agent
        model.save("td3_donkey")
        print("done training")


    env.close()