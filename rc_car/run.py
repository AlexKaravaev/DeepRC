import gym_donkeycar
import numpy as np
import torch
import gym

from rc_car.models.utils import pil2tensor
from rc_car.models.cnn import CNNAutoPilot

if __name__ == "__main__":
    sim_path = "/home/robot/dev/DonkeySimLinux/donkey_sim.x86_64"

    #Make an environment test our trained policy
    env = gym.make("donkey-mountain-track-v0", exe_path=sim_path, port=9091)
    model = CNNAutoPilot()
    model.load_state_dict(torch.load('./rc_car/model'))
    obs = env.reset()
    while True:
        
        obs = pil2tensor(obs, np.float32).reshape(1,3,120,160)
        
        action = model.forward(obs)
        action = [action[0].item(), action[1].item()]
        #print(action)
        obs, rewards, dones, info = env.step(action)
        #print(info)
        env.render()