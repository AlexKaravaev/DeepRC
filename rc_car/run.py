import gym_donkeycar
import numpy as np
import torch
import gym
import argparse

from rc_car.car_interface.interface import GymDonkeyInterface
from rc_car.models.utils import pil2tensor
from rc_car.models.cnn import CNNAutoPilot

if __name__ == "__main__":
    supported_models = {"cnn": CNNAutoPilot}

    parser = argparse.ArgumentParser(description='Run autopilot of the rc-car.')

    parser.add_argument('--model', type=str, 
                    help='Name of the input model file')
    parser.add_argument('--model-type', type=str, 
                        help='Type of the model to train', choices=supported_models.keys())
    parser.add_argument('--save_logs', type=bool)  
    parser.add_argument('--sim-path', type=str, help='Full path to donkey_sim.x86_64 file')

    sim_path = "/home/robot/dev/DonkeySimLinux/donkey_sim.x86_64"

    #Make an environment test our trained policy
    car_interface = GymDonkeyInterface("donkey-mountain-track-v0", sim_path,9091, './logs')
    model = CNNAutoPilot()
    model.load_state_dict(torch.load('./rc_car/good_model'))
    obs = car_interface.reset()
    while True:
        
        obs = pil2tensor(obs, np.float32).reshape(1,3,120,160)
        
        action = model.forward(obs)
        action = [action[0].item(), action[1].item()]
        #print(action)
        obs, rewards, dones, info = car_interface.step(action)
        #print(info)
        car_interface.render()
    