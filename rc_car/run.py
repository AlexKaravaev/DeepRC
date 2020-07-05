import gym_donkeycar
import numpy as np
import torch
import gym
import argparse

from rc_car.car_interface.sim_interface import GymDonkeyInterface
from rc_car.car_interface.car_interface import TorchInterface

from rc_car.models.utils import pil2tensor
from rc_car.models.models import supported_models
from rc_car.models.cnn import CNNAutoPilot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run autopilot of the rc-car.')

    parser.add_argument('--model', type=str, 
                    help='Name of the input model file')
    parser.add_argument('--model-type', type=str, 
                        help='Type of the model to train', choices=supported_models.keys())
    parser.add_argument('--save-logs', type=bool)  
    parser.add_argument('--sim-path', type=str, help='Full path to donkey_sim.x86_64 file')

    args = parser.parse_args()

    #Make an environment test our trained policy
    sim = GymDonkeyInterface("donkey-mountain-track-v0", args.sim_path,9091, './logs')
    obs = sim.reset()

    car = TorchInterface(args.model_type, args.model)
    
    while True:
        action = car.step(obs)
        obs, rewards, dones, info = sim.step(action)
        sim.render()
    