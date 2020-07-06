import torch
import numpy as np
import typing

from rc_car.models.models import supported_models
from rc_car.models.utils import pil2tensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CarInterface(object):
    def __init__(self, model_name: str):
        """Base class that is simulating interface to rc car

        Args:
            model_name (str): model_name. Can be Torch model, simple PID etc..
        """
    
    def step(self, obs:np.ndarray)->list:
        """ Make one step with image

        Args:
            obs (np.ndarray): image observation

        Returns:
            List: [throttle, angle]
        """
        raise NotImplementedError

class TorchInterface(CarInterface):

    def __init__(self, model_name:str, model_path:str):
        """ Interface for Torch models

        Args:
            model_name (str): model name from supported
            model_path (str): path to checkpoint

        Raises:
            NotImplementedError: [description]
        """
        try:
            self.model = supported_models[model_name]()
        except KeyError:
            raise NotImplementedError(f"Model {model_name} is not supported yet")

        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)

    def step(self, obs:np.ndarray)->list:
        obs_tensor = pil2tensor(obs, np.float32).reshape(1,3,160,120)
        
        obs_tensor = obs_tensor.to(device)

        action = self.model.forward(obs_tensor)
        action = [action[0].item(), action[1].item()]

        return action

class RealRcInterface(CarInterface):
    
    def __init__(self):
        """ Interface for real rc car
        """