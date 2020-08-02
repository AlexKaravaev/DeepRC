import gym_donkeycar
import atexit
import json
import datetime
import gym
import numpy as np
import os

class CarInterface(object):
    """ Base class of car interface """
    def __init__(self):
        """ Initialize environment

        Raises:
            NotImplementedError: Pure virtual
        """
        raise NotImplementedError()

    
    def step(self, action: np.ndarray):
        """ One step into simulator

        Args:
            action (np.ndarray): [steering_angle, throttle]

        Raises:
            NotImplementedError: Pure virtual
        """
        raise NotImplementedError()
    
    

class GymDonkeyInterface(CarInterface):

    def __init__(self, env_name: str,
                 exe_path: str, port: int, log_path: str=None, cam_res: tuple=(800,288,3)):
        """ Interface for OpenAi gym donkeycar

        Args:
            exe_path (str): path to simulator binary donkey_sim.x86_64
            port (int): port for connection
            env_name (str, optional): Track name. 
            log_path (str, optional): Path to save this run logs. Defaults to None.
        """
        self.env = gym.make(env_name, exe_path=exe_path, port=port,cam_resolution=cam_res)
        self.cam_config = {
            "fov":90,
            "fish_eye_x":0,
            "fish_eye_y":0,
            "img_w" : cam_res[0],
            "img_h" : cam_res[1],
            "img_d" : cam_res[2],
            "img_enc": 0,
            "offset_x": 0,
            "offset_y": 0.5,
            "offset_z": 1,
            "rot_x": 0
        }
        self.env.viewer.set_cam_config(**self.cam_config)

        self.path_to_save_logs = None
        if log_path:
            self.path_to_save_logs = log_path
            if not os.path.exists(self.path_to_save_logs):
                os.makedirs(self.path_to_save_logs)
            self.run_info = {
                "positions": [],
                "speeds"   : [],
                "ctes"     : []
            }
        atexit.register(self.__save_logs)

    def step(self, action: np.ndarray):
        obs, reward, dones, info = self.env.step(action)
        if (self.path_to_save_logs):
            self.run_info["positions"].append(info['pos'])
            self.run_info["speeds"].append(info['speed'])
            self.run_info["ctes"].append(info['cte'])
       
        return obs, reward, dones, info
    

    def __save_logs(self):
        if not self.path_to_save_logs:
            return
        filename = self.path_to_save_logs + "/" +\
                    datetime.datetime.now().isoformat(timespec='seconds')+"_runlog.json"
        
        with open(filename,'x') as f:
            json.dump(self.run_info, f, ensure_ascii=False)



    def render(self):
        """ Renders open ai gym simulator
        """
        self.env.render()

    def reset(self):
        """ Resets car position in simulator
        """
        return self.env.reset()

class RealRcInterface(CarInterface):
    """ Class for handling connection to real life RC car """
    def __init__(self):
        raise NotImplementedError()

    def step(self, action: np.ndarray):
        raise NotImplementedError()