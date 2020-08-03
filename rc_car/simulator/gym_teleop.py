#!/usr/bin/env python
# manual
import pyglet
import random
import numpy as np
import cv2
import argparse
import sys
import math
import time

from PIL import Image
from pyglet.window import key
from rc_car.simulator.sim_interface import GymDonkeyInterface
from rc_car.vision.lane_detector import SimpleLaneDetector, UFNetLaneDetector
from rc_car.datastore.dataset import  DonkeyDatasetRecorder
from donkeycar.parts.behavior import BehaviorPart

dataset_record_options = [
        'simple', # Simple donkeycar dataset(Img + json file containing image, throttle and anlge
        'behavior', # Adding to the simple dataset behaviour part
        'None' # Record nothing
    ]

lane_detectors = [
    'None',
    'SimpleCV', # Simple detector using hough transform and edge detection
    'UF'        # Ultrafast lane detector
]

behavior_state = [
    'Right_Lane',
    'Left_Lane'
]

class TeleopClient:
    def __init__(self, n_episodes, env, dataset_record_opt, lane_detector, dataset_dir):
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
        self.obs = np.empty([0,0])

        self.write_prev_time = None
        self.state = None
        self.bp = None
        self.dataset_writer = None
        self.is_writing = False

        if dataset_record_opt == "behavior":
            self.bp = BehaviorPart(behavior_state)
            if dataset_dir:
                self.dataset_writer = DonkeyDatasetRecorder(dataset_dir)
            else:
                print("Not recording any dataset, because dir is not specified")

        # TODO Better to write one config file, than using args
        if lane_detector == "None":
            self.ld_detector = None
        elif lane_detector == "SimpleCV":
            self.ld_detector = SimpleLaneDetector()
        elif lane_detector == "UF":
            self.ld_detector = UFNetLaneDetector('./weights/culane_18.pth')

    def cv2glet(self, img):
        '''Assumes image is in BGR color space. Returns a pyimg object'''

        img = cv2.resize(img, (160,120))
        rows, cols, channels = img.shape
        raw_img = Image.fromarray(img).tobytes()

        top_to_bottom_flag = -1
        bytes_per_row = channels*cols
        pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   format='BGR', 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
        return pyimg

    def info(self):

        if not self.state:
            return ""

        info = '\n'.join(['%s:: %s' % (key, value) for (key, value) in self.state.items()])
        info += '\n %s:: %s' % ("Dataset is recording", self.is_writing)
        return info

    def form_info(self, obs, action):
        state = {}
        if self.bp:
            bp_info = self.bp.run()
            state["behavior/state"] = bp_info[0]
            state["behavior/label"] = bp_info[1]
            state["behavior/one_hot_state_array"] = bp_info[2]
        state["user/mode"] = "user"
        state["user/throttle"] = action[1]
        state["user/angle"] = action[0]
        return state

    def latest_frames_with_detected_lanes(self):
        if not self.obs.any():
            return None, None

        self.orig_obs = self.obs
        if self.ld_detector:
            self.obs = self.ld_detector.detect_lanes(self.obs)
        return self.cv2glet(self.obs), self.cv2glet(self.orig_obs)

    def update(self, dt):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """

        action = np.array([0.0, 0.0])

        const_val = 0.65

        # Throttle and angle val will be with some random noise
        throttle_val = const_val + random.uniform(-0.2,0.2)
        angle_val = const_val + random.uniform(-0.1,0.1)
        if key_handler[key.UP]:
            action += np.array([0.0, throttle_val])
        if key_handler[key.DOWN]:
            action += np.array([0.0, -throttle_val])
        if key_handler[key.LEFT]:
            action += np.array([-angle_val, +0.0])
        if key_handler[key.RIGHT]:
            action += np.array([angle_val, 0.0])
        if key_handler[key.SPACE]:
            action += np.array([0, 0])

        if key_handler[key.C] and self.bp:
            self.bp.increment_state()

        action = [math.copysign(1,action[0])*min(abs(1), abs(action[0])),
                  math.copysign(1,action[1])*min(abs(1), abs(action[1]))]
        # Speed boost
        if key_handler[key.LSHIFT]:
            action *= 1.5
        if key_handler[key.S]:
            self.is_writing = not self.is_writing
        obs, reward, done, info = self.env.step(action)
        #obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        self.obs = obs
        self.episode_reward += reward
        self.state = self.form_info(obs,action)

        # Write only 15 fps
        if self.dataset_writer and self.is_writing:

            fps_15_in_ms = 0.0067
            fps_7_in_ms  = fps_15_in_ms/2
            fps_30_in_ms = fps_15_in_ms*2
            if not self.write_prev_time or time.time() - self.write_prev_time >= fps_30_in_ms:
                self.dataset_writer.write_state(obs, self.state)
                self.write_prev_time = time.time()

        self.step_count += 1

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
            self.rewards        = np.array(self.rewards)
            self.episode_starts = np.array(self.episode_starts[:-1])
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
    ld_detector = "None"
    parser = argparse.ArgumentParser(description='Run autopilot of the rc-car.')

    parser.add_argument('--sim-path', type=str, help='Full path to donkey_sim.x86_64 file')
    parser.add_argument('--record_dataset', type=str,choices=dataset_record_options, help='Type of recording dataset', default='None')
    parser.add_argument('--path_to_dataset', type=str)

    args = parser.parse_args()

    #Make an environment test our trained policy
    sim = GymDonkeyInterface("donkey-mountain-track-v0", args.sim_path,9091)
    obs = sim.reset()
    #Make an environment test our trained policy

    
    window = pyglet.window.Window()
    keys = key.KeyStateHandler()
    # Register a keyboard handler
    key_handler = key.KeyStateHandler()
    window.push_handlers(key_handler)

    
    uw = TeleopClient(50, sim, args.record_dataset, ld_detector,args.path_to_dataset)


    @window.event
    def on_draw():
        window.clear()
        lane_video, video = uw.latest_frames_with_detected_lanes()
        info = uw.info()
        label = pyglet.text.Label(info,
                                  font_name='Times New Roman',
                                  font_size=10,
                                  x=window.width-window.width//3,
                                  y=window.height-window.height//3,
                                  anchor_x='center',
                                  anchor_y='center',
                                  width=30,
                                  multiline=True)

        if video:

            video.blit(10,10)
            lane_video.blit(10,200)
        label.draw()
    pyglet.clock.schedule_interval(uw.update, 1.0 / 120.0)


    # Enter main event loop
    pyglet.app.run()

    env.close()