import pathlib
import numpy as np
import typing
import json
import torch
from skimage import io
from rc_car.models.utils import pil2tensor
from torch.utils.data import Dataset, DataLoader

class DonkeyCarDataset(Dataset):


    def __init__(self, dirs: typing.List[pathlib.Path]):
        """
        Args:
            dirs (string):  Directories containing files record_{}.json(with throttle and angle)
                            and {}_cam-image-array_.jpg(containing corresponding image)
        """
        self.dir_list = dirs
        self.json_files = []
        self.__collate_dirs()

    def __collate_dirs(self):
        """ Collate directories and get contatining json file names """
        
        for _dir in self.dir_list:
            if not _dir.exists():
                raise Exception(f"Path {_dir} for training data not found")
        
            for file in _dir.iterdir():

                if not file.suffix == ".json" or file.name == 'meta.json':
                    continue
                json_file = None
                with open(file, 'r+') as f:
                    json_file = json.load(f)
                
                
                    # Check if previously was rewritten
                if '/' in json_file['cam/image_array']:
                    self.json_files.append(file.absolute())
                    continue
                json_file['cam/image_array'] = str(file.parents[0].absolute()) + '/' + json_file['cam/image_array']
                with open(file, 'w+') as f:
                    json.dump(json_file, f)

                self.json_files.append(file.absolute())

        
    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        corresponding_json = None
        with open(self.json_files[idx], 'r') as f:
            corresponding_json = json.load(f)
        
        img_name = corresponding_json['cam/image_array']

        image = io.imread(img_name)
        
        throttle, angle = corresponding_json['user/throttle'], \
                            corresponding_json['user/angle']

        sample = {'image': pil2tensor(image, np.float32), 'throttle': throttle, 'angle': angle}

        return sample