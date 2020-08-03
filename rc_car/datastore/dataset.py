
import os
import json
import urllib.request
import numpy as np
import cv2
import logging
import time

from abc import ABCMeta, abstractmethod
from labelbox import Client, Dataset

def convert_labelbox_to_culane(path_to_json, path_for_save):
    """
    converts labels from labelbox.com to tusimple dataset
    """
    # Check for correctness
    if not os.path.exists(path_for_save):
        os.makedirs(path_for_save)
    path_for_save = path_for_save if path_for_save[-1] == '/' else path_for_save+'/'

    client = Client()
    with open(path_to_json, 'r') as f:
        labels = json.load(f)

    id_to_lanes = {}
    for label in labels:
        uuid = label['External ID']

        # Collect lanes
        lanes = []
        for object in label["Label"]['objects']:
            if "line" in object:
                lanes.append([])
                for point in object["line"]:
                    lanes[-1].extend([point["x"], point["y"]])

        #assert len(lanes) == 3
        id_to_lanes[uuid] = lanes
    datasets = client.get_datasets(where=Dataset.name == "Mountain-Road")
    for dataset_it in datasets:
        dataset = dataset_it
    for data_row in dataset.data_rows():
        id = data_row.external_id
        try:
            lanes_for_id = id_to_lanes[id]
        except KeyError:
            # This image is not labeled yet
            continue
        image = urllib.request.urlretrieve(data_row.row_data, path_for_save + id)

        with open(path_for_save + id.split('.')[0] + '.lines.txt', "w") as out_lanes:
            for i, lane in enumerate(lanes_for_id):
                apdx = '\n' if i != len(lanes_for_id)-1 else ''
                out_lanes.write(" ".join(str(coord) for coord in lane) + apdx)


class DatasetRecorder(metaclass=ABCMeta):

    @abstractmethod
    def write_state(self):
        pass

class DonkeyDatasetRecorder(DatasetRecorder):

    def __init__(self, write_dir):
        self.state = None
        self.image = None
        self.write_dir = os.path.expanduser(write_dir)

        exists = os.path.exists(self.write_dir)
        if not exists:
            os.makedirs(self.write_dir)

        self.i = self.get_last_idx()
        logging.info(f"Starting with {self.i} index")

    def get_last_idx(self):
        files = os.listdir(self.write_dir)
        record_list = [file for file in files if file.endswith('.json')]
        if len(record_list) == 0:
            return 1

        sort_key = lambda fn: int(fn.split('_')[-1].split('.')[0].split('.')[0])
        record_list = sorted(record_list, key=sort_key)

        # Split record_1.json and return only number
        return sort_key(record_list[-1])

    def write_state(self, image:np.ndarray, state:dict):
        img_name = f"{self.i}_cam-image_array_.jpg"
        cv2.imwrite(self.write_dir + "/" + img_name, image)
        state["cam/image_array"] = img_name

        with open(f"{self.write_dir}/record_{self.i}.json", 'w') as f:
            json.dump(state, f)

        self.i += 1

if __name__=="__main__":
    convert_labelbox_to_culane('/home/robot/dev/DeepRC/data/dataset.json', './lanes_dataset/')

