
import os
import json
import urllib.request
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

        assert len(lanes) == 3
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

if __name__=="__main__":
    convert_labelbox_to_culane('/home/robot/dev/DeepRC/data/export-2020-07-22T14_33_24.352Z.json', './lanes_dataset/')

