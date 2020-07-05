import glob
import logging
import pathlib
import os
import argparse
import torch

from rc_car.models.models import supported_models
from rc_car.models.cnn import CNNAutoPilot
from rc_car.models.trainer import train
from rc_car.datastore.donkeycar_dataloader import DonkeyCarDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train autopilot of the rc-car.')

    parser.add_argument('--model', type=str, 
                    help='Name of the output model file')
    parser.add_argument('--model-type', type=str, 
                        help='Type of the model to train', choices=supported_models.keys())
    parser.add_argument('--path-to-training-data', type=str, nargs='+',help='Path to directories containing training data\n \
                        Data in format like in example directory')     
    parser.add_argument('--augment', type=bool, help='If augment images')

    args = parser.parse_args()
    
    training_data_dirs = [pathlib.Path(_dir) for _dir in args.path_to_training_data]
    dataset = DonkeyCarDataset(training_data_dirs,args.augment)

    if len(dataset) == 0:
        raise Exception(f"No training data at dirs {args.path_to_training_data} found. Check your paths")
    train_len = int(0.8*len(dataset))
    val_len   = len(dataset) - train_len

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_len, val_len))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=128, shuffle=False,
                                             num_workers=4)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=128, shuffle=False,
                                              num_workers=4)

    logging.info(f"Train dataset len: {len(train_dataset)}, val dataset len: {len(val_dataset)}")
    try:
        model = supported_models[args.model_type]().to(device)
    except KeyError:
        print(f"Model {args.model_type} not supported for now. Available models are {supported_models.keys()}")

    
    # Loss and optimizer
    learning_rate = 0.0001
    num_epochs = 1
    criterion_angle = torch.nn.MSELoss()
    criterion_throttle = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model,criterion_angle,criterion_throttle,
            optimizer,train_dataloader,val_dataloader,
            1, learning_rate, args.model,device)
    