import glob
import logging
import pathlib
import os
import argparse
import torch
import json

from rc_car.models.models import supported_models
from rc_car.models.cnn import CNNAutoPilot
from rc_car.models.trainer import train
from rc_car.datastore.donkeycar_dataloader import DonkeyCarDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train autopilot of the rc-car.')

    parser.add_argument('--model', type=str, 
                    help='Name of the output model file', default='Model')
    parser.add_argument('--model-type', type=str, 
                        help='Type of the model to train', choices=supported_models.keys(), default=supported_models["cnn"])
    parser.add_argument('--path-to-training-data', type=str, nargs='+',help='Path to directories containing training data\n \
                        Data in format like in example directory', required=True)     
    parser.add_argument('--augment', type=bool, help='If augment images')
    parser.add_argument('--hyperparam-file', type=str, help='Path to json hyperparams', required=True)
    
    args = parser.parse_args()
    
    logging.info(f"Augment ? {args.augment}") 
    
    
    training_data_dirs = [pathlib.Path(_dir) for _dir in args.path_to_training_data]
    dataset = DonkeyCarDataset(training_data_dirs,args.augment)

    with open(args.hyperparam_file, 'r') as f:
        train_conf = json.load(f)

    logging.info(f" Starting with these params {train_conf}, model type {args.model_type}")
    
    if len(dataset) == 0:
        raise Exception(f"No training data at dirs {args.path_to_training_data} found. Check your paths")
    
    train_len = int(0.75*len(dataset))
    val_len   = len(dataset) - train_len

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_len, val_len))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=train_conf["batch_size"], shuffle=True,
                                             num_workers=4)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=train_conf["batch_size"], shuffle=True,
                                              num_workers=4)

    logging.info(f"Train dataset len: {len(train_dataset)}, val dataset len: {len(val_dataset)}")
    try:
        model = supported_models[args.model_type]().to(device)
    except KeyError:
        print(f"Model {args.model_type} not supported for now. Available models are {supported_models.keys()}")

    
    # Loss and optimizer
    learning_rate = train_conf["learning_rate"]
    num_epochs = train_conf["num_epochs"]
    criterion_angle = torch.nn.MSELoss()
    criterion_throttle = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model,criterion_angle,criterion_throttle,
            optimizer,train_dataloader,val_dataloader,
            num_epochs, learning_rate, args.model,device, train_conf["patience_for_early_stopping"])
    