
import pathlib
import torch
import os
import logging

from rc_car.models.utils import EarlyStopping

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def train(model: torch.nn.Module , criterion_angle: torch.nn.functional,
          criterion_throttle: torch.nn.functional, optimizer: torch.optim,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          num_epochs: int, lr: float, tb_log: str,
          device: str, patience: int):
    
    tb_log = './tb_logs/' + tb_log
    writer = SummaryWriter(tb_log)

    path = pathlib.Path(tb_log).absolute()
    model_name = path.name + '.pt'
    model_path = str(path) + "/" + model_name
    logging.info(f"Saving model to {model_path}")
    # Define early stopping callback
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                    path=model_path)

    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        for i, goal_vector in enumerate(train_dataloader):
            
            images = goal_vector['image'].to(device)
            throttles = goal_vector['throttle'].to(device)
            angles = goal_vector['angle'].to(device)
            
            out_angle, out_throttle = model(images)

            angle_loss = criterion_angle(out_angle, angles)
            throttle_loss = criterion_throttle(out_throttle, throttles)
            loss = angle_loss + throttle_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch)

        for i, goal_vector in enumerate(val_dataloader):
            images = goal_vector['image'].to(device)
            throttles = goal_vector['throttle'].to(device)
            angles = goal_vector['angle'].to(device)
            
            with torch.no_grad():
                val_out_angle, val_out_throttle = model(images)
                val_angle_loss = criterion_angle(val_out_angle, angles)
                val_throttle_loss = criterion_throttle(val_out_throttle, throttles)
                val_loss = val_angle_loss + val_throttle_loss

        writer.add_scalar('Loss/valid', val_loss.item(), epoch)
        
        logging.info ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), val_loss.item()))
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break
    else:
        torch.save(model.state_dict(), model_path)
        logging.info("Finished training")
    