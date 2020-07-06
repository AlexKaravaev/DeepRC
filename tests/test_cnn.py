import numpy as np
import torch
from rc_car.models.cnn import CNNAutoPilot

if __name__ == "__main__":
    model = CNNAutoPilot()
    batch_size, C, H, W = 1, 3, 160, 120
    x  = torch.randn(batch_size, C, H, W)
    output = model(x)
    assert output