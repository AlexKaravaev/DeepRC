import numpy as np
import torch

def crop_input(input_shape, roi):
    height = input_shape[0]
    new_height = height - roi[0] - roi[1]
    return (new_height, input_shape[1], input_shape[2])
    
def pil2tensor(image,dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False) )