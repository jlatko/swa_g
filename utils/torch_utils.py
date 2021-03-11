import torch
import os
from collections import OrderedDict
from utils.experiment_utils import get_available_epochs

def to_gpu(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

def load_model_weights(path, epoch=None):
    if epoch is None:
        # if multiple checkpoints get the latest
        epochs = get_available_epochs(path)
        if epochs:
            pth = os.path.join(path, f'model_{max(epochs)}.pth')
        else:
            pth = os.path.join(path, f'model.pth')
    else:
        pth = os.path.join(path, f'model_{epoch}.pth')
    print(f'Loading model from {pth}')
    return torch.load(pth)

def get_flattened_params(state_dict):
  flattened_params = []
  for _, weight in state_dict.items():
    flattened_params.append(torch.flatten(weight))
  flattened_params = torch.cat(flattened_params)
  return flattened_params

def flattened_to_state_dict(flattened_params, state_dict):
    new_state_dict = OrderedDict()
    i = 0
    for k, v in state_dict.items():
      w_shape = v.shape
      w_size = np.prod(w_shape)
      new_state_dict[k] = torch.reshape(flattened_params[int(i):int(i+w_size)], w_shape)
      i += w_size
    return new_state_dict
