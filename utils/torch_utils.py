import torch
import os

def to_gpu(x):
    if torch.cuda.is_available():
        x.cuda()

def load_model_weights(path, epoch=None):
    if epoch is None:
        # if multiple checkpoints get the latest
        epochs = [int(f.split('.')[0].split('_')[1]) for f in os.listdir(path) if f.startswith('model_')]
        if epochs:
            pth = os.path.join(path, f'model_{max(epochs)}.pth')
        else:
            pth = os.path.join(path, f'model.pth')
    else:
        pth = os.path.join(path, f'model_{epoch}.pth')
    print(f'Loading model from {pth}')
    return torch.load(pth)
