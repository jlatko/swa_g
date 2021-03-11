import os

def get_available_epochs(path):
    return [int(f.split('.')[0].split('_')[1]) for f in os.listdir(path) if f.startswith('model_')]