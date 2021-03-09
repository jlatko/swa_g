from models.get_model import get_model
from utils.get_datasets import get_datasets
from utils.experiment_utils import evaluate, train
from utils.torch_utils import to_gpu, load_model_weights
from utils.get_scheduler import get_scheduler
from evaluators.evaluator import Evaluator
import torch
from modules.model_saver import ModelSaver
import os
import json

CONFIG = {
    "dataset": "MNIST",
    "pretrained": True,
    "freeze": True,
    "n_classes": 10,
    "n_epochs": 10,
    "model_name": "mnist_vgg16",
    "optimizer_name": "Adam",
    "optimizer_kwargs": {
        "lr": 3e-4,
    },
    "load_model_path": None,
    "save_model": True,
    "scheduler": None,
    "scheduler_kwargs": {},
    "saver_kwargs": {
        "mode": 'epochs',
        "epochs_to_save": [0,1,2,3,4,5,6,7,8,9]
    },
}

def get_valid_path(experiment_path):
    if os.path.exists(experiment_path):
        print(f'Experiment {experiment_path} already exists, will use {experiment_path+"_0"}')
        experiment_path += '_0'
        return get_valid_path(experiment_path) # check recursive
    else:
        return experiment_path

def run_training(
    experiment_path,
    dataset,
    pretrained,
    freeze,
    n_classes,
    n_epochs,
    model_name,
    optimizer_name,
    optimizer_kwargs,
    load_model_path,
    save_model,
    scheduler,
    scheduler_kwargs,
    saver_kwargs,
):
    # INIT
    experiment_path = get_valid_path(experiment_path)
    os.mkdir(experiment_path)
    config = locals()
    with open(os.path.join(experiment_path, 'config.json'), 'w') as fh:
        print(config)
        json.dump(config, fh)

    train_loader, test_loader = get_datasets(
        dataset, batch_size_train=256, batch_size_test=1024
    )

    # model
    model = get_model(model_name, pretrained, n_classes, freeze)
    if load_model_path:
        weights = load_model_weights(load_model_path)
        model.load_state_dict(weights)
    
    to_gpu(model)

    # init optimizer
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    loss = torch.nn.CrossEntropyLoss()

    # evaluators
    train_evaluator = Evaluator(n_classes, "TRAIN", path=experiment_path)
    trainval_evaluator = Evaluator(n_classes, "TRAINVAL", path=experiment_path)
    test_evaluator = Evaluator(n_classes, "TEST", path=experiment_path)
    saver = ModelSaver(experiment_path, **saver_kwargs)
    
    if scheduler is not None:
        scheduler = get_scheduler(scheduler, train_loader, optimizer, **scheduler_kwargs)

    for i in range(n_epochs):
        # train
        train(i, train_loader, model, train_evaluator, optimizer, loss, scheduler)
        # trainval
        evaluate(i, train_loader, model, trainval_evaluator, loss)
        # test
        test_metrics = evaluate(i, test_loader, model, test_evaluator, loss) # this could be validation set
        
        # save (here will probably swag stuff go)
        if save_model:
            saver.save_if_needed(model, i, test_metrics) # TODO: don't do early stopping on test set

