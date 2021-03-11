from models.get_model import get_model
from utils.get_datasets import get_datasets
from utils.experiment_utils import evaluate, train
from utils.torch_utils import to_gpu, load_model_weights
from utils.get_scheduler import get_scheduler
from evaluators.evaluator import Evaluator
from modules.swa import apply_swa
from utils.update_bn import update_batch_normalization
import torch
from modules.model_saver import ModelSaver
import os
import json

CONFIG = {
    "dataset": "MNIST",
    "ood_dataset": None,
    "n_classes": 10,
    "model_name": "mnist_vgg16",
    "load_model_path": None,
    "epochs": None,
    "mode": "base", # "base" | "swa" | "swag"
    "K": None
}

def run_evaluation(
    experiment_path,
    dataset,
    n_classes,
    model_name,
    load_model_path,
    K
):
    # INIT
    # experiment_path = get_valid_path(experiment_path)
    os.mkdir(experiment_path)
    config = locals()
    with open(os.path.join(experiment_path, 'config.json'), 'w') as fh:
        print(config)
        json.dump(config, fh)

    train_loader, test_loader = get_datasets(
        dataset, batch_size_train=256, batch_size_test=1024
    )

    # model
    model = get_model(model_name, False, n_classes, False)
    if mode == "base":
        weights = load_model_weights(load_model_path, epoch=epochs)
        model.load_state_dict(weights)
    elif mode == "swa":
        apply_swa(model, load_model_path, K=K)
        update_batch_normalization(model, train_loader)
    elif mode == "swag":
        raise NotImplementedError
    else:
        raise ValueError('wrong mode')

    to_gpu(model)

    # evaluation
    loss = torch.nn.CrossEntropyLoss()
    test_evaluator = Evaluator(n_classes, "TEST")
    test_metrics = evaluate(i, test_loader, model, test_evaluator, loss) # this could be validation set
    print(test_metrics)


    return test_evaluator, train_loader, test_loader, model, config
