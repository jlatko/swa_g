import torch
from utils.experiment_utils import evaluate
import pandas as pd
import os

def get_point_along_axis(w0, w1, loc):
    d = w1 - w0
    abs_dist = loc * torch.norm(d)
    w = w0 + loc * d
    return w, abs_dist

def evaluate_along_axis(ex_path, e0, e1, test_loader, model, evaluator, loss, step=0.1, margin=0.2):
    w0 = torch.load(os.path.join(ex_path, f'model_{e0}.pth'))
    w1 = torch.load(os.path.join(ex_path, f'model_{e1}.pth'))
    w0f = get_flattened_params(w0)
    w1f = get_flattened_params(w1)
    results = []
    for loc in np.arange(-margin, 1 + margin + step, step):
        print('evaluating at ', loc)
        wif, abs_dist = get_point_along_axis(w0f, w1f, loc=loc)
        wi = flattened_to_state_dict(wif, w0)
        model.load_state_dict(wi)
        # to_gpu(model)
        result = evaluate(0, test_loader, model, evaluator, loss)
        result['loc'] = loc
        result['abs_dist'] = abs_dist.cpu().numpy()
        results.append(result)
    return pd.DataFrame(results)

