import torch 

def get_schedule_fn(cycle_size, min_fraction=0.1):
  def _schedule_fn(step):
    return 1 - (1 - min_fraction) * ((step % cycle_size) / cycle_size)
  return _schedule_fn

def get_scheduler(name, train_loader, optimizer, cycle_length=1, min_lr=1e-6):
    # calculate length of a cycle for the given dataset
    epoch_steps = len(train_loader)
    cycle_length *= epoch_steps 

    if name == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_length, T_mult=1, eta_min=min_lr)
    elif name == 'LambdaLR':
        lr_fraction = min_lr / optimizer.param_groups[0]["lr"]
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_schedule_fn(cycle_length, lr_fraction))
    else:
        raise ValueError(f'Scheduler {name} is not supported')