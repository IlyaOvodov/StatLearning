
import torch

from .find_lr import FindLrScheduler


def create_scheduler(config, opt, model=None, loss_fn=None, train_loader=None, log_dir=None):
    if config.scheduler.type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.EPOCHS, eta_min=config.scheduler.LR_MIN)
    elif config.scheduler.type == 'LinearLR':
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=config.scheduler.LR_MIN/config.BASE_LR, total_iters=config.EPOCHS)
    elif config.scheduler.type == 'FindLr':
        assert model is not None and loss_fn is not None and train_loader is not None, "FindLr requires model, loss_fn, train_loader"
        scheduler = FindLrScheduler(opt, model, loss_fn, train_loader,
                                    start_lr=config.scheduler.FindLr.LR_LOW_BOUND,
                                    end_lr=config.scheduler.FindLr.LR_UPPER_BOUND,
                                    num_steps=config.scheduler.FindLr.NUM_STEPS,
                                    log_dir=log_dir)
    elif config.scheduler.type is None:
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type {config.scheduler.type}")
        
    # iters_per_epoch = len(train_loader)
    # lr_schedule = np.interp(np.arange((config.EPOCHS+1) * iters_per_epoch),
    #                         [0, 5 * iters_per_epoch, config.EPOCHS * iters_per_epoch],
    #                         [0, 1, 0])
    # scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    
    return scheduler