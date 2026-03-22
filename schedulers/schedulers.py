
import torch

def create_scheduler(config, opt):
    if config.scheduler.type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.EPOCHS, eta_min=config.scheduler.LR_MIN)
    elif config.scheduler.type == 'LinearLR':
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=config.scheduler.LR_MIN/config.BASE_LR, total_iters=config.EPOCHS)
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