from .sgd_with_stats import SGDProxy, SGDWithStats, SGDWithStatsFixed

def create_optimizer(config, model):
    if config.opt.type == 'SGDWithStatsFixed':
        opt = SGDWithStatsFixed(config=config, params=model.parameters(), lr=config.BASE_LR, momentum=config.opt.MOMENTUM, weight_decay=config.opt.WEIGHT_DECAY, lr_grow=config.opt.LR_GROW, lr_shrink=config.opt.LR_SHRINK, selection_method=config.opt.SELECTION_METHOD)
    elif config.opt.type == 'SGD':
        opt = SGDProxy(config=config, params=model.parameters(), lr=config.BASE_LR, momentum=config.opt.MOMENTUM, weight_decay=config.opt.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer type {config.opt.type}")
    return opt