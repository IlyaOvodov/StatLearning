import numpy as np
import os
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as tvmodels
import time

from tqdm import tqdm

from utils.config_processor import config
import model
from resnet_k_upd import ResNet18 as ResNet18_kuangliu
import loaders
from sgd_with_stats import SGDWithStats, SGDWithStatsFixed

config.init(default_config_path='configs/default.yaml')

BASE_LOG_DIR = Path(__file__).parent / config.BASE_LOG_DIR
# EXPERIMENT = f'fix_opt/{config.opt.SELECTION_METHOD}prm_largebs{config.LARGE_BATCH}_smallbs{config.SMALL_BATCH}_lr{config.BASE_LR}_grow{config.opt.LR_GROW}_shrink{config.opt.LR_SHRINK}_m{config.opt.MOMENTUM}'
# EXPERIMENT = f'{config.model.type}_{config.opt.type}_bs{config.LARGE_BATCH}_sbs{config.SMALL_BATCH}_lr{config.BASE_LR}_m{config.opt.MOMENTUM}_ls{config.LABEL_SMOOTH}{"_clip" if config.CLIP_PROB.ENABLED else ""}'
EXPERIMENT = f'{config.model.type}_bs{config.LARGE_BATCH}_epochs{config.EPOCHS}_schd{config.scheduler.type}_lr{config.BASE_LR}_minLR{config.scheduler.LR_MIN}_ls{config.LABEL_SMOOTH}{"_clip"+(str(config.CLIP_PROB.LEVEL) if config.CLIP_PROB.LEVEL is not None else "") if config.CLIP_PROB.ENABLED else ""}'
LOG_DIR = BASE_LOG_DIR / EXPERIMENT
assert not os.path.exists(LOG_DIR), f"Directory {LOG_DIR} already exists!"
print(str(LOG_DIR))

assert config.LARGE_BATCH % config.SMALL_BATCH == 0, "config.LARGE_BATCH size must be divisible by config.SMALL_BATCH size"

train_loader, test_loader = loaders.create_cifar_loaders(config.SMALL_BATCH, use_amp=False)

if config.model.type == 'ResNet18':
    model = tvmodels.resnet18()
elif config.model.type == 'ResNet34':
    model = tvmodels.resnet34()
elif config.model.type == 'ResNet18_kuangliu':
    model = ResNet18_kuangliu() # https://github.com/kuangliu/pytorch-cifar
elif config.model.type == 'tiny':
    model = model.create_model()
model=model.cuda()

loss_fn_CE = CrossEntropyLoss()
def CELossWithClip():
    def loss_fn(out, labs):
        eps = 1e-8
        num_classes = out.shape[1]
        probs = F.softmax(out, dim=1)
        if config.CLIP_PROB.ENABLED:
            level = config.CLIP_PROB.LEVEL if config.CLIP_PROB.LEVEL is not None else config.LABEL_SMOOTH
            probs = torch.clamp(probs, max(eps, level/num_classes), 1 - max(eps, level))
        else:
            probs = torch.clamp(probs, min=eps, max=1 - eps)
        one_hot_targets = F.one_hot(labs, num_classes=num_classes).float()
        if config.LABEL_SMOOTH:
            one_hot_targets = (1 - config.LABEL_SMOOTH) * one_hot_targets + config.LABEL_SMOOTH / num_classes
        loss = -torch.mean(torch.sum(one_hot_targets * torch.log(probs), dim=1))
        return loss
    return loss_fn
loss_fn = CELossWithClip()

def train():
    if config.opt.type == 'SGDWithStatsFixed':
        opt = SGDWithStatsFixed(model.parameters(), lr=config.BASE_LR, momentum=config.opt.MOMENTUM, weight_decay=config.opt.WEIGHT_DECAY, lr_grow=config.opt.LR_GROW, lr_shrink=config.opt.LR_SHRINK, selection_method=config.opt.SELECTION_METHOD)
    elif config.opt.type == 'SGD':
        opt = SGD(model.parameters(), lr=config.BASE_LR, momentum=config.opt.MOMENTUM, weight_decay=config.opt.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer type {config.opt.type}")
    
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
    
    writer = SummaryWriter(log_dir=BASE_LOG_DIR / EXPERIMENT)

    global_step = 0
    prev_eval_step = 0
    iteration_no = 0
    opt.zero_grad(set_to_none=True)  # Initialize gradients at the start of epoch
    batch_split = config.LARGE_BATCH // config.SMALL_BATCH
    loss, loss_CE = 0., 0.,
    # epoch cycle
    for ep in range(config.EPOCHS):
        epoch_start_time = time.time()
        pbar = tqdm(train_loader, postfix={'epoch': ep})
        # itrations cycle
        for ims, labs in pbar:
            iteration_no += 1
            bs = len(ims)
            global_step += bs
            out = model(ims)
            loss_i = loss_fn(out, labs) * bs / config.LARGE_BATCH  # use actual bs, not batch_split
            loss += loss_i
            loss_CE += loss_fn_CE(out, labs) * bs / config.LARGE_BATCH
            if hasattr(opt, 'update_before_backward'):  # SGDWithStats
                opt.update_before_backward()
            loss_i.backward()
            if hasattr(opt, 'update_after_backward'):  # SGDWithStats
                opt.update_after_backward()
            if iteration_no % batch_split == 0:
                
                if batch_split != 1:
                    if hasattr(opt, 't_value'):
                        all_t_values = torch.cat([opt.t_value(p).reshape(-1) for p in model.parameters()])
                        max_t_value = all_t_values.max().item()
                        mean_t_value = all_t_values.mean().item()
                        med_t_value = torch.quantile(all_t_values, 0.5).item()
                        max90_t_value = torch.quantile(all_t_values, 0.9).item()
                        writer.add_scalar('train/t_value', mean_t_value, global_step)
                        writer.add_scalar('train/t_value_max', max_t_value, global_step)
                        writer.add_scalar('train/t_value_max90', max90_t_value, global_step)
                        writer.add_scalar('train/t_value_med', med_t_value, global_step)
                        # for name, p in model.named_parameters():
                        #     print(f"{name}: {opt.t_value(p).mean().item()}")

                opt.step()

                if isinstance(opt, SGDWithStatsFixed):
                    ls_scles = [opt.state[p]["lr_scale"].reshape(-1) for p in model.parameters()]
                    ls_scles = torch.cat(ls_scles)
                    writer.add_scalar('train/ls_scale_mean', ls_scles.mean().item(), global_step)
                    writer.add_scalar('train/ls_scale_max', ls_scles.max().item(), global_step)
                    writer.add_scalar('train/ls_scale_min', ls_scles.min().item(), global_step)

                opt.zero_grad(set_to_none=True)
                writer.add_scalar('train/loss', loss.mean().item(), global_step)
                writer.add_scalar('train/loss_CE', loss_CE.mean().item(), global_step)
                writer.add_scalar('train/lr', opt.param_groups[0]['lr'], global_step)
                writer.add_scalar('train/epoch', ep, global_step)
                loss = 0
                loss_CE = 0
                if scheduler is not None and not config.scheduler.BY_EPOCH:
                    scheduler.step()
        # end of itrations cycle
        epoch_time = time.time() - epoch_start_time
        if scheduler is not None and config.scheduler.BY_EPOCH:
            scheduler.step()
        writer.add_scalar('train/epoch_time', epoch_time, global_step)
        if global_step >= prev_eval_step + config.VAL_STEP:
            eval(writer, global_step)
            prev_eval_step = global_step
    # end of epoch cycle
    print(f"{EXPERIMENT} finished")

def eval(writer, global_step):
    model.eval()
    metrix = dict()
    with torch.no_grad():
        total_correct, total_correct_flip, total_num, loss, loss_CE = 0., 0., 0., 0., 0.,
        pbar = tqdm(test_loader)
        for ims, labs in pbar:
            total_num += ims.shape[0]
            out = model(ims)
            total_correct += out.argmax(1).eq(labs).sum().cpu().item()
            metrix['accuracy'] = total_correct / total_num * 100
            if config.USE_TTA_EVAL:
                out_flip = (out + model(torch.fliplr(ims))) / 2. # Test-time augmentation
                total_correct_flip += out_flip.argmax(1).eq(labs).sum().cpu().item()
                metrix['accuracy with TTA'] = total_correct_flip / total_num * 100
            loss += loss_fn(out, labs)
            metrix['loss'] = loss.cpu().item() / total_num
            loss_CE += loss_fn_CE(out, labs)
            metrix['loss_CE'] = loss_CE.cpu().item() / total_num
            pbar.set_postfix(metrix)
    model.train()
    for k, v in metrix.items():
        writer.add_scalar(f'test/{k}', v, global_step)
    return metrix

if __name__=='__main__':
    train()