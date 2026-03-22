import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import time

from tqdm import tqdm

from utils.config_processor import config
from models.model import create_model
from utils.loaders import create_cifar_loaders
from losses.losses import create_loss
from optimizers.optimizers import create_optimizer
from schedulers.schedulers import create_scheduler
from utils.metrics import MetricLogger

config.init(default_config_path='configs/default.yaml')

BASE_LOG_DIR = Path(__file__).parent / config.BASE_LOG_DIR
# EXPERIMENT = f'fix_opt/{config.opt.SELECTION_METHOD}prm_largebs{config.LARGE_BATCH}_smallbs{config.SMALL_BATCH}_lr{config.BASE_LR}_grow{config.opt.LR_GROW}_shrink{config.opt.LR_SHRINK}_m{config.opt.MOMENTUM}'
# EXPERIMENT = f'{config.model.type}_{config.opt.type}_bs{config.LARGE_BATCH}_sbs{config.SMALL_BATCH}_lr{config.BASE_LR}_m{config.opt.MOMENTUM}_ls{config.LABEL_SMOOTH}{"_clip" if config.CLIP_PROB.ENABLED else ""}'
# EXPERIMENT = f'{config.model.type}_bs{config.LARGE_BATCH}_epochs{config.EPOCHS}_schd{config.scheduler.type}_lr{config.BASE_LR}_minLR{config.scheduler.LR_MIN}_ls{config.LABEL_SMOOTH}{"_clip"+(str(config.CLIP_PROB.LEVEL) if config.CLIP_PROB.LEVEL is not None else "") if config.CLIP_PROB.ENABLED else ""}' \
#     f'{("_loss"+str(config.CLIP_LOSS.LOW_THRESHOLD)+"-"+str(config.CLIP_LOSS.HIGH_THRESHOLD) + (("e"+str(config.CLIP_LOSS.START_EPOCH)) if config.CLIP_LOSS.START_EPOCH else "")) if config.CLIP_LOSS.ENABLED else ""}'
EXPERIMENT = f'{config.model.type}_bs{config.LARGE_BATCH}_epochs{config.EPOCHS}_schd{config.scheduler.type}_lr{config.BASE_LR}_minLR{config.scheduler.LR_MIN}_{config.EXPERIMENT_SUFFIX or ""}'

LOG_DIR = BASE_LOG_DIR / EXPERIMENT
LOG_DIR.mkdir(parents=True, exist_ok=False)
print(str(LOG_DIR))
with open(f'{LOG_DIR}/config.yaml', 'w') as f:
    f.write(config.dump())

assert config.LARGE_BATCH % config.SMALL_BATCH == 0, "config.LARGE_BATCH size must be divisible by config.SMALL_BATCH size"

train_loader, test_loader = create_cifar_loaders(config.SMALL_BATCH, use_amp=False)
model = create_model(config)
loss_fn = create_loss(config)
opt = create_optimizer(config, model)
scheduler = create_scheduler(config, opt)
metrics = MetricLogger()
writer = SummaryWriter(log_dir=BASE_LOG_DIR / EXPERIMENT)

def train():
    global_step = 0
    prev_eval_step = 0
    iteration_no = 0
    opt.zero_grad(set_to_none=True)  # Initialize gradients at the start of epoch
    batch_split = config.LARGE_BATCH // config.SMALL_BATCH
    # epoch cycle
    for ep in range(config.EPOCHS):
        metrics.reset()
        loss_fn.reset_epoch()
        total_num, loss, loss_CE = 0, 0., 0.,
        epoch_start_time = time.time()
        fwd_time = 0
        bwd_time = 0
        pbar = tqdm(train_loader, postfix={'epoch': ep})
        # itrations cycle
        for ims, labs in pbar:
            iteration_no += 1
            bs = len(ims)
            total_num += bs
            global_step += bs

            fwd_start_time = time.time()
            out = model(ims)
            loss_i = loss_fn(out, labs) * bs / config.LARGE_BATCH  # use actual bs, not batch_split
            fwd_time += time.time() - fwd_start_time

            bwd_start_time = time.time()
            opt.update_before_backward()
            loss_i.backward()
            opt.update_after_backward()

            if iteration_no % batch_split == 0:

                opt.step()

                metrics.update('train', **opt.get_metrics())
                metrics.update('train', **loss_fn.get_metrics())
                writer.add_scalar('train/epoch', ep, global_step)

                opt.zero_grad(set_to_none=True)
                if scheduler is not None and not config.scheduler.BY_EPOCH:
                    scheduler.step()
                metrics.to_writer(writer, global_step)
                metrics.reset()
                    
            bwd_time += time.time() - bwd_start_time
        # end of itrations cycle
        if scheduler is not None and config.scheduler.BY_EPOCH:
            scheduler.step()

        writer.add_scalar('train/epoch_time', time.time() - epoch_start_time, global_step)
        writer.add_scalar('train/fwd_time', fwd_time, global_step)
        writer.add_scalar('train/bwd_time', bwd_time, global_step)
        if global_step >= prev_eval_step + config.VAL_STEP:
            eval(writer, global_step)
            prev_eval_step = global_step
    # end of epoch cycle
    print(f"{EXPERIMENT} finished")

def eval(writer, global_step):
    model.eval()
    metrix = MetricLogger()
    with torch.no_grad():
        total_correct, total_correct_flip, total_num = 0., 0., 0.,
        pbar = tqdm(test_loader)
        for iter_no, (ims, labs) in enumerate(pbar):
            total_num += ims.shape[0]
            out = model(ims)
            total_correct += out.argmax(1).eq(labs).sum().cpu().item()
            if config.USE_TTA_EVAL:
                out_flip = (out + model(torch.fliplr(ims))) / 2. # Test-time augmentation
                total_correct_flip += out_flip.argmax(1).eq(labs).sum().cpu().item()
            loss_fn(out, labs)
            metrix.update('test', **loss_fn.get_metrics())
            pbar.set_postfix(metrix.get_metrics())
    model.train()
    metrix.update('test', **{'accuracy':total_correct / total_num * 100,})
    if config.USE_TTA_EVAL:
        metrix.update('test', **{'accuracy with TTA':total_correct_flip / total_num * 100})
    metrix.to_writer(writer, global_step)
    return metrix

if __name__=='__main__':
    train()