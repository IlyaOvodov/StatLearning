import numpy as np
from pathlib import Path
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import model
import loaders
import stat_scheduler2

USE_STAT_LR = True
EPOCHS = 24
BATCH_SIZE = 10
BASE_LR = 0.001 # if USE_STAT_LR else 0.5  #GVNC
VAL_PERIOD = 2
USE_AMP = True

sched_params = dict(
jitter=1,
jitter_pvalue_thr=0.05,
step_lr_scale=1,
use_detailed_stat=False,
)

BASE_LOG_DIR = Path(__file__).parent / 'results'
if USE_STAT_LR:
    EXPERIMENT = f'stat_bs-{BATCH_SIZE}_lr-{BASE_LR}_{" ".join([k + "_"+str(v) for k,v in sched_params.items()])}'
else:
    EXPERIMENT = f'baseline_bs-{BATCH_SIZE}_lr-{BASE_LR}'

train_loader, test_loader = loaders.create_cifar_loaders(BATCH_SIZE, use_amp=USE_AMP)
model = model.create_model()

def train():
    opt = SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = len(train_loader)

    if USE_STAT_LR:
        scheduler = stat_scheduler2.StatLRSceduler(opt, **sched_params)
    scaler = GradScaler(enabled=USE_AMP)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    
    writer = SummaryWriter(log_dir=BASE_LOG_DIR / EXPERIMENT)

    global_iter = -1
    logget_step = 0.
    
    for ep in range(EPOCHS):
        pbar = tqdm(train_loader, postfix={'epoch': ep})
        for ims, labs in pbar:
            global_iter += 1
            logget_step = global_iter
            writer.add_scalar('epoch', ep, logget_step)
            writer.add_scalar('train/lr', opt.param_groups[0]['lr'], logget_step)
                        
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                out = model(ims)
                loss = loss_fn(out, labs)
            writer.add_scalar('train/loss', loss.mean().item(), logget_step)

            if USE_STAT_LR:
                writer.add_scalar('train/base_lr', scheduler.base_lrs[0], logget_step)
                scheduler.step(loss=loss)

            scaler.scale(loss).mean().backward()
            scaler.step(opt)
            scaler.update()

        if ep % VAL_PERIOD == 0 or ep == EPOCHS-1:
            metrix = eval()
            writer.add_scalar('test/accuracy', metrix['accuracy'], logget_step)
            # writer.add_scalars('test/accuracy', {"with TTA": metrix['accuracy with TTA']}, logget_step)


def eval():
    model.eval()
    with torch.no_grad():
        total_correct, total_correct_flip, total_num = 0., 0., 0.
        pbar = tqdm(test_loader)
        for ims, labs in pbar:
            with autocast():
                out = model(ims)
                out_flip = (out + model(torch.fliplr(ims))) / 2. # Test-time augmentation
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_correct_flip += out_flip.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
            accuracy = total_correct / total_num * 100
            accuracy_tta = total_correct_flip / total_num * 100
            metrix = {'accuracy': accuracy, 'accuracy with TTA': accuracy_tta}
            pbar.set_postfix(metrix)
    model.train()
    return metrix

if __name__=='__main__':
    train()