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
import stat_scheduler_twin as slr

USE_STAT_LR = True
EPOCHS = 24
BATCH_SIZE = 10  # 512 for baseline
BASE_LR = 0.00001 if USE_STAT_LR else 0.5  # GVNC 0.5 for baseline
VAL_PERIOD = 2
USE_AMP = False

LOG_FILE=None
#LOG_FILE='/home/jovyan/ovodov/stat_lr/log/log-twin.txt'

sched_params = dict(
jitter=1,
jitter_pvalue_thr=0.05,
step_lr_scale=0.1,
use_detailed_stat=False,
)

BASE_LOG_DIR = Path(__file__).parent / 'results'
if USE_STAT_LR:
    EXPERIMENT = f'stat_twin_bin_momentum_0_bs-{BATCH_SIZE}_lr-{BASE_LR}_{" ".join([k + "_"+str(v) for k,v in sched_params.items()])}'
else:
    EXPERIMENT = f'baseline_momentum_0_bs-{BATCH_SIZE}_lr-{BASE_LR}'

train_loader, test_loader = loaders.create_cifar_loaders(BATCH_SIZE, use_amp=USE_AMP)
model_lo = model.create_model()
model_hi = model.create_model()
model_hi.load_state_dict(model_lo.state_dict())

def train():
    opt_lo = SGD(model_lo.parameters(), lr=BASE_LR, momentum=0, weight_decay=5e-4)  # GVNC momentum=0.9 for baseline
    opt_hi = SGD(model_hi.parameters(), lr=BASE_LR, momentum=0, weight_decay=5e-4)  # GVNC momentum=0.9 for baseline
    iters_per_epoch = len(train_loader)

    if USE_STAT_LR:
        scheduler = slr.StatLRSceduler(opt_lo, opt_hi, log_file=LOG_FILE, **sched_params)
    else:
        lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                                [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
                                [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(opt_lo, lr_schedule.__getitem__)
        
    scaler_lo = GradScaler(enabled=USE_AMP)
    scaler_hi = GradScaler(enabled=USE_AMP)
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
            writer.add_scalar('train/lr_lo', opt_lo.param_groups[0]['lr'], logget_step)
            writer.add_scalar('train/lr_hi', opt_hi.param_groups[0]['lr'], logget_step)
                        
            opt_lo.zero_grad(set_to_none=True)
            opt_hi.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                # model_lo == model_hi 
                out = model_lo(ims)
                loss_lo = loss_fn(out, labs)
                out = model_hi(ims)
                loss_hi = loss_fn(out, labs)
                assert loss_lo.sum() == loss_hi.sum()
                prev_loss = loss_lo
            writer.add_scalar('train/loss', loss_lo.mean().item(), logget_step)
            writer.add_scalar('train/base_lr', scheduler.base_lrs[0], logget_step)

            scaler_lo.scale(loss_lo).mean().backward()
            scaler_lo.step(opt_lo)
            scaler_lo.update()
            scaler_hi.scale(loss_hi).mean().backward()
            scaler_hi.step(opt_hi)
            scaler_hi.update()
            
            if USE_STAT_LR:
                with autocast(enabled=USE_AMP):
                    with torch.no_grad():
                        model_lo.eval()  # ?
                        out = model_lo(ims)
                        loss_lo = loss_fn(out, labs)
                        model_lo.train()
                        model_hi.eval()  # ?
                        out = model_hi(ims)
                        loss_hi = loss_fn(out, labs)
                        model_hi.train()
                        writer.add_scalar('train/loss_delta', loss_hi.mean().item() - loss_lo.mean().item(), logget_step)
                        use_hi = scheduler.step(prev_loss=prev_loss, loss_lo=loss_lo, loss_hi=loss_hi)
                        if use_hi:
                            model_lo.load_state_dict(model_hi.state_dict()) 
                        else: 
                            model_hi.load_state_dict(model_lo.state_dict())
            else:
                scheduler.step()
        if ep % VAL_PERIOD == 0 or ep == EPOCHS-1:
            metrix = eval()
            writer.add_scalar('test/accuracy', metrix['accuracy'], logget_step)
            # writer.add_scalars('test/accuracy', {"with TTA": metrix['accuracy with TTA']}, logget_step)


def eval():
    model_lo.eval()
    with torch.no_grad():
        total_correct, total_correct_flip, total_num = 0., 0., 0.
        pbar = tqdm(test_loader)
        for ims, labs in pbar:
            with autocast():
                out = model_lo(ims)
                out_flip = (out + model_lo(torch.fliplr(ims))) / 2. # Test-time augmentation
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_correct_flip += out_flip.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
            accuracy = total_correct / total_num * 100
            accuracy_tta = total_correct_flip / total_num * 100
            metrix = {'accuracy': accuracy, 'accuracy with TTA': accuracy_tta}
            pbar.set_postfix(metrix)
    model_lo.train()
    return metrix

if __name__=='__main__':
    train()