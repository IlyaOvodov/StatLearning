import numpy as np
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time

from tqdm import tqdm

import model
import loaders
from sgd_with_stats import SGDWithStats, SGDWithStatsFixed
DEBUG = True

LARGE_BATCH = 32
SMALL_BATCH = 8
BASE_LR = 0.001
LR_GROW = 0.01
LR_SHRINK = 0.01
SELECTION_METHOD = 'tvalue2'
MOMENTUM = 0 #0.9
BASE_LOG_DIR = Path(__file__).parent / 'results/statopt'
EXPERIMENT = f'fix_opt/{SELECTION_METHOD}prm_largebs{LARGE_BATCH}_smallbs{SMALL_BATCH}_lr{BASE_LR}_grow{LR_GROW}_shrink{LR_SHRINK}_m{MOMENTUM}'
print(EXPERIMENT)

EPOCHS = 10
VAL_STEP = 25000
USE_TTA_EVAL = False

assert LARGE_BATCH % SMALL_BATCH == 0, "LARGE_BATCH size must be divisible by SMALL_BATCH size"

train_loader, test_loader = loaders.create_cifar_loaders(SMALL_BATCH, use_amp=False)
model = model.create_model()

def train():
    opt = SGDWithStatsFixed(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=5e-4, lr_grow=LR_GROW, lr_shrink=LR_SHRINK, selection_method=SELECTION_METHOD)
    iters_per_epoch = len(train_loader)
    # lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
    #                         [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
    #                         [0, 1, 0])
    # scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scheduler = None
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    
    writer = SummaryWriter(log_dir=BASE_LOG_DIR / EXPERIMENT)

    global_step = 0
    prev_eval_step = 0
    iteration_no = 0
    opt.zero_grad(set_to_none=True)  # Initialize gradients at the start of epoch
    batch_split = LARGE_BATCH // SMALL_BATCH
    loss = 0
    for ep in range(EPOCHS):
        epoch_start_time = time.time()
        pbar = tqdm(train_loader, postfix={'epoch': ep})
        for ims, labs in pbar:
            iteration_no += 1
            bs = len(ims)
            global_step += bs
            out = model(ims)
            loss_i = loss_fn(out, labs) * bs / LARGE_BATCH
            loss += loss_i
            opt.update_before_backward()
            loss_i.backward()
            opt.update_after_backward()
            if iteration_no % batch_split == 0:
                
                if batch_split != 1:
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
                loss = 0
            if scheduler is not None:
                scheduler.step()
            writer.add_scalar('train/lr', opt.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/epoch', ep, global_step)
        epoch_time = time.time() - epoch_start_time
        writer.add_scalar('train/epoch_time', epoch_time, global_step)
        if global_step >= prev_eval_step + VAL_STEP:
            eval(writer, global_step)
            prev_eval_step = global_step
    print(f"{EXPERIMENT} finished")

def eval(writer, global_step):
    model.eval()
    metrix = dict()
    with torch.no_grad():
        total_correct, total_correct_flip, total_num = 0., 0., 0.
        pbar = tqdm(test_loader)
        for ims, labs in pbar:
            total_num += ims.shape[0]
            out = model(ims)
            total_correct += out.argmax(1).eq(labs).sum().cpu().item()
            metrix['accuracy'] = total_correct / total_num * 100
            if USE_TTA_EVAL:
                out_flip = (out + model(torch.fliplr(ims))) / 2. # Test-time augmentation
                total_correct_flip += out_flip.argmax(1).eq(labs).sum().cpu().item()
                metrix['accuracy with TTA'] = total_correct_flip / total_num * 100
            pbar.set_postfix(metrix)
    model.train()
    for k, v in metrix.items():
        writer.add_scalar(f'test/{k}', v, global_step)
    return metrix

if __name__=='__main__':
    train()