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

BATCH_SIZE = 512
BASE_LR = 0.1
MOMENTUM = 0.9
BASE_LOG_DIR = Path(__file__).parent / 'results/statopt'
EXPERIMENT = f'baseline_bs{BATCH_SIZE}_lr{BASE_LR}_m{MOMENTUM}'

EPOCHS = 24
VAL_STEP = 25000
USE_TTA_EVAL = False

train_loader, test_loader = loaders.create_cifar_loaders(BATCH_SIZE, use_amp=False)
model = model.create_model()

def train():
    opt = SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=5e-4)
    iters_per_epoch = len(train_loader)
    lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                            [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    
    writer = SummaryWriter(log_dir=BASE_LOG_DIR / EXPERIMENT)

    global_step = 0
    prev_eval_step = 0
    for ep in range(EPOCHS):
        epoch_start_time = time.time()
        pbar = tqdm(train_loader, postfix={'epoch': ep})
        for ims, labs in pbar:
            global_step += BATCH_SIZE
            opt.zero_grad(set_to_none=True)
            out = model(ims)
            loss = loss_fn(out, labs)
            loss.backward()
            opt.step()
            scheduler.step()
            writer.add_scalar('train/loss', loss.mean().item(), global_step)
            writer.add_scalar('train/lr', opt.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/epoch', ep, global_step)
        epoch_time = time.time() - epoch_start_time
        writer.add_scalar('train/epoch_time', epoch_time, global_step)
        if global_step >= prev_eval_step + VAL_STEP:
            eval(writer, global_step)
            prev_eval_step = global_step

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