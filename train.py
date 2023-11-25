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

EPOCHS = 24
BATCH_SIZE = 512
BASE_LR = 0.5
VAL_PERIOD = 2
USE_AMP = True
BASE_LOG_DIR = Path(__file__).parent / 'results'
EXPERIMENT = f'baseline_bs-{BATCH_SIZE}_lr-{BASE_LR}'

train_loader, test_loader = loaders.create_cifar_loaders(BATCH_SIZE, use_amp=USE_AMP)
model = model.create_model()

def train():
    opt = SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = len(train_loader)
    lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                            [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler(enabled=USE_AMP)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    
    writer = SummaryWriter(log_dir=BASE_LOG_DIR / EXPERIMENT)

    global_step = -1
    for ep in range(EPOCHS):
        pbar = tqdm(train_loader, postfix={'epoch': ep})
        for ims, labs in pbar:
            global_step += BATCH_SIZE
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                out = model(ims)
                loss = loss_fn(out, labs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            writer.add_scalar('train/loss', loss.mean().item(), global_step)
            writer.add_scalar('train/lr', opt.param_groups[0]['lr'], global_step)
            writer.add_scalar('epoch', ep, global_step)
        if ep % VAL_PERIOD == 0:
            metrix = eval()
            writer.add_scalars('test', metrix, global_step)


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
            metrix = {'accuracy': accuracy, 'accuracy with TTA:': accuracy_tta}
            pbar.set_postfix(metrix)
    model.train()
    return metrix

if __name__=='__main__':
    train()