from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import torch
BASE_LOG_DIR = Path(__file__).parent / 'results/tmp'
EXPERIMENT = f'tst'

writer = SummaryWriter(log_dir=BASE_LOG_DIR / EXPERIMENT)

grad_sqr = torch.zeros([])
num_updates = torch.zeros([])
grad = torch.zeros([])

for i in range(100):
    prev_grad = grad.clone()
    random_value = np.random.normal(loc=0.1, scale=10)    
    grad += random_value
    new_grad = grad - prev_grad
    grad_sqr.add_(torch.square(new_grad))
    num_updates.add_(torch.ones([], dtype=torch.int16))
    if num_updates > 1:
        p_value = grad.abs()/num_updates/torch.sqrt(grad_sqr/num_updates)
        s2 = (grad_sqr - torch.square(grad)/num_updates)/(num_updates-1)
        s = torch.sqrt(s2)
        t_value = grad.abs()/s*torch.sqrt(num_updates)
        writer.add_scalar('p_value', p_value, i)
        writer.add_scalar('t_value', t_value, i)
        print(i+1, p_value.item(), t_value.item(), grad.abs()/num_updates, torch.sqrt(grad_sqr/num_updates))