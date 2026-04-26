import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class CELossWithClip(CrossEntropyLoss):
    """
    Вариант CE loss с clip с вариантами:
    - LABEL_SMOOTH (обычный)
    - CLIP_PROB (вместе с метками обрезаем вероятности)
    - CLIP_LOSS (выбираем только семплы, дающие лосс в интервале)
    См. Мои идеи/Optimizers/25-07-26 Clip вместо label_smooth
    При CLIP_PROB, CLIP_LOSS  совпадает с CE Loss
    """
    def __init__(self, config):
        super().__init__()
        self.epoch = -1  # epoch number 0..
        self.config = config
        self.metrics = {}
        self.reset_epoch()

    def get_metrics(self):
        return self.metrics

    def __call__(self, out, labs):
        config = self.config
        eps = 1e-8
        num_classes = out.shape[1]
        probs = F.softmax(out, dim=1)
        if config.CLIP_PROB.ENABLED:
            level = config.CLIP_PROB.LEVEL if config.CLIP_PROB.LEVEL is not None else config.LABEL_SMOOTH
            levels = (torch.zeros_like(probs) + level/num_classes, torch.ones_like(probs) + 1 - level)
            skips = ((probs <= levels[0]) | (probs >= levels[1])).min(1).values
            self.skipped += skips.sum().item()
            self.total += skips.shape[0] 
            probs = torch.clamp(probs, levels[0].clamp(min=eps), levels[1].clamp(max=1-eps))
        else:
            probs = torch.clamp(probs, min=eps, max=1 - eps)
        one_hot_targets = F.one_hot(labs, num_classes=num_classes).float()
        if config.LABEL_SMOOTH:
            one_hot_targets = (1 - config.LABEL_SMOOTH) * one_hot_targets + config.LABEL_SMOOTH / num_classes
        loss = -torch.sum(one_hot_targets * torch.log(probs), dim=1)
        if config.CLIP_LOSS.ENABLED and (self.epoch >= (config.CLIP_LOSS.START_EPOCH or 0)):
            loss, loss_indices = loss.sort(dim=0)
            low_idx = int(loss.shape[0] * config.CLIP_LOSS.LOW_THRESHOLD)
            high_idx = int(loss.shape[0] * config.CLIP_LOSS.HIGH_THRESHOLD)
            loss = loss[low_idx:high_idx]
        loss = torch.mean(loss)
        self.metrics['loss'] = loss.item()
        if config.CLIP_PROB.ENABLED:
            self.metrics['skips'] = self.skipped/self.total if self.total > 0 else 0.0
        self.metrics['loss_CE'] = super().__call__(out, labs).mean().item()
        
        return loss

    def reset_epoch(self):
        self.skipped = 0
        self.total = 0
        self.epoch += 1

def create_loss(config):
    # if config.CLIP_PROB.ENABLED:
    return CELossWithClip(config)
    # elif config.CLIP_LOSS.ENABLED:
    #     return CELossWithClip(config)
    # else:
    #     return CrossEntropyLoss()