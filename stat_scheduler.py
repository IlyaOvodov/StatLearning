from collections import namedtuple
from copy import copy
import math
import random
from scipy import stats
import torch

LossRec = namedtuple('LossRec', ['step', 'is_hi', 'is_hi_is_random', 'd_sum', 'd_sqr_sum', 'n'])

class StatLRSceduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False, jitter=1, jitter_pvalue_thr=0.05, use_detailed_stat=True):
        self.jitter = jitter
        self.jitter_pvalue_thr = jitter_pvalue_thr
        self.use_detailed_stat = use_detailed_stat
        self.deltas = []  # list of LossRec by iterations 
        """
        As train loop looks like:
            loss = loss(model.params)  # loss=model(imput)
            model.params = params(opt.lr)  # opt.step
            self.base_lr, self.is_hi = ...(loss)
            opt.lr = opt.lr(self.base_lr, self.is_hi) # scheduler_step
        loss passed to self.step() reflects self.base_lr and self.is_hi on 2 iters before.
        So we need a quasi-queue to store state for 2 iters.   
        """
        self.next_is_hi = False  # first iteration does not produse record in self.deltas. Just do it with smaller LR
        self.prev_is_hi = None
        self.next_is_hi_is_random = False
        self.prev_is_hi_is_random = None
        self.prev_loss = None
        self.evaluation_start_index = 0  # index at deltas and delta_means when statistics started
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        k = 1 + self.jitter
        if not self.next_is_hi:
            k = 1 / k
        return [lr * k for lr in self.base_lrs]

    def process_new_loss(self, loss):
        if loss is not None:
            loss = loss.detach().type(torch.DoubleTensor)
            if loss.isnan().any():
                print('loss is nan')        
        if self.prev_loss is not None:
            delta = self.prev_loss - loss
            delta_sqr = delta * delta
            self.deltas.append( LossRec(step=self._step_count -1,  # for debugging
                                        is_hi=self.prev_is_hi,
                                        is_hi_is_random=self.prev_is_hi_is_random,  # for debugging
                                        d_sum=delta.sum().item(),
                                        d_sqr_sum=delta_sqr.sum().item(),
                                        n=len(delta)))
        self.prev_loss = loss
        
    def _update_is_hi(self):
        self.next_is_hi_is_random = not self.next_is_hi_is_random
        if self.next_is_hi_is_random:
            self.next_is_hi = random.random() > 0.5
        else:
            self.next_is_hi = not self.next_is_hi
            
        #self.base_lrs = [lr+0.0001 for lr in self.base_lrs]  #GVNC

    def _eval_jitter(self, n, s, s2):
        d0, d1 = (s[i]/n[i] for i in (0,1))
        s0, s1 = ( (s2[i] - s[i]*s[i] / n[i]) / (n[i] - 1) / n[i] for i in (0,1) )   # s^2/n 
        t = (d1 - d0) / math.sqrt(s1 + s0)
        nu = int( (s1 + s0)**2 / (  s1**2/(n[1]-1) + s0**2/(n[0]-1) ) )
        p_value = stats.t.sf(abs(t), nu)  # p-value for any side
        if p_value > self.jitter_pvalue_thr:
            return 0, p_value
        else:
            return 1 if t > 0 else -1, p_value

    
    def _evaluate_lr_change(self, detailed=None, update=True):
        if detailed is None:
            detailed = self.use_detailed_stat
        n = [0,0]
        s = [0,0]
        s2 = [0,0]
        jitter_tested = False
        stop_tested = False
        for i, loss_rec in enumerate(self.deltas[: : -1]):
            s[loss_rec.is_hi] += loss_rec.d_sum
            if detailed:
                n[loss_rec.is_hi] += loss_rec.n
                s2[loss_rec.is_hi] += loss_rec.d_sqr_sum               
            else:
                n[loss_rec.is_hi] += 1
                s2[loss_rec.is_hi] += loss_rec.d_sum*loss_rec.d_sum

            if len(self.deltas) - i - 1 <= self.evaluation_start_index:
                jitter_tested = True
            
            stop_tested = True  # TODO
            
            if jitter_tested and stop_tested:
                break
                
        if jitter_tested and n[0] == n[1] and n[0] >= 2 and n[1] >= 2:
            decision, p_value = self._eval_jitter(n, s, s2)
            print(f'detailed: {detailed}, n: {n[0]}, p_value: {p_value}, decision: {decision}')
            if update and decision:
                k = 1 + self.jitter
                if decision < 0:
                    k = 1 / k
                self.base_lrs =  [lr * k for lr in self.base_lrs]
                self.evaluation_start_index = len(self.deltas) + 2
                
                print(f'Updates LR -> {self.base_lrs}')
        
        

    def step(self, epoch=None, loss=None):
        if loss is None:
            assert self.last_epoch == -1, 'incorrect call to StatLRSceduler: missed step(loss=) argument'  # except initial_step
        curr_is_hi = self.next_is_hi
        curr_is_hi_is_random = self.next_is_hi_is_random
        self.process_new_loss(loss)
        self._evaluate_lr_change()
        self._update_is_hi()
        super().step(epoch)
        self.prev_is_hi = curr_is_hi
        self.prev_is_hi_is_random = curr_is_hi_is_random
        
