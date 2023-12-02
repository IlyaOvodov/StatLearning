from collections import namedtuple
from copy import copy
import math
from pathlib import Path
import random
from scipy import stats
import torch

LossRec = namedtuple('LossRec', ['step', 'is_hi', 'is_hi_is_random', 'd_sum', 'd_sqr_sum', 'n'])

class StatLRSceduler:

    def __init__(self, optimizer, last_epoch=-1, verbose=False,
                 jitter=1, jitter_pvalue_thr=0.05, step_lr_scale=0.1, use_detailed_stat=False,
                 log_file=None):
        self.jitter = jitter
        self.jitter_pvalue_thr = jitter_pvalue_thr
        self.step_lr_scale = step_lr_scale
        self.use_detailed_stat = use_detailed_stat
        self.log_file = log_file
        self.deltas = []  # list of LossRec by iterations 
        """
        train loop looks like:
            loss = loss(model.params)                 # loss=model(imput)
            self.base_lr, self.is_hi = ...(loss)      #scheduler_step
            opt.lr = opt.lr(self.base_lr, self.is_hi) # scheduler_step
            model.params = params(opt.lr)             # opt.step
        """
        self.is_hi = False  # first iteration does not produse record in self.deltas. Just do it with smaller LR
        self.is_hi_is_random = False
        self.step_no = -1
        self.prev_loss = None
        self.prev_loss_sum = None  # for debug
        self.evaluation_start_index = 0  # index at deltas and delta_means when statistics started
        self.optimizer = optimizer
        self.base_lrs = []
        for i, pg in enumerate(self.optimizer.param_groups):
            self.base_lrs.append(pg['lr']) 
        if self.log_file:
            with Path(self.log_file).open('w') as f:
                f.write('self.step_no\tlast_rec.is_hi\tlast_rec.is_hi_is_random\tlast_rec.d_sum\tlast_rec.d_sqr_sum\tlast_rec.n\t' +
                    'self.evaluation_start_index\tn[0]\tn[1]\ts[0]\ts[1]\ts2[0]\ts2[1]\t'+
                    'decision\tp_value\td0\td1\tmath.sqrt(s0)\tmath.sqrt(s1)\n')                

    def get_lr(self):
        k = 1 + self.jitter
        if not self.is_hi:
            k = 1 / k
        return [lr * k for lr in self.base_lrs]

    def _process_new_loss(self, loss):
        assert  loss is not None
        if loss is not None:
            loss = loss.detach().type(torch.DoubleTensor)
            if loss.isnan().any():
                print('loss is nan')        
        if self.prev_loss is not None:
            delta = self.prev_loss - loss  # mmore is better
            delta_sqr = delta * delta
            self.deltas.append( LossRec(step=self.step_no,  # for debugging
                                        is_hi=self.is_hi,
                                        is_hi_is_random=self.is_hi_is_random,  # for debugging
                                        d_sum=delta.sum().item(),
                                        d_sqr_sum=delta_sqr.sum().item(),
                                        n=len(delta)))
        self.prev_loss = loss
        self.prev_loss_sum = loss.sum().item()

    def _log(self, n, s, s2, decision, p_value, d0, d1, s0, s1):
        if not self.log_file:
            return
        with Path(self.log_file).open('a') as f:
            last_rec = self.deltas[-1]
            f.write(f'{self.step_no}\t{last_rec.is_hi}\t{last_rec.is_hi_is_random}\t{last_rec.d_sum}\t{last_rec.d_sqr_sum}\t{last_rec.n}\t' +
                    f'{self.evaluation_start_index}\t{n[0]}\t{n[1]}\t{s[0]}\t{s[1]}\t{s2[0]}\t{s2[1]}\t' +
                    f'{decision}\t{p_value}\t{d0}\t{d1}\t{math.sqrt(s0)}\t{math.sqrt(s1)}\n')

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
            decision, p_value, d0, d1, s0, s1 = self._eval_jitter(n, s, s2)
            self._log(n, s, s2, decision, p_value, d0, d1, s0, s1)
            print(f'detailed: {detailed}, n: {n[0]}, p_value: {p_value}, decision: {decision}')
            if update and decision:
                k = 1 + self.step_lr_scale
                if decision < 0:
                    k = 1 / k
                self.base_lrs =  [lr * k for lr in self.base_lrs]
                self.evaluation_start_index = len(self.deltas) + 2
                
                print(f'Updates LR -> {self.base_lrs}')
        
    def _eval_jitter(self, n, s, s2):
        d0, d1 = (s[i]/n[i] for i in (0,1))
        s0, s1 = ( (s2[i] - s[i]*s[i] / n[i]) / (n[i] - 1) / n[i] for i in (0,1) )   # s^2/n 
        t = (d1 - d0) / math.sqrt(s1 + s0)
        nu = int( (s1 + s0)**2 / (  s1**2/(n[1]-1) + s0**2/(n[0]-1) ) )
        p_value = stats.t.sf(abs(t), nu)  # p-value for any side
        if p_value > self.jitter_pvalue_thr:
            return 0, p_value, d0, d1, s0, s1
        else:
            return 1 if t > 0 else -1, p_value, d0, d1, s0, s1

    def _update_is_hi(self):
        self.is_hi_is_random = not self.is_hi_is_random
        if self.is_hi_is_random:
            self.is_hi = random.random() > 0.5
        else:
            self.is_hi = not self.is_hi
            
        #self.base_lrs = [lr*1.001 for lr in self.base_lrs]  #GVNC

    def _update_opt_lr(self):
        lrs = self.get_lr()
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = lrs[i]

    def step(self, loss=None):
        self.step_no += 1
        self._process_new_loss(loss)  # fill deltas
        self._evaluate_lr_change()    # eval stat 
        self._update_is_hi()
        self._update_opt_lr()          
        
