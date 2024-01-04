from collections import namedtuple
from copy import copy
import math
from pathlib import Path
import random
from scipy import stats
import torch

LossRec = namedtuple('LossRec', ['step', 'd_sum_lo', 'd_sqr_sum_lo', 'd_sum_hi', 'd_sqr_sum_hi', 'n'])

class StatLRSceduler:

    def __init__(self, opt_lo, opt_hi, last_epoch=-1, verbose=False,
                 jitter=1, jitter_pvalue_thr=0.05, step_lr_scale=0.1, use_detailed_stat=False,
                 log_file=None):
        self.opt_lo = opt_lo
        self.opt_hi = opt_hi
        self.jitter = jitter
        self.jitter_pvalue_thr = jitter_pvalue_thr
        self.step_lr_scale = step_lr_scale
        self.use_detailed_stat = use_detailed_stat
        self.log_file = log_file
        self.deltas = []  # list of LossRec by iterations 
        self.step_no = -1
        self.evaluation_start_index = 0  # index at deltas and delta_means when statistics started
        self.base_lrs = []
        for i, pg in enumerate(self.opt_lo.param_groups):
            self.base_lrs.append(pg['lr']) 
        self._update_opt_lr()
        if self.log_file:
            with Path(self.log_file).open('w') as f:
                f.write('self.step_no\tlast_rec.d_sum_lo\tlast_rec.d_sqr_sum_lo\tlast_rec.d_sum_hi\tlast_rec.d_sqr_sum_hi\tlast_rec.n\t' +
                    'self.evaluation_start_index\tn\ts[0]\ts[1]\ts2[0]\ts2[1]\t'+
                    'decision\tp_value\td0\td1\tmath.sqrt(s0)\tmath.sqrt(s1)\n')                

    def _get_lr(self, is_hi):
        k = 1 + self.jitter
        if not is_hi:
            k = 1 / k
        return [lr * k for lr in self.base_lrs]

    def _process_new_losses(self, prev_loss, loss_lo, loss_hi):
        for is_hi, loss in enumerate((loss_lo, loss_hi,)):
            assert  loss is not None
            if loss.isnan().any():
                print('loss is nan')        
        if prev_loss is not None:
            delta_lo = prev_loss - loss_lo.detach().type(torch.DoubleTensor)
            delta_sqr_lo = delta_lo * delta_lo
            delta_hi = prev_loss - loss_hi.detach().type(torch.DoubleTensor)
            delta_sqr_hi = delta_hi * delta_hi
            self.deltas.append( LossRec(step=self.step_no,  # for debugging
                                        d_sum_lo=delta_lo.sum().item(),
                                        d_sqr_sum_lo=delta_sqr_lo.sum().item(),
                                        d_sum_hi=delta_hi.sum().item(),
                                        d_sqr_sum_hi=delta_sqr_hi.sum().item(),
                                        n=len(delta_lo)))

    def _log(self, n, s, s2, decision = None, p_value=None, d0=None, d1=None, s0=None, s1=None):
        if not self.log_file or not self.deltas:
            return
        with Path(self.log_file).open('a') as f:
            last_rec = self.deltas[-1]
            if s0 is not None:
                s0  =math.sqrt(s0)
            if s1 is not None:
                s1  =math.sqrt(s1)
            f.write(f'{self.step_no}\t{last_rec.d_sum_lo}\t{last_rec.d_sqr_sum_lo}\t{last_rec.d_sum_hi}\t{last_rec.d_sqr_sum_hi}\t{last_rec.n}\t' +
                    f'{self.evaluation_start_index}\t{n}\t{s[0]}\t{s[1]}\t{s2[0]}\t{s2[1]}\t' +
                    f'{decision}\t{p_value}\t{d0}\t{d1}\t{s0}\t{s1}\n')

    def _evaluate_lr_change(self, detailed=None, update=True):
        if detailed is None:
            detailed = self.use_detailed_stat
        n = 0
        s = [0,0]
        s2 = [0,0]
        jitter_tested = False
        stop_tested = False
        decision, p_value, d0, d1, s0, s1 = None,None,None,None,None,None
        for i, loss_rec in enumerate(self.deltas[: : -1]):
            s[0] += loss_rec.d_sum_lo
            s[1] += loss_rec.d_sum_hi
            if detailed:
                n += loss_rec.n
                s2[0] += loss_rec.d_sqr_sum_lo               
                s2[1] += loss_rec.d_sqr_sum_hi               
            else:
                n += 1
                s2[0] += loss_rec.d_sum_lo*loss_rec.d_sum_lo
                s2[1] += loss_rec.d_sum_hi*loss_rec.d_sum_hi

            if len(self.deltas) - i - 1 <= self.evaluation_start_index:
                jitter_tested = True
            else:
                if not jitter_tested and n >= 2:
                    decision, p_value, d0, d1, s0, s1 = self._eval_jitter_xi(n, s, s2)
                    #print(f'detailed: {detailed}, n: {n}, p_value: {p_value}, decision: {decision}')
                    if update and decision:
                        k = 1 + self.step_lr_scale
                        if decision < 0:
                            k = 1 / k
                        self.base_lrs =  [lr * k for lr in self.base_lrs]
                        self.evaluation_start_index = len(self.deltas) + 2
                        print(f'Updates LR -> {self.base_lrs} i={i}')
                        jitter_tested = True

            stop_tested = True  # TODO
            
            if jitter_tested and stop_tested:
                break
                
        self._log(n, s, s2, decision, p_value, d0, d1, s0, s1)
        return decision

    def _eval_jitter_xi(self, n, s, s2):
        d0, d1 = (s[i]/n for i in (0,1))
        s0, s1 = ( (s2[i] - s[i]*s[i] / n) / (n - 1) / n for i in (0,1) )   # s^2/n 
        t = (d1 - d0) / math.sqrt(s1 + s0)
        nu = int( (s1 + s0)**2 / (  s1**2/(n-1) + s0**2/(n-1) ) )
        p_value = stats.t.sf(abs(t), nu)  # p-value for any side
        if p_value > self.jitter_pvalue_thr:
            return 0, p_value, d0, d1, s0, s1
        else:
            return 1 if t > 0 else -1, p_value, d0, d1, s0, s1
        
    def _eval_jitter(self, n, s, s2):
        d0, d1 = (s[i]/n for i in (0,1))
        s0, s1 = ( (s2[i] - s[i]*s[i] / n) / (n - 1) / n for i in (0,1) )   # s^2/n 
        t = (d1 - d0) / math.sqrt(s1 + s0)
        p_value = abs(t)  # GVNC
        if p_value < 2:
            return 0, p_value, d0, d1, s0, s1
        else:
            return 1 if t > 0 else -1, p_value, d0, d1, s0, s1
        

    def _update_opt_lr(self):
        for is_hi, opt in enumerate((self.opt_lo, self.opt_hi,)):
            lrs = self._get_lr(is_hi)
            for i, pg in enumerate(opt.param_groups):
                pg['lr'] = lrs[i]

    def step(self, prev_loss, loss_lo, loss_hi):
        self.step_no += 1
        self._process_new_losses(prev_loss.detach().type(torch.DoubleTensor), loss_lo, loss_hi)  # fill deltas
        decision = self._evaluate_lr_change()    # eval stat
        if decision: 
            self._update_opt_lr()
        
        use_hi = random.random() > 0.5
        return use_hi
        
