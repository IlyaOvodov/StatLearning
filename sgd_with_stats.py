from typing import cast, List, Optional, Union, Dict, Any
import torch
from torch import Tensor
from torch.optim import SGD
from torch.optim.optimizer import _use_grad_for_differentiable

def zer_state_tensor(state: Dict, name: str):
    if name in state:
        state[name].zero_()

class SGDWithStats(SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        
    @_use_grad_for_differentiable
    def update_before_backward(self):
        for group in self.param_groups:
            for p in group["params"]:
                self._param_update_before_backward(p)
                
    @_use_grad_for_differentiable
    def update_after_backward(self):
        for group in self.param_groups:
            for p in group["params"]:
                self._param_update_after_backward(p)
            
    def _param_update_before_backward(self, p):
        state = self.state[p]
        if p.grad is None:
            zer_state_tensor(state, "prev_grad")
            zer_state_tensor(state, "grad_sqr")
            zer_state_tensor(state, "num_updates")
        else:
            if state.get("prev_grad") is None:
                state["prev_grad"] = p.grad.clone()
            else:
                state["prev_grad"][:] = p.grad

    def _param_update_after_backward(self, p):
        state = self.state[p]
        new_grad = p.grad - state.get("prev_grad", 0)
        if state.get("grad_sqr") is None:
            state["grad_sqr"] = torch.square(new_grad)
        else:
            state["grad_sqr"].add_(torch.square(new_grad))
        state["num_updates"] = state.get("num_updates", 0) + torch.ones([], dtype=torch.int16)

    def t_value(self, p):
        state = self.state[p]
        num_updates = state["num_updates"]
        grad_sqr = state["grad_sqr"]
        s2 = (grad_sqr - torch.square(p.grad)/num_updates)/(num_updates-1+1e-6)
        s = torch.sqrt(s2)
        t_value = p.grad.abs()/s*torch.sqrt(num_updates)
        return t_value
        
class SGDWithStatsFixed(SGDWithStats):
    def __init__(self, *args, lr_grow = 0.01, lr_shrink = 0.02, min_step=0.000001, max_step=0.1, selection_method = 'meangrad', **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_grow = lr_grow
        self.lr_shrink = lr_shrink
        self.max_step = max_step
        self.min_step = min_step
        self.selection_method = selection_method

    @_use_grad_for_differentiable
    def step(self):
        loss = None
        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )
            assert not has_sparse_grad, "Sparse gradients are not supported"

            self._single_tensor_sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss


    def _single_tensor_sgd(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
    ):
        for i, param in enumerate(params):
            state = self.state[param]
            grad = grads[i] if not maximize else -grads[i]
            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(grad).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf
            USE_MEAN_GRAD = self.selection_method == 'meangrad'
            USE_MEAN_GRAD2 = self.selection_method == 'meangrad2'
            USE_T_VALUE = self.selection_method == 'tvalue'
            grad_sign = torch.sign(grad).to(torch.int)
            prev_sign = state.get("prev_sign", torch.zeros_like(grad_sign))
            lr_scale = state.get("lr_scale", torch.zeros_like(grad) + 1)
            if USE_MEAN_GRAD:
                mean_grad = grad.abs().mean()
                grad_sign = (grad_sign*(grad.abs() > mean_grad)).to(torch.int)
                validity_mask = (grad_sign!=0) & (prev_sign!=0)
            elif USE_MEAN_GRAD2:
                mean_grad = grad.abs().mean()
                validity_mask = (grad.abs() > mean_grad)
            else:
                validity_mask = 1
            mask = validity_mask & (grad_sign == prev_sign)
            lr_scale.mul_(1 + mask*self.lr_grow).clip_(max=self.max_step/lr)
            step = -grad_sign*lr*lr_scale
            mask = validity_mask & (grad_sign != prev_sign)
            lr_scale.div_(1+mask*self.lr_shrink).clip_(min=self.min_step/lr)
            param.add_(step)
            if weight_decay != 0:
                param.add_(param, alpha=-lr*weight_decay)
            if USE_MEAN_GRAD:
                state["prev_sign"] = grad_sign
            else:
                state["prev_sign"] = grad_sign*validity_mask + prev_sign*~validity_mask
            state["lr_scale"] = lr_scale
