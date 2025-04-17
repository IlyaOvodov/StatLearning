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
        