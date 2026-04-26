from typing import Dict, List, Optional

import torch
from torch import nn


class Analyzer:
    """Tracks per-parameter sum of squared gradients via backward hooks on parameters."""

    def __init__(self, config, model: nn.Module):
        self.config = config
        self.model = model
        self.grad_sq_sum: Dict[str, float] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_param_hooks()

    def _register_param_hooks(self) -> None:
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            def make_hook(pname: str):
                def hook(grad: Optional[torch.Tensor]) -> None:
                    if grad is None:
                        return
                    # self.grad_sq_sum[pname] = grad.detach().pow(2).sum().item()
                    p.grad_sq_sum = grad.detach().pow(2).sum().item()

                return hook

            h = p.register_hook(make_hook(name))
            self._hook_handles.append(h)
            break
        
        def clipped_relu_hook(m,i,o):
            return i

        # for modules in self.model.modules():
        #     modules.register_full_backward_hook(clipped_relu_hook)

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # def clear_grad_sq_sum(self) -> None:
    #     self.grad_sq_sum.clear()

    # def analyze(self) -> Dict[str, float]:
    #     return dict(self.grad_sq_sum)
