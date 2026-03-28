import copy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FindLrScheduler:
    """LR Finder: save state -> run exponential LR search -> find optimal -> restore state -> set LR.
    Batches are cached on first step() to avoid creating new iterators each epoch (prevents thread accumulation)."""

    def __init__(self, optimizer, model, loss_fn, train_loader, start_lr, end_lr, num_steps, log_dir=None):
        self.optimizer = optimizer
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_steps = num_steps
        self.log_dir = Path(log_dir) if log_dir else None
        self._batches = None
        self._step_count = 0

    def step(self):
        # 1. Save state
        model_state = copy.deepcopy(self.model.state_dict())
        opt_state = copy.deepcopy(self.optimizer.state_dict())

        # 2. Cache batches on first run (single iterator creation, no accumulation)
        if self._batches is None:
            data_iter = iter(self.train_loader)
            self._batches = []
            for _ in range(self.num_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                self._batches.append(copy.deepcopy(batch))

        # 3. Run LR search over cached batches
        lr_history = []
        loss_history = []
        update_before = getattr(self.optimizer, 'update_before_backward', lambda: None)
        update_after = getattr(self.optimizer, 'update_after_backward', lambda: None)

        for i in range(self.num_steps):
            lr = self.start_lr * (self.end_lr / self.start_lr) ** (i / max(1, self.num_steps - 1))
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            lr_history.append(lr)

            ims, labs = self._batches[i]

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(ims)
            loss = self.loss_fn(out, labs)
            update_before()
            loss.backward()
            update_after()
            self.optimizer.step()

            loss_history.append(loss.item())

        # 4. Select optimal LR
        n = len(loss_history)
        start_loss = sum(loss_history[:max(1, n // 10)]) / max(1, n // 10)
        max_idx = 0
        for i in range(n):
            if loss_history[i] <= start_loss:
                max_idx = i

        best_lr = 0.3 * lr_history[max_idx]

        # 5. Restore state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(opt_state)

        # 6. Set chosen LR
        for pg in self.optimizer.param_groups:
            pg['lr'] = best_lr
        print(f"FindLr: selected lr={best_lr:.2e} (max_idx={max_idx})")

        # 7. Plot loss_history
        if self.log_dir:
            fig, ax = plt.subplots()
            ax.plot(lr_history, loss_history, 'b-')
            ax.set_xscale('log')
            ax.set_xlabel('LR')
            ax.set_ylabel('Loss')
            ax.axvline(x=best_lr, color='r', linestyle='--', label=f'selected={best_lr:.2e}')
            ax.legend()
            fig.savefig(self.log_dir / f'find_lr_ep{self._step_count}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        self._step_count += 1
