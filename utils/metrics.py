from collections import defaultdict
import torch

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.value_sums = defaultdict(float)
        self.value_counts = defaultdict(int)
        self.delimiter = delimiter
        
    def reset(self):
        self.value_sums.clear()
        self.value_counts.clear()

    def update(self, mode, **kwargs):
        for k, v in kwargs.items():
            k = f"{mode}/{k}"
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.value_sums[k] += v
            self.value_counts[k] += 1
            
    def get_value(self, name):
        return self.value_sums[name] / self.value_counts[name] if self.value_counts[name] > 0 else 0.0
    
    def get_metrics(self):
        return {name: self.get_value(name) for name in sorted(self.value_sums.keys())}
    
    def to_writer(self, writer, global_step):
        for name in sorted(self.value_sums.keys()):
            writer.add_scalar(name, self.get_value(name), global_step)

    def __str__(self):
        loss_str = []
        for name in sorted(self.value_sums.keys()):
            loss_str.append(
                f"{name}: {self.get_value(name):.4f} ({self.value_counts[name]})"
            )
        return self.delimiter.join(loss_str)

