import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup: int,
        decay: int,
        final_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup = warmup
        self.decay = decay
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if step < self.warmup:
            return [base_lr * step / self.warmup for base_lr in self.base_lrs]
        elif step <= self.decay:
            cosine_decay = 0.5 * (
                1
                + math.cos(math.pi * (step - self.warmup) / (self.decay - self.warmup))
            )
            return [
                self.final_lr + (base_lr - self.final_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
        else:
            return [self.final_lr for _ in self.base_lrs]
