import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        decay_steps: int,
        final_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        elif step <= self.decay_steps:
            cosine_decay = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (step - self.warmup_steps)
                    / (self.decay_steps - self.warmup_steps)
                )
            )
            return [
                self.final_lr + (base_lr - self.final_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
        else:
            return [self.final_lr for _ in self.base_lrs]
