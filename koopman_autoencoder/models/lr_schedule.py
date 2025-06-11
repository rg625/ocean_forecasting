# models/lr_schedule.py
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmup(_LRScheduler):
    """
    Implements a learning rate schedule with a linear warmup phase followed by
    a cosine decay phase.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup: int,
        decay: int,
        final_lr: float,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: The optimizer.
            warmup: Number of warmup steps.
            decay: Total number of steps for cosine decay (must be > warmup).
            final_lr: The final learning rate after decay.
            last_epoch: The index of the last epoch.
        """
        if warmup < 0:
            raise ValueError("Warmup steps must be non-negative.")
        if decay <= warmup:
            raise ValueError("Decay steps must be greater than warmup steps.")

        self.warmup = warmup
        self.decay = decay
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if step < self.warmup:
            # Linear warmup
            warmup_factor = step / self.warmup if self.warmup > 0 else 1.0
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        if step <= self.decay:
            # Cosine decay
            progress = (step - self.warmup) / (self.decay - self.warmup)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.final_lr + (base_lr - self.final_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]

        # After decay, stick to the final learning rate
        return [self.final_lr for _ in self.base_lrs]
