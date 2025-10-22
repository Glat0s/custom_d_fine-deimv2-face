import math
from functools import partial
from torch.optim.optimizer import Optimizer

def flat_cosine_schedule_with_warmup(
    current_iter: int,
    warmup_iter: int,
    flat_iter: int,
    total_iter: int,
    no_aug_iter: int,
    base_lr: float,
    min_lr: float,
):
    """
    Computes learning rate with linear warmup, flat phase, and cosine decay.
    """
    if current_iter < warmup_iter:
        # Linear warmup
        return base_lr * (current_iter + 1) / warmup_iter
    elif warmup_iter <= current_iter < flat_iter:
        # Flat phase
        return base_lr
    else:
        # Cosine decay phase
        # The decay phase starts after flat_iter and ends at total_iter - no_aug_iter
        decay_total_steps = total_iter - flat_iter - no_aug_iter
        if decay_total_steps <= 0: # Handle cases where flat phase is long
            return base_lr

        current_decay_step = current_iter - flat_iter
        cosine_decay = 0.5 * (1 + math.cos(math.pi * current_decay_step / decay_total_steps))
        return min_lr + (base_lr - min_lr) * cosine_decay

class FlatCosineLRScheduler:
    """
    Learning rate scheduler with linear warm-up, an optional flat phase, and cosine decay.
    This implementation more closely matches common practices.
    """
    def __init__(self, optimizer: Optimizer, lr_gamma: float, iter_per_epoch: int, total_epochs: int,
                 warmup_iter: int, flat_epochs: int, no_aug_epochs: int):

        self.optimizer = optimizer
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]

        total_iter = int(iter_per_epoch * total_epochs)
        flat_iter = int(iter_per_epoch * flat_epochs)
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)
        
        # Ensure flat_iter is not less than warmup_iter
        if flat_iter < warmup_iter:
             flat_iter = warmup_iter

        self.lr_func = partial(
            flat_cosine_schedule_with_warmup,
            warmup_iter=warmup_iter,
            flat_iter=flat_iter,
            total_iter=total_iter,
            no_aug_iter=no_aug_iter,
        )
        print(f"Scheduler Config: total_iter={total_iter}, warmup_iter={warmup_iter}, flat_iter={flat_iter}, no_aug_iter={no_aug_iter}")

    def step(self, current_iter: int, optimizer: Optimizer):
        """
        Updates the learning rate of the optimizer at the current iteration.
        """
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = self.lr_func(
                current_iter=current_iter,
                base_lr=self.base_lrs[i],
                min_lr=self.min_lrs[i]
            )
        return optimizer
