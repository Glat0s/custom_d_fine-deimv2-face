import math
from functools import partial


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales

def flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, current_iter, init_lr, min_lr):
    """
    Computes the learning rate using a warm-up, flat, and cosine decay schedule.
    """
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr
    elif current_iter >= total_iter - no_aug_iter:
        return min_lr
    else:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - flat_iter) /
                                           (total_iter - flat_iter - no_aug_iter)))
        return min_lr + (init_lr - min_lr) * cosine_decay


class FlatCosineLRScheduler:
    """
    Learning rate scheduler with warm-up, optional flat phase, and cosine decay.
    """
    def __init__(self, optimizer, lr_gamma, iter_per_epoch, total_epochs,
                 warmup_iter, flat_epochs, no_aug_epochs):
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]

        total_iter = int(iter_per_epoch * total_epochs)
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)
        flat_iter = int(iter_per_epoch * flat_epochs)

        self.lr_func = partial(flat_cosine_schedule, total_iter, warmup_iter, flat_iter, no_aug_iter)

    def step(self, current_iter, optimizer):
        """
        Updates the learning rate of the optimizer at the current iteration.
        """
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = self.lr_func(current_iter, self.base_lrs[i], self.min_lrs[i])
        return optimizer
