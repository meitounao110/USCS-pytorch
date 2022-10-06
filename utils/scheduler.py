from torch.optim.lr_scheduler import _LRScheduler, StepLR


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-7):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


class WarmUpPolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, warmup_steps, power=0.9, last_epoch=-1, min_lr=1e-7):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        super(WarmUpPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [max(base_lr * (self.last_epoch / self.warmup_steps), self.min_lr)
                    for base_lr in self.base_lrs]
        else:
            return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                    for base_lr in self.base_lrs]
