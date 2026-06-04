"""Optimizers and learning-rate schedulers.

Contains:
  - RAdam optimizer (Rectified Adam)
  - get_cosine_schedule_with_warmup – batch-level cosine LR with linear warmup
"""

import math
from typing import Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    ReduceLROnPlateau,
)


# ------------------------------------------------------------------ RAdam --


class RAdam(optim.Optimizer):
    """Rectified Adam optimizer.

    Reference: "On the Variance of the Adaptive Learning Rate and Beyond"
    (Liu et al., 2019).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for _ in range(10)]
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                state["step"] += 1

                buffered = self.buffer[int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = step_size

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * group["lr"])

                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group["lr"])

                p.data.copy_(p_data_fp32)

        return loss


# -------------------------------------------------- cosine warmup scheduler --


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine LR schedule with linear warmup (batch-level stepping).

    During warmup the LR rises linearly from 0 → base LR.
    After warmup it follows a cosine curve down to min_lr_ratio * base_lr.

    Args:
        optimizer:           Optimizer whose LR will be scheduled.
        num_warmup_steps:    Number of warmup steps.
        num_training_steps:  Total training steps.
        num_cycles:          Number of cosine cycles (default 0.5 = half cosine).
        last_epoch:          The index of the last epoch (for resuming).
        min_lr_ratio:        Minimum LR as a fraction of base LR (default 0.0).

    Returns:
        A ``LambdaLR`` scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_factor = max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
        return max(min_lr_ratio, cosine_factor)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# -------------------------------------------------- scheduler factory ------


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Build the LR scheduler specified in *cfg*.

    Args:
        optimizer:        The optimizer.
        cfg:              Config dict with keys ``scheduler_type``, ``epochs``,
                          ``warmup_epochs``, ``patience``, ``min_lr``.
        steps_per_epoch:  Number of optimizer steps per epoch (len(train_loader)).

    Returns:
        ``(scheduler, scheduler_batch_step)`` where *scheduler_batch_step* is
        ``True`` if the scheduler must be stepped every batch (not every epoch).
    """
    scheduler = None
    scheduler_batch_step = False

    if not cfg.get("use_lr_scheduler", True):
        return scheduler, scheduler_batch_step

    stype = cfg.get("scheduler_type", "cosine_warmup")

    if stype == "cosine_warmup":
        total_steps = steps_per_epoch * cfg["epochs"]
        warmup_steps = steps_per_epoch * cfg["warmup_epochs"]
        min_lr_ratio = cfg.get("min_lr", 1e-7) / max(cfg["learning_rate"], 1e-10)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, min_lr_ratio=min_lr_ratio
        )
        scheduler_batch_step = True

    elif stype == "cosine_restarts":
        # CosineAnnealingWarmRestarts: periodically resets LR to help escape local minima
        t0 = steps_per_epoch * cfg.get("restart_period_epochs", 25)
        t_mult = int(cfg.get("restart_t_mult", 2))
        eta_min = cfg.get("min_lr", 1e-7)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, t0), T_mult=t_mult, eta_min=eta_min
        )
        scheduler_batch_step = True

    elif stype == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["patience"], eta_min=cfg["min_lr"])

    elif stype == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=cfg["min_lr"]
        )

    return scheduler, scheduler_batch_step
