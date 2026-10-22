"""Minimal BertAdam compatibility layer for the frozen MOGB official clone.

The scheduler/warmup behavior is intentionally only an approximation of the
historical ``pytorch-pretrained-bert`` optimizer and is not bit-identical.
"""

from __future__ import annotations

from typing import Iterable

import torch


def _warmup_linear(progress: float, warmup: float) -> float:
    if warmup > 0 and progress < warmup:
        return progress / warmup
    return max(1.0 - progress, 0.0)


class BertAdam(torch.optim.AdamW):
    """Approximate the model_pipeline BertAdam API on top of modern ``torch``."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        warmup: float = -1.0,
        t_total: int = -1,
        schedule: str = "warmup_linear",
        b1: float = 0.9,
        b2: float = 0.999,
        e: float = 1e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        if schedule != "warmup_linear":
            raise ValueError(f"Unsupported model_pipeline BertAdam schedule: {schedule}")
        super().__init__(
            params=params,
            lr=lr,
            betas=(b1, b2),
            eps=e,
            weight_decay=weight_decay,
        )
        self.warmup = warmup
        self.t_total = t_total
        self.schedule = schedule
        self.max_grad_norm = max_grad_norm
        self._step_count_compat = 0
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    def _scheduled_lr(self, group: dict) -> float:
        base_lr = group["initial_lr"]
        if self.t_total is None or self.t_total <= 0:
            return base_lr
        progress = self._step_count_compat / float(self.t_total)
        return base_lr * _warmup_linear(progress, self.warmup)

    def get_lr(self):
        return [self._scheduled_lr(group) for group in self.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        if self.max_grad_norm and self.max_grad_norm > 0:
            params = [
                param
                for group in self.param_groups
                for param in group["params"]
                if param.grad is not None
            ]
            if params:
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        for group in self.param_groups:
            group["lr"] = self._scheduled_lr(group)

        loss = super().step(closure=closure)
        self._step_count_compat += 1
        return loss


__all__ = ["BertAdam"]
