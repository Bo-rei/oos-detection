"""Minimal ``utils.util`` bridge for the frozen MOGB official clone."""

from __future__ import annotations

import importlib


class _NoOpSummaryWriter:
    """No-op stand-in for the historical tensorboard writer."""

    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


summary_writer = _NoOpSummaryWriter()


def _model_pipeline_util_module():
    try:
        return importlib.import_module("util")
    except ModuleNotFoundError:
        return None


def __getattr__(name):
    if name == "summary_writer":
        return summary_writer
    model_pipeline_util = _model_pipeline_util_module()
    if model_pipeline_util is not None and hasattr(model_pipeline_util, name):
        return getattr(model_pipeline_util, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["summary_writer"]
