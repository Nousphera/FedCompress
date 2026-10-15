"""Explicit experiment settings shared by the launcher and Flower Apps."""

import argparse
import math
from pathlib import Path

import torch

from .models import CLASSES

METHODS = ("fedcompress", "client", "fedavg")


def defaults():
    return {
        "method": "fedcompress",
        "dataset": "cifar10",
        "nr-clients": 20,
        "rounds": 20,
        "participation": 1.0,
        "local-epochs": 10,
        "warmup-epochs": 5,
        "server-epochs": 10,
        "batch-size": 128,
        "server-batch-size": 256,
        "learning-rate": 0.001,
        "cluster-learning-rate": 0.0005,
        "server-learning-rate": 0.0001,
        "temperature": 2.0,
        "min-clusters": 8,
        "max-clusters": 30,
        "cluster-step": 1,
        "window": 3,
        "patience": 3,
        "score-tolerance": 0.01,
        "quantity-skew": 0.25,
        "validation-fraction": 0.1,
        "seed": 42,
        "torch-threads": 1,
        "ood-limit": 50000,
        "device": "cpu",
        "data-root": "data",
        "model-root": "checkpoints",
        "output": "results/run.json",
        "registration-timeout": 120.0,
        "reply-timeout": 86400.0,
    }


def validate(cfg):
    cfg = dict(cfg)
    if cfg["method"] not in METHODS or cfg["dataset"] not in CLASSES:
        raise ValueError("Unknown method or dataset")
    positive = (
        "nr-clients",
        "rounds",
        "local-epochs",
        "batch-size",
        "server-batch-size",
        "min-clusters",
        "max-clusters",
        "cluster-step",
        "window",
        "patience",
        "torch-threads",
    )
    for key in positive:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("warmup-epochs", "server-epochs", "ood-limit", "seed"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key, value in cfg.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if (
        not 0 < cfg["participation"] <= 1
        or not 0 <= cfg["quantity-skew"] <= 1
        or not 0 < cfg["validation-fraction"] < 1
    ):
        raise ValueError("Invalid partition or participation setting")
    if cfg["min-clusters"] < 2 or cfg["max-clusters"] < cfg["min-clusters"]:
        raise ValueError("Invalid cluster bounds")
    if (
        not 0 <= cfg["warmup-epochs"] < cfg["local-epochs"]
        and cfg["method"] != "fedavg"
    ):
        raise ValueError("warmup-epochs must be smaller than local-epochs")
    if cfg["method"] == "fedcompress" and cfg["server-epochs"] < 1:
        raise ValueError("FedCompress requires server distillation epochs")
    if (
        cfg["server-epochs"] < 0
        or cfg["ood-limit"] < 0
        or cfg["seed"] < 0
        or cfg["score-tolerance"] < 0
    ):
        raise ValueError("Negative experiment setting")
    for key in (
        "learning-rate",
        "cluster-learning-rate",
        "server-learning-rate",
        "temperature",
        "registration-timeout",
        "reply-timeout",
    ):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps")
    for key in ("data-root", "model-root", "output"):
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    return cfg


def parser():
    p = argparse.ArgumentParser(
        description="Launch FedCompress or its baselines as a native Flower App."
    )
    for key, value in defaults().items():
        options = {"type": type(value), "default": value}
        if key == "method":
            options["choices"] = METHODS
        if key == "dataset":
            options["choices"] = tuple(CLASSES)
        if key == "device":
            options.update(
                choices=("cpu", "cuda", "mps"),
                default="cuda" if torch.cuda.is_available() else "cpu",
            )
        if key == "output":
            options["default"] = None
        p.add_argument("--" + key, **options)
    p.add_argument("--parallel-clients", type=int, default=1)
    p.add_argument("--client-cpus", type=int, default=1)
    p.add_argument("--client-gpus", type=float)
    p.add_argument("--superlink", default="local")
    p.add_argument("--dry-run", action="store_true")
    return p


def build_config(args):
    cfg = {key: getattr(args, key.replace("-", "_")) for key in defaults()}
    cfg["output"] = (
        cfg["output"]
        or f"results/{cfg['method']}-{cfg['dataset']}-seed{cfg['seed']}.json"
    )
    return validate(cfg)
