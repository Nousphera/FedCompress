"""Create and load reproducible, untrained paper-backbone initializations."""

import argparse
from pathlib import Path

import torch

from .models import CLASSES, make_model


def checkpoint_path(root, dataset, seed):
    return Path(root) / f"{dataset}-seed{seed}.pt"


def prepare(dataset, root="checkpoints", seed=42):
    path = checkpoint_path(root, dataset, seed)
    if path.exists():
        load(dataset, root, seed)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    model = make_model(dataset, seed)
    temporary = path.with_suffix(".part")
    torch.save(
        {"dataset": dataset, "seed": seed, "state_dict": model.state_dict()}, temporary
    )
    temporary.replace(path)
    return path


def load(dataset, root, seed):
    path = checkpoint_path(root, dataset, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepare initialization: python scripts/prepare_model.py --dataset {dataset} --seed {seed}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["dataset"] != dataset or checkpoint["seed"] != seed:
        raise ValueError("Model initialization metadata mismatch")
    model = make_model(dataset, seed)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite initialization")
    return model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=tuple(CLASSES))
    p.add_argument("--root", type=Path, default=Path("checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.seed < 0:
        p.error("seed must be nonnegative")
    for name in (args.dataset,) if args.dataset else CLASSES:
        print(prepare(name, args.root, args.seed))
