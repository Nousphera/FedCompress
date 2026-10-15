"""Local training, frozen-teacher self-compression and evaluation."""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .clustering import cluster_model, effective_rank, strip_clustering
from .data import loader
from .models import AUDIO


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return torch.device(name)


def regularization(model, audio=False):
    zero = next(model.parameters()).new_zeros(())
    if audio:
        return zero
    return 1e-4 * sum(
        (
            m.weight.square().sum()
            for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))
        ),
        start=zero,
    )


def distillation_loss(student, teacher, temperature=2.0):
    return (
        F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


@torch.no_grad()
def evaluate(model, dataset, cfg):
    model.eval()
    total = correct = 0
    loss = 0.0
    target = device(cfg["device"])
    model.to(target)
    for x, y in loader(
        dataset, cfg["batch-size"] * 4, cfg["seed"], audio=cfg["dataset"] in AUDIO
    ):
        logits = model(x.to(target))
        y = y.to(target)
        loss += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        total += len(y)
    if not total:
        raise ValueError("Evaluation data are empty")
    return {"loss": loss / total, "accuracy": correct / total, "examples": total}


@torch.no_grad()
def representation_score(model, dataset, cfg):
    model.eval()
    embeddings = []
    for x, _ in loader(
        dataset, cfg["batch-size"], cfg["seed"], audio=cfg["dataset"] in AUDIO
    ):
        embeddings.append(model.features(x.to(device(cfg["device"]))).cpu())
    return effective_rank(torch.cat(embeddings))


def train_client(model, train, valid, cfg, clusters, round_id, cid):
    seed = cfg["seed"] + 100003 * round_id + cid
    seed_everything(seed, cfg["torch-threads"])
    model.to(device(cfg["device"]))
    audio = cfg["dataset"] in AUDIO
    data = loader(train, cfg["batch-size"], seed, True, audio)
    clustered = cfg["method"] != "fedavg"
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["learning-rate"],
        eps=1e-7,
    )
    last_loss = 0.0
    last_count = 0
    for epoch in range(cfg["local-epochs"]):
        if clustered and epoch == cfg["warmup-epochs"]:
            cluster_model(model, clusters, seed)
            optimizer = torch.optim.Adam(
                (p for p in model.parameters() if p.requires_grad),
                lr=cfg["cluster-learning-rate"],
                eps=1e-7,
            )
        model.train()
        last_loss = 0.0
        last_count = 0
        for x, y in data:
            x, y = x.to(device(cfg["device"])), y.to(device(cfg["device"]))
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + regularization(model, audio)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite client loss")
            loss.backward()
            optimizer.step()
            last_loss += float(loss.detach()) * len(y)
            last_count += len(y)
    strip_clustering(model)
    metrics = {
        "train-loss": last_loss / last_count,
        "score": representation_score(model, valid, cfg),
        "examples": len(train),
    }
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, metrics


def self_compress(model, auxiliary, cfg, clusters, round_id):
    seed = cfg["seed"] + 200003 * round_id
    seed_everything(seed, cfg["torch-threads"])
    target = device(cfg["device"])
    # Teacher is never updated, including its BatchNorm buffers.
    teacher = copy.deepcopy(model).to(target).eval().requires_grad_(False)
    student = copy.deepcopy(model).to(target)
    cluster_model(student, clusters, seed)
    optimizer = torch.optim.Adam(
        (p for p in student.parameters() if p.requires_grad),
        lr=cfg["server-learning-rate"],
        eps=1e-7,
    )
    data = loader(
        auxiliary, cfg["server-batch-size"], seed, True, cfg["dataset"] in AUDIO
    )
    total_loss = 0.0
    count = 0
    for _ in range(cfg["server-epochs"]):
        student.train()
        total_loss = 0.0
        count = 0
        for x in data:
            x = x.to(target)
            with torch.no_grad():
                targets = teacher(x)
            optimizer.zero_grad(set_to_none=True)
            loss = distillation_loss(student(x), targets, cfg["temperature"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite server loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(x)
            count += len(x)
    if not count:
        raise ValueError("Empty auxiliary dataset")
    return strip_clustering(student).cpu(), {
        "distillation-loss": total_loss / count,
        "auxiliary-examples": len(auxiliary),
    }
