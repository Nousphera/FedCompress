"""Hard weight sharing with learnable centroids and straight-through gradients."""

import math
from collections import deque

import torch
from torch import nn
from torch.nn.utils import parametrize


def initial_centroids(weight, count, seed):
    """Seeded k-means++ initialization, with no unbounded N x C allocation."""
    values = weight.detach().flatten().cpu().float()
    count = min(count, torch.unique(values).numel())
    generator = torch.Generator().manual_seed(seed)
    centers = [values[torch.randint(values.numel(), (1,), generator=generator).item()]]
    distance = (values - centers[0]).square()
    for _ in range(1, count):
        index = torch.multinomial(distance, 1, generator=generator).item()
        centers.append(values[index])
        distance = torch.minimum(distance, (values - centers[-1]).square())
    return torch.stack(centers).to(weight.device, weight.dtype)


def nearest(weight, centroids):
    sorted_centers, order = centroids.detach().sort()
    boundaries = (sorted_centers[:-1] + sorted_centers[1:]) / 2
    return order[torch.bucketize(weight.detach().contiguous(), boundaries)]


class SharedWeights(nn.Module):
    def __init__(self, weight, clusters, seed):
        super().__init__()
        self.centroids = nn.Parameter(initial_centroids(weight, clusters, seed))

    def forward(self, weight):
        shared = self.centroids[nearest(weight, self.centroids)]
        # Exactly shared in the forward pass; identity gradient to the latent
        # weights and summed gradients to centroids, as in the original wrapper.
        return shared + (weight - weight.detach())


def cluster_model(model, clusters, seed):
    if clusters < 2:
        raise ValueError("At least two clusters are required")
    for index, layer in enumerate(list(model.modules())):
        # Original implementation wraps Conv2D, excluding depthwise kernels.
        if isinstance(layer, nn.Conv2d) and layer.groups == 1:
            if parametrize.is_parametrized(layer):
                raise ValueError("Model already clustered")
            parametrize.register_parametrization(
                layer, "weight", SharedWeights(layer.weight, clusters, seed + index)
            )
    return model


def strip_clustering(model):
    for layer in list(model.modules()):
        if parametrize.is_parametrized(layer, "weight"):
            parametrize.remove_parametrizations(
                layer, "weight", leave_parametrized=True
            )
    return model


def effective_rank(embeddings):
    z = embeddings.detach().cpu().double()
    if z.ndim != 2 or not torch.isfinite(z).all():
        raise ValueError("Embeddings must be a finite matrix")
    z = torch.nn.functional.normalize(z, dim=1)
    singular = torch.linalg.svdvals(z)
    total = singular.sum()
    if total == 0:
        return 0.0
    p = singular / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


class ClusterSchedule:
    def __init__(
        self, minimum=8, maximum=30, window=3, patience=3, tolerance=0.01, step=1
    ):
        if (
            min(minimum, window, patience, step) < 1
            or maximum < minimum
            or tolerance < 0
        ):
            raise ValueError("Invalid cluster schedule")
        self.count, self.maximum = minimum, maximum
        self.scores = deque(maxlen=window)
        self.patience, self.tolerance, self.step = patience, tolerance, step
        self.previous, self.stalled = None, 0

    def update(self, score):
        if not math.isfinite(score) or score < 0:
            raise ValueError("Invalid effective rank")
        self.scores.append(score)
        if len(self.scores) < self.scores.maxlen:
            return self.count
        mean = sum(self.scores) / len(self.scores)
        if self.previous is not None:
            self.stalled = (
                self.stalled + 1 if mean - self.previous <= self.tolerance else 0
            )
        self.previous = mean
        if self.stalled >= self.patience:
            self.count = min(self.maximum, self.count + self.step)
            self.stalled = 0
        return self.count
