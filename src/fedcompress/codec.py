"""Lossless, pickle-free model payloads for both Flower communication directions."""

import zipfile
from io import BytesIO

import numpy as np
import torch


def encode(state, compressed=True):
    arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    stream = BytesIO()
    (np.savez_compressed if compressed else np.savez)(stream, **arrays)
    return stream.getvalue()


def decode(payload, expected):
    if not isinstance(payload, bytes):
        raise ValueError("Expected serialized weight bytes")
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            budget = sum(x.numel() * x.element_size() + 2048 for x in expected.values())
            if (
                len(archive.infolist()) != len(expected)
                or sum(x.file_size for x in archive.infolist()) > budget
            ):
                raise ValueError("Unexpected model payload size")
        with np.load(BytesIO(payload), allow_pickle=False) as data:
            if set(data.files) != set(expected):
                raise ValueError("Model parameter names do not match")
            state = {}
            for name, template in expected.items():
                value = data[name]
                if (
                    value.shape != tuple(template.shape)
                    or value.dtype != template.cpu().numpy().dtype
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid parameter {name}")
                state[name] = torch.from_numpy(value.copy())
            return state
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError("Invalid model payload") from exc


def weighted_average(updates):
    """Sample-weighted floating state; integer BN counters use the maximum."""
    total, accum, integers, template = 0, {}, {}, None
    for state, count in updates:
        if type(count) is not int or count <= 0:
            raise ValueError("Client example counts must be positive integers")
        if template is None:
            template = state
        if set(state) != set(template):
            raise ValueError("Inconsistent model state")
        for name, tensor in state.items():
            reference = template[name]
            if (
                tensor.shape != reference.shape
                or tensor.dtype != reference.dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid model state: {name}")
            tensor = tensor.detach().cpu()
            if tensor.is_floating_point():
                if name not in accum:
                    accum[name] = torch.zeros_like(tensor, dtype=torch.float64)
                accum[name].add_(tensor.double(), alpha=count)
            else:
                integers[name] = (
                    tensor.clone()
                    if name not in integers
                    else torch.maximum(integers[name], tensor)
                )
        total += count
    if not total:
        raise ValueError("No client updates")
    return {
        name: (accum[name] / total).to(tensor.dtype)
        if name in accum
        else integers[name]
        for name, tensor in template.items()
    }
