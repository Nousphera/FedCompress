"""Sample-weighted FL with adaptive clustering and server self-compression."""

import json
import math
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

from .clustering import ClusterSchedule
from .codec import decode, encode, weighted_average
from .config import defaults, validate
from .data import AuxiliaryData, load_dataset
from .model_assets import checkpoint_path, load
from .models import AUDIO
from .preparation import digest
from .preparation import validate as validate_data
from .training import evaluate, seed_everything, self_compress

app = ServerApp()


def selected_nodes(nodes, participation, rng):
    # Flower IDs are unsigned 64-bit integers: NumPy may coerce a mixed signed/
    # unsigned list to float64. Sample positions to preserve exact node IDs.
    indexes = rng.choice(
        len(nodes), max(1, math.floor(len(nodes) * participation)), replace=False
    )
    return sorted(nodes[int(i)] for i in indexes)


def wait_for_nodes(grid, count, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        nodes = sorted(grid.get_node_ids())
        if len(nodes) > count:
            raise ValueError("More Flower nodes than configured clients")
        if len(nodes) == count:
            return nodes
        time.sleep(0.25)
    raise TimeoutError("Not all Flower clients registered")


def collect(grid, messages, cfg, expected, round_id):
    pending = {m.metadata.dst_node_id for m in messages}
    seen = set()
    client_ids = set()
    updates = []
    metrics = []
    for reply in grid.send_and_receive(messages, timeout=cfg["reply-timeout"]):
        node = reply.metadata.src_node_id
        if node not in pending or node in seen or reply.has_error():
            raise RuntimeError(f"Invalid or failed reply from Flower node {node}")
        seen.add(node)
        if reply.metadata.group_id != str(round_id):
            raise ValueError("Reply group mismatch")
        row = dict(reply.content["metrics"])
        cid, count = row["client-id"], row["examples"]
        if (
            type(cid) is not int
            or not 0 <= cid < cfg["nr-clients"]
            or cid in client_ids
            or type(count) is not int
            or count < 1
            or row["round"] != round_id
        ):
            raise ValueError("Invalid client metrics")
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in row.values()
        ):
            raise ValueError("Nonfinite client metrics")
        client_ids.add(cid)
        payload = reply.content["weights"]["payload"]
        if row["payload-bytes"] != len(payload):
            raise ValueError("Payload size mismatch")
        updates.append((decode(payload, expected), count))
        metrics.append(row)
    if seen != pending:
        raise RuntimeError("Missing client replies; round was not aggregated")
    return weighted_average(updates), metrics


def run_server(grid, cfg):
    cfg = validate(defaults() | cfg)
    seed_everything(cfg["seed"], cfg["torch-threads"])
    output = Path(cfg["output"])
    checkpoint = output.with_suffix(".npz")
    if output.exists() or checkpoint.exists():
        raise FileExistsError("Choose a new output path")
    validate_data(cfg["dataset"], cfg["data-root"], full=False)
    test = load_dataset(cfg["dataset"], cfg["data-root"], "test")
    model = load(cfg["dataset"], cfg["model-root"], cfg["seed"])
    auxiliary = None
    if cfg["method"] == "fedcompress":
        auxiliary_name = "librispeech" if cfg["dataset"] in AUDIO else "stylegan"
        validate_data(auxiliary_name, cfg["data-root"], full=False)
        auxiliary = AuxiliaryData(
            cfg["data-root"],
            auxiliary_name,
            28 if cfg["dataset"] == "pathmnist" else 32,
            cfg["ood-limit"],
        )
    nodes = wait_for_nodes(grid, cfg["nr-clients"], cfg["registration-timeout"])
    schedule = ClusterSchedule(
        cfg["min-clusters"],
        cfg["max-clusters"],
        cfg["window"],
        cfg["patience"],
        cfg["score-tolerance"],
        cfg["cluster-step"],
    )
    rng = np.random.default_rng(cfg["seed"])
    history = []
    initial = evaluate(model, test, cfg)
    for rnd in range(1, cfg["rounds"] + 1):
        clusters = schedule.count
        selected = selected_nodes(nodes, cfg["participation"], rng)
        expected = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload = encode(expected)
        messages = [
            Message(
                RecordDict(
                    {
                        "weights": ConfigRecord({"payload": payload}),
                        "config": ConfigRecord({"round": rnd, "clusters": clusters}),
                    }
                ),
                dst_node_id=n,
                message_type="train",
                group_id=str(rnd),
            )
            for n in selected
        ]
        state, client_metrics = collect(grid, messages, cfg, expected, rnd)
        model.load_state_dict(state)
        aggregated = evaluate(model, test, cfg)
        count = sum(m["examples"] for m in client_metrics)
        score = sum(m["score"] * m["examples"] for m in client_metrics) / count
        train_loss = (
            sum(m["train-loss"] * m["examples"] for m in client_metrics) / count
        )
        compression = {}
        if auxiliary is not None:
            model, compression = self_compress(model, auxiliary, cfg, clusters, rnd)
        final = evaluate(model, test, cfg) if auxiliary is not None else aggregated
        if cfg["method"] != "fedavg":
            schedule.update(score)
        row = {
            "round": rnd,
            "clusters": clusters if cfg["method"] != "fedavg" else None,
            "next-clusters": schedule.count,
            "train-loss": train_loss,
            "effective-rank": score,
            "aggregated": aggregated,
            "evaluation": final,
            "uplink-bytes": sum(m["payload-bytes"] for m in client_metrics),
            "downlink-bytes": len(payload) * len(selected),
            "clients": client_metrics,
            **compression,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "round": rnd,
                    "accuracy": final["accuracy"],
                    "clusters": row["clusters"],
                }
            ),
            flush=True,
        )
    result = {
        "schema": 1,
        "config": cfg,
        "initial-evaluation": initial,
        "rounds": history,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "flower": version("flwr"),
        },
        "data-manifest-sha256": digest(
            Path(cfg["data-root"]) / cfg["dataset"] / "manifest.json"
        ),
        "initialization-sha256": digest(
            checkpoint_path(cfg["model-root"], cfg["dataset"], cfg["seed"])
        ),
        "byte-accounting": "Lossless NPZ/DEFLATE model payloads in both directions for every method; excludes Flower and network overhead.",
        "auxiliary-examples": len(auxiliary) if auxiliary is not None else 0,
    }
    if auxiliary is not None:
        result["auxiliary-manifest-sha256"] = digest(
            Path(cfg["data-root"]) / auxiliary_name / "manifest.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    final_payload = encode(model.state_dict())
    with checkpoint.open("xb") as stream:
        stream.write(final_payload)
    result["final-payload-bytes"] = len(final_payload)
    result["checkpoint-sha256"] = digest(checkpoint)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_server(grid, context.run_config)
