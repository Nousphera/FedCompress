"""Native Flower ClientApp: local data remain on their assigned client."""

import math

from flwr.app import ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .codec import decode, encode
from .config import defaults, validate
from .data import client_data
from .models import make_model
from .training import train_client

app = ClientApp()


@app.train()
def train(message: Message, context: Context) -> Message:
    cfg = validate(defaults() | context.run_config)
    cid = int(context.node_config["partition-id"])
    if not 0 <= cid < cfg["nr-clients"]:
        raise ValueError("Invalid client partition")
    params = message.content["config"]
    round_id, clusters = params["round"], params["clusters"]
    if (
        type(round_id) is not int
        or not 1 <= round_id <= cfg["rounds"]
        or type(clusters) is not int
        or not cfg["min-clusters"] <= clusters <= cfg["max-clusters"]
    ):
        raise ValueError("Invalid round configuration")
    model = make_model(cfg["dataset"], cfg["seed"])
    model.load_state_dict(
        decode(message.content["weights"]["payload"], model.state_dict())
    )
    train_data, valid_data = client_data(cfg, cid)
    state, metrics = train_client(
        model, train_data, valid_data, cfg, clusters, round_id, cid
    )
    payload = encode(state)
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError("Nonfinite client metrics")
    return Message(
        RecordDict(
            {
                "weights": ConfigRecord({"payload": payload}),
                "metrics": MetricRecord(
                    {
                        **metrics,
                        "client-id": cid,
                        "round": round_id,
                        "payload-bytes": len(payload),
                    }
                ),
            }
        ),
        reply_to=message,
    )
