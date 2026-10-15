# Testing

Activate the environment from the [README](../README.md#installation), then run:

```bash
python -m pip check
python -m pytest -q
```

The suite covers convolution clustering, fixed-teacher server distillation, effective-rank aggregation, plateau scheduling, lossless payloads, image/audio adapters, source-bound label caches, transactional preparation, and stable Flower client identity.
Generated inputs keep regression tests small and deterministic. Dataset tests
exercise import, validation, content hashes, and consumption by the training
loader without downloading complete corpora.

## Native Flower smoke check

```bash
flwr run .
python scripts/check_run.py runs/flower-smoke --rounds 2
```

Use a new output directory for repeated runs. The checker inspects saved results
and requires completed rounds and finite final parameters. It supplements the
CLI exit status with checks on training outputs.

[GitHub Actions](https://github.com/Nousphera/FedCompress/actions) runs the CPU checks
on Linux with Python 3.11. See each run for its commit, dependency installation,
and test output.

Audio/TFDS dependencies are included in `requirements.txt`, so the standard
installation also supports the data-adapter checks.
