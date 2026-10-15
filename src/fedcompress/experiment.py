"""Build a temporary Flower App and launch it with explicit experiment settings."""

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import tomli_w

from .config import build_config, parser
from .data import client_data
from .model_assets import load
from .models import AUDIO
from .preparation import validate


@contextmanager
def flower_app(cfg):
    package = Path(__file__).resolve().parent
    dist = distribution("fedcompress")
    descriptor = {
        "build-system": {
            "requires": ["setuptools>=77", "wheel"],
            "build-backend": "setuptools.build_meta",
        },
        "project": {
            "name": "fedcompress",
            "version": dist.version,
            "description": dist.metadata["Summary"],
            "requires-python": ">=3.11,<3.14",
            "dependencies": dist.requires or [],
        },
        "tool": {
            "setuptools": {"packages": {"find": {"where": ["src"]}}},
            "flwr": {
                "app": {
                    "publisher": "nousphera",
                    "components": {
                        "serverapp": "fedcompress.server_app:app",
                        "clientapp": "fedcompress.client_app:app",
                    },
                    "config": cfg,
                }
            },
        },
    }
    with TemporaryDirectory(prefix="fedcompress-flower-") as directory:
        root = Path(directory)
        target = root / "src/fedcompress"
        target.mkdir(parents=True)
        for source in package.glob("*.py"):
            shutil.copy2(source, target / source.name)
        (root / "pyproject.toml").write_text(tomli_w.dumps(descriptor))
        yield root


def command(args, cfg, root, run_path):
    if " " in sys.executable:
        raise ValueError(
            "Ray needs a Python environment without spaces in its path. Create the named environment shown in README.md."
        )
    binary = Path(sys.executable).with_name("flwr")
    if args.parallel_clients < 1 or args.client_cpus < 1:
        raise ValueError("Client resources must be positive")
    gpus = (
        args.client_gpus
        if args.client_gpus is not None
        else (1.0 if cfg["device"] == "cuda" else 0.0)
    )
    if gpus < 0 or (cfg["device"] == "cuda" and gpus <= 0):
        raise ValueError("CUDA clients require positive --client-gpus")
    resources = f"num-supernodes={cfg['nr-clients']} client-resources-num-cpus={args.client_cpus} client-resources-num-gpus={gpus} init-args-num-cpus={args.client_cpus * args.parallel_clients}"
    return [
        str(binary),
        "run",
        str(root),
        args.superlink,
        "--run-config",
        str(run_path),
        "--federation-config",
        resources,
        "--stream",
    ]


@contextmanager
def local_runtime(args, env, log_path):
    """Own the local service lifecycle; never reuse another environment's daemon."""
    if args.superlink != "local":
        yield env
        return

    def available_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    port, control = available_port(), available_port()
    while control == port:
        control = available_port()
    with TemporaryDirectory(prefix="fedcompress-runtime-") as directory:
        env = env.copy()
        env.update(
            FLWR_HOME=directory,
            FLWR_LOCAL_SUPERLINK_HTTP_API_PORT=str(port),
            FLWR_LOCAL_CONTROL_API_PORT=str(control),
            FLWR_DISABLE_UPDATE_CHECK="1",
        )
        binary = Path(sys.executable).with_name("flower-superlink")
        cmd = [
            str(binary),
            "--insecure",
            "--simulation",
            "--isolation",
            "subprocess",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--control-api-address",
            f"127.0.0.1:{control}",
        ]
        with Path(log_path).open("x") as log:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Flower service failed; see {log_path}")
                    try:
                        with urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=0.5
                        ) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Flower startup timed out; see {log_path}")
                    time.sleep(0.2)
                yield env
            finally:
                # This group was created above and contains only this run's service.
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = build_config(args)
        path = Path(cfg["output"])
        run_path = path.with_suffix(".run.toml")
        with flower_app(cfg) as root:
            cmd = command(args, cfg, root, run_path)
            if args.dry_run:
                print(json.dumps({"config": cfg, "command": cmd}, indent=2))
                return
            validate(cfg["dataset"], cfg["data-root"])
            client_data(cfg, 0)
            load(cfg["dataset"], cfg["model-root"], cfg["seed"])
            if cfg["method"] == "fedcompress":
                validate(
                    "librispeech" if cfg["dataset"] in AUDIO else "stylegan",
                    cfg["data-root"],
                )
            for candidate in (
                path,
                path.with_suffix(".npz"),
                run_path,
                path.with_suffix(".flower.log"),
            ):
                if candidate.exists():
                    raise FileExistsError(f"Output already exists: {candidate}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with run_path.open("x") as stream:
                stream.write(tomli_w.dumps(cfg))
            env = os.environ.copy()
            env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
            env["PATH"] = (
                str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            )
            print(shlex.join(cmd), flush=True)
            with local_runtime(
                args, env, path.with_suffix(".flower.log")
            ) as runtime_env:
                subprocess.run(cmd, check=True, env=runtime_env)
        if not path.is_file() or not path.with_suffix(".npz").is_file():
            raise RuntimeError("Flower ended without a saved result and checkpoint")
        print(f"Saved {path}", flush=True)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"Experiment failed: {exc}\n")
