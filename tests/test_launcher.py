"""Exercise the real launcher with a tiny child in place of the ML runtime."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from configs import Params


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fedcompress tests ")
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.repo = self.parent / "repository"
        (self.repo / "src").mkdir(parents=True)
        for name in ("run.py", "configs.py"):
            shutil.copy2(ROOT / name, self.repo / name)
        (self.repo / "src" / "main.py").write_text(
            "import json, os, sys\n"
            "print('LAUNCH:' + json.dumps({'args': sys.argv[1:], "
            "'executable': sys.executable, 'cwd': os.getcwd()}))\n"
            "sys.exit(int(os.environ.get('TEST_CHILD_EXIT', '0')))\n"
        )

    def launch(self, *args, outside=False, extra_env=None):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        environment.update(extra_env or {})
        return subprocess.run(
            [sys.executable, "-S", str(self.repo / "run.py"), *args],
            cwd=self.parent if outside else self.repo,
            env=environment,
            capture_output=True,
            text=True,
        )

    def child_payload(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("LAUNCH:")]
        self.assertEqual(len(lines), 1, result.stdout)
        return json.loads(lines[0].removeprefix("LAUNCH:"))

    def test_default_launches_a_supported_method(self):
        payload = self.child_payload(self.launch())
        self.assertIn("--server_compression", payload["args"])
        self.assertIn("--client_compression", payload["args"])

    def test_method_flags_preserve_the_three_approaches(self):
        for method, client, server in (
            ("fedavg", False, False),
            ("client", True, False),
            ("fedcompress", True, True),
        ):
            with self.subTest(method=method):
                args = self.child_payload(self.launch("--methods", method))["args"]
                self.assertEqual("--client_compression" in args, client)
                self.assertEqual("--server_compression" in args, server)
                self.assertEqual(args[args.index("--epochs") + 1], "5")

    def test_launch_is_independent_of_working_directory_and_path_python(self):
        fake_bin = self.parent / "bin"
        fake_bin.mkdir()
        fake_python = fake_bin / "python3"
        fake_python.write_text("#!/bin/sh\nexit 99\n")
        fake_python.chmod(0o755)
        payload = self.child_payload(self.launch(
            "--methods", "fedavg", outside=True,
            extra_env={"PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", "")},
        ))
        self.assertEqual(Path(payload["executable"]).resolve(), Path(sys.executable).resolve())
        args = payload["args"]
        for option, suffix in (
            ("--results_dir", "assets/results"),
            ("--model_dir", "assets/saved_models"),
            ("--init_model_fp", "assets/init_models/resnet20_cifar10_random_1.h5"),
        ):
            self.assertEqual(Path(args[args.index(option) + 1]), (self.repo / suffix).resolve())

    def test_child_failure_is_reported_as_launcher_failure(self):
        result = self.launch("--methods", "fedavg", extra_env={"TEST_CHILD_EXIT": "23"})
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)

    def test_experiment_ids_are_created_per_configuration(self):
        self.assertNotEqual(Params().random_id, Params().random_id)


if __name__ == "__main__":
    unittest.main()
