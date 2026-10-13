"""Small utility contracts; no TensorFlow, Flower or Ray runtime is used."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from src import utils


ROOT = Path(__file__).resolve().parents[1]


class UtilityTests(unittest.TestCase):
    def test_lightweight_helpers_import_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-S", "-c", "from src.utils import none_or_int; assert none_or_int('12') == 12"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_timeout_conversion(self):
        self.assertIsNone(utils.none_or_int("None"))
        self.assertEqual(utils.none_or_int("60"), 60)
        self.assertEqual(utils.none_or_int("0"), 0)
        with self.assertRaises(ValueError):
            utils.none_or_int("invalid")

    def test_client_model_sizes_survive_metric_recording(self):
        aggregated = utils.weighted_average_train_metrics([
            (2, {"train_loss": 1.0, "train_accuracy": 0.5, "model_size": 3.0}),
            (3, {"train_loss": 2.0, "train_accuracy": 1.0, "model_size": 5.0}),
        ])
        recorded = utils.set_train_metrics(aggregated)
        self.assertEqual(recorded.get("client_model_size"), 4.0)
        self.assertEqual(recorded.get("clients_model_size"), [3.0, 5.0])
        self.assertAlmostEqual(recorded["train_loss"], 1.6)
        self.assertAlmostEqual(recorded["train_accuracy"], 0.8)

    def test_file_size_measurement_removes_temporary_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "weights.h5"
            source.write_bytes(b"weights" * 100)
            reference = directory / "reference.zip"
            with zipfile.ZipFile(reference, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(source)
            expected_size = reference.stat().st_size / 1000
            reference.unlink()
            with patch.object(tempfile, "tempdir", str(directory)):
                size = utils.get_gzipped_model_size(str(source))
            self.assertEqual(size, expected_size)
            self.assertEqual(set(directory.iterdir()), {source})

    def test_model_size_measurement_removes_temporary_artifacts(self):
        class FakeModel:
            def save(self, filename, include_optimizer):
                self.assert_optimizer = include_optimizer
                Path(filename).write_bytes(b"model data" * 100)

        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tempfile, "tempdir", directory):
                self.assertGreater(utils.get_gzipped_model_size_from_model(model), 0)
            self.assertFalse(model.assert_optimizer)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failed_model_save_removes_temporary_artifacts(self):
        class FailingModel:
            def save(self, filename, include_optimizer):
                Path(filename).write_bytes(b"partial model")
                raise RuntimeError("save failed")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tempfile, "tempdir", directory):
                with self.assertRaisesRegex(RuntimeError, "save failed"):
                    utils.get_gzipped_model_size_from_model(FailingModel())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
