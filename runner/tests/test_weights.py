"""The readiness check for model files (apps.missing_weights): a deleted or incomplete model is not "ready"."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import apps
from apps import App, missing_weights


class MissingWeightsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cache = root / "hub"
        self.folder = root / "app"
        self.folder.mkdir()
        self.app = App("fake", "Fake", "Local", "max p", folder=str(self.folder))

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self, repo="org/model", files=("model.safetensors",), blobs=True):
        """A cache entry laid out as huggingface_hub does: snapshot files are symlinks into blobs/."""
        base = self.cache / f"models--{repo.replace('/', '--')}"
        snap = base / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (base / "blobs").mkdir(exist_ok=True)
        for i, name in enumerate(files):
            blob = base / "blobs" / f"blob{i}"
            if blobs:
                blob.write_text("x")
            os.symlink(blob, snap / name)
        return base

    def check(self, weights):
        with mock.patch.dict(apps.WEIGHTS, {"fake": weights}):
            return missing_weights(self.app, cache=self.cache)

    def test_complete_repo_is_ready(self):
        self.snapshot(files=("config.json", "model.safetensors"))
        self.assertIsNone(self.check(("hf:org/model",)))

    def test_repo_not_downloaded(self):
        self.assertIn("not in the Hugging Face cache", self.check(("hf:org/model",)))

    def test_deleted_blobs_are_reported(self):  # a cleaned cache leaves the snapshot links pointing at nothing
        self.snapshot(blobs=False)
        self.assertIn("missing file", self.check(("hf:org/model",)))

    def test_snapshot_without_weights(self):
        self.snapshot(files=("config.json",))
        self.assertIn("no weight file", self.check(("hf:org/model",)))

    def test_local_file_and_broken_symlink(self):
        self.assertIn("model file missing", self.check(("path:models/*/READY.json",)))
        (self.folder / "models" / "rev").mkdir(parents=True)
        (self.folder / "models" / "rev" / "READY.json").write_text("{}")
        self.assertIsNone(self.check(("path:models/*/READY.json",)))
        os.symlink(self.folder / "gone.gguf", self.folder / "link.gguf")
        self.assertIn("model file missing", self.check(("path:link.gguf",)))

    def test_repo_from_a_setting(self):
        self.snapshot(repo="org/other")
        with mock.patch.dict(os.environ, {"FAKE_MODEL": "org/other"}):
            self.assertIsNone(self.check(("hf-setting:FAKE_MODEL:org/model",)))
        self.assertIn("org/model", self.check(("hf-setting:FAKE_MODEL:org/model",)))

    def test_every_registered_spec_is_well_formed(self):
        for name, specs in apps.WEIGHTS.items():
            self.assertTrue(name in apps.APPS_BY_NAME or any(a.folder == name for a in apps.APPS), name)
            for spec in specs:
                self.assertIn(spec.partition(":")[0], {"hf", "hf-setting", "path"}, spec)


if __name__ == "__main__":
    unittest.main()
