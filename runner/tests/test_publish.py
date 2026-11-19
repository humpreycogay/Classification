import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_all


class PublishGuardTest(unittest.TestCase):
    """A smaller run never replaces a larger published reports/<task>.html by accident (phase 2, item 1)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(run_all, "REPORTS_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def mark(self, n):
        run_all.latest_marker_path("ag_news").write_text(json.dumps({"n": n, "run": "x"}))

    def test_publishes_without_marker(self):
        self.assertEqual(run_all.should_publish("ag_news", 200), (True, None))

    def test_smaller_run_not_published(self):
        self.mark(1000)
        ok, why = run_all.should_publish("ag_news", 200)
        self.assertFalse(ok)
        self.assertIn("--publish-smaller", why)

    def test_same_or_larger_and_forced_runs_published(self):
        self.mark(1000)
        self.assertTrue(run_all.should_publish("ag_news", 1000)[0])
        self.assertTrue(run_all.should_publish("ag_news", 7600)[0])
        self.assertTrue(run_all.should_publish("ag_news", 200, force=True)[0])


if __name__ == "__main__":
    unittest.main()
