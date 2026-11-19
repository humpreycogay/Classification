import unittest

from apps import APPROACHES, APPS, APPS_BY_NAME


class ApproachTest(unittest.TestCase):
    """Every app falls in exactly one comparison category (phase 2, item 3)."""

    def test_every_app_has_a_known_approach(self):
        for task in ("ag_news", "banking77"):
            for app in APPS:
                self.assertIn(app.approach_on(task), APPROACHES, app.name)

    def test_known_examples(self):
        expected = {"tfidf_logreg": "classical supervised", "embedsup_logreg": "classical supervised",
                    "scratch_cnn": "from scratch", "finetune_deberta": "fine-tuned encoder",
                    "nli": "zero-shot", "jev": "zero-shot"}
        for name, approach in expected.items():
            self.assertEqual(APPS_BY_NAME[name].approach_on("ag_news"), approach, name)

    def test_in_domain_follows_trained_on_tasks(self):
        for app in APPS:
            for task in app.trained_on_tasks:
                self.assertEqual(app.approach_on(task), "in-domain checkpoint", app.name)


if __name__ == "__main__":
    unittest.main()
