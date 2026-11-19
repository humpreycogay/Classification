import unittest
from unittest import mock

from classifier import HEAD_MAX_LEN, MAX_LEN, LayaClassifier


class FakeRouter:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return {"answers": {"label": {"choice": "a", "answer_confidence": 0.75,
                                         "probabilities": {"a": 0.75, "b": 0.25}}}}

    def predict_batch(self, requests, batch_size):
        self.calls.append((requests, batch_size))
        return [self.predict(r["state"], r["questions"]) for r in requests]


class LayaBudgetTest(unittest.TestCase):
    def classifier(self):
        with mock.patch("classifier.Router") as router, mock.patch("classifier.try_to_load_from_cache", return_value="/tmp/x"):
            fake = FakeRouter()
            router.return_value = fake
            return LayaClassifier({"a": "A", "b": "B"}, "pick", model="english", device="cpu"), fake

    def test_single_forwards_expanded_budgets(self):
        classifier, router = self.classifier()
        classifier.predict("x")
        self.assertEqual(router.calls[0][2]["max_len"], MAX_LEN)
        self.assertEqual(router.calls[0][2]["head_max_len"], HEAD_MAX_LEN)

    def test_batch_puts_expanded_budgets_on_every_request(self):
        classifier, router = self.classifier()
        classifier.predict_many(["x", "y"], batch_size=2)
        requests = router.calls[0][0]
        self.assertTrue(all(r["max_len"] == 1024 and r["head_max_len"] == 512 for r in requests))


if __name__ == "__main__":
    unittest.main()