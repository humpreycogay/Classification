"""Adapter contract tests; no weights or network access required."""

import unittest

from classifier import D1Classifier


class FakeModel:
    def __init__(self):
        self.calls = []

    @staticmethod
    def response(probabilities):
        return {"answers": {"label": {"type": "choice", "choice": "Business",
                                        "confidence": max(probabilities.values()),
                                        "probabilities": probabilities}},
                "usage": {"input_tokens": 10, "output_tokens": 0}}

    def system_one(self, text, questions):
        self.calls.append((text, questions))
        return self.response({"Business": 0.8, "Sports": 0.2})

    def system_one_batch(self, requests):
        self.calls.append(requests)
        return [self.response({"Business": 0.25, "Sports": 0.75}) for _ in requests]


class D1ClassifierTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.clf = D1Classifier({"business": "Companies and markets", "sports": "Sporting events"},
                                "Choose the topic", device="cpu", model=self.model)

    def test_native_choice_schema_and_probability_mapping(self):
        pred = self.clf.predict("Stocks rose")
        self.assertEqual((pred.label, pred.confidence), ("business", 0.8))
        self.assertEqual(pred.probabilities, {"business": 0.8, "sports": 0.2})
        question = self.model.calls[0][1]["label"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["criteria"], {"Business": "Companies and markets", "Sports": "Sporting events"})

    def test_packed_batches_preserve_order_and_size(self):
        predictions = self.clf.predict_many(["a", "b", "c"], batch_size=2)
        self.assertEqual([prediction.label for prediction in predictions], ["sports"] * 3)
        self.assertEqual([len(call) for call in self.model.calls], [2, 1])

    def test_rejects_unexpected_native_options(self):
        self.model.system_one = lambda *_: self.model.response({"Business": 1.0})
        with self.assertRaisesRegex(RuntimeError, "unexpected choice names"):
            self.clf.predict("text")


if __name__ == "__main__":
    unittest.main()