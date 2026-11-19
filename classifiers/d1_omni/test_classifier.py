import unittest
from classifier import D1OmniClassifier

class Fake:
    def system_one(self, *_): return {"answers": {"label": {"probabilities": {"Business": .8, "Sports": .2}}}}
    def system_one_batch(self, rows): return [self.system_one() for _ in rows]

class TestAdapter(unittest.TestCase):
    def test_mapping_and_batch(self):
        c = D1OmniClassifier({"business": "markets", "sports": "games"}, "pick", device="cpu", model=Fake())
        self.assertEqual(c.predict("x").label, "business")
        self.assertEqual(len(c.predict_many(["a", "b"])), 2)

if __name__ == "__main__": unittest.main()