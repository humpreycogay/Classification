import unittest
from classifier import KevClassifier
class Fake:
 model=type("M",(),{"backend":"fake"})()
 def __call__(self,record):
  question=record["questions"]["label"]
  assert question["label"] in question["criteria"]
  assert question["src"]=="classification_benchmark_choice"
  return {"probabilities":{"label":{"Business":.75,"Sports":.25}}}
class Tests(unittest.TestCase):
 def test_mapping(self):
  c=KevClassifier({"business":"markets","sports":"games"},"pick",predictor=Fake())
  self.assertEqual((c.predict("x").label,c.predict("x").confidence),("business",.75))
if __name__=="__main__": unittest.main()