import unittest
from classifier import D3Classifier
class Fake:
 def system_one(self,**_): return {"answers":{"label":{"probabilities":{"Business":.6,"Sports":.4}}}}
class Tests(unittest.TestCase):
 def test_mapping(self):
  c=D3Classifier({"business":"markets","sports":"games"},"pick",device="cpu",model=Fake())
  self.assertEqual(c.predict("x").label,"business")
 def test_normalization(self):
  c=D3Classifier({"business":"markets","sports":"games"},"pick",device="cpu",model=Fake()); c.model.system_one=lambda **_:{"answers":{"label":{"probabilities":{"Business":.6,"Sports":.3}}}}
  with self.assertRaisesRegex(RuntimeError,"invalid"): c.predict("x")
if __name__=="__main__": unittest.main()