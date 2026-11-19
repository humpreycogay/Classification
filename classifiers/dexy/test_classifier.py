import unittest
from classifier import DexyClassifier
class Fake:
 def predict(self,_): return {"answers":{"label":{"probabilities":{"Business":.65,"Sports":.35}}}}
class Tests(unittest.TestCase):
 def test_mapping(self):
  c=DexyClassifier({"business":"markets","sports":"games"},"pick",device="cpu",model=Fake())
  self.assertEqual(c.predict("x").probabilities,{"business":.65,"sports":.35})
if __name__=="__main__": unittest.main()