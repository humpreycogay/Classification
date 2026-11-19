import unittest
from classifier import NaluriClassifier
class Module:
    @staticmethod
    def predict(*_args,**_kwargs): return {"label":{"Business":.7,"Sports":.3}}
class Tests(unittest.TestCase):
    def test_release_mapping(self):
        c=NaluriClassifier({"business":"markets","sports":"games"},"pick",device="cpu",runtime=(Module(),None,None))
        self.assertEqual(c.predict("x").probabilities,{"business":.7,"sports":.3})
    def test_limit(self):
        with self.assertRaisesRegex(ValueError,"2-20"): NaluriClassifier({str(i):str(i) for i in range(21)},"pick",runtime=(Module(),None,None))
if __name__=="__main__": unittest.main()