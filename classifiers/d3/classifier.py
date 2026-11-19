"""Decision 3.0 text classifier using the checkpoints' portable AutoModel runtime."""
from __future__ import annotations
import math, os, sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0,os.environ.get("BENCHMARK_SHARED_DIR") or str(Path(__file__).resolve().parents[2]/"shared"))
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
import torch  # noqa:E402
from task import current_task  # noqa:E402

MODELS={
 "edge":("vllm-sr/d3-edge","bee5e872f75b9c5d1405767bf244a6948b95c0af"),
 "lite":("vllm-sr/d3-lite","6c01fba14e74f63b70e1ef442b0b6d1b934339d5"),
 "nano":("vllm-sr/d3-nano","8198c9f763b046605b2f4827b4999971fa2c92c4"),
}
MAX_OPTIONS=255
TASK=current_task(); TASK_LABELS={s["id"]:s["description"] for s in TASK.labels}; LABEL_NAMES={s["id"]:s["name"] for s in TASK.labels}
@dataclass
class Prediction: label:str; confidence:float; probabilities:dict[str,float]
def default_device(): return "cuda" if torch.cuda.is_available() else "cpu"

class D3Classifier:
 def __init__(self,labels,instruction,variant="edge",device=None,model=None):
  if variant not in MODELS: raise ValueError(f"unknown d3 variant {variant!r}")
  if not 2<=len(labels)<=MAX_OPTIONS: raise ValueError(f"d3 supports 2-{MAX_OPTIONS} options")
  self.variant=variant; self.device=device or os.getenv("D3_DEVICE") or default_device()
  if self.device not in {"cuda","mps","cpu"}: raise ValueError("D3_DEVICE must be cuda, mps or cpu")
  self.label_of={LABEL_NAMES.get(k,k):k for k in labels}
  if len(self.label_of)!=len(labels): raise ValueError("label names must be unique")
  self.question={"label":{"type":"choice","instructions":instruction,"criteria":{LABEL_NAMES.get(k,k):v for k,v in labels.items()}}}
  if model is None:
   from transformers import AutoModel
   repo,rev=MODELS[variant]
   model=AutoModel.from_pretrained(repo,revision=rev,trust_remote_code=True,device=self.device)
  self.model=model
 def _convert(self,response):
  answer=response["answers"]["label"]
  if "error" in answer: raise RuntimeError(f"d3 failed: {answer['error']}: {answer.get('message','')}")
  native=answer["probabilities"]
  if set(native)!=set(self.label_of): raise RuntimeError(f"d3 returned {len(native)} unexpected options")
  values=[float(v) for v in native.values()]
  if not all(math.isfinite(v) and v>=0 for v in values) or abs(sum(values)-1)>1e-5: raise RuntimeError("invalid d3 probabilities")
  probabilities={self.label_of[k]:float(v) for k,v in native.items()}; label=max(probabilities,key=probabilities.get)
  return Prediction(label,probabilities[label],probabilities)
 def predict(self,text): return self._convert(self.model.system_one(state=text,questions=self.question))
 def predict_many(self,texts): return [self.predict(t) for t in texts]
if __name__=="__main__": print(D3Classifier(TASK_LABELS,TASK.instruction).predict(" ".join(sys.argv[1:])))