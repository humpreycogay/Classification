"""Kev-0.8B classifier through Kev's released pointer-head runtime and Apple MLX backend."""
from __future__ import annotations
import os,sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0,os.environ.get("BENCHMARK_SHARED_DIR") or str(Path(__file__).resolve().parents[2]/"shared"))
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
from task import current_task  # noqa:E402

MODEL_REPO="jaredpalmer/kev-0.8b"
MODEL_REVISION="bf75a6a8848ea6960ff2ed108d9ed44c2941174f"
CODE_COMMIT="45923b7a3460b6d36358e2e143455902c1eb856b"
BASE_REPO="Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
MAX_OPTIONS=255; MAX_STATE_TOKENS=7552; MAX_BRANCH_TOKENS=8192
TASK=current_task(); TASK_LABELS={s["id"]:s["description"] for s in TASK.labels}; LABEL_NAMES={s["id"]:s["name"] for s in TASK.labels}
@dataclass
class Prediction: label:str; confidence:float; probabilities:dict[str,float]

class KevClassifier:
 def __init__(self,labels,instruction,predictor=None):
  if not 2<=len(labels)<=MAX_OPTIONS: raise ValueError(f"Kev supports 2-{MAX_OPTIONS} options")
  self.labels=labels; self.device="mps" if sys.platform=="darwin" else "cuda"
  self.label_of={LABEL_NAMES.get(k,k):k for k in labels}
  if len(self.label_of)!=len(labels): raise ValueError("label names must be unique")
  criteria={LABEL_NAMES.get(k,k):v for k,v in labels.items()}
  # LocalPredictor materializes the same labelled-record shape used by Kev's
  # evaluator.  label/src are required bookkeeping fields; api_request strips
  # them before the serving-path prompt is encoded, so this placeholder cannot
  # leak the answer or affect inference.
  self.question={"label":{"type":"choice","instructions":instruction,"criteria":criteria,
                          "label":next(iter(criteria)),"src":"classification_benchmark_choice"}}
  if predictor is None:
   from kev.checkpoint import LoadOptions
   from kev.predictors import LocalPredictor
   opts=LoadOptions(backend="mlx" if sys.platform=="darwin" else "auto")
   context={"max_state":MAX_STATE_TOKENS,"max_branch":MAX_BRANCH_TOKENS,"max_packed":MAX_BRANCH_TOKENS*2}
   predictor=LocalPredictor(f"{MODEL_REPO}@{MODEL_REVISION}",self.device,opts,context=context)
   self.device=predictor.model.backend
  self.predictor=predictor
 def predict(self,text):
  result=self.predictor({"state":text,"questions":self.question})
  native=result["probabilities"]["label"]
  if set(native)!=set(self.label_of): raise RuntimeError("Kev returned unexpected options")
  probabilities={self.label_of[k]:float(v) for k,v in native.items()}; label=max(probabilities,key=probabilities.get)
  return Prediction(label,probabilities[label],probabilities)
 def predict_many(self,texts): return [self.predict(t) for t in texts]
if __name__=="__main__": print(KevClassifier(TASK_LABELS,TASK.instruction).predict(" ".join(sys.argv[1:])))