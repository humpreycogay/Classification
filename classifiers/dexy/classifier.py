"""Experimental Dexy-Decision-9B classifier using its supplied JevVerbalizerHead runtime."""
from __future__ import annotations
import importlib.util,os,sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0,os.environ.get("BENCHMARK_SHARED_DIR") or str(Path(__file__).resolve().parents[2]/"shared"))
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
import torch  # noqa:E402
from huggingface_hub import snapshot_download  # noqa:E402
from task import current_task  # noqa:E402
MODEL_REPO="Patrick-18/Dexy-Decision-9B"; MODEL_REVISION="afee4baa7ef921221a0e74f695a150edd6887ce2"
MAX_OPTIONS=256; MAX_CONTEXT_TOKENS=65536
TASK=current_task(); TASK_LABELS={s["id"]:s["description"] for s in TASK.labels}; LABEL_NAMES={s["id"]:s["name"] for s in TASK.labels}
@dataclass
class Prediction: label:str; confidence:float; probabilities:dict[str,float]
def default_device(): return "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
def load_runtime(device):
 root=Path(snapshot_download(MODEL_REPO,revision=MODEL_REVISION))
 spec=importlib.util.spec_from_file_location("dexy_decision_runtime",root/"decision_runtime.py"); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
 dtype="bfloat16" if device!="cpu" else "float32"
 return module.DecisionModel(root,device=device,dtype=dtype,attn_impl="sdpa",max_model_len=MAX_CONTEXT_TOKENS)
class DexyClassifier:
 def __init__(self,labels,instruction,device=None,model=None):
  if not 2<=len(labels)<=MAX_OPTIONS: raise ValueError(f"Dexy supports 2-{MAX_OPTIONS} options")
  self.device=device or os.getenv("DEXY_DEVICE") or default_device(); self.label_of={LABEL_NAMES.get(k,k):k for k in labels}
  self.question={"label":{"type":"choice","instructions":instruction,"criteria":{LABEL_NAMES.get(k,k):v for k,v in labels.items()}}}
  self.model=model or load_runtime(self.device)
 def predict(self,text):
  answer=self.model.predict({"state":text,"questions":self.question})["answers"]["label"]; native=answer["probabilities"]
  if set(native)!=set(self.label_of): raise RuntimeError("Dexy returned unexpected options")
  probabilities={self.label_of[k]:float(v) for k,v in native.items()}; label=max(probabilities,key=probabilities.get)
  return Prediction(label,probabilities[label],probabilities)
 def predict_many(self,texts): return [self.predict(t) for t in texts]
if __name__=="__main__": print(DexyClassifier(TASK_LABELS,TASK.instruction).predict(" ".join(sys.argv[1:])))