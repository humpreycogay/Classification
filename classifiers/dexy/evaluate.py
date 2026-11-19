"""Evaluate Dexy-Decision-9B (experimental; CUDA release, opt-in MPS path)."""
import os,sys
from pathlib import Path
APP_DIR=Path(__file__).resolve().parent
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or APP_DIR.parents[1] / "shared")
sys.path.insert(0, str(SHARED_DIR))
from task import select_task_from_argv
select_task_from_argv()
from classifier import *
from harness import Setup,main
def adapter(args,ctx):
 c=DexyClassifier(TASK_LABELS,ctx.task.instruction)
 return Setup(model=f"{MODEL_REPO}@{MODEL_REVISION[:7]} on {c.device} (experimental)",classifier=c,sequential=True,predict=lambda texts,v=None:c.predict_many(texts),model_repo=MODEL_REPO,
 metadata={"dexy":{"model":MODEL_REPO,"revision":MODEL_REVISION,"device":c.device,"context_window":MAX_CONTEXT_TOKENS,"max_options":MAX_OPTIONS,
 "status":"CUDA release; Apple MPS path is unverified and high-risk on 32 GB","scoring":"final-token verbalizer log-probability + FP32 bias and release temperature"}})
if __name__=="__main__": main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=["torch","transformers","accelerate","pandas","httpx"])