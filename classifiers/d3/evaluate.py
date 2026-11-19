"""Evaluate Decision 3.0 (default d3-edge; --model edge|lite|nano)."""
import os,sys
from pathlib import Path
APP_DIR=Path(__file__).resolve().parent
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or APP_DIR.parents[1] / "shared")
sys.path.insert(0, str(SHARED_DIR))
from task import select_task_from_argv
select_task_from_argv()
from classifier import D3Classifier,MODELS,TASK_LABELS
from harness import Setup,main
def add_arguments(p): p.add_argument("--model",choices=MODELS,default="edge")
def adapter(args,ctx):
 c=D3Classifier(TASK_LABELS,ctx.task.instruction,args.model); repo,rev=MODELS[args.model]
 return Setup(model=f"{repo}@{rev[:7]} on {c.device}",classifier=c,sequential=True,predict=lambda texts,v=None:c.predict_many(texts),
  model_repo=repo,results=args.model,metadata={"d3":{"model":repo,"revision":rev,"device":c.device,"max_options":255,
  "mps_status":"experimental; release defaults to CUDA or CPU","scoring":"255-way FP32 code readout; native normalization; no truncation"}})
if __name__=="__main__": main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=["torch","transformers","safetensors","pandas","httpx"], add_arguments=add_arguments)