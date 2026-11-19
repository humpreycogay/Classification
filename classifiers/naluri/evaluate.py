"""Evaluate a Naluri encoder (default 308M; --model 279m|308m)."""
import os, sys
from pathlib import Path
APP_DIR=Path(__file__).resolve().parent
SHARED_DIR = Path(os.environ.get("BENCHMARK_SHARED_DIR") or APP_DIR.parents[1] / "shared")
sys.path.insert(0, str(SHARED_DIR))
from task import select_task_from_argv
select_task_from_argv()
from classifier import HEAD_MAX_LENGTH, MAX_LENGTH, MODELS, TASK_LABELS, NaluriClassifier
from harness import Setup, main
def add_arguments(p): p.add_argument("--model", choices=MODELS, default="308m")
def adapter(args, ctx):
    c=NaluriClassifier(TASK_LABELS, ctx.task.instruction, variant=args.model)
    repo, rev=MODELS[args.model]
    return Setup(model=f"{repo}@{rev[:7]} on {c.device}", classifier=c, sequential=True,
      predict=lambda texts,v=None:c.predict_many(texts), model_repo=repo, results=args.model,
      metadata={"naluri":{"model":repo,"revision":rev,"device":c.device,"max_length":MAX_LENGTH,
      "head_max_length":HEAD_MAX_LENGTH,"license":"CC-BY-NC-4.0","scoring":"mask-marker pointer head; release temperature"}})
if __name__=="__main__": main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=["torch","transformers","safetensors","pandas","httpx"], add_arguments=add_arguments)