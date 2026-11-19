"""Evaluate Kev-0.8B (in-domain trained on AG News and Banking77)."""
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
 c=KevClassifier(TASK_LABELS,ctx.task.instruction)
 return Setup(model=f"{MODEL_REPO}@{MODEL_REVISION[:7]} on {c.device}",classifier=c,sequential=True,predict=lambda texts,v=None:c.predict_many(texts),model_repo=BASE_REPO,
  metadata={"kev":{"model":MODEL_REPO,"revision":MODEL_REVISION,"base_model":BASE_REPO,"base_revision":BASE_REVISION,"code_commit":CODE_COMMIT,"device":c.device,
  "validated_context":MAX_BRANCH_TOKENS,"max_state_tokens":MAX_STATE_TOKENS,"scoring":"release LoRA and FP32 pointer head; shipped temperature"}})
if __name__=="__main__": main(adapter, app_dir=APP_DIR, doc=__doc__, key_packages=["kev","torch","transformers","peft","mlx","mlx-lm","pandas","httpx"])