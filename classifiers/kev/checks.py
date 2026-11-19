import sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from classifier import KevClassifier,TASK,TASK_LABELS
APP_DIR=Path(__file__).resolve().parent
SPLITS={"ag_news":("fancyzhx/ag_news","eb185aade064a813bc0b7f42de02595523103ca4"),"banking77":("mteb/banking77","18072d2685ea682290f7b8924d94c62acc19c0b2")}
def overlap():
 import overlap as shared_overlap
 out={}
 for task,(repo,rev) in SPLITS.items():
  path=hf_hub_download(repo,"data/train-00000-of-00001.parquet",repo_type="dataset",revision=rev)
  out[task]=shared_overlap.check(task,{"train":pd.read_parquet(path)["text"].tolist()},note=f"{repo}@{rev[:7]} train split")
 shared_overlap.write(APP_DIR,out)
def probes():
 from probes import check_probes
 check_probes(KevClassifier(TASK_LABELS,TASK.instruction).predict_many,name="Kev probes")
{"overlap":overlap,"probes":probes}[sys.argv[1]]()