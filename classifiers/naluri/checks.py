from classifier import NaluriClassifier,TASK,TASK_LABELS
from probes import check_probes
for v in ("279m","308m"): check_probes(NaluriClassifier(TASK_LABELS,TASK.instruction,variant=v).predict_many,name=f"Naluri {v} probes")