from classifier import DexyClassifier,TASK,TASK_LABELS
from probes import check_probes
check_probes(DexyClassifier(TASK_LABELS,TASK.instruction).predict_many,name="Dexy probes")