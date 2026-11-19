from classifier import D1OmniClassifier, TASK, TASK_LABELS
from probes import check_probes
check_probes(D1OmniClassifier(TASK_LABELS, TASK.instruction).predict_many, name="d1-omni probe check")