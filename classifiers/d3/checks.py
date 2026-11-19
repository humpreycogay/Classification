from classifier import D3Classifier,TASK,TASK_LABELS
from probes import check_probes
check_probes(D3Classifier(TASK_LABELS,TASK.instruction,"edge").predict_many,name="d3-edge probes")