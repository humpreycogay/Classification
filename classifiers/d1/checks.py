"""Setup-time probe check for LiquidAI d1-3B."""

from classifier import TASK, TASK_LABELS, D1Classifier
from probes import check_probes


if __name__ == "__main__":
    check_probes(D1Classifier(TASK_LABELS, TASK.instruction).predict_many, name="d1 probe check")