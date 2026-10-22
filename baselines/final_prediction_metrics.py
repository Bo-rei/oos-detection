"""Common full-test metrics for the final Ours and official MOGB predictions."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def prediction_metrics(gold, predicted, known):
    gold, predicted = np.asarray(gold), np.asarray(predicted)
    if gold.shape != predicted.shape or not len(gold):
        raise ValueError('Missing or misaligned predictions')
    truth, rejected = gold == '__oos__', predicted == '__oos__'
    return dict(oos_f1=float(f1_score(truth, rejected, zero_division=0)),
        known_f1=float(f1_score(gold, predicted, labels=list(known), average='macro', zero_division=0)),
        accuracy=float(accuracy_score(gold, predicted)),
        oos_precision=float(precision_score(truth, rejected, zero_division=0)),
        oos_recall=float(recall_score(truth, rejected, zero_division=0)),
        known_recall=float((~rejected[~truth]).mean()),
        false_accept_rate=float((~rejected[truth]).mean()),
        known_count=int((~truth).sum()), oos_count=int(truth.sum()),
        oos_false_accepts=int((truth & ~rejected).sum()),
        known_false_rejects=int((~truth & rejected).sum()))
