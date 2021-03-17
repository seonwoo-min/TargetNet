# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import numpy as np


def compute_confusion_matrices(outputs, labels):
    """
    Compute a binary confusion matrix
      [TN_k FN_k]
      [FP_k TP_k]
    """

    A = np.zeros((2, 2))
    for i in range(len(labels)):
        if labels[i] == 1 and outputs[i] == 1: # TP
            A[1, 1] += 1
        elif labels[i] == 0 and outputs[i] == 1: # FP
            A[1, 0] += 1
        elif labels[i] == 1 and outputs[i] == 0: # FN
            A[0, 1] += 1
        elif labels[i] == 0 and outputs[i] == 0: # TN
            A[0, 0] += 1

    return A


def compute_metrics(labels, outputs):
    """ compute set-wise evaluation metrics """
    A = compute_confusion_matrices(labels, outputs)
    tp, fp, fn, tn = A[1, 1], A[0, 1], A[1, 0], A[0, 0]

    acc = float(tp + tn) / float(tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0
    pr = float(tp) / float(tp + fp) if tp + fp > 0 else 0
    re = float(tp) / float(tp + fn) if tp + fn > 0 else 0
    ne = float(tn) / float(fn + tn) if fn + tn > 0 else 0
    sp = float(tn) / float(fp + tn) if fp + tn > 0 else 0
    f1 = float(2 * tp) / float(2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0

    return tp, fp, fn, tn, acc, pr, re, ne, sp, f1
