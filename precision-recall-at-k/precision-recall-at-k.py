import numpy as np

def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended = np.asarray(recommended)
    relevant = np.asarray(relevant)
    top_k = recommended[:k]
    count = len(np.intersect1d(top_k, relevant))
    precision = count / float(k)
    recall = count / float(len(relevant))
    return [precision, recall]
    