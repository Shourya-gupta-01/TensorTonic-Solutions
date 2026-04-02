import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    proportions = counts.astype('float64') / len(y)
    return -np.sum(proportions * np.log2(proportions))