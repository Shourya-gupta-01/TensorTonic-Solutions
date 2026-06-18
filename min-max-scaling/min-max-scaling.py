import numpy as np

def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    data = np.asarray(data)
    minn = np.min(data, axis = 0, keepdims = True)
    maxx = np.max(data, axis = 0, keepdims = True)
    rangee = maxx - minn

    rangee[rangee == 0] = 1.0
    scaled = (data - minn) / rangee
    return scaled.tolist()