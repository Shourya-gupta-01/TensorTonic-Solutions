import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)

    if x.ndim == 1:
        numerator = np.exp(x - np.max(x))
        denominator = float(np.sum(numerator))
        return numerator / denominator

    if x.ndim == 2:
        numerator = np.exp(x - np.max(x, axis = 1, keepdims = True))
        denominator = np.sum(numerator, axis = 1, keepdims = True)
        return numerator / denominator