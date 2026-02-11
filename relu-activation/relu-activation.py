import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x_arr = np.asarray(x, dtype=float)

    # Ensure scalar becomes shape (1,)
    if x_arr.ndim == 0:
        x_arr = x_arr.reshape(1)

    return np.maximum(0.0, x_arr)