import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    minn = np.full(D, np.inf)
    maxx = np.full(D, -np.inf)

    return {'minn':minn, 'maxx':maxx}

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X_batch = np.asarray(X_batch, dtype = float)
    minn = np.min(X_batch, axis = 0, keepdims = True)
    maxx = np.max(X_batch, axis = 0, keepdims = True)
    state['minn'] = np.minimum(state['minn'], minn)
    state['maxx'] = np.maximum(state['maxx'], maxx)

    rangee = state['maxx'] - state['minn']
    rangee[rangee == 0] = eps

    return (X_batch - state['minn']) / rangee