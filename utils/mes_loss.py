import numpy as np
def mse_loss(predictions, actual):
    return np.mean((actual - predictions) ** 2)
