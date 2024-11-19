import numpy as np
def mse_loss(predictions, actual):
    return np.mean((predictions - actual) ** 2)
