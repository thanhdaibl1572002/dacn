import numpy as np
def cross_entropy_loss(predictions, actual):
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(actual * np.log(predictions) + (1 - actual) * np.log(1 - predictions))