import numpy as np
def cross_entropy_loss(predictions, actual):
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(actual * np.log(predictions) + (1 - actual) * np.log(1 - predictions))