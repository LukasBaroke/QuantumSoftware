import numpy as np

def compute_observable(config):
    """
    Compute an observable (e.g., magnetization).
    
    :param config: Spin configuration
    :return: Magnetization value
    """
    return np.sum(config) / len(config)
