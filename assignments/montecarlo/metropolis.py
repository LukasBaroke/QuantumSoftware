import numpy as np
from .bitstring import BitString


class Metropolis:
    def __init__(self, beta: float):
        """
        Initialize Metropolis algorithm.
        
        :param beta: Inverse temperature (1/kT)
        """
        self.beta = beta

    def step(self, energy_fn, config):
        """
        Perform a single Metropolis update.

        :param energy_fn: Function to compute energy of a configuration
        :param config: Current configuration
        :return: New configuration after update
        """
        new_config = config.copy()
        idx = np.random.randint(len(config))
        new_config[idx] *= -1  # Flip a spin

        delta_E = energy_fn(new_config) - energy_fn(config)
        if delta_E < 0 or np.random.rand() < np.exp(-self.beta * delta_E):
            return new_config
        return config
