import numpy as np
from montecarlo.metropolis import Metropolis

def test_metropolis():
    def energy(config):
        return -np.sum(config)

    metropolis = Metropolis(beta=1.0)
    config = np.array([1, -1, 1, -1])
    new_config = metropolis.step(energy, config)

    assert len(new_config) == len(config)
