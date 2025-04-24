import numpy as np
from .bitstring import BitString

class MonteCarlo:
    def __init__(self, hamiltonian):
        """
        Initialize Monte Carlo simulation with an Ising Hamiltonian.
        """
        self.hamiltonian = hamiltonian
        self.N = len(hamiltonian.G.nodes)  # Corrected attribute name
        self.config = self.initialize_configuration()

    def initialize_configuration(self):
        """
        Initialize a random spin configuration.
        """
        return BitString(self.N)  # Assuming BitString can initialize randomly

    def metropolis_step(self, T):
        """
        Perform a single Metropolis update step.
        """
        for _ in range(self.N):  # Loop over all sites
            flip_index = np.random.randint(0, self.N)  # Choose a site to flip
            new_config = self.config.copy()
            new_config.flip_site(flip_index)  # Flip a single spin

            # Compute energy change
            E_old = self.hamiltonian.energy(self.config)
            E_new = self.hamiltonian.energy(new_config)
            dE = E_new - E_old

            # Accept or reject the new configuration
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                self.config = new_config  # Accept move

    def run(self, T, n_samples=10000, n_burn=1000):
        """
        Run Monte Carlo simulation.

        :param T: Temperature
        :param n_samples: Number of samples to collect
        :param n_burn: Burn-in period
        :return: Arrays of energies and magnetizations
        """
        energies = []
        magnetizations = []

        # Burn-in phase
        for _ in range(n_burn):
            self.metropolis_step(T)

        # Sampling phase
        for _ in range(n_samples):
            self.metropolis_step(T)
            E = self.hamiltonian.energy(self.config)
            M = sum([1 if bit == 1 else -1 for bit in self.config.config])  # Magnetization
            energies.append(E)
            magnetizations.append(M)

        return np.array(energies), np.array(magnetizations)
