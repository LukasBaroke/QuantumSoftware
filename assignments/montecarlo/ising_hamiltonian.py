import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .bitstring import BitString



class IsingHamiltonian:
    def __init__(self, G: nx.Graph):
        """
        Initialize the Ising Hamiltonian with a given graph.
        
        :param G: NetworkX graph where edges represent interactions.
        """
        self.G = G
        self.mus = None  # External field values
        self.J = np.zeros((len(G.nodes), len(G.nodes)))  # Interaction matrix
        
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            self.J[u, v] = weight
            self.J[v, u] = weight  # Ensure symmetry
 
    def energy(self, config: BitString) -> float:
        """
        Compute the energy of a given spin configuration.
        
        :param config: A BitString representing the spin configuration.
        :return: Energy of the given configuration.
        """
        energy = 0.0

        # Iterate over edges in the graph
        for u, v, data in self.G.edges(data=True):
            weight = data.get('weight', 1)  # Default weight is 1 if not specified
            spin_u = 1 if config.config[u] == 1 else -1
            spin_v = 1 if config.config[v] == 1 else -1
            energy += weight * spin_u * spin_v  # Interaction term

        # External field term
        if self.mus is not None:
            energy += np.dot(self.mus, np.where(config.config == 1, 1, -1))

        return energy




    def set_mu(self, mus: np.array):
        """
        Set the external field values.
        
        :param mus: Numpy array of external field values.
        """
        if len(mus) != len(self.G.nodes()):
            raise ValueError("Length of mus must match number of nodes in G.")
        self.mus = mus



    def compute_average_values(self, T: float):
        """
        Compute average values such as magnetization and energy over a given temperature.
        
        :param T: Temperature value.
        :return: Tuple containing (avg_energy, avg_magnetization, heat_capacity, magnetization_susceptibility)
        """
        if T <= 0:
            raise ValueError("Temperature must be positive.")

        num_nodes = len(self.G.nodes())
        config = BitString(num_nodes)

        # Lists to store energy and magnetization values
        energies = []
        magnetizations = []
        weights = []

        # Compute all possible configurations
        for i in range(2**num_nodes):
            config.set_integer_config(i)
            E = self.energy(config)
            M = sum([1 if bit == 1 else -1 for bit in config.config])
            
            energies.append(E)
            magnetizations.append(M)
            weights.append(-E / T)

        # Apply log-sum-exp trick for numerical stability
        max_weight = max(weights)  # Find the max exponent value to shift the weights
        weights = np.exp(np.array(weights) - max_weight)  # Shift and exponentiate
        partition_function = np.sum(weights)

        # Compute expectation values
        avg_energy = np.sum(np.array(energies) * weights) / partition_function
        avg_magnetization = np.sum(np.array(magnetizations) * weights) / partition_function

        # Compute squared expectation values
        energy_squared = np.sum(np.array(energies)**2 * weights) / partition_function
        magnetization_squared = np.sum(np.array(magnetizations)**2 * weights) / partition_function

        # Heat capacity: C = (⟨E²⟩ - ⟨E⟩²) / T²
        heat_capacity = (energy_squared - avg_energy**2) / T**2

        # Magnetization susceptibility: χ = (⟨M²⟩ - ⟨M⟩²) / T
        magnetization_susceptibility = (magnetization_squared - avg_magnetization**2) / T

        return avg_energy, avg_magnetization, heat_capacity, magnetization_susceptibility


