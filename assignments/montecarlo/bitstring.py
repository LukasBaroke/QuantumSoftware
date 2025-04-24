import numpy as np

class BitString:
    """
    Simple class to implement a config of bits
    """
    def __init__(self, N):
        self.N = N
        self.config = np.zeros(N, dtype=int) 

    def __repr__(self):
        return "".join(map(str, self.config))

    def __eq__(self, other):        
        return np.array_equal(self.config, other.config)
    
    def __len__(self):
        return len(self.config)

    def on(self):
        """Return number of bits that are on"""
        return np.sum(self.config)

    def off(self):
        """Return number of bits that are off"""
        return len(self.config) - self.on()

    def flip_site(self, i):
        """Flip the bit at site i"""
        if i < 0 or i >= self.N:
            raise ValueError(f"Index i={i} out of bounds for BitString of size {self.N}")
        self.config[i] = 1 - self.config[i]

    def integer(self):
        """Return the decimal integer corresponding to BitString"""
        return int("".join(map(str, self.config)), 2)

    def set_config(self, s: list[int]):
        """Set the config from a list of integers"""
        if len(s) != self.N:
            raise ValueError("Input list must be of length N")
        self.config = np.array(s, dtype=int)
        
    def set_integer_config(self, dec: int):
        """
        Convert a decimal integer to binary and set it as the configuration.
        """
        if dec >= 2**self.N:
            raise ValueError(f"Decimal value {dec} exceeds the range for a bitstring of length {self.N}")
        binary_rep = format(dec, f'0{self.N}b')  # Convert to binary with leading zeros
        self.config = np.array([int(bit) for bit in binary_rep], dtype=int)

    def copy(self):
        """Return a copy of the current BitString."""
        new_bitstring = BitString(self.N)
        new_bitstring.config = self.config.copy()  # Create a copy of the numpy array
        return new_bitstring
