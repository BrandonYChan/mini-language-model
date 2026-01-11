from dataclasses import dataclass, field
from typing import Annotated
import numpy as np

@dataclass
class PositionEncoder:
    embedding_dim: Annotated[int, "Must be divisible by 2 (even number)"]
    sequence_length: Annotated[int, "Number of tokens in the sequence"]
    _position_matrix: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    
    def __post_init__(self):
        self._add_positions(self.sequence_length)

    @property
    def position_matrix(self):
        return self._position_matrix
    
    def _add_positions(self, num_positions: int) -> None:
        """Generate positional encodings for positions 0 to num_positions-1."""
        positions = np.arange(num_positions).reshape(-1, 1)
        dimensions = np.arange(self.embedding_dim).reshape(1, -1)
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        exponent = 2 * (dimensions // 2) / self.embedding_dim
        denominator = 10000 ** exponent
        
        encoding = np.zeros((num_positions, self.embedding_dim))
        encoding[:, 0::2] = np.sin(positions / denominator[:, 0::2])
        encoding[:, 1::2] = np.cos(positions / denominator[:, 1::2])
        
        self._position_matrix = encoding