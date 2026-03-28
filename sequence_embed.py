from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

@dataclass
class SequenceEncoder:
    """Encode entire sequence of tokens"""
    _embedding_dim: int
    _sequence: List[str]
    _embedding_matrix: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    _token_matrix: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    _position_matrix: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    _sequence_matrix: np.ndarray = field(default_factory = lambda: np.array([]).reshape(0, 0))
    _token_map: Dict[str, int] = field(default_factory=dict)
    
    def _add_token(self, token: str) -> None:
        """Add token to embedding matrix"""
        if not token in self._token_map:
            vector = np.random.uniform(-1, 1, size = self._embedding_dim)
            self._token_matrix = np.vstack([self._token_matrix, [vector]])
            self._sequence_matrix = np.vstack([self._sequence_matrix, [vector]])
            self._token_map[token] = len(self._token_matrix) - 1
        else:
            token_vector = self._token_matrix[self._token_map[token]] 
            self._sequence_matrix = np.vstack([self._sequence_matrix, [token_vector]])

    def _add_position_encodings(self, sequence_length: int) -> None:
        """Generate sinusoidal positional encodings for positions 0 to num_positions-1."""
        positions = np.arange(sequence_length).reshape(-1, 1)
        dimensions = np.arange(self._embedding_dim).reshape(1, -1)
        exponent = 2 * (dimensions // 2) / self._embedding_dim
        denominator = 10000 ** exponent

        encoding = np.zeros((sequence_length, self._embedding_dim))
        encoding[:, 0::2] = np.sin(positions / denominator[:, 0::2])
        encoding[:, 1::2] = np.cos(positions / denominator[:, 1::2])
        
        self._position_matrix = encoding
    
    def __post_init__(self) -> None:
        """Populate encoding matrices"""
        self._add_position_encodings(len(self._sequence))
        for token in self._sequence:
            self._add_token(token)
        self._embedding_matrix = np.add(self._sequence_matrix, self._position_matrix)
    
    @property
    def sequence(self):
        return self._sequence
    
    @property
    def embedding_dim(self):
        return self._embedding_dim
    
    @property
    def sequence_length(self):
        return len(self._sequence)
    
    @property
    def embedding_matrix(self):
        return self._embedding_matrix
    
    @property
    def position_matrix(self):
        return self._position_matrix

    @property
    def token_matrix(self) -> np.ndarray:
        return self._token_matrix
    
    @property
    def sequence_matrix(self) -> np.ndarray:
        return self._sequence_matrix

    @property
    def token_map(self) -> Dict[str, int]:
        return self._token_map