from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import random

@dataclass
class TokenEncoder:
    embedding_dim: int
    _token_matrix: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    _sequence_matrix: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    token_map: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        # Initialize matrices with correct shape
        self._token_matrix = np.array([]).reshape(0, self.embedding_dim)
        self._sequence_matrix = np.array([]).reshape(0, self.embedding_dim)

    def _vectorize(self) -> List[float]:
        return [random.uniform(-1, 1) for _ in range(self.embedding_dim)]

    def _get_embedding(self, token: str) -> np.ndarray:
        idx = self.token_map[token]
        return self._token_matrix[idx]

    @property
    def token_lookup_matrix(self) -> np.ndarray:
        return self._token_matrix
    
    @property
    def token_sequence_matrix(self) -> np.ndarray:
        return self._sequence_matrix

    def add_token(self, token: str) -> None:
        """Add token to embedding matrix"""
        if not token in self.token_map:
            vector = self._vectorize()
            self._token_matrix = np.vstack([self._token_matrix, [vector]])
            self._sequence_matrix = np.vstack([self._sequence_matrix, [vector]])
            self.token_map[token] = len(self._token_matrix) - 1
        else:
            token_vector = self._get_embedding(token)
            self._sequence_matrix = np.vstack([self._sequence_matrix, token_vector])
