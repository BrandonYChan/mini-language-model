from dataclasses import dataclass
from typing import Union, List, Dict
import numpy as np
import random

@dataclass
class TokenLookupMatrix:
    vector_length: int = 3
    _token_matrix: np.ndarray = np.array([])
    token_map: Dict[str, int] = {}

    def _vectorize(self) -> np.ndarray:
        return np.array([random.uniform(-1, 1) for i in range(self.vector_length)])

    def _unique_vector(self) -> np.ndarray:
        vectorized_token = self._vectorize()
        while vectorized_token in self._token_matrix.values():
            vectorized_token = self._vectorize()
        return vectorized_token
    
    @property
    def token_matrix(self):
        return self._token_matrix

    def add_token(self, token) -> bool:
        self._token_matrix = np.append(self._token_matrix, self._unique_vector())
        self.token_map[token] = len(self._token_matrix) - 1
        return True
        
    def contains_token(self, token: str) -> bool:
        return token in self.token_map.keys()