from dataclasses import dataclass, field
from typing import Union, List, Dict
import numpy as np
import random

@dataclass
class TokenLookupMatrix:
    vector_length: int = 3
    _token_matrix: List = field(default_factory=list)
    token_map: Dict[str, int] = field(default_factory=dict)

    def _vectorize(self) -> List:
        return [random.uniform(-1, 1) for _ in range(self.vector_length)]

    def _unique_vector(self) -> List:
        vectorized_token = self._vectorize()
        while vectorized_token in self._token_matrix:
            vectorized_token = self._vectorize()
        return vectorized_token
    
    @property
    def token_matrix(self) -> np.ndarray:
        return np.array(self._token_matrix)

    def add_token(self, token) -> bool:
        self._token_matrix.append(self._unique_vector())
        self.token_map[token] = len(self._token_matrix) - 1
        return True
        
    def contains_token(self, token: str) -> bool:
        return token in self.token_map.keys()