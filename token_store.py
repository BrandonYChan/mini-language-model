from dataclasses import dataclass
from typing import Union, List, Dict
import numpy as np
import random

@dataclass
class TokenLookupTable:
    vector_length:int = 3
    lookup_table: Dict[int, np.ndarray] = {}
    token_map: Dict[str, int] = {}

    def _vectorize(self) -> np.ndarray:
        return np.array([random.uniform(-1, 1) for i in range(self.vector_length)])

    def _unique_vector(self) -> np.ndarray:
        vectorized_token = self._vectorize()
        while vectorized_token in self.lookup_table.values():
            vectorized_token = self._vectorize()
        return vectorized_token

    def add(self, token) -> bool:
        max_id = max(self.lookup_table.keys()) if len(self.lookup_table.keys()) > 0 else -1
        self.lookup_table[max_id + 1] = self._unique_vector()
        self.token_map[token] = max_id
        return True
        
    def contains(self, token: str) -> bool:
        return token in self.token_map.keys()