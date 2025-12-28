from dataclasses import dataclass
from typing import Dict, Annotated
import numpy as np
import math

@dataclass
class PositionEncodingMatrix:
    embedding_dim: Annotated[int, "Must be divisible by 2 (even number)"]
    _position_matrix: np.ndarray = np.array([])
    
    @property
    def position_matrix(self):
        return self._position_matrix
    
    def add_position(self, token_id: int) -> None:
        try:
            index = len(self._position_matrix)
            inner = token_id/(10000**(index/self.embedding_dim))
            encoding = math.cos(inner) if index % 2 == 0 else math.sin(inner)
            np.append(self._position_matrix, encoding)
        except Exception as e:
            raise e