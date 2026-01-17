from dataclasses import dataclass, field
import numpy as np
import math

@dataclass
class Attention:
    embedding_dim: int
    sequence_length: int
    num_heads: int
    d_k: int = 0
    _q_weights: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    _k_weights: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    _v_weights: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))

    def _assign_qkv(self):
        """Initialize QKV weights"""
        self.d_k = self.embedding_dim // self.num_heads
        bound = math.sqrt(6 / (self.embedding_dim + self.d_k))
        
        self._q_weights = np.random.uniform(-bound, bound, (self.embedding_dim, self.sequence_length))
        self._k_weights = np.random.uniform(-bound, bound, (self.embedding_dim, self.sequence_length))
        self._v_weights = np.random.uniform(-bound, bound, (self.embedding_dim, self.sequence_length))

    def _softmax(self) -> np.ndarray:

        return np.array([])

    def _attention(self) -> np.ndarray:

        return np.array([])
    
    @property
    def q_weights(self):
        return self._q_weights
    
    @property
    def k_weights(self):
        return self._k_weights
    
    @property
    def v_weights(self):
        return self._v_weights
    