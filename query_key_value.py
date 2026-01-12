from dataclasses import dataclass, field
import numpy as np
import math
import random

@dataclass
class QKV:
    """Query, Key, Value matrix weight class"""
    embedding_dim: int
    sequence_length: int
    num_heads: int
    d_k: int = 0
    _q_weights: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    _k_weights: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    _v_weights: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))

    def __post_init__(self):
        self.d_k = self.embedding_dim // self.num_heads
        bound = math.sqrt(6 / (self.embedding_dim + self.d_k))
        
        self._q_weights = np.random.uniform(-bound, bound, (self.embedding_dim, self.sequence_length))
        self._k_weights = np.random.uniform(-bound, bound, (self.embedding_dim, self.sequence_length))
        self._v_weights = np.random.uniform(-bound, bound, (self.embedding_dim, self.sequence_length))

    @property
    def q_weights(self):
        return self._q_weights
    
    @property
    def k_weights(self):
        return self._k_weights
    
    @property
    def v_weights(self):
        return self._v_weights
    