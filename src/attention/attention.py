from dataclasses import dataclass, field
import numpy as np
from src.sequence_embed.sequence_embed import SequenceEncoder
from typing import Dict

@dataclass
class Attention:
    sequence_embedding: SequenceEncoder
    num_heads: int
    d_k: int = 0
    qkv: Dict = field(default_factory=lambda: 
                          {'q': np.array([]).reshape(0, 0), 'q_weights': np.array([]).reshape(0, 0), 
                          'k': np.array([]).reshape(0, 0), 'k_weights': np.array([]).reshape(0, 0), 
                          'v': np.array([]).reshape(0, 0), 'v_weights': np.array([]).reshape(0, 0)
                          })

    def __post_init__(self):
        self._init_qkv()

    def _init_qkv(self):
        self.d_k = self.sequence_embedding.embedding_dim // self.num_heads
        bound = np.sqrt(6 / (self.sequence_embedding.embedding_dim + self.d_k))
        embedding_dim = self.sequence_embedding.embedding_dim
        sequence_length = self.sequence_embedding.sequence_length

        for prefix in ['q', 'k', 'v']:
            self.qkv[f"{prefix}_weights"] = np.random.uniform(-bound, bound, (embedding_dim , sequence_length))
            self.qkv[prefix] = self.sequence_embedding.embedding_matrix @ self.qkv[f"{prefix}_weights"]

    def _softmax(self, logits) -> np.ndarray:
        results = np.array([])
        for logit in logits:
            result = np.exp(logit) / np.sum(np.exp(logits))
            results = np.vstack([results, result])
        return results

    def _attention(self) -> np.ndarray:
        logits = self.qkv['q'] @ np.transpose(self.qkv['k']) / np.sqrt(self.d_k)
        softmax_result = self._softmax(logits)
        attention_result = softmax_result @ self.qkv['v']
        return attention_result
    