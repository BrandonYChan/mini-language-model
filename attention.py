from dataclasses import dataclass
import numpy as np
from query_key_value import QKV
import math

@dataclass
class Attention:
    qkv: QKV

    def _softmax(self) -> np.ndarray:

        return np.array([])


    def _attention(self) -> np.ndarray:

        return np.array([])