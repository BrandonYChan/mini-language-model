from dataclasses import dataclass, field
from typing import List, Optional
from token_embedding import TokenEncoder
from position_embedding import PositionEncoder
import numpy as np

@dataclass
class SequenceEncoder:
    embedding_dim: int
    sequence: List[str]
    token_encoder: Optional[TokenEncoder] = None
    position_encoder: Optional[PositionEncoder] = None
    _sequence_matrix: np.ndarray = field(default_factory = lambda: np.array([]).reshape(0, 0))

    def __post_init__(self) -> None:
        self.token_encoder = TokenEncoder(self.embedding_dim)
        self.position_encoder = PositionEncoder(self.embedding_dim, len(self.sequence))

        for token in self.sequence:
            self.token_encoder.add_token(token)

    @property
    def sequence_matrix(self) -> np.ndarray:
        return self._sequence_matrix

