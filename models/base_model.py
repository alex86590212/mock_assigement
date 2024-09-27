from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, PrivateAttr


class Model(BaseModel, ABC):
    _parameters: dict = PrivateAttr(default_factory=dict)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass
