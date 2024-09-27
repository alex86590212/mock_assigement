from collections import Counter

from models.base_model import Model
from pydantic import Field, field_validator
import copy

import numpy as np

class KNearestNeighbors(Model):
    k: int = Field(title="Number of neighbors", default=3)

    @field_validator("k")
    def k_greater_than_zero(cls, value):
        if value <= 0:
            raise ValueError("k must be greater than 0")

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        self._parameters = {"observations": observations, "ground_truth": ground_truth}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        if self._parameters is None:
            raise ValueError("Model not fitted. Call 'fit' with arguments")

        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observations: np.ndarray) -> any:
        distances = np.linalg.norm(
            observations - self._parameters["observations"], axis=1
        )
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[: self.k]
        k_nearest_labels = [self._parameters["ground_truth"][i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    @property
    def parameters(self):
        return copy.deepcopy(self._parameters)