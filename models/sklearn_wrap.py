from sklearn.linear_model import Lasso as SkLasso
import numpy as np
from models.base_model import Model, PrivateAttr

class Lasso(Model):
    _lasso_model: SkLasso = PrivateAttr()

    def __init__(self) -> None:
        super().__init__()
        self._lasso_model = SkLasso()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        self._lasso_model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._lasso_model.predict(observations)