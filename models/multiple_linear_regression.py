import numpy as np
from models.base_model import Model

class MultipleLinearRegression(Model):
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        # Ensure that observations and ground_truth are NumPy arrays of type float64
        X = np.hstack((np.ones((observations.shape[0], 1)), np.array(observations, dtype=np.float64)))
        y = np.array(ground_truth, dtype=np.float64)  # Ensure y is a NumPy array of floats

        # Transpose X
        X_transpose = np.transpose(X)
        
        # Perform the normal equation
        self._parameters["coefficients"] = np.linalg.pinv(X_transpose @ X) @ X_transpose @ y

    def predict(self, observations: np.ndarray) -> np.ndarray:
        # Add the bias term (ones column) and predict
        X = np.hstack((np.ones((observations.shape[0], 1)), np.array(observations, dtype=np.float64)))
        coefficients = self._parameters["coefficients"]
        prediction = np.dot(X, coefficients)

        return prediction
