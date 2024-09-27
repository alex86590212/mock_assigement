import numpy as np
import pandas as pd
from models.multiple_linear_regression import MultipleLinearRegression
from models.sklearn_wrap import Lasso
from models.k_nearest_neighbors import KNearestNeighbors

def main():
    # Load the dataset
    dp = pd.read_csv("oop-24-25-assignment-1-group-8\data\Real estate.csv")
    
    
    observations = dp.iloc[:, :-1].to_numpy() 
    ground_truth = dp.iloc[:, -1].to_numpy()  
    
    print("=== Multiple Linear Regression ===")
    mlr = MultipleLinearRegression()
    mlr.fit(observations, ground_truth)
    mlr_predictions = mlr.predict(observations)
    print("MLR Predictions:", mlr_predictions)

    # 2. Lasso Regression
    print("\n=== Lasso Regression ===")
    lasso = Lasso()
    lasso.fit(observations, ground_truth)
    lasso_predictions = lasso.predict(observations)
    print("Lasso Predictions:", lasso_predictions)

    # 3. K-Nearest Neighbors (KNN)
    print("\n=== K-Nearest Neighbors (KNN) ===")
    knn = KNearestNeighbors()
    knn.fit(observations, ground_truth)
    knn_predictions = knn.predict(observations)
    print("KNN Predictions:", knn_predictions)

if __name__ == "__main__":
    main()