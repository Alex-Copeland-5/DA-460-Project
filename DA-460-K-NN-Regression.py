# Alex Copleand
# DA-460
# K-NN Script

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===== USER CONFIGURATION =====
EXCEL_PATH = "C:\\Users\\Alex\\Desktop\\DA Project Take 2\\skyblock_dataset.xlsx"           # put the file in the same folder as this .py
SHEET_NAME = "Skyblock"                        # the sheet with your data
TARGET_COLUMN = "Networth"                     # outcome column
K_VALUES = [1,3,5,7,9,11,13,15,17,19]                 # the k value you want to use
BANDWIDTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # the b values you want to compare
# ==============================

def parse_networth(value):
    """
    Convert strings like '1.8B' or '182.5M' to billions.
    Example:
      '1.8B'  -> 1.8
      '182.5M' -> 0.1825
    """
    if isinstance(value, str):
        s = value.strip().upper()
        if s.endswith("B"):
            return float(s[:-1])
        elif s.endswith("M"):
            return float(s[:-1]) / 1000.0
        else:
            # if it's plain numeric, convert to billions
            v = float(s)
            return v / 1e9
    return float(value) / 1e9


def load_data(path, sheet_name, target_column):
    # header=1 → the real header is on the second row
    df = pd.read_excel(path, sheet_name=sheet_name, header=1)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Columns: {list(df.columns)}")

    # Convert Networth → billions
    if target_column == "Networth":
        df[target_column] = df[target_column].apply(parse_networth)

    y = df[target_column].astype(float).to_numpy()

    # X = numeric columns except target
    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=[np.number])

    return X.to_numpy(), y


def compute_distance_matrix(X):
    """Euclidean distance matrix"""
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            dist = math.sqrt(np.dot(diff, diff))
            D[i, j] = dist
            D[j, i] = dist

    return D


def loocv_knn_regression(distance_matrix, y, k, bandwidth):
    n = len(y)
    total_sse = 0.0

    for test_idx in range(n):
        dists = distance_matrix[test_idx].copy()
        dists[test_idx] = np.inf

        nn_idx = np.argsort(dists)[:k]
        nn_dists = dists[nn_idx]
        nn_y = y[nn_idx]

        weights = np.exp(-nn_dists / bandwidth)
        weighted_sum = np.sum(weights * nn_y)
        weight_total = np.sum(weights)

        if weight_total == 0:
            y_pred = np.mean(nn_y)
        else:
            y_pred = weighted_sum / weight_total

        total_sse += (y[test_idx] - y_pred) ** 2

    return total_sse


def predict_all(distance_matrix, y, k, bandwidth):
    """Predict every observation using LOOCV for the BEST model."""
    n = len(y)
    predictions = np.zeros(n)

    for test_idx in range(n):
        dists = distance_matrix[test_idx].copy()
        dists[test_idx] = np.inf

        nn_idx = np.argsort(dists)[:k]
        nn_dists = dists[nn_idx]
        nn_y = y[nn_idx]

        weights = np.exp(-nn_dists / bandwidth)
        weighted_sum = np.sum(weights * nn_y)
        weight_total = np.sum(weights)

        if weight_total == 0:
            y_pred = np.mean(nn_y)
        else:
            y_pred = weighted_sum / weight_total

        predictions[test_idx] = y_pred

    return predictions


def main():
    # Load data
    X, y = load_data(EXCEL_PATH, SHEET_NAME, TARGET_COLUMN)
    D = compute_distance_matrix(X)

    print(f"Loaded {X.shape[0]} observations and {X.shape[1]} predictors.\n")

    # Find best (k, b)
    best_k = None
    best_b = None
    best_sse = None

    for k in K_VALUES:
        print(f"=== Testing k = {k} ===")
        for b in BANDWIDTHS:
            sse = loocv_knn_regression(D, y, k, b)
            print(f"  b = {b}: SSE = {sse:.6f}")
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_k = k
                best_b = b
        print()

    print(f"\nBest model: k = {best_k}, b = {best_b}, SSE = {best_sse:.6f}\n")

    # Compute predictions using BEST MODEL
    preds = predict_all(D, y, best_k, best_b)

    # Print Actual vs Predicted
    print("=== Actual vs Predicted (in Billions) ===")
    for i in range(len(y)):
        print(f"Obs {i+1}: Actual = {y[i]:.4f}B   Predicted = {preds[i]:.4f}B   Error = {y[i] - preds[i]:.4f}B")

    # Plot error distribution
    errors = y - preds
    plt.hist(errors, bins=30, edgecolor='black')
    plt.title("Prediction Error Distribution (in Billions)")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")

    plt.show()


if __name__ == "__main__":
    main()
