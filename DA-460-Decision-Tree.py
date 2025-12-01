import math
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split


# ================== USER CONFIGURATION ==================
EXCEL_PATH = "C:\\Users\\Alex\\Desktop\\DA Project Take 2\\skyblock_dataset.xlsx"   # Excel file in the SAME folder as this script
SHEET_NAME = "Skyblock"               # Sheet with your Skyblock data
TARGET_COLUMN = "Networth"            # Numeric outcome we want to predict
# Tree hyperparameters
MAX_DEPTH = None        # None = full depth; or try 3, 4, 5, etc.
MIN_SAMPLES_LEAF = 1    # Minimum number of samples per leaf
MIN_SAMPLES_SPLIT = 2   # Minimum samples required to split an internal node
CCP_ALPHA = 0.0         # Cost-complexity pruning (post-pruning). >0 prunes more.
RANDOM_STATE = 42       # For reproducibility
# ========================================================
# Plotting options
PLOT_MAX_DEPTH = None   # e.g., set to 3 to limit displayed depth  
USE_GRAPHVIZ_EXPORT = True  # If True, export a high-quality PNG via Graphviz
PLOT_MARGINS = None
PLOT_FIGSIZE = (34, 16)  # Wider canvas to spread nodes
PLOT_DPI = 100           # Higher DPI for clarity
WRAP_FEATURE_NAMES = False  # Wrap long feature names to multiple lines
WRAP_WIDTH = 12             # Characters per line for wrapping

# Graphviz layout tuning (applies when USE_GRAPHVIZ_EXPORT is True)
GRAPHVIZ_TUNE_LAYOUT = True
GRAPHVIZ_RANKSEP = 1.1   # Vertical spacing between ranks (levels)
GRAPHVIZ_NODESEP = 1.5   # Horizontal spacing between nodes at same level
GRAPHVIZ_SIZE = "100,50"  # Canvas size in inches (width,height) - set to "100,50" for very wide output


def parse_networth(value):
    """
    Convert strings like '1.8B' or '182.5M' to numeric in *billions*.

    Examples:
      '1.8B'   -> 1.8
      '182.5M' -> 0.1825
      '9000000000' -> 9.0
    """
    if isinstance(value, str):
        s = value.strip().upper()
        if s.endswith("B"):
            # '1.8B' -> 1.8
            return float(s[:-1])
        elif s.endswith("M"):
            # '182.5M' -> 0.1825
            return float(s[:-1]) / 1000.0
        else:
            # plain numeric string, assume raw coins and convert to billions
            v = float(s)
            return v / 1e9
    # already numeric: assume raw coins, convert to billions
    return float(value) / 1e9


def load_data(path, sheet_name, target_column):
    """
    Load the Skyblock data:

    - Uses row 2 (index 1) as the header, because the file has two header rows.
    - Converts Networth to *billions*.
    - Returns X (numeric predictors), y (Networth in billions), and feature names.
    """
    # header=1 means: row index 1 is the header row (second line in Excel)
    df = pd.read_excel(path, sheet_name=sheet_name, header=1)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. "
                         f"Available columns: {list(df.columns)}")

    # Convert Networth -> billions
    if target_column == "Networth":
        df[target_column] = df[target_column].apply(parse_networth)

    # y: target as a 1D numpy array
    y = df[target_column].astype(float).to_numpy()

    # X: all numeric predictors except the target (this will drop Username automatically)
    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=[np.number])   # keep only numeric columns

    feature_names = list(X.columns)
    return X.to_numpy(), y, feature_names


def main():
    X, y, feature_names = load_data(EXCEL_PATH, SHEET_NAME, TARGET_COLUMN)
    n, d = X.shape
    print(f"Loaded {n} observations with {d} numeric predictors.")
    print("Outcome: Networth (in BILLIONS of coins).\n")

    # === Train/test split ===
    # 80% train, 20% test – only used to pick the best depth using SSE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=== Depth sweep using SSE on TEST set ===")
    best_depth = None
    best_sse = None

    # Try depths 1 through 10 and compute SSE on the test set
    for depth in range(1, 11):
        tree = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=depth,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_samples_split=MIN_SAMPLES_SPLIT,
            ccp_alpha=CCP_ALPHA,
            random_state=42
        )
        tree.fit(X_train, y_train)
        preds = tree.predict(X_test)
        sse = np.sum((y_test - preds) ** 2)

        print(f"max_depth = {depth:2d}   SSE(test) = {sse:.3f}")

        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_depth = depth

    print(f"\nBest depth by test SSE: {best_depth} (SSE = {best_sse:.3f})\n")

    # === Fit final tree using the best depth on ALL data ===
    final_tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=best_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
        ccp_alpha=CCP_ALPHA,
        random_state=42
    )
    final_tree.fit(X, y)

    # In-sample predictions for reporting (like before)
    y_pred = final_tree.predict(X)
    errors = y - y_pred
    sse_full = np.sum(errors ** 2)

    print("=== Final Decision Tree (using best depth) ===")
    print(f"Chosen max_depth:       {best_depth}")
    print(f"Tree depth (actual):    {final_tree.get_depth()}")
    print(f"Number of leaves:       {final_tree.get_n_leaves()}")
    print(f"Training SSE (all data): {sse_full:.6f}\n")

    # Feature importances
    importances = final_tree.feature_importances_
    print("=== Feature Importances ===")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"{name:16s}: {imp:.4f}")
    print()

    # Actual vs Predicted
    print("=== Actual vs Predicted (Networth in BILLIONS) ===")
    for i in range(n):
        print(
            f"Obs {i+1:3d}: "
            f"Actual = {y[i]:8.4f}B   "
            f"Predicted = {y_pred[i]:8.4f}B   "
            f"Error = {errors[i]:8.4f}B"
        )

    # Optional: plot the tree
    plot_feature_names = feature_names
    if WRAP_FEATURE_NAMES:
        plot_feature_names = [textwrap.fill(n, WRAP_WIDTH) for n in feature_names]

    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plot_tree(
        final_tree,
        feature_names=plot_feature_names,
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=PLOT_MAX_DEPTH
    )
    plt.title(f"Decision Tree for Networth (max_depth={best_depth})")
    plt.tight_layout()
    # Optional precise margins around the axes area
    if PLOT_MARGINS:
        plt.subplots_adjust(**PLOT_MARGINS)
    plt.show()

    # Optional: export a cleaner layout using Graphviz
    if USE_GRAPHVIZ_EXPORT:
        try:
            from sklearn.tree import export_graphviz
            import graphviz
            import os
            
            # Add Graphviz to PATH if not already there
            graphviz_path = r'C:\Program Files\Graphviz\bin'
            if graphviz_path not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + graphviz_path

            dot = export_graphviz(
                final_tree,
                out_file=None,
                feature_names=plot_feature_names,
                filled=True,
                rounded=True,
                precision=3
            )

            # Optionally, inject layout attributes to spread nodes further apart
            if GRAPHVIZ_TUNE_LAYOUT:
                header = "digraph Tree {"
                attrs = f"  graph [ranksep={GRAPHVIZ_RANKSEP}, nodesep={GRAPHVIZ_NODESEP}, size=\"{GRAPHVIZ_SIZE}\"];\n"
                print(f"DEBUG: Looking for '{header}' in DOT file...")
                print(f"DEBUG: Found header: {header in dot}")
                if header in dot:
                    dot = dot.replace(header, header + "\n" + attrs, 1)
                    print(f"DEBUG: Injected attributes: ranksep={GRAPHVIZ_RANKSEP}, nodesep={GRAPHVIZ_NODESEP}, size={GRAPHVIZ_SIZE}")
                # Print first 500 chars of modified DOT to verify
                print("DEBUG: DOT file beginning:")
                print(dot[:500])

            graph = graphviz.Source(dot)
            output_path = "networth_tree"
            graph.render(output_path, format="png", cleanup=True)
            print(f"Graphviz export saved to {output_path}.png")
        except Exception as e:
            print("Graphviz export failed. Install 'graphviz' and 'pydotplus'.")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
