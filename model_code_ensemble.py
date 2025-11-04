# networth_decision_tree_ensemble.py
import os
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

# ---------- CONFIG ----------
# Put your Excel path here (same folder or absolute path).
EXCEL_PATH = "./skyblock_dataset.xlsx"
OUTPUT_DIR = "artifacts"
MAX_DEPTH = 5
TEST_SIZE = 0.1
NUM_ITERATIONS = 500  # Number of models to train and average
SAVE_BEST_MODEL = True  # Whether to save the best performing model
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- helpers ---
def parse_networth_to_billions(x):
    """Convert values like '1.8B', '900M', 120000000 into *billions* (float).
        - '1.8B' -> 1.8
        - '900M' -> 0.9
        - 120000000 -> 0.12

    Strings without suffix are assumed to be absolute dollars.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x) / 1e9
    s = str(x).strip().upper().replace(",", "")
    try:
        if s.endswith("B"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1_000  # millions -> billions
        # plain number string => absolute dollars
        return float(s) / 1e9
    except:
        return np.nan


def billions_to_pretty(b):
    """
    Convert a *billions* float back to a pretty label:
    - >= 1.0 -> '{:.1f}B'
    - else   -> '{:.0f}M'
    """
    if pd.isna(b):
        return None
    if b >= 1.0:
        return f"{b:.1f}B"
    m = b * 1000.0  # billions -> millions
    return f"{int(round(m))}M" if abs(m - round(m)) < 0.05 else f"{m:.1f}M"


# --- load data ---
# Many spreadsheets like yours have the real header on row 2 -> header=1.
df = pd.read_excel(EXCEL_PATH, header=1)

# Clean up columns: drop 'Unnamed: ...' and trim names
df = df[[c for c in df.columns if not str(c).lower().startswith("unnamed")]]
df.columns = [str(c).strip() for c in df.columns]

# expected features & target
feature_cols = [
    "SB Level",
    "Combat Level",
    "Farming Level",
    "Fishing Level",
    "Mining Level",
    "Foraging Level",
    "Enchanting Level",
    "Alchemy Level",
]
target_col = "Networth"

missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}\nFound columns: {list(df.columns)}")

# Build X (numeric) and y (networth in billions)
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = df[target_col].apply(parse_networth_to_billions)

# Drop rows with NaNs
mask = X.notna().all(axis=1) & y.notna()
dropped = len(df) - mask.sum()
if dropped > 0:
    print(f"Dropping {dropped} row(s) due to missing values in features/target.")

X = X[mask]
y = y[mask]

print(f"\n=== Running {NUM_ITERATIONS} iterations to average feature importances ===")

# Store results from each iteration
all_importances = []
all_r2_scores = []
all_mae_scores = []
all_models = []
best_r2 = -np.inf
best_model = None
best_model_info = None

for iteration in range(NUM_ITERATIONS):
    # Use different random state for each iteration
    random_state = randint(0, 100000)

    # Split with this iteration's random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)

    # Train model
    model = DecisionTreeRegressor(criterion="squared_error", max_depth=MAX_DEPTH, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae_b = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    all_importances.append(model.feature_importances_)
    all_r2_scores.append(r2)
    all_mae_scores.append(mae_b)

    # Track best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_info = {
            "iteration": iteration + 1,
            "r2": r2,
            "mae": mae_b,
            "random_state": random_state,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    # Progress indicator
    if (iteration + 1) % 10 == 0:
        print(f"Completed {iteration + 1}/{NUM_ITERATIONS} iterations...")

# Calculate averaged feature importances
avg_importances = np.mean(all_importances, axis=0)
std_importances = np.std(all_importances, axis=0)

# Calculate average performance metrics
avg_r2 = np.mean(all_r2_scores)
std_r2 = np.std(all_r2_scores)
avg_mae = np.mean(all_mae_scores)
std_mae = np.std(all_mae_scores)

print(f"\n=== Ensemble Results (averaged over {NUM_ITERATIONS} iterations) ===")
print(f"Average R² Score: {avg_r2:.3f} ± {std_r2:.3f}")
print(f"Average MAE: {avg_mae:.3f} ± {std_mae:.3f} (≈ {billions_to_pretty(avg_mae)} error)")
print(f"Best R² Score: {best_r2:.3f} (iteration {best_model_info['iteration']})")

# Create comprehensive feature importance DataFrame
fi_ensemble = pd.DataFrame(
    {
        "feature": feature_cols,
        "avg_importance": avg_importances,
        "std_importance": std_importances,
        "cv_importance": std_importances / avg_importances,  # coefficient of variation
    }
).sort_values("avg_importance", ascending=False)

# Add confidence intervals (assuming normal distribution)
fi_ensemble["importance_ci_lower"] = fi_ensemble["avg_importance"] - 1.96 * fi_ensemble["std_importance"]
fi_ensemble["importance_ci_upper"] = fi_ensemble["avg_importance"] + 1.96 * fi_ensemble["std_importance"]

print(f"\n=== Averaged Feature Importances (with confidence intervals) ===")
for _, row in fi_ensemble.iterrows():
    print(
        f"{row['feature']:20s}: {row['avg_importance']:.4f} ± {row['std_importance']:.4f} "
        f"(CV: {row['cv_importance']:.2f})"
    )

# Save ensemble feature importances
fi_ensemble_path = os.path.join(OUTPUT_DIR, "networth_feature_importances_ensemble.csv")
fi_ensemble.to_csv(fi_ensemble_path, index=False)
print(f"\nSaved ensemble feature importances -> {fi_ensemble_path}")

# Create detailed results DataFrame with all iterations
detailed_results = pd.DataFrame(
    {"iteration": range(1, NUM_ITERATIONS + 1), "r2_score": all_r2_scores, "mae_score": all_mae_scores}
)

# Add individual feature importances as columns
for i, feature in enumerate(feature_cols):
    detailed_results[f"{feature}_importance"] = [imp[i] for imp in all_importances]

detailed_results_path = os.path.join(OUTPUT_DIR, "networth_detailed_results.csv")
detailed_results.to_csv(detailed_results_path, index=False)
print(f"Saved detailed results from all iterations -> {detailed_results_path}")

# Create visualization of feature importance stability
plt.figure(figsize=(15, 10))

# Plot 1: Feature importances over iterations
plt.subplot(2, 2, 1)
for i, feature in enumerate(feature_cols):
    importances_by_iteration = [imp[i] for imp in all_importances]
    plt.plot(range(1, NUM_ITERATIONS + 1), importances_by_iteration, label=feature, alpha=0.7, linewidth=1)
plt.xlabel("Iteration")
plt.ylabel("Feature Importance")
plt.title("Feature Importance Stability Across Iterations")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, alpha=0.3)

# Plot 2: Average importances with error bars
plt.subplot(2, 2, 2)
y_pos = np.arange(len(feature_cols))
plt.barh(y_pos, fi_ensemble["avg_importance"], xerr=fi_ensemble["std_importance"], capsize=5)
plt.yticks(y_pos, fi_ensemble["feature"])
plt.xlabel("Average Importance")
plt.title("Average Feature Importances (with std dev)")
plt.grid(True, alpha=0.3)

# Plot 3: R² scores over iterations
plt.subplot(2, 2, 3)
plt.plot(range(1, NUM_ITERATIONS + 1), all_r2_scores, "b-", alpha=0.7)
plt.axhline(avg_r2, color="r", linestyle="--", label=f"Average: {avg_r2:.3f}")
plt.xlabel("Iteration")
plt.ylabel("R² Score")
plt.title("Model Performance Across Iterations")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Coefficient of variation for importances
plt.subplot(2, 2, 4)
cv_data = fi_ensemble.sort_values("cv_importance")
plt.barh(range(len(cv_data)), cv_data["cv_importance"])
plt.yticks(range(len(cv_data)), cv_data["feature"])
plt.xlabel("Coefficient of Variation")
plt.title("Feature Importance Stability (lower = more stable)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
ensemble_plot_path = os.path.join(OUTPUT_DIR, "networth_ensemble_analysis.png")
plt.savefig(ensemble_plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved ensemble analysis plot -> {ensemble_plot_path}")

if SAVE_BEST_MODEL:
    # Save the best performing model and its artifacts
    print(f"\n=== Best Model Details (Iteration {best_model_info['iteration']}) ===")
    print(f"R² Score: {best_model_info['r2']:.3f}")
    print(f"MAE: {best_model_info['mae']:.3f}")

    # Save best model
    best_model_path = os.path.join(OUTPUT_DIR, "networth_best_model.joblib")
    dump(best_model, best_model_path)
    print(f"Saved best model -> {best_model_path}")

    # Save best model tree visualization
    plt.figure(figsize=(22, 12))
    plot_tree(best_model, feature_names=feature_cols, filled=True, rounded=True)
    plt.title(f"Best Decision Tree (R²={best_model_info['r2']:.3f}, Iteration {best_model_info['iteration']})")
    plt.tight_layout()
    best_tree_png_path = os.path.join(OUTPUT_DIR, "networth_best_tree.png")
    plt.savefig(best_tree_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved best tree visualization -> {best_tree_png_path}")

    # Save best model rules
    best_rules = export_text(best_model, feature_names=feature_cols)
    best_rules_path = os.path.join(OUTPUT_DIR, "networth_best_tree_rules.txt")
    with open(best_rules_path, "w") as f:
        f.write(f"Best Model (Iteration {best_model_info['iteration']}, R²={best_model_info['r2']:.3f})\n")
        f.write("=" * 50 + "\n\n")
        f.write(best_rules)
    print(f"Saved best model rules -> {best_rules_path}")

    # Save best model predictions preview
    best_preview = best_model_info["X_test"].copy()
    best_preview["Actual"] = [billions_to_pretty(v) for v in best_model_info["y_test"].values]
    best_preview["Predicted"] = [billions_to_pretty(v) for v in best_model_info["y_pred"]]
    best_preview_csv = os.path.join(OUTPUT_DIR, "networth_best_predictions_preview.csv")
    best_preview.to_csv(best_preview_csv, index=False)
    print(f"Saved best model predictions -> {best_preview_csv}")

print(f"\n=== Summary ===")
print(f"Completed {NUM_ITERATIONS} iterations of model training")
print(f"Most stable features (lowest coefficient of variation):")
stable_features = fi_ensemble.nsmallest(3, "cv_importance")
for _, row in stable_features.iterrows():
    print(f"  {row['feature']}: CV = {row['cv_importance']:.3f}")
print(f"Most important features (highest average importance):")
important_features = fi_ensemble.nlargest(3, "avg_importance")
for _, row in important_features.iterrows():
    print(f"  {row['feature']}: {row['avg_importance']:.4f} ± {row['std_importance']:.4f}")
