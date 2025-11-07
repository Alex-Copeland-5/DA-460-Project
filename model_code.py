# networth_decision_tree.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

# ---------- CONFIG ----------
# Put your Excel path here (same folder or absolute path).
EXCEL_PATH = "C:/Users/Alex/Desktop/DA-460 Project/skyblock_dataset.xlsx"
OUTPUT_DIR = "artifacts"
MAX_DEPTH = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- helpers ---
def parse_networth_to_billions(x):
    """
    Convert values like '1.8B', '900M', 120000000 into *billions* (float).
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
    'SB Level', 'Combat Level', 'Farming Level', 'Fishing Level',
    'Mining Level', 'Foraging Level', 'Enchanting Level', 'Alchemy Level'
]
target_col = 'Networth'

missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}\nFound columns: {list(df.columns)}")

# Build X (numeric) and y (networth in billions)
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = df[target_col].apply(parse_networth_to_billions)

# Drop rows with NaNs
mask = X.notna().all(axis=1) & y.notna()
dropped = len(df) - mask.sum()
if dropped > 0:
    print(f"Dropping {dropped} row(s) due to missing values in features/target.")

X = X[mask]
y = y[mask]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Model
model = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae_b = mean_absolute_error(y_test, y_pred)      # in billions
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluation (target in *billions*) ===")
print(f"Mean Absolute Error: {mae_b:.3f} (≈ {billions_to_pretty(mae_b)} error)")
print(f"R² Score: {r2:.3f}")

# Preview predictions in pretty units
preview = X_test.copy()
preview['Actual'] = [billions_to_pretty(v) for v in y_test.values]
preview['Predicted'] = [billions_to_pretty(v) for v in y_pred]
print("\n=== Sample predictions (pretty units) ===")
print(preview.head(10).to_string())

# Save feature importances
fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
fi_path = os.path.join(OUTPUT_DIR, "networth_feature_importances.csv")
fi.to_csv(fi_path, index=False)
print(f"\nSaved feature importances -> {fi_path}")

# Save tree plot
plt.figure(figsize=(22, 12))
plot_tree(model, feature_names=feature_cols, filled=True, rounded=True)
plt.title("Decision Tree for Networth (target measured in billions)")
plt.tight_layout()
tree_png_path = os.path.join(OUTPUT_DIR, "networth_tree.png")
plt.savefig(tree_png_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved tree image -> {tree_png_path}")

# Save textual rules
rules = export_text(model, feature_names=feature_cols)
rules_path = os.path.join(OUTPUT_DIR, "networth_tree_rules.txt")
with open(rules_path, "w") as f:
    f.write(rules)
print(f"Saved tree rules -> {rules_path}")

# Save model
model_path = os.path.join(OUTPUT_DIR, "networth_tree_model.joblib")
dump(model, model_path)
print(f"Saved model -> {model_path}")

# Optional: also export the preview with pretty units
preview_csv = os.path.join(OUTPUT_DIR, "networth_predictions_preview.csv")
preview.to_csv(preview_csv, index=False)
print(f"Saved prediction preview -> {preview_csv}")
