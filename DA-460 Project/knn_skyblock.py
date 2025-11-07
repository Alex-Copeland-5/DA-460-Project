#!/usr/bin/env python3
import argparse
import math
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_money_to_billions(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x) / 1e9
    s = str(x).strip().replace(",", "").replace(" ", "")
    s = re.sub(r"^[^\d\-\.]+", "", s)
    m = re.match(r"^(-?\d+(\.\d+)?)([BMKbmk])?$", s)
    if m:
        val = float(m.group(1))
        suf = m.group(3).upper() if m.group(3) else None
        if suf == "B":
            return val
        if suf == "M":
            return val / 1e3
        if suf == "K":
            return val / 1e6
        return val / 1e9
    try:
        return float(s) / 1e9
    except Exception:
        return np.nan


def resolve_target_column(df: pd.DataFrame, requested: str) -> str:
    norm_cols = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    norm_req = re.sub(r"[^a-z0-9]", "", requested.lower())
    aliases = ["networth", "net_worth", "networthusd", "networthbillions"]
    if norm_req in norm_cols:
        return norm_cols[norm_req]
    for a in [norm_req] + aliases:
        if a in norm_cols:
            return norm_cols[a]
    for c in df.columns:
        s = re.sub(r"[^a-z0-9]", "", str(c).lower())
        if "net" in s and "worth" in s:
            return c
    raise KeyError(f"Could not find target column matching '{requested}'. Columns: {list(df.columns)}")


def load_excel(path: str, sheet: str | None, header_row: int | None) -> pd.DataFrame:
    if sheet is not None and header_row is not None:
        return pd.read_excel(path, sheet_name=sheet, header=header_row)
    # autodetect sheet
    sheets = pd.read_excel(path, sheet_name=None)
    # choose sheet with more columns
    best_name = max(sheets.keys(), key=lambda n: (sheets[n].shape[1], sheets[n].shape[0]))
    df = sheets[best_name]
    # if first cell looks like a banner, try header=1
    try:
        if isinstance(df.iloc[0, 0], str) and any(k in df.iloc[0, 0].lower() for k in ["da-460", "project"]):
            df = pd.read_excel(path, sheet_name=best_name, header=1)
    except Exception:
        pass
    return df


def load_and_prepare(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()
    # drop unnamed columns
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df.columns = [str(c).strip() for c in df.columns]

    # resolve target
    target_col = resolve_target_column(df, target_col)

    # parse target if string
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].apply(parse_money_to_billions)

    # choose numeric features
    non_feature_like = {target_col, "Username", "User", "Name", "Player", "IGN"}
    feature_cols = []
    for c in df.columns:
        if c in non_feature_like:
            continue
        if c == target_col:
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().mean() >= 0.5:
            df[c] = coerced
            feature_cols.append(c)

    df = df.dropna(subset=[target_col])
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def build_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor())
    ])


def run_grid(X, y, fast: bool=False) -> GridSearchCV:
    pipe = build_model()
    grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
        "knn__leaf_size": [15, 30, 45],
    }
    gs = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=3 if fast else 5, n_jobs=1 if fast else -1, refit=True)
    gs.fit(X, y)
    return gs


def save_artifacts(model: Pipeline, X_train, y_train, X_test, y_test, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    pred = pd.DataFrame({"actual_billions": y_test.values, "predicted_billions": y_pred}, index=y_test.index)
    pred["abs_error_billions"] = (pred["predicted_billions"] - pred["actual_billions"]).abs()
    csv_path = os.path.join(out_dir, "knn_predictions.csv")
    pred.to_csv(csv_path, index=True)

    # plot
    fig = plt.figure()
    plt.scatter(pred["actual_billions"], pred["predicted_billions"])
    low = min(pred["actual_billions"].min(), pred["predicted_billions"].min())
    high = max(pred["actual_billions"].max(), pred["predicted_billions"].max())
    plt.plot([low, high], [low, high])
    plt.xlabel("Actual Networth (billions)")
    plt.ylabel("Predicted Networth (billions)")
    plt.title("K-NN Regression: Actual vs Predicted")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "knn_actual_vs_pred.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    report_path = os.path.join(out_dir, "knn_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"R2: {r2:.4f}\nRMSE (billions): {rmse:.4f}\nMAE (billions): {mae:.4f}\n")

    return csv_path, plot_path, report_path, r2, rmse, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="skyblock_dataset.xlsx")
    ap.add_argument("--sheet", type=str, default="Skyblock")
    ap.add_argument("--header-row", type=int, default=1)
    ap.add_argument("--target", type=str, default="Networth")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="knn_outputs")
    ap.add_argument("--fast", action="store_true", help="Use a smaller hyperparameter grid for quick runs")
    args = ap.parse_args()

    # --- Fix: auto-locate Excel file in same folder ---
    if not os.path.isfile(args.data):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible = os.path.join(script_dir, args.data)
    if os.path.isfile(possible):
        args.data = possible


    ext = os.path.splitext(args.data)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = load_excel(args.data, args.sheet, args.header_row)
    else:
        df = pd.read_csv(args.data)

    X, y, feats = load_and_prepare(df, args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    gs = run_grid(X_train, y_train, fast=args.fast)
    best = gs.best_estimator_

    csv_path, plot_path, report_path, r2, rmse, mae = save_artifacts(best, X_train, y_train, X_test, y_test, args.out_dir)

    print("Best params:", gs.best_params_)
    print("Features:", feats)
    print(f"R2: {r2:.4f}  RMSE(billions): {rmse:.4f}  MAE(billions): {mae:.4f}")
    print("Artifacts:")
    print(report_path)
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
