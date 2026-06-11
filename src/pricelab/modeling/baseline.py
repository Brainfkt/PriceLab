from __future__ import annotations

import numpy as np
import pandas as pd


def predict_recent_median(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict units with a product-level median and a global fallback."""

    global_median = float(train["units_sold"].median()) if len(train) else 0.0
    product_median = train.groupby("product_id")["units_sold"].median().to_dict()
    values = test["product_id"].map(product_median).fillna(global_median)
    return values.astype(float).to_numpy()


def regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    err = true - pred
    mae = float(np.mean(np.abs(err))) if len(true) else np.nan
    rmse = float(np.sqrt(np.mean(err**2))) if len(true) else np.nan
    denominator = float(np.sum(np.abs(true)))
    wmape = float(np.sum(np.abs(err)) / denominator) if denominator > 0 else np.nan
    smape_denominator = np.abs(true) + np.abs(pred)
    valid_smape = smape_denominator > 0
    smape = (
        float(np.mean(2 * np.abs(err[valid_smape]) / smape_denominator[valid_smape]))
        if valid_smape.any()
        else np.nan
    )
    variance = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1 - np.sum(err**2) / variance) if variance > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "wmape": wmape, "smape": smape, "r2": r2}
