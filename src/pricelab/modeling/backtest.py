from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pricelab.features.build import build_model_frame
from pricelab.modeling.baseline import predict_recent_median, regression_metrics
from pricelab.modeling.demand_model import predict_demand, train_demand_model


@dataclass
class BacktestResult:
    valid: bool
    metrics: dict[str, float]
    fold_metrics: pd.DataFrame
    product_metrics: pd.DataFrame
    message: str = ""


def make_time_splits(dates: list[pd.Timestamp], n_splits: int = 3) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    unique_dates = sorted(pd.to_datetime(pd.Series(dates).dropna().unique()))
    n_dates = len(unique_dates)
    if n_dates < 12:
        return []
    test_size = max(4, n_dates // (n_splits + 4))
    min_train = max(8, int(n_dates * 0.45))
    splits: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for idx in range(n_splits):
        test_start_idx = min_train + idx * test_size
        test_end_idx = min(test_start_idx + test_size, n_dates)
        if test_end_idx <= test_start_idx or test_start_idx >= n_dates:
            break
        train_end = unique_dates[test_start_idx - 1]
        test_start = unique_dates[test_start_idx]
        test_end = unique_dates[test_end_idx - 1]
        splits.append((pd.Timestamp(train_end), pd.Timestamp(test_start), pd.Timestamp(test_end)))
    return splits


def run_backtest(df: pd.DataFrame, n_splits: int = 3, random_state: int = 42) -> BacktestResult:
    frame = _ensure_feature_frame(df)
    splits = make_time_splits(list(frame["date"]), n_splits=n_splits)
    if not splits:
        return BacktestResult(
            valid=False,
            metrics={},
            fold_metrics=pd.DataFrame(),
            product_metrics=pd.DataFrame(),
            message="Not enough dated observations for temporal backtesting.",
        )

    fold_rows: list[dict[str, float | int]] = []
    pred_rows: list[pd.DataFrame] = []
    for fold_idx, (train_end, test_start, test_end) in enumerate(splits, start=1):
        train = frame[frame["date"] <= train_end].copy()
        test = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)].copy()
        if len(train) < 20 or test.empty:
            continue
        model = train_demand_model(train, random_state=random_state + fold_idx)
        pred = predict_demand(model, test)
        baseline = predict_recent_median(train, test)
        metrics = regression_metrics(test["units_sold"], pred)
        baseline_metrics = regression_metrics(test["units_sold"], baseline)
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "wmape": metrics["wmape"],
                "r2": metrics["r2"],
                "baseline_wmape": baseline_metrics["wmape"],
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    "product_id": test["product_id"].astype(str).to_numpy(),
                    "actual": test["units_sold"].astype(float).to_numpy(),
                    "predicted": pred,
                    "baseline": baseline,
                }
            )
        )

    if not fold_rows:
        return BacktestResult(
            valid=False,
            metrics={},
            fold_metrics=pd.DataFrame(),
            product_metrics=pd.DataFrame(),
            message="Backtest splits were generated but no fold had enough rows.",
        )

    fold_metrics = pd.DataFrame(fold_rows)
    all_predictions = pd.concat(pred_rows, ignore_index=True)
    model_metrics = regression_metrics(all_predictions["actual"], all_predictions["predicted"])
    baseline_metrics = regression_metrics(all_predictions["actual"], all_predictions["baseline"])
    product_metrics = _product_metrics(all_predictions)
    metrics = {
        **model_metrics,
        "baseline_wmape": baseline_metrics["wmape"],
        "wmape_gain_vs_baseline": baseline_metrics["wmape"] - model_metrics["wmape"],
        "fold_count": float(len(fold_metrics)),
    }
    return BacktestResult(valid=True, metrics=metrics, fold_metrics=fold_metrics, product_metrics=product_metrics)


def _product_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for product_id, group in predictions.groupby("product_id"):
        metrics = regression_metrics(group["actual"], group["predicted"])
        baseline = regression_metrics(group["actual"], group["baseline"])
        rows.append(
            {
                "product_id": str(product_id),
                "observations": int(len(group)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "wmape": metrics["wmape"],
                "r2": metrics["r2"],
                "baseline_wmape": baseline["wmape"],
                "wmape_gain_vs_baseline": baseline["wmape"] - metrics["wmape"],
            }
        )
    return pd.DataFrame(rows)


def _ensure_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if {"log_price", "rolling_units_4", "week_sin"}.issubset(df.columns):
        return df.copy()
    return build_model_frame(df, weekly=False)

