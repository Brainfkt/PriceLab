from __future__ import annotations

import numpy as np
import pandas as pd

from pricelab.features.calendar import add_calendar_features
from pricelab.features.price import add_price_features


GROUP_KEYS = ["product_id", "product_name", "category", "channel", "region"]


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["week_start"] = out["date"].dt.to_period("W-MON").dt.start_time

    optional_aggs: dict[str, tuple[str, str]] = {}
    for col in ["competitor_price", "weather_index"]:
        if col in out.columns:
            optional_aggs[col] = (col, "mean")
    for col in ["marketing_spend", "traffic", "returns"]:
        if col in out.columns:
            optional_aggs[col] = (col, "sum")
    for col in ["customer_segment"]:
        if col in out.columns:
            optional_aggs[col] = (col, _mode_or_unknown)

    grouped_obj = out.groupby(GROUP_KEYS + ["week_start"], dropna=False)
    try:
        grouped = grouped_obj.apply(_aggregate_group, include_groups=False)
    except TypeError:
        grouped = grouped_obj.apply(_aggregate_group)
    grouped = grouped.reset_index().rename(columns={"week_start": "date"})

    for col, (_, agg) in optional_aggs.items():
        if col not in grouped.columns:
            extra = out.groupby(GROUP_KEYS + ["week_start"], dropna=False).agg(value=(col, agg)).reset_index()
            grouped[col] = extra["value"].to_numpy()

    return grouped.sort_values(["product_id", "channel", "region", "date"]).reset_index(drop=True)


def build_model_frame(df: pd.DataFrame, weekly: bool = True) -> pd.DataFrame:
    frame = aggregate_to_weekly(df) if weekly else df.copy()
    frame = add_calendar_features(frame)
    frame = add_price_features(frame)
    frame = add_lag_features(frame)
    return frame.replace([np.inf, -np.inf], np.nan)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["product_id", "channel", "region", "date"]).copy()
    group = out.groupby(["product_id", "channel", "region"], dropna=False)
    out["lag_units_1"] = group["units_sold"].shift(1)
    out["lag_price_1"] = group["price"].shift(1)
    out["rolling_units_4"] = group["units_sold"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    out["rolling_price_4"] = group["price"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    out["price_change_pct"] = (out["price"] / out["lag_price_1"] - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
    out["rolling_units_4"] = out["rolling_units_4"].fillna(out["units_sold"].median())
    out["rolling_price_4"] = out["rolling_price_4"].fillna(out["price"].median())
    out["lag_units_1"] = out["lag_units_1"].fillna(out["rolling_units_4"])
    out["lag_price_1"] = out["lag_price_1"].fillna(out["rolling_price_4"])
    return out


def model_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_candidates = [
        "price",
        "log_price",
        "discount_rate",
        "promo_discount_interaction",
        "competitor_gap",
        "log_competitor_price",
        "log_marketing_spend",
        "log_traffic",
        "weather_index",
        "stockout_flag",
        "holiday_flag",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",
        "lag_units_1",
        "lag_price_1",
        "rolling_units_4",
        "rolling_price_4",
        "price_change_pct",
    ]
    categorical_candidates = ["product_id", "category", "channel", "region", "customer_segment"]
    numeric = [col for col in numeric_candidates if col in df.columns]
    categorical = [col for col in categorical_candidates if col in df.columns]
    return numeric, categorical


def _aggregate_group(group: pd.DataFrame) -> pd.Series:
    units = group["units_sold"].astype(float).sum()
    weights = group["units_sold"].astype(float).clip(lower=0)
    if float(weights.sum()) <= 0:
        weights = None
    price = _weighted_average(group["price"], weights)
    cost = _weighted_average(group["cost"], weights) if "cost" in group.columns else np.nan
    stock = group["stock_available"].astype(float).sum() if "stock_available" in group.columns else np.nan
    discount = group["discount_rate"].astype(float).mean() if "discount_rate" in group.columns else 0.0
    promo = bool(group["promotion_flag"].astype(bool).max()) if "promotion_flag" in group.columns else False
    holiday = bool(group["holiday_flag"].astype(bool).max()) if "holiday_flag" in group.columns else False

    row: dict[str, object] = {
        "units_sold": float(units),
        "price": float(price),
        "cost": float(cost) if not pd.isna(cost) else np.nan,
        "stock_available": float(stock) if not pd.isna(stock) else np.nan,
        "promotion_flag": promo,
        "discount_rate": float(discount),
        "holiday_flag": holiday,
    }
    for col in ["competitor_price", "weather_index"]:
        if col in group.columns:
            row[col] = float(group[col].astype(float).mean())
    for col in ["marketing_spend", "traffic", "returns"]:
        if col in group.columns:
            row[col] = float(group[col].astype(float).sum())
    if "customer_segment" in group.columns:
        row["customer_segment"] = _mode_or_unknown(group["customer_segment"])
    return pd.Series(row)


def _weighted_average(values: pd.Series, weights: pd.Series | None) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if weights is None:
        return float(numeric.mean())
    clean_weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    if float(clean_weights.sum()) <= 0:
        return float(numeric.mean())
    return float(np.average(numeric.fillna(numeric.mean()), weights=clean_weights))


def _mode_or_unknown(series: pd.Series) -> str:
    modes = series.dropna().astype(str).mode()
    return str(modes.iloc[0]) if not modes.empty else "Unknown"
