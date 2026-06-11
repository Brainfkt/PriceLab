from __future__ import annotations

import numpy as np
import pandas as pd

from pricelab.features.calendar import add_calendar_features
from pricelab.features.price import add_price_features


GROUP_KEYS = ["product_id", "product_name", "category", "channel", "region"]


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["week_start"] = out["date"].dt.to_period("W-SUN").dt.start_time
    defaults = {
        "cost": np.nan,
        "stock_available": np.nan,
        "promotion_flag": False,
        "discount_rate": 0.0,
        "holiday_flag": False,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    if out.empty:
        columns = GROUP_KEYS + [
            "date",
            "units_sold",
            "price",
            "cost",
            "stock_available",
            "promotion_flag",
            "discount_rate",
            "holiday_flag",
        ]
        for col in ["competitor_price", "weather_index", "marketing_spend", "traffic", "returns", "customer_segment"]:
            if col in out.columns:
                columns.append(col)
        return pd.DataFrame(columns=columns)

    keys = GROUP_KEYS + ["week_start"]
    units = pd.to_numeric(out["units_sold"], errors="coerce").fillna(0).clip(lower=0)
    price = pd.to_numeric(out["price"], errors="coerce")
    cost = pd.to_numeric(out["cost"], errors="coerce") if "cost" in out.columns else pd.Series(np.nan, index=out.index)
    price_weight = units.where(price.notna(), 0)
    cost_weight = units.where(cost.notna(), 0)
    out["_price_weight"] = price_weight
    out["_price_weighted"] = price.fillna(0) * price_weight
    out["_cost_weight"] = cost_weight
    out["_cost_weighted"] = cost.fillna(0) * cost_weight

    aggs: dict[str, tuple[str, str]] = {
        "units_sold": ("units_sold", "sum"),
        "price_mean": ("price", "mean"),
        "price_weight": ("_price_weight", "sum"),
        "price_weighted": ("_price_weighted", "sum"),
        "cost_mean": ("cost", "mean"),
        "cost_weight": ("_cost_weight", "sum"),
        "cost_weighted": ("_cost_weighted", "sum"),
        "stock_available": ("stock_available", "sum"),
        "promotion_flag": ("promotion_flag", "max"),
        "discount_rate": ("discount_rate", "mean"),
        "holiday_flag": ("holiday_flag", "max"),
    }
    for col in ["competitor_price", "weather_index"]:
        if col in out.columns:
            aggs[col] = (col, "mean")
    for col in ["marketing_spend", "traffic", "returns"]:
        if col in out.columns:
            aggs[col] = (col, "sum")

    grouped = out.groupby(keys, dropna=False).agg(**aggs).reset_index()
    grouped["price"] = np.where(
        grouped["price_weight"] > 0,
        grouped["price_weighted"] / grouped["price_weight"],
        grouped["price_mean"],
    )
    grouped["cost"] = np.where(
        grouped["cost_weight"] > 0,
        grouped["cost_weighted"] / grouped["cost_weight"],
        grouped["cost_mean"],
    )
    grouped = grouped.drop(
        columns=[
            "price_mean",
            "price_weight",
            "price_weighted",
            "cost_mean",
            "cost_weight",
            "cost_weighted",
        ]
    )
    if "customer_segment" in out.columns:
        segment = out.groupby(keys, dropna=False)["customer_segment"].agg(_mode_or_unknown).reset_index(name="customer_segment")
        grouped = grouped.merge(segment, on=keys, how="left")
    grouped = grouped.rename(columns={"week_start": "date"})

    return grouped.sort_values(["product_id", "channel", "region", "date"]).reset_index(drop=True)


def build_model_frame(df: pd.DataFrame, weekly: bool | str = "auto") -> pd.DataFrame:
    source_grain = infer_temporal_grain(df)
    use_weekly = source_grain == "daily" if weekly == "auto" else bool(weekly)
    frame = aggregate_to_weekly(df) if use_weekly else df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["source_grain"] = source_grain
    frame["model_grain"] = "weekly" if use_weekly else "native"
    frame = add_calendar_features(frame)
    frame = add_price_features(frame)
    frame = add_lag_features(frame)
    return frame.replace([np.inf, -np.inf], np.nan)


def infer_temporal_grain(df: pd.DataFrame) -> str:
    if df.empty or "date" not in df.columns:
        return "unknown"
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    group_cols = [col for col in ["product_id", "channel", "region"] if col in frame.columns]
    gaps: list[float] = []
    if group_cols:
        for _, group in frame.dropna(subset=["date"]).groupby(group_cols, dropna=False):
            dates = group["date"].drop_duplicates().sort_values()
            if len(dates) >= 2:
                gaps.extend(dates.diff().dropna().dt.days.astype(float).tolist())
    else:
        dates = frame["date"].dropna().drop_duplicates().sort_values()
        gaps.extend(dates.diff().dropna().dt.days.astype(float).tolist())
    if not gaps:
        return "unknown"
    median_gap = float(np.median(gaps))
    if median_gap <= 2:
        return "daily"
    if 5 <= median_gap <= 9:
        return "weekly"
    if 25 <= median_gap <= 35:
        return "monthly"
    return "irregular"


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["product_id", "channel", "region", "date"]).copy()
    group = out.groupby(["product_id", "channel", "region"], dropna=False)
    out["lag_units_1"] = group["units_sold"].shift(1)
    out["lag_price_1"] = group["price"].shift(1)
    out["rolling_units_4"] = group["units_sold"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    out["rolling_price_4"] = group["price"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    out["price_change_pct"] = (out["price"] / out["lag_price_1"] - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
    price_changed = group["price"].diff().abs().fillna(0) > 0.005
    out["_price_change_marker"] = out["date"].where(price_changed)
    out["_last_price_change_date"] = out.groupby(["product_id", "channel", "region"], dropna=False)["_price_change_marker"].ffill()
    out["days_since_price_change"] = (
        (pd.to_datetime(out["date"]) - pd.to_datetime(out["_last_price_change_date"])).dt.days.fillna(0).clip(lower=0)
    )
    out = out.drop(columns=["_price_change_marker", "_last_price_change_date"])
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
        "days_since_price_change",
        "day_of_week",
    ]
    categorical_candidates = ["product_id", "category", "channel", "region", "customer_segment", "season", "price_bucket", "promo_depth_bucket"]
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
