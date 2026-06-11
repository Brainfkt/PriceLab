from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price"] = out["price"].astype(float)
    out["units_sold"] = out["units_sold"].astype(float)
    out["log_price"] = np.log(out["price"].clip(lower=0.01))
    out["log_units"] = np.log1p(out["units_sold"].clip(lower=0))
    out["revenue"] = out["units_sold"] * out["price"]

    if "cost" in out.columns:
        out["unit_margin"] = out["price"] - out["cost"]
        out["gross_margin"] = (out["price"] - out["cost"]) * out["units_sold"]
        out["margin_rate"] = np.where(out["price"] > 0, (out["price"] - out["cost"]) / out["price"], np.nan)
    else:
        out["unit_margin"] = np.nan
        out["gross_margin"] = np.nan
        out["margin_rate"] = np.nan

    if "competitor_price" in out.columns:
        out["competitor_gap"] = np.where(
            out["competitor_price"] > 0,
            (out["price"] - out["competitor_price"]) / out["competitor_price"],
            0.0,
        )
        out["log_competitor_price"] = np.log(out["competitor_price"].clip(lower=0.01))
    else:
        out["competitor_gap"] = 0.0

    if "stock_available" in out.columns:
        out["stockout_flag"] = out["stock_available"] <= 0
        out["stock_pressure"] = np.where(
            out["units_sold"] > 0,
            out["stock_available"] / out["units_sold"].replace(0, np.nan),
            np.nan,
        )
    else:
        out["stockout_flag"] = False
        out["stock_pressure"] = np.nan

    if "returns" in out.columns:
        out["net_units_sold"] = (out["units_sold"] - out["returns"].fillna(0)).clip(lower=0)
    else:
        out["net_units_sold"] = out["units_sold"]

    if "marketing_spend" in out.columns:
        out["log_marketing_spend"] = np.log1p(out["marketing_spend"].clip(lower=0))
    else:
        out["log_marketing_spend"] = 0.0

    if "traffic" in out.columns:
        out["log_traffic"] = np.log1p(out["traffic"].clip(lower=0))
    else:
        out["log_traffic"] = 0.0

    for col, default in [("discount_rate", 0.0), ("promotion_flag", False), ("holiday_flag", False), ("weather_index", 0.0)]:
        if col not in out.columns:
            out[col] = default

    out["price_bucket"] = _bucket_by_quantile(out["price"], max_bins=5, prefix="P")
    out["promo_depth_bucket"] = _promo_depth_bucket(out["discount_rate"].fillna(0))
    out["promo_discount_interaction"] = out["promotion_flag"].astype(float) * out["discount_rate"].fillna(0)
    return out


def _bucket_by_quantile(series: pd.Series, max_bins: int, prefix: str) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    unique = int(clean.nunique(dropna=True))
    if unique < 2:
        return pd.Series([f"{prefix}1"] * len(series), index=series.index)
    bins = min(max_bins, unique)
    try:
        bucket = pd.qcut(clean, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series([f"{prefix}1"] * len(series), index=series.index)
    return bucket.fillna(0).astype(int).add(1).map(lambda value: f"{prefix}{value}")


def _promo_depth_bucket(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    return pd.cut(
        clean,
        bins=[-0.001, 0.0, 0.10, 0.20, 0.35, np.inf],
        labels=["none", "light", "medium", "deep", "extreme"],
        include_lowest=True,
    ).astype(str)
