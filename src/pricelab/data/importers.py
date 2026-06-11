from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from pricelab.config import OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from pricelab.data.mapping import infer_column_mapping, missing_required_columns
from pricelab.schemas import ColumnMapping


def load_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    return pd.read_csv(source)


def standardize_columns(raw: pd.DataFrame, mapping: ColumnMapping | None = None) -> pd.DataFrame:
    if mapping is None:
        mapping = infer_column_mapping(raw.columns)

    missing_mappings = missing_required_columns(mapping)
    if missing_mappings:
        raise ValueError("Missing required mappings: " + ", ".join(missing_mappings))

    rename_map = {source: canonical for canonical, source in mapping.as_dict().items()}
    df = raw.rename(columns=rename_map).copy()
    keep = [col for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if col in df.columns]
    df = df[keep]

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns after mapping: " + ", ".join(missing_columns))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["units_sold", "price", "cost", "stock_available", "discount_rate", "competitor_price", "marketing_spend", "traffic", "returns", "weather_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["promotion_flag", "holiday_flag"]:
        if col in df.columns:
            df[col] = _to_bool_series(df[col])

    for col in ["product_id", "product_name", "category", "channel", "region", "customer_segment"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown").astype(str)

    if "channel" not in df.columns:
        df["channel"] = "All"
    if "region" not in df.columns:
        df["region"] = "All"
    if "discount_rate" not in df.columns:
        df["discount_rate"] = 0.0
    if "promotion_flag" not in df.columns:
        df["promotion_flag"] = False
    if "holiday_flag" not in df.columns:
        df["holiday_flag"] = False
    if "returns" not in df.columns:
        df["returns"] = 0.0

    df = df.sort_values(["product_id", "channel", "region", "date"]).reset_index(drop=True)
    return df


def load_and_standardize(source: str | Path | BinaryIO, mapping: ColumnMapping | None = None) -> pd.DataFrame:
    raw = load_csv(source)
    return standardize_columns(raw, mapping)


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    truthy = {"1", "true", "t", "yes", "y", "oui", "promo", "promoted"}
    falsy = {"0", "false", "f", "no", "n", "non", "", "nan", "none"}
    values = series.astype(str).str.strip().str.lower()
    out = values.map(lambda value: True if value in truthy else False if value in falsy else bool(value))
    return out.fillna(False).astype(bool)
