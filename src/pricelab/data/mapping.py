from __future__ import annotations

import re

import pandas as pd

from pricelab.config import OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from pricelab.schemas import ColumnMapping


ALIASES: dict[str, set[str]] = {
    "date": {"date", "day", "week", "order_date", "sales_date", "timestamp"},
    "product_id": {"product_id", "sku", "id_product", "item_id", "article_id", "productcode"},
    "product_name": {"product_name", "name", "item_name", "article", "label", "designation"},
    "category": {"category", "cat", "department", "family", "product_category"},
    "units_sold": {"units_sold", "quantity", "qty", "sales_units", "volume", "units", "sold"},
    "price": {"price", "unit_price", "selling_price", "avg_price", "asp"},
    "cost": {"cost", "unit_cost", "cogs", "purchase_cost"},
    "stock_available": {"stock_available", "stock", "inventory", "available_stock", "on_hand"},
    "promotion_flag": {"promotion_flag", "promo", "is_promo", "promotion", "promo_flag"},
    "discount_rate": {"discount_rate", "discount", "markdown", "discount_pct"},
    "channel": {"channel", "sales_channel", "store_type"},
    "region": {"region", "area", "market", "country", "zone"},
    "competitor_price": {"competitor_price", "comp_price", "market_price", "rival_price"},
    "marketing_spend": {"marketing_spend", "ad_spend", "media_spend", "campaign_spend"},
    "traffic": {"traffic", "visits", "sessions", "footfall"},
    "holiday_flag": {"holiday_flag", "holiday", "is_holiday", "bank_holiday"},
    "customer_segment": {"customer_segment", "segment", "customer_type"},
    "returns": {"returns", "returned_units", "refund_units"},
    "weather_index": {"weather_index", "weather", "temperature_index"},
}


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return normalized


def infer_column_mapping(columns: list[str] | pd.Index) -> ColumnMapping:
    normalized_to_original = {normalize_column_name(col): str(col) for col in columns}
    mapping: dict[str, str | None] = {}
    for canonical in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        match = None
        for alias in ALIASES.get(canonical, {canonical}):
            if normalize_column_name(alias) in normalized_to_original:
                match = normalized_to_original[normalize_column_name(alias)]
                break
        mapping[canonical] = match
    return ColumnMapping(**mapping)


def missing_required_columns(mapping: ColumnMapping) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if not getattr(mapping, column)]


def mapping_options(df: pd.DataFrame) -> dict[str, list[str]]:
    columns = [""] + [str(column) for column in df.columns]
    return {canonical: columns for canonical in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
