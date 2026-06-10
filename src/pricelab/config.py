from __future__ import annotations

from dataclasses import dataclass


REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "category",
    "units_sold",
    "price",
    "cost",
    "stock_available",
    "promotion_flag",
]

OPTIONAL_COLUMNS = [
    "discount_rate",
    "channel",
    "region",
    "competitor_price",
    "marketing_spend",
    "traffic",
    "holiday_flag",
    "customer_segment",
    "returns",
    "weather_index",
]

KEY_COLUMNS = ["date", "product_id", "channel", "region"]
OBJECTIVES = ["volume", "revenue", "margin", "prudence"]


@dataclass(frozen=True)
class ReliabilityThresholds:
    min_history_days: int = 45
    full_history_days: int = 180
    min_price_points: int = 3
    full_price_points: int = 10
    low_price_cv: float = 0.02
    full_price_cv: float = 0.12
    hard_stockout_rate: float = 0.50
    bad_stockout_rate: float = 0.40
    full_stockout_rate: float = 0.05
    bad_promo_rate: float = 0.70
    full_promo_rate: float = 0.20
    normal_score: float = 75.0
    cautious_score: float = 55.0
    simulation_only_score: float = 35.0


THRESHOLDS = ReliabilityThresholds()

