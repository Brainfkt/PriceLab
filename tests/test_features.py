from pricelab.data.demo_generator import generate_demo_dataset
from pricelab.data.importers import standardize_columns
import pandas as pd

from pricelab.features.build import build_model_frame, infer_temporal_grain


def test_feature_frame_adds_business_and_time_features():
    df = standardize_columns(generate_demo_dataset(seed=1, n_products=2, periods=8))
    frame = build_model_frame(df, weekly=True)
    for column in ["log_price", "log_units", "revenue", "gross_margin", "week_sin", "rolling_units_4"]:
        assert column in frame.columns
    assert frame["rolling_units_4"].isna().sum() == 0
    assert frame["revenue"].sum() > 0


def test_weekly_aggregation_uses_monday_week_start():
    df = standardize_columns(generate_demo_dataset(seed=2, n_products=1, periods=1))
    frame = build_model_frame(df, weekly=True)
    assert frame["date"].dt.weekday.eq(0).all()


def test_v2_features_and_auto_grain_detection():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=14, freq="D"),
            "product_id": ["A"] * 14,
            "product_name": ["Widget"] * 14,
            "category": ["Tools"] * 14,
            "channel": ["Online"] * 14,
            "region": ["North"] * 14,
            "units_sold": [10 + i for i in range(14)],
            "price": [10, 10, 11, 11, 11, 12, 12, 12, 11, 11, 13, 13, 13, 13],
            "cost": [5.0] * 14,
            "stock_available": [100.0] * 14,
            "promotion_flag": [False, False, True, True, False, False, False, True, False, False, False, True, False, False],
            "discount_rate": [0, 0, 0.1, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0, 0],
        }
    )
    df = standardize_columns(raw)
    assert infer_temporal_grain(df) == "daily"
    frame = build_model_frame(df, weekly="auto")
    for column in ["unit_margin", "price_bucket", "promo_depth_bucket", "day_of_week", "season", "days_since_price_change"]:
        assert column in frame.columns
    assert frame["model_grain"].eq("weekly").all()
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])

    weekly = raw.copy()
    weekly["date"] = pd.date_range("2025-01-06", periods=14, freq="W-MON").astype(str)
    weekly_frame = build_model_frame(standardize_columns(weekly), weekly="auto")
    assert weekly_frame["source_grain"].eq("weekly").all()
    assert weekly_frame["model_grain"].eq("native").all()
    assert pd.api.types.is_datetime64_any_dtype(weekly_frame["date"])
