import pandas as pd

from pricelab.data.importers import standardize_columns
from pricelab.data.quality import scan_quality


def test_quality_scanner_detects_invalid_price_and_low_variation():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5, freq="W"),
            "product_id": ["A"] * 5,
            "product_name": ["Widget"] * 5,
            "category": ["Tools"] * 5,
            "units_sold": [1, 2, 3, 4, 5],
            "price": [10, 10, 10, 10, 0],
            "cost": [4] * 5,
            "stock_available": [10] * 5,
            "promotion_flag": [False] * 5,
        }
    )
    df = standardize_columns(raw)
    report = scan_quality(df)
    codes = {issue.code for issue in report.issues}
    assert "non_positive_price" in codes
    assert "low_price_variation" in codes

