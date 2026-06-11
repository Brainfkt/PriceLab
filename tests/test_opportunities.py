import numpy as np
import pandas as pd

from pricelab.analytics.opportunities import scan_catalogue_opportunities


def test_opportunity_scanner_adds_action_category_and_score():
    prices = np.linspace(8, 16, 40)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40, freq="W"),
            "product_id": ["A"] * 40,
            "product_name": ["Widget"] * 40,
            "category": ["Tools"] * 40,
            "channel": ["Online"] * 40,
            "region": ["North"] * 40,
            "units_sold": 120 * (prices / 10) ** -1.1,
            "price": prices,
            "cost": [5.0] * 40,
            "stock_available": [999.0] * 40,
            "promotion_flag": [False] * 40,
            "discount_rate": [0.0] * 40,
        }
    )
    metrics = pd.DataFrame({"product_id": ["A"], "wmape": [0.15], "baseline_wmape": [0.25]})
    result = scan_catalogue_opportunities(df, product_backtest_metrics=metrics)
    assert not result.empty
    assert {"action_category", "opportunity_score", "reason"}.issubset(result.columns)
    assert result["opportunity_score"].iloc[0] >= 0
