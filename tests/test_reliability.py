import pandas as pd

from pricelab.modeling.reliability import compute_reliability


def _base_product(prices):
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(prices), freq="W"),
            "product_id": ["A"] * len(prices),
            "product_name": ["Widget"] * len(prices),
            "category": ["Tools"] * len(prices),
            "channel": ["Online"] * len(prices),
            "region": ["North"] * len(prices),
            "units_sold": [20] * len(prices),
            "price": prices,
            "cost": [5.0] * len(prices),
            "stock_available": [50.0] * len(prices),
            "promotion_flag": [False] * len(prices),
        }
    )


def test_reliability_blocks_almost_constant_price():
    df = _base_product([10.0] * 30)
    result = compute_reliability(df, "A")
    assert result.level == "blocked"
    assert any("constant" in block.lower() for block in result.hard_blocks)


def test_reliability_blocks_margin_without_cost():
    df = _base_product([8 + i * 0.2 for i in range(30)]).drop(columns=["cost"])
    result = compute_reliability(df, "A", objective="margin")
    assert result.level == "blocked"
    assert any("Cost" in block for block in result.hard_blocks)

