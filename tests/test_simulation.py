import numpy as np
import pandas as pd

from pricelab.modeling.optimization import find_price_recommendation
from pricelab.modeling.simulation import simulate_price_scenario


def _sim_data():
    prices = np.linspace(8, 16, 40)
    units = 120 * (prices / 10) ** -1.2
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40, freq="W"),
            "product_id": ["A"] * 40,
            "product_name": ["Widget"] * 40,
            "category": ["Tools"] * 40,
            "channel": ["Online"] * 40,
            "region": ["North"] * 40,
            "units_sold": units,
            "price": prices,
            "cost": [5.0] * 40,
            "stock_available": [999.0] * 40,
            "promotion_flag": [False] * 40,
            "discount_rate": [0.0] * 40,
        }
    )


def test_scenario_blocks_far_extrapolation():
    result = simulate_price_scenario(_sim_data(), "A", 100.0)
    assert result.status == "blocked"
    assert any("outside" in warning.lower() for warning in result.warnings)


def test_optimizer_returns_recommendation_or_cautious_result():
    result = find_price_recommendation(_sim_data(), "A", objective="revenue")
    assert result.status in {"recommended", "cautious", "simulation_only"}
    if result.status != "simulation_only":
        assert result.recommended_price is not None
        assert result.lower_price <= result.recommended_price <= result.upper_price

