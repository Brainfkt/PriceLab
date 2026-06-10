import numpy as np
import pandas as pd

from pricelab.features.build import build_model_frame
from pricelab.modeling.elasticity import fit_loglog_elasticity


def test_elasticity_is_negative_on_controlled_demand_curve():
    prices = np.linspace(8, 16, 40)
    units = 120 * (prices / 10) ** -1.4
    raw = pd.DataFrame(
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
    frame = build_model_frame(raw, weekly=False)
    result = fit_loglog_elasticity(frame, "A", bootstrap_samples=0)
    assert result.elasticity < -0.5
    assert result.n_obs == 40
    assert result.price_points == 40

