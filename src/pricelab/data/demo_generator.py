from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORIES = {
    "Audio": (29.0, 249.0),
    "Home": (12.0, 129.0),
    "Beauty": (8.0, 89.0),
    "Sports": (15.0, 179.0),
}
CHANNELS = ["Online", "Retail"]
REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Core", "Premium", "Deal seekers"]


def generate_demo_dataset(
    seed: int = 42,
    n_products: int = 60,
    periods: int = 78,
    freq: str = "W-MON",
) -> pd.DataFrame:
    """Generate a realistic synthetic pricing dataset.

    The simulation encodes product-specific elasticities, seasonality,
    promotions, regional/channel effects, stockouts, competitors, and noisy
    demand. It is deterministic for a given seed and safe for tests.
    """

    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq=freq)
    products = _make_product_catalogue(rng, n_products)
    rows: list[dict[str, object]] = []

    for _, product in products.iterrows():
        base_price = float(product["base_price"])
        cost = float(product["cost"])
        base_demand = float(product["base_demand"])
        elasticity = float(product["true_elasticity"])

        for date_idx, date in enumerate(dates):
            week_angle = 2 * math.pi * date.isocalendar().week / 52
            category_seasonality = 1 + product["seasonality_strength"] * math.sin(
                week_angle + product["seasonality_phase"]
            )
            holiday_flag = bool(date.month == 12 or (date.month == 11 and date.day >= 20))
            weather_index = float(np.clip(rng.normal(0, 0.8), -2.5, 2.5))

            for channel in CHANNELS:
                channel_multiplier = 1.14 if channel == "Online" else 0.92
                for region in REGIONS:
                    region_multiplier = {
                        "North": 1.02,
                        "South": 0.95,
                        "East": 1.08,
                        "West": 1.00,
                    }[region]
                    segment = rng.choice(SEGMENTS, p=[0.58, 0.22, 0.20])
                    segment_multiplier = {
                        "Core": 1.00,
                        "Premium": 0.84,
                        "Deal seekers": 1.18,
                    }[segment]

                    promo_probability = 0.12 + (0.06 if segment == "Deal seekers" else 0.0)
                    promotion_flag = bool(rng.random() < promo_probability)
                    discount_rate = float(rng.uniform(0.06, 0.28) if promotion_flag else 0.0)
                    trend = 1 + 0.04 * date_idx / max(periods - 1, 1)
                    base_price_move = rng.normal(0, 0.035)
                    promo_adjustment = -discount_rate
                    price = base_price * trend * (1 + base_price_move + promo_adjustment)
                    price = float(max(price, cost * 1.08, 1.0))

                    competitor_noise = rng.normal(0, 0.055)
                    competitor_price = float(max(base_price * (1 + competitor_noise), cost))
                    marketing_spend = float(
                        rng.gamma(6, 9) * (1.8 if promotion_flag else 1.0) * channel_multiplier
                    )
                    traffic = float(
                        rng.gamma(42, 20)
                        * channel_multiplier
                        * region_multiplier
                        * (1.25 if promotion_flag else 1.0)
                    )

                    price_effect = (price / base_price) ** elasticity
                    competitor_effect = (competitor_price / price) ** 0.18
                    promo_effect = 1.18 if promotion_flag else 1.0
                    holiday_effect = 1.18 if holiday_flag and product["category"] in {"Beauty", "Audio"} else 1.0
                    marketing_effect = 1 + min(marketing_spend / 1200, 0.22)
                    traffic_effect = 1 + min((traffic - 700) / 5000, 0.16)
                    weather_effect = 1 + weather_index * (0.018 if product["category"] == "Sports" else 0.004)

                    expected_units = (
                        base_demand
                        * category_seasonality
                        * channel_multiplier
                        * region_multiplier
                        * segment_multiplier
                        * price_effect
                        * competitor_effect
                        * promo_effect
                        * holiday_effect
                        * marketing_effect
                        * traffic_effect
                        * weather_effect
                    )
                    expected_units = max(expected_units, 0.2)
                    units = float(rng.poisson(expected_units))

                    available_stock = float(max(rng.normal(expected_units * 1.45, expected_units * 0.22), 0))
                    stockout_shock = rng.random() < 0.035
                    if stockout_shock:
                        available_stock = float(rng.uniform(0, max(units * 0.55, 1)))
                    units_sold = float(min(units, available_stock))
                    returns = float(rng.binomial(int(round(units_sold)), 0.015)) if units_sold > 0 else 0.0

                    rows.append(
                        {
                            "date": date.date().isoformat(),
                            "product_id": product["product_id"],
                            "product_name": product["product_name"],
                            "category": product["category"],
                            "units_sold": round(units_sold, 2),
                            "price": round(price, 2),
                            "cost": round(cost, 2),
                            "stock_available": round(available_stock, 2),
                            "promotion_flag": promotion_flag,
                            "discount_rate": round(discount_rate, 3),
                            "channel": channel,
                            "region": region,
                            "competitor_price": round(competitor_price, 2),
                            "marketing_spend": round(marketing_spend, 2),
                            "traffic": round(traffic, 2),
                            "holiday_flag": holiday_flag,
                            "customer_segment": segment,
                            "returns": round(returns, 2),
                            "weather_index": round(weather_index, 3),
                        }
                    )

    return pd.DataFrame(rows)


def save_demo_dataset(path: str | Path, seed: int = 42) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_dataset(seed=seed).to_csv(out_path, index=False)
    return out_path


def _make_product_catalogue(rng: np.random.Generator, n_products: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    category_names = list(CATEGORIES)
    for idx in range(n_products):
        category = category_names[idx % len(category_names)]
        low, high = CATEGORIES[category]
        base_price = float(rng.uniform(low, high))
        margin_rate = float(rng.uniform(0.35, 0.68))
        cost = base_price * (1 - margin_rate)
        base_demand = float(rng.uniform(12, 85) * (high / base_price) ** 0.18)
        elasticity = float(-rng.uniform(0.55, 2.35))
        rows.append(
            {
                "product_id": f"P{idx + 1:03d}",
                "product_name": f"{category} Product {idx + 1:03d}",
                "category": category,
                "base_price": base_price,
                "cost": cost,
                "base_demand": base_demand,
                "true_elasticity": elasticity,
                "seasonality_strength": float(rng.uniform(0.02, 0.18)),
                "seasonality_phase": float(rng.uniform(0, 2 * math.pi)),
            }
        )
    return pd.DataFrame(rows)

