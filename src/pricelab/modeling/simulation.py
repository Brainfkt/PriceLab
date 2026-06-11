from __future__ import annotations

import numpy as np
import pandas as pd

from pricelab.modeling.elasticity import ElasticityResult, fit_loglog_elasticity
from pricelab.modeling.reliability import compute_reliability
from pricelab.schemas import ScenarioResult


def simulate_price_scenario(
    df: pd.DataFrame,
    product_id: str,
    price: float,
    objective: str = "revenue",
    elasticity_result: ElasticityResult | None = None,
    product_backtest_metrics: pd.DataFrame | None = None,
    discount_rate: float = 0.0,
    channel: str | None = None,
    region: str | None = None,
    month: int | None = None,
    stock_available: float | None = None,
) -> ScenarioResult:
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty:
        return ScenarioResult(
            product_id=str(product_id),
            price=float(price),
            reference_price=0.0,
            predicted_units=0.0,
            predicted_revenue=0.0,
            predicted_margin=None,
            low_units=0.0,
            high_units=0.0,
            reliability_score=0.0,
            status="blocked",
            context={},
            warnings=["No data for this product."],
        )

    reliability = compute_reliability(
        df,
        product_id,
        product_backtest_metrics=product_backtest_metrics,
        scenario_price=price,
        objective=objective,
    )
    if elasticity_result is None:
        elasticity_result = fit_loglog_elasticity(product, product_id)

    reference, context_warnings = _scenario_reference_window(product, channel=channel, region=region, month=month)
    reference_price = float(np.average(reference["price"], weights=reference["units_sold"].clip(lower=0) + 1))
    reference_units = float(reference["units_sold"].median())
    elasticity = float(elasticity_result.elasticity) if np.isfinite(elasticity_result.elasticity) else -1.0
    warnings = list(reliability.reasons) + list(elasticity_result.warnings) + context_warnings
    if elasticity > 0:
        warnings.append("Positive elasticity was capped for scenario simulation.")
        elasticity = -0.05

    price_ratio = max(float(price) / max(reference_price, 0.01), 0.01)
    predicted_units = max(reference_units * (price_ratio**elasticity), 0.0)
    promo_lift = _promotion_lift(product)
    clean_discount = float(max(0.0, min(discount_rate, 0.80)))
    if clean_discount > 0:
        predicted_units *= 1 + promo_lift * min(clean_discount / 0.20, 1.5)
    if stock_available is not None and np.isfinite(stock_available):
        stock_cap = max(float(stock_available), 0.0)
        if predicted_units > stock_cap:
            warnings.append("Predicted demand is capped by the selected stock level.")
            predicted_units = stock_cap
    predicted_revenue = predicted_units * float(price)
    cost = _reference_cost(product)
    predicted_margin = None if cost is None else (float(price) - cost) * predicted_units

    uncertainty_band = 0.08 + (1 - reliability.score / 100) * 0.45
    low_units = max(predicted_units * (1 - uncertainty_band), 0.0)
    high_units = predicted_units * (1 + uncertainty_band)
    status = "blocked" if reliability.level == "blocked" else "ok"
    if reliability.level == "simulation_only":
        status = "simulation_only"
    if reliability.level == "cautious":
        status = "cautious"

    return ScenarioResult(
        product_id=str(product_id),
        price=round(float(price), 2),
        reference_price=round(reference_price, 2),
        predicted_units=round(float(predicted_units), 2),
        predicted_revenue=round(float(predicted_revenue), 2),
        predicted_margin=round(float(predicted_margin), 2) if predicted_margin is not None else None,
        low_units=round(float(low_units), 2),
        high_units=round(float(high_units), 2),
        reliability_score=reliability.score,
        status=status,
        context={
            "discount_rate": round(clean_discount, 3),
            "channel": channel or "All",
            "region": region or "All",
            "month": month,
            "stock_available": round(float(stock_available), 2) if stock_available is not None and np.isfinite(stock_available) else None,
            "reference_rows": int(len(reference)),
        },
        warnings=list(dict.fromkeys(warnings + reliability.hard_blocks)),
    )


def _reference_window(product: pd.DataFrame, periods: int = 8) -> pd.DataFrame:
    ordered = product.sort_values("date").copy()
    if "stock_available" in ordered.columns:
        no_stockout = ordered[ordered["stock_available"] > 0]
        if len(no_stockout) >= max(3, periods // 2):
            ordered = no_stockout
    return ordered.tail(periods) if len(ordered) >= periods else ordered


def _scenario_reference_window(
    product: pd.DataFrame,
    channel: str | None,
    region: str | None,
    month: int | None,
    periods: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    candidate = product.copy()
    warnings: list[str] = []
    filters: list[tuple[str, object]] = []
    if channel and channel != "All" and "channel" in candidate.columns:
        filters.append(("channel", channel))
    if region and region != "All" and "region" in candidate.columns:
        filters.append(("region", region))
    if month and "month" in candidate.columns:
        filters.append(("month", int(month)))

    for col, value in filters:
        filtered = candidate[candidate[col].astype(str) == str(value)] if col != "month" else candidate[candidate[col] == value]
        if len(filtered) >= max(4, periods // 2):
            candidate = filtered
        else:
            warnings.append(f"Not enough comparable rows for {col}={value}; using broader history.")
    return _reference_window(candidate, periods=periods), warnings


def _promotion_lift(product: pd.DataFrame) -> float:
    if "promotion_flag" not in product.columns:
        return 0.15
    clean = product.copy()
    clean["promotion_flag"] = clean["promotion_flag"].astype(bool)
    promo = clean[clean["promotion_flag"]]
    non_promo = clean[~clean["promotion_flag"]]
    if len(promo) < 4 or len(non_promo) < 4:
        return 0.15
    baseline = float(non_promo["units_sold"].median())
    if baseline <= 0:
        return 0.15
    lift = float(promo["units_sold"].median() / baseline - 1)
    return float(max(0.0, min(lift, 0.80)))


def _reference_cost(product: pd.DataFrame) -> float | None:
    if "cost" not in product.columns:
        return None
    costs = product["cost"].dropna()
    costs = costs[costs > 0]
    if costs.empty:
        return None
    return float(costs.median())
