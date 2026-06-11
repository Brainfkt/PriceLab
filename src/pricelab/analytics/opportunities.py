from __future__ import annotations

import pandas as pd

from pricelab.modeling.optimization import find_price_recommendation
from pricelab.modeling.reliability import compute_reliability


def scan_catalogue_opportunities(
    df: pd.DataFrame,
    objective: str = "revenue",
    product_backtest_metrics: pd.DataFrame | None = None,
    limit: int = 25,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for product_id, product in df.groupby("product_id"):
        product_id_str = str(product_id)
        current_price = float(product["price"].median())
        recommendation = find_price_recommendation(
            df,
            product_id_str,
            objective=objective,
            product_backtest_metrics=product_backtest_metrics,
            grid_size=25,
        )
        reliability = compute_reliability(
            df,
            product_id_str,
            product_backtest_metrics=product_backtest_metrics,
            objective=objective,
        )
        price_delta = None
        if recommendation.recommended_price is not None and current_price > 0:
            price_delta = recommendation.recommended_price / current_price - 1
        action, action_reason, opportunity_score = _classify_opportunity(product, recommendation.status, reliability.score, price_delta)
        rows.append(
            {
                "product_id": product_id_str,
                "product_name": str(product["product_name"].mode().iloc[0]) if "product_name" in product.columns else product_id_str,
                "category": str(product["category"].mode().iloc[0]) if "category" in product.columns else "Unknown",
                "action_category": action,
                "opportunity_score": opportunity_score,
                "status": recommendation.status,
                "reliability_score": recommendation.reliability_score,
                "current_price": round(current_price, 2),
                "recommended_price": recommendation.recommended_price,
                "price_delta_pct": round(price_delta * 100, 1) if price_delta is not None else None,
                "expected_revenue": recommendation.expected_revenue,
                "expected_margin": recommendation.expected_margin,
                "reason": action_reason or ("; ".join(recommendation.reasons[:2]) if recommendation.reasons else reliability.level),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    status_rank = {"recommended": 0, "cautious": 1, "simulation_only": 2, "blocked": 3}
    result["_status_rank"] = result["status"].map(status_rank).fillna(9)
    return (
        result.sort_values(["_status_rank", "opportunity_score", "reliability_score", "expected_revenue"], ascending=[True, False, False, False])
        .drop(columns=["_status_rank"])
        .head(limit)
        .reset_index(drop=True)
    )


def _classify_opportunity(product: pd.DataFrame, status: str, reliability_score: float, price_delta: float | None) -> tuple[str, str, float]:
    stockout_rate = float((product.get("stock_available", pd.Series(1, index=product.index)) <= 0).mean())
    promo_rate = float(product.get("promotion_flag", pd.Series(False, index=product.index)).astype(bool).mean())
    margin_rate = None
    if {"price", "cost"}.issubset(product.columns):
        prices = pd.to_numeric(product["price"], errors="coerce").replace(0, float("nan"))
        costs = pd.to_numeric(product["cost"], errors="coerce")
        margin_rate = float(((prices - costs) / prices).median())

    if status == "blocked":
        return "Data Insufficient", "Recommendation blocked by reliability rules.", 0.0
    if stockout_rate >= 0.25:
        return "Stock-Constrained", f"Stockout rate is {stockout_rate:.1%}; observed demand may be censored.", round(reliability_score * 0.45, 1)
    if promo_rate >= 0.35:
        return "Promo Trap", f"Promotion rate is {promo_rate:.1%}; separate normal price from promo price before acting.", round(reliability_score * 0.55, 1)
    if price_delta is None:
        return "Data Insufficient", "No reliable target price corridor was produced.", round(reliability_score * 0.35, 1)
    delta_abs = abs(price_delta)
    base_score = min(100.0, reliability_score * (0.5 + min(delta_abs, 0.25) * 2.0))
    if price_delta >= 0.03:
        return "Raise Price Candidate", f"Recommended price is {price_delta:.1%} above current median.", round(base_score, 1)
    if price_delta <= -0.03:
        return "Discount Candidate", f"Recommended price is {abs(price_delta):.1%} below current median.", round(base_score, 1)
    if margin_rate is not None and margin_rate >= 0.45:
        return "Margin Hero", f"Median margin rate is {margin_rate:.1%} with stable recommended price.", round(reliability_score * 0.65, 1)
    return "Stable Price Product", "Recommended price is close to current median.", round(reliability_score * 0.50, 1)
