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
        rows.append(
            {
                "product_id": product_id_str,
                "product_name": str(product["product_name"].mode().iloc[0]) if "product_name" in product.columns else product_id_str,
                "category": str(product["category"].mode().iloc[0]) if "category" in product.columns else "Unknown",
                "status": recommendation.status,
                "reliability_score": recommendation.reliability_score,
                "current_price": round(current_price, 2),
                "recommended_price": recommendation.recommended_price,
                "price_delta_pct": round(price_delta * 100, 1) if price_delta is not None else None,
                "expected_revenue": recommendation.expected_revenue,
                "expected_margin": recommendation.expected_margin,
                "reason": "; ".join(recommendation.reasons[:2]) if recommendation.reasons else reliability.level,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    status_rank = {"recommended": 0, "cautious": 1, "simulation_only": 2, "blocked": 3}
    result["_status_rank"] = result["status"].map(status_rank).fillna(9)
    return (
        result.sort_values(["_status_rank", "reliability_score", "expected_revenue"], ascending=[True, False, False])
        .drop(columns=["_status_rank"])
        .head(limit)
        .reset_index(drop=True)
    )

