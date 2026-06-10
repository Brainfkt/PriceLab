from __future__ import annotations

import numpy as np
import pandas as pd

from pricelab.modeling.elasticity import fit_loglog_elasticity
from pricelab.modeling.reliability import compute_reliability
from pricelab.modeling.simulation import simulate_price_scenario
from pricelab.schemas import PriceRecommendation, PricingObjective


def find_price_recommendation(
    df: pd.DataFrame,
    product_id: str,
    objective: str = "revenue",
    product_backtest_metrics: pd.DataFrame | None = None,
    grid_size: int = 41,
) -> PriceRecommendation:
    objective_enum = PricingObjective(objective)
    base_reliability = compute_reliability(
        df,
        product_id,
        product_backtest_metrics=product_backtest_metrics,
        objective=objective_enum.value,
    )
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty:
        return _blocked(product_id, objective_enum, 0.0, ["No data for this product."])
    if base_reliability.level == "blocked":
        return _blocked(product_id, objective_enum, base_reliability.score, base_reliability.hard_blocks + base_reliability.reasons)

    prices = product["price"].dropna().astype(float)
    low = float(prices.quantile(0.10))
    high = float(prices.quantile(0.90))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(prices.min())
        high = float(prices.max())
    if high <= low:
        return _blocked(product_id, objective_enum, base_reliability.score, ["Observed price corridor is too narrow."])

    candidates = np.linspace(low, high, grid_size)
    elasticity = fit_loglog_elasticity(product, product_id)
    simulations = [
        simulate_price_scenario(
            df,
            product_id,
            float(candidate),
            objective=objective_enum.value,
            elasticity_result=elasticity,
            product_backtest_metrics=product_backtest_metrics,
        )
        for candidate in candidates
    ]
    rows = pd.DataFrame([_model_dump(sim) for sim in simulations])
    rows = rows[rows["status"] != "blocked"].copy()
    if objective_enum == PricingObjective.MARGIN and rows["predicted_margin"].isna().all():
        return _blocked(product_id, objective_enum, base_reliability.score, ["Cost is required for margin optimization."])
    if rows.empty:
        return _blocked(product_id, objective_enum, base_reliability.score, ["All candidate prices were blocked by reliability rules."])

    rows["objective_value"] = _objective_values(rows, product, objective_enum)
    best = rows.sort_values(["objective_value", "reliability_score"], ascending=False).iloc[0]
    best_value = float(best["objective_value"])
    tolerance = 0.98 if best_value > 0 else 1.02
    near_best = rows[rows["objective_value"] >= best_value * tolerance].copy()
    if near_best.empty:
        near_best = rows.loc[[best.name]]

    status = "recommended" if float(best["reliability_score"]) >= 75 else "cautious"
    if float(best["reliability_score"]) < 55:
        status = "simulation_only"

    reasons = base_reliability.reasons + [warning for warning in best.get("warnings", []) if warning]
    return PriceRecommendation(
        product_id=str(product_id),
        objective=objective_enum,
        status=status,
        recommended_price=round(float(best["price"]), 2),
        lower_price=round(float(near_best["price"].min()), 2),
        upper_price=round(float(near_best["price"].max()), 2),
        expected_units=round(float(best["predicted_units"]), 2),
        expected_revenue=round(float(best["predicted_revenue"]), 2),
        expected_margin=round(float(best["predicted_margin"]), 2) if pd.notna(best["predicted_margin"]) else None,
        reliability_score=round(float(best["reliability_score"]), 1),
        reasons=list(dict.fromkeys(reasons)),
    )


def _objective_values(rows: pd.DataFrame, product: pd.DataFrame, objective: PricingObjective) -> pd.Series:
    if objective == PricingObjective.VOLUME:
        return rows["predicted_units"].astype(float)
    if objective == PricingObjective.REVENUE:
        return rows["predicted_revenue"].astype(float)
    if objective == PricingObjective.MARGIN:
        return rows["predicted_margin"].fillna(-np.inf).astype(float)
    reference_price = float(product["price"].median())
    distance_penalty = (rows["price"].astype(float) / reference_price - 1).abs()
    return rows["predicted_revenue"].astype(float) * (rows["reliability_score"].astype(float) / 100) * (1 - distance_penalty.clip(upper=0.6))


def _blocked(product_id: str, objective: PricingObjective, score: float, reasons: list[str]) -> PriceRecommendation:
    return PriceRecommendation(
        product_id=str(product_id),
        objective=objective,
        status="blocked",
        reliability_score=round(float(score), 1),
        reasons=list(dict.fromkeys(reasons)),
    )


def _model_dump(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()
