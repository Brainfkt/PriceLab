from __future__ import annotations

import numpy as np
import pandas as pd

from pricelab.config import THRESHOLDS
from pricelab.schemas import ReliabilityResult


WEIGHTS = {
    "history": 0.15,
    "variation": 0.13,
    "dispersion": 0.10,
    "data_quality": 0.15,
    "stock": 0.08,
    "promotion": 0.07,
    "model": 0.20,
    "extrapolation": 0.07,
    "stability": 0.03,
    "cost": 0.02,
}


def compute_reliability(
    df: pd.DataFrame,
    product_id: str,
    product_backtest_metrics: pd.DataFrame | None = None,
    scenario_price: float | None = None,
    objective: str = "revenue",
) -> ReliabilityResult:
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    reasons: list[str] = []
    strengths: list[str] = []
    hard_blocks: list[str] = []

    if product.empty:
        return ReliabilityResult(
            product_id=str(product_id),
            score=0.0,
            level="blocked",
            components={key: 0.0 for key in WEIGHTS},
            strengths=[],
            reasons=["No data for this product."],
            hard_blocks=["No product history."],
        )

    dates = pd.to_datetime(product["date"], errors="coerce").dropna()
    history_days = int((dates.max() - dates.min()).days) if len(dates) >= 2 else 0
    price_points = int(product["price"].nunique())
    price_mean = float(product["price"].mean()) if len(product) else 0.0
    price_cv = float(product["price"].std(ddof=0) / price_mean) if price_mean > 0 else 0.0
    stockout_rate = float((product.get("stock_available", pd.Series(index=product.index, data=1)) <= 0).mean())
    promo_rate = float(product.get("promotion_flag", pd.Series(index=product.index, data=False)).astype(bool).mean())
    units_mean = float(product["units_sold"].mean()) if "units_sold" in product.columns and len(product) else 0.0
    units_cv = float(product["units_sold"].std(ddof=0) / units_mean) if units_mean > 0 else 0.0
    has_product_backtest = _has_product_backtest(product_id, product_backtest_metrics)

    components = {
        "history": _linear_component(history_days, THRESHOLDS.min_history_days, THRESHOLDS.full_history_days),
        "variation": _linear_component(price_points, THRESHOLDS.min_price_points, THRESHOLDS.full_price_points),
        "dispersion": _linear_component(price_cv, THRESHOLDS.low_price_cv, THRESHOLDS.full_price_cv),
        "data_quality": _data_quality_component(product, reasons),
        "stock": _inverse_component(stockout_rate, THRESHOLDS.full_stockout_rate, THRESHOLDS.bad_stockout_rate),
        "promotion": _inverse_component(promo_rate, THRESHOLDS.full_promo_rate, THRESHOLDS.bad_promo_rate),
        "model": _model_component(product_id, product_backtest_metrics, reasons),
        "extrapolation": _extrapolation_component(product, scenario_price, hard_blocks),
        "stability": _stability_component(units_cv, reasons),
        "cost": _cost_component(product, objective, hard_blocks),
    }

    if history_days < THRESHOLDS.min_history_days:
        hard_blocks.append("History is shorter than 45 days.")
    if price_points < THRESHOLDS.min_price_points:
        hard_blocks.append("Fewer than 3 distinct prices are observed.")
    if price_cv <= THRESHOLDS.low_price_cv:
        hard_blocks.append("Price is almost constant.")
    if stockout_rate > THRESHOLDS.hard_stockout_rate:
        hard_blocks.append("Stockouts are too frequent to infer demand.")

    if components["history"] < 1:
        reasons.append(f"History covers {history_days} days.")
    if components["variation"] < 1:
        reasons.append(f"Only {price_points} distinct prices are observed.")
    if components["dispersion"] < 1:
        reasons.append(f"Price coefficient of variation is {price_cv:.1%}.")
    if components["stock"] < 1:
        reasons.append(f"Stockout rate is {stockout_rate:.1%}.")
    if components["promotion"] < 1:
        reasons.append(f"Promotion rate is {promo_rate:.1%}.")
    if components["history"] >= 0.9:
        strengths.append(f"History covers {history_days} days.")
    if components["variation"] >= 0.9:
        strengths.append(f"{price_points} distinct prices are observed.")
    if components["data_quality"] >= 0.95:
        strengths.append("Product data quality is strong.")
    if components["stock"] >= 0.95:
        strengths.append("Stockouts are rare.")
    if components["promotion"] >= 0.95:
        strengths.append("Promotions do not dominate the history.")
    if components["model"] >= 0.75:
        strengths.append("Temporal backtest supports the model.")
    if components["stability"] >= 0.75:
        strengths.append("Demand volatility is manageable.")

    score = 100 * sum(WEIGHTS[name] * components[name] for name in WEIGHTS)
    if not has_product_backtest and not hard_blocks:
        score = min(score, THRESHOLDS.cautious_score - 1.0)
    if hard_blocks:
        score = min(score, 34.0)
    score = float(max(0.0, min(100.0, score)))
    level = _level(score, hard_blocks)
    return ReliabilityResult(
        product_id=str(product_id),
        score=round(score, 1),
        level=level,
        components={key: round(float(value), 3) for key, value in components.items()},
        strengths=_dedupe(strengths),
        reasons=_dedupe(reasons),
        hard_blocks=_dedupe(hard_blocks),
    )


def _linear_component(value: float, low: float, high: float) -> float:
    if value >= high:
        return 1.0
    if value <= low:
        return 0.0
    return float((value - low) / (high - low))


def _inverse_component(value: float, full: float, bad: float) -> float:
    if value <= full:
        return 1.0
    if value >= bad:
        return 0.0
    return float(1 - (value - full) / (bad - full))


def _model_component(product_id: str, metrics: pd.DataFrame | None, reasons: list[str]) -> float:
    if metrics is None or metrics.empty or "product_id" not in metrics.columns:
        reasons.append("No product-level temporal backtest is available yet.")
        return 0.35
    row = metrics[metrics["product_id"].astype(str) == str(product_id)]
    if row.empty:
        reasons.append("No product-level temporal backtest is available yet.")
        return 0.35
    wmape = float(row.iloc[0].get("wmape", np.nan))
    baseline_wmape = float(row.iloc[0].get("baseline_wmape", np.nan))
    if not np.isfinite(wmape):
        reasons.append("Backtest metric is invalid.")
        return 0.0
    quality = 1 - (wmape - 0.12) / (0.50 - 0.12)
    quality = float(max(0.0, min(1.0, quality)))
    if np.isfinite(baseline_wmape) and baseline_wmape > 0:
        improvement = (baseline_wmape - wmape) / baseline_wmape
        improvement_score = float(max(0.0, min(1.0, (improvement + 0.05) / 0.25)))
        if improvement < 0:
            reasons.append("Model underperforms the temporal baseline.")
    else:
        improvement_score = 0.4
    if wmape > 0.45:
        reasons.append(f"Backtest wMAPE is high ({wmape:.1%}).")
    return 0.7 * quality + 0.3 * improvement_score


def _data_quality_component(product: pd.DataFrame, reasons: list[str]) -> float:
    penalties = 0.0
    checked = [col for col in ["units_sold", "price", "cost", "stock_available"] if col in product.columns]
    for col in checked:
        missing_rate = float(product[col].isna().mean())
        if missing_rate:
            penalties += min(0.25, missing_rate)
            reasons.append(f"{col} missing rate is {missing_rate:.1%}.")
    if {"price", "cost"}.issubset(product.columns):
        valid_cost = product["cost"].notna() & product["price"].notna()
        if valid_cost.any():
            negative_margin_rate = float((product.loc[valid_cost, "cost"] >= product.loc[valid_cost, "price"]).mean())
            if negative_margin_rate > 0:
                penalties += min(0.30, negative_margin_rate)
                reasons.append(f"{negative_margin_rate:.1%} of rows have cost greater than or equal to price.")
    if "returns" in product.columns:
        returns_issue_rate = float((product["returns"].fillna(0) > product["units_sold"].fillna(0)).mean())
        if returns_issue_rate > 0:
            penalties += min(0.20, returns_issue_rate)
            reasons.append(f"{returns_issue_rate:.1%} of rows have returns above units sold.")
    return float(max(0.0, min(1.0, 1.0 - penalties)))


def _stability_component(units_cv: float, reasons: list[str]) -> float:
    component = _inverse_component(units_cv, full=0.35, bad=1.25)
    if component < 1:
        reasons.append(f"Demand coefficient of variation is {units_cv:.1%}.")
    return component


def _has_product_backtest(product_id: str, metrics: pd.DataFrame | None) -> bool:
    if metrics is None or metrics.empty or "product_id" not in metrics.columns:
        return False
    row = metrics[metrics["product_id"].astype(str) == str(product_id)]
    if row.empty:
        return False
    wmape = float(row.iloc[0].get("wmape", np.nan))
    return bool(np.isfinite(wmape))


def _extrapolation_component(product: pd.DataFrame, scenario_price: float | None, hard_blocks: list[str]) -> float:
    if scenario_price is None:
        return 1.0
    prices = product["price"].dropna().astype(float)
    if prices.empty:
        hard_blocks.append("Observed price range is unavailable.")
        return 0.0
    min_price = float(prices.min())
    max_price = float(prices.max())
    p10 = float(prices.quantile(0.10))
    p90 = float(prices.quantile(0.90))
    price = float(scenario_price)
    if price < 0.7 * min_price or price > 1.3 * max_price:
        hard_blocks.append("Scenario price is too far outside observed prices.")
        return 0.0
    if p10 <= price <= p90:
        return 1.0
    if min_price <= price <= max_price:
        return 0.7
    return 0.35


def _cost_component(product: pd.DataFrame, objective: str, hard_blocks: list[str]) -> float:
    if objective != "margin":
        return 1.0
    if "cost" not in product.columns or product["cost"].dropna().empty:
        hard_blocks.append("Cost is required for margin optimization.")
        return 0.0
    if (product["cost"].dropna() <= 0).all():
        hard_blocks.append("Positive cost is required for margin optimization.")
        return 0.0
    return 1.0


def _level(score: float, hard_blocks: list[str]) -> str:
    if hard_blocks:
        return "blocked"
    if score >= THRESHOLDS.normal_score:
        return "normal"
    if score >= THRESHOLDS.cautious_score:
        return "cautious"
    if score >= THRESHOLDS.simulation_only_score:
        return "simulation_only"
    return "blocked"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out
