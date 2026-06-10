from __future__ import annotations

import pandas as pd

from pricelab.analytics.product import product_summary
from pricelab.modeling.elasticity import ElasticityResult
from pricelab.schemas import PriceRecommendation, ReliabilityResult


def build_product_markdown_report(
    df: pd.DataFrame,
    product_id: str,
    reliability: ReliabilityResult,
    recommendation: PriceRecommendation,
    elasticity: ElasticityResult | None = None,
) -> str:
    summary = product_summary(df, product_id)
    lines = [
        f"# PriceLab Product Report - {summary.get('product_name', product_id)}",
        "",
        "## Executive Summary",
        f"- Product: `{product_id}`",
        f"- Category: {summary.get('category', 'Unknown')}",
        f"- Reliability score: {reliability.score}/100 ({reliability.level})",
        f"- Recommendation status: {recommendation.status}",
    ]
    if recommendation.recommended_price is not None:
        lines.extend(
            [
                f"- Recommended price: {recommendation.recommended_price:.2f}",
                f"- Recommended range: {recommendation.lower_price:.2f} to {recommendation.upper_price:.2f}",
                f"- Expected units: {recommendation.expected_units:.2f}",
                f"- Expected revenue: {recommendation.expected_revenue:.2f}",
            ]
        )
        if recommendation.expected_margin is not None:
            lines.append(f"- Expected margin: {recommendation.expected_margin:.2f}")
    lines.extend(
        [
            "",
            "## Product KPIs",
            f"- Observations: {summary.get('observations', 0)}",
            f"- History: {summary.get('first_date', 'n/a')} to {summary.get('last_date', 'n/a')}",
            f"- Units sold: {summary.get('units', 0)}",
            f"- Revenue: {summary.get('revenue', 0)}",
            f"- Gross margin: {summary.get('gross_margin', 'n/a')}",
            f"- Observed price range: {summary.get('min_price', 'n/a')} to {summary.get('max_price', 'n/a')}",
            f"- Distinct prices: {summary.get('price_points', 0)}",
            "",
            "## Elasticity",
        ]
    )
    if elasticity is not None:
        ci = "n/a"
        if elasticity.ci_low is not None and elasticity.ci_high is not None:
            ci = f"{elasticity.ci_low:.2f} to {elasticity.ci_high:.2f}"
        lines.extend(
            [
                f"- Estimated elasticity: {elasticity.elasticity:.2f}",
                f"- Bootstrap interval: {ci}",
                f"- Training observations: {elasticity.n_obs}",
                f"- Train R2: {elasticity.r2_train:.3f}",
            ]
        )
    else:
        lines.append("- Elasticity was not estimated.")
    lines.extend(["", "## Reliability Drivers"])
    for name, value in reliability.components.items():
        lines.append(f"- {name}: {value:.2f}")
    if reliability.hard_blocks:
        lines.extend(["", "## Hard Blocks"])
        lines.extend([f"- {reason}" for reason in reliability.hard_blocks])
    if reliability.reasons:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {reason}" for reason in reliability.reasons])
    if recommendation.reasons:
        lines.extend(["", "## Recommendation Notes"])
        lines.extend([f"- {reason}" for reason in recommendation.reasons])
    lines.extend(
        [
            "",
            "## Statistical Caveat",
            "This report estimates conditional price response from observational data. It is not proof of causal elasticity.",
        ]
    )
    return "\n".join(lines)

