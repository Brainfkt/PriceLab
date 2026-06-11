from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pricelab.modeling.elasticity import fit_loglog_elasticity
from pricelab.modeling.optimization import find_price_recommendation
from pricelab.modeling.reliability import compute_reliability
from pricelab.modeling.simulation import simulate_price_scenario
from pricelab.modeling.backtest import BacktestResult
from pricelab.analytics.product import price_performance_bins
from pricelab.analytics.promotions import promotion_depth_summary, promotion_timing_effect
from pricelab.reporting.html import build_html_report_with_figures
from pricelab.reporting.markdown import build_product_markdown_report
from pricelab.ui.components import (
    price_bin_chart,
    price_units_chart,
    product_backtest_error_chart,
    promotion_depth_chart,
    promotion_timing_chart,
    revenue_margin_chart,
    scenario_curve_chart,
)


def render_export_page(
    df: pd.DataFrame,
    product_id: str,
    objective: str,
    product_metrics: pd.DataFrame | None,
    backtest: BacktestResult | None = None,
) -> None:
    st.subheader("Export product report")
    reliability = compute_reliability(df, product_id, product_backtest_metrics=product_metrics, objective=objective)
    recommendation = find_price_recommendation(df, product_id, objective=objective, product_backtest_metrics=product_metrics)
    elasticity = fit_loglog_elasticity(df, product_id)
    markdown_report = _report_context(df, product_id, objective) + "\n\n" + build_product_markdown_report(df, product_id, reliability, recommendation, elasticity)
    figures = [
        ("Price and units history", price_units_chart(df, product_id)),
        ("Revenue and margin history", revenue_margin_chart(df, product_id)),
        ("Observed price performance", price_bin_chart(price_performance_bins(df, product_id))),
        ("Promotion depth", promotion_depth_chart(promotion_depth_summary(df, product_id))),
        ("Promotion timing", promotion_timing_chart(promotion_timing_effect(df, product_id))),
    ]
    scenario_figure = _recommended_scenario_figure(df, product_id, objective, recommendation.recommended_price, product_metrics)
    if scenario_figure is not None:
        figures.append(("Recommended price scenario", scenario_figure))
    if backtest is not None and backtest.valid:
        figures.append(("Product-level backtest error", product_backtest_error_chart(backtest.product_metrics)))
    html_report = build_html_report_with_figures(
        markdown_report,
        figures,
    )
    st.download_button("Download Markdown report", markdown_report, file_name=f"pricelab_{product_id}.md", mime="text/markdown")
    st.download_button("Download HTML report", html_report, file_name=f"pricelab_{product_id}.html", mime="text/html")
    st.text_area("Report preview", markdown_report, height=520)


def _report_context(df: pd.DataFrame, product_id: str, objective: str) -> str:
    product = df[df["product_id"].astype(str) == str(product_id)]
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    rows = len(product)
    start = pd.to_datetime(product["date"], errors="coerce").min()
    end = pd.to_datetime(product["date"], errors="coerce").max()
    lines = [
        "## Report Context",
        f"- Generated at: {generated_at}",
        f"- Objective: {objective}",
        f"- Product rows: {rows}",
    ]
    if pd.notna(start) and pd.notna(end):
        lines.append(f"- Product history: {start.date().isoformat()} to {end.date().isoformat()}")
    lines.append("- Caveat: observational data estimates association, not guaranteed causal price response.")
    return "\n".join(lines)


def _recommended_scenario_figure(
    df: pd.DataFrame,
    product_id: str,
    objective: str,
    recommended_price: float | None,
    product_metrics: pd.DataFrame | None,
):
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty:
        return None
    min_price = float(product["price"].min())
    max_price = float(product["price"].max())
    low = round(max(min_price * 0.7, 0.01), 2)
    high = round(max_price * 1.3, 2)
    latest_date = pd.to_datetime(product["date"], errors="coerce").max()
    current_price = float(product[pd.to_datetime(product["date"], errors="coerce") == latest_date]["price"].median())
    selected_price = float(recommended_price) if recommended_price is not None else float(product["price"].median())
    rows = []
    for candidate in np.linspace(low, high, 31):
        scenario = simulate_price_scenario(
            df,
            product_id,
            float(candidate),
            objective=objective,
            product_backtest_metrics=product_metrics,
        )
        rows.append(scenario.model_dump() if hasattr(scenario, "model_dump") else scenario.dict())
    return scenario_curve_chart(
        pd.DataFrame(rows),
        current_price=current_price,
        selected_price=selected_price,
        recommended_price=recommended_price,
        observed_low=min_price,
        observed_high=max_price,
    )
