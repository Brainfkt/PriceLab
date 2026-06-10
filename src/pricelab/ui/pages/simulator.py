from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.modeling.optimization import find_price_recommendation
from pricelab.modeling.simulation import simulate_price_scenario
from pricelab.ui.components import compact_number


def render_simulator_page(
    df: pd.DataFrame,
    product_id: str,
    objective: str,
    product_metrics: pd.DataFrame | None,
) -> None:
    st.subheader("Scenario simulator and optimal price finder")
    product = df[df["product_id"].astype(str) == str(product_id)]
    if product.empty:
        st.warning("No product selected.")
        return
    min_price = float(product["price"].min())
    max_price = float(product["price"].max())
    median_price = float(product["price"].median())
    low = round(max(min_price * 0.7, 0.01), 2)
    high = round(max_price * 1.3, 2)
    median_price = round(min(max(median_price, low), high), 2)
    scenario_price = st.slider("Scenario price", min_value=low, max_value=high, value=median_price, step=0.01)

    scenario = simulate_price_scenario(
        df,
        product_id,
        scenario_price,
        objective=objective,
        product_backtest_metrics=product_metrics,
    )
    cols = st.columns(5)
    cols[0].metric("Status", scenario.status)
    cols[1].metric("Predicted units", compact_number(scenario.predicted_units))
    cols[2].metric("Revenue", compact_number(scenario.predicted_revenue))
    cols[3].metric("Margin", compact_number(scenario.predicted_margin))
    cols[4].metric("Reliability", f"{scenario.reliability_score:.0f}/100")
    st.caption(f"Prediction interval proxy: {scenario.low_units:,.1f} to {scenario.high_units:,.1f} units.")
    if scenario.warnings:
        st.warning("; ".join(scenario.warnings))

    recommendation = find_price_recommendation(
        df,
        product_id,
        objective=objective,
        product_backtest_metrics=product_metrics,
    )
    st.write("Optimal price finder")
    if recommendation.status == "blocked":
        st.error("Recommendation blocked: " + "; ".join(recommendation.reasons))
    elif recommendation.status == "simulation_only":
        st.warning("Simulation only: " + "; ".join(recommendation.reasons))
    else:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Recommended price", f"{recommendation.recommended_price:.2f}")
        r2.metric("Range", f"{recommendation.lower_price:.2f} - {recommendation.upper_price:.2f}")
        r3.metric("Expected revenue", compact_number(recommendation.expected_revenue))
        r4.metric("Reliability", f"{recommendation.reliability_score:.0f}/100")
        if recommendation.reasons:
            st.info("; ".join(recommendation.reasons))
