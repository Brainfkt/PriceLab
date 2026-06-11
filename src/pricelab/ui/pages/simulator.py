from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from pricelab.modeling.optimization import find_price_recommendation
from pricelab.modeling.simulation import simulate_price_scenario
from pricelab.ui.components import compact_number, scenario_curve_chart


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
    latest_date = pd.to_datetime(product["date"]).max()
    current_price = float(product[pd.to_datetime(product["date"]) == latest_date]["price"].median())
    low = round(max(min_price * 0.7, 0.01), 2)
    high = round(max_price * 1.3, 2)
    median_price = round(min(max(median_price, low), high), 2)
    scenario_price = st.slider("Scenario price", min_value=low, max_value=high, value=median_price, step=0.01)
    c1, c2, c3, c4 = st.columns(4)
    discount_pct = c1.slider("Discount", min_value=0, max_value=60, value=0, step=1, format="%d%%")
    discount_rate = discount_pct / 100
    channel_options = ["All"] + sorted(product["channel"].dropna().astype(str).unique().tolist()) if "channel" in product.columns else ["All"]
    region_options = ["All"] + sorted(product["region"].dropna().astype(str).unique().tolist()) if "region" in product.columns else ["All"]
    channel = c2.selectbox("Channel", channel_options)
    region = c3.selectbox("Region", region_options)
    month_options = [0] + list(range(1, 13))
    month = c4.selectbox("Month", month_options, format_func=lambda value: "All" if value == 0 else pd.Timestamp(2024, int(value), 1).month_name())
    stock_default = float(product["stock_available"].dropna().median()) if "stock_available" in product.columns and product["stock_available"].notna().any() else 0.0
    stock_available = st.number_input("Available stock for scenario", min_value=0.0, value=round(stock_default, 2), step=1.0)

    scenario = simulate_price_scenario(
        df,
        product_id,
        scenario_price,
        objective=objective,
        product_backtest_metrics=product_metrics,
        discount_rate=discount_rate,
        channel=None if channel == "All" else channel,
        region=None if region == "All" else region,
        month=None if month == 0 else int(month),
        stock_available=stock_available,
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
    curve_rows = []
    for candidate in np.linspace(low, high, 31):
        row = simulate_price_scenario(
            df,
            product_id,
            float(candidate),
            objective=objective,
            product_backtest_metrics=product_metrics,
            discount_rate=discount_rate,
            channel=None if channel == "All" else channel,
            region=None if region == "All" else region,
            month=None if month == 0 else int(month),
            stock_available=stock_available,
        )
        curve_rows.append(row.model_dump() if hasattr(row, "model_dump") else row.dict())
    st.plotly_chart(
        scenario_curve_chart(
            pd.DataFrame(curve_rows),
            current_price=current_price,
            selected_price=scenario_price,
            recommended_price=recommendation.recommended_price,
            observed_low=min_price,
            observed_high=max_price,
        ),
        use_container_width=True,
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
