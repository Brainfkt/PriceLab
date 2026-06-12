from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pricelab.modeling.optimization import find_price_recommendation
from pricelab.modeling.simulation import simulate_price_scenario
from pricelab.ui.components import app_header, compact_number, dense_dataframe, scenario_curve_chart, section_header, status_pills


def render_simulator_page(
    df: pd.DataFrame,
    product_id: str | None,
    objective: str,
    product_metrics: pd.DataFrame | None,
) -> None:
    product = df[df["product_id"].astype(str) == str(product_id)]
    if product.empty:
        app_header("Scenario simulator", "Select a product in the left rail before simulating a price.", [("No product", "bad")])
        st.warning("No product selected.")
        return

    product_name = str(product["product_name"].mode().iloc[0]) if "product_name" in product.columns else str(product_id)
    app_header(
        "Scenario simulator",
        f"Compare candidate prices for {product_name} against the active {objective} objective.",
        [(f"Product {product_id}", ""), (objective.title(), "ok")],
    )
    min_price = float(product["price"].min())
    max_price = float(product["price"].max())
    median_price = float(product["price"].median())
    latest_date = pd.to_datetime(product["date"]).max()
    current_price = float(product[pd.to_datetime(product["date"]) == latest_date]["price"].median())
    low = round(max(min_price * 0.7, 0.01), 2)
    high = round(max_price * 1.3, 2)
    median_price = round(min(max(median_price, low), high), 2)
    stock_default = float(product["stock_available"].dropna().median()) if "stock_available" in product.columns and product["stock_available"].notna().any() else 0.0

    section_header("Scenario inputs", "Submit intentionally to update the scenario output and comparison table.")
    with st.form("scenario_form"):
        c1, c2, c3 = st.columns([1.2, 0.8, 1])
        scenario_price = c1.slider("Scenario price", min_value=low, max_value=high, value=median_price, step=0.01)
        discount_pct = c2.slider("Discount", min_value=0, max_value=60, value=0, step=1, format="%d%%")
        stock_available = c3.number_input("Available stock", min_value=0.0, value=round(stock_default, 2), step=1.0)
        c4, c5, c6 = st.columns(3)
        channel_options = ["All"] + sorted(product["channel"].dropna().astype(str).unique().tolist()) if "channel" in product.columns else ["All"]
        region_options = ["All"] + sorted(product["region"].dropna().astype(str).unique().tolist()) if "region" in product.columns else ["All"]
        channel = c4.selectbox("Channel", channel_options)
        region = c5.selectbox("Region", region_options)
        month_options = [0] + list(range(1, 13))
        month = c6.selectbox("Month", month_options, format_func=lambda value: "All" if value == 0 else pd.Timestamp(2024, int(value), 1).month_name())
        submitted = st.form_submit_button("Run scenario", type="primary", width="stretch")

    latest = st.session_state.get("latest_scenario", {})
    if submitted or not latest or str(latest.get("product_id")) != str(product_id) or latest.get("objective") != objective:
        scenario = simulate_price_scenario(
            df,
            product_id,
            scenario_price,
            objective=objective,
            product_backtest_metrics=product_metrics,
            discount_rate=discount_pct / 100,
            channel=None if channel == "All" else channel,
            region=None if region == "All" else region,
            month=None if month == 0 else int(month),
            stock_available=stock_available,
        )
        row = _model_dump(scenario)
        row["objective"] = objective
        st.session_state["latest_scenario"] = row
        comparisons = st.session_state.setdefault("scenario_comparisons", [])
        comparisons.append(row)
        st.session_state["scenario_comparisons"] = comparisons[-8:]

    scenario_row = st.session_state.get("latest_scenario", {})
    cols = st.columns(5)
    cols[0].metric("Status", scenario_row.get("status", "n/a"))
    cols[1].metric("Predicted units", compact_number(scenario_row.get("predicted_units")))
    cols[2].metric("Revenue", compact_number(scenario_row.get("predicted_revenue")))
    cols[3].metric("Margin", compact_number(scenario_row.get("predicted_margin")))
    cols[4].metric("Reliability", f"{scenario_row.get('reliability_score', 0):.0f}/100")
    if scenario_row:
        st.caption(f"Prediction interval proxy: {scenario_row.get('low_units', 0):,.1f} to {scenario_row.get('high_units', 0):,.1f} units.")
    warnings = scenario_row.get("warnings") or []
    if warnings:
        st.warning("; ".join(warnings))

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
            discount_rate=float((scenario_row.get("context") or {}).get("discount_rate", 0.0)),
            channel=None if (scenario_row.get("context") or {}).get("channel") == "All" else (scenario_row.get("context") or {}).get("channel"),
            region=None if (scenario_row.get("context") or {}).get("region") == "All" else (scenario_row.get("context") or {}).get("region"),
            month=(scenario_row.get("context") or {}).get("month"),
            stock_available=(scenario_row.get("context") or {}).get("stock_available"),
        )
        curve_rows.append(_model_dump(row))
    section_header("Scenario curve", "Observed corridor, selected scenario, and recommendation marker.")
    st.plotly_chart(
        scenario_curve_chart(
            pd.DataFrame(curve_rows),
            current_price=current_price,
            selected_price=float(scenario_row.get("price", median_price)),
            recommended_price=recommendation.recommended_price,
            observed_low=min_price,
            observed_high=max_price,
        ),
        width="stretch",
    )

    section_header("Optimal price finder", "Guarded recommendation using the current objective.")
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

    comparisons = pd.DataFrame(st.session_state.get("scenario_comparisons", []))
    if not comparisons.empty:
        status_pills([(f"{len(comparisons)} scenarios retained", "ok")])
        dense_dataframe(
            comparisons[[col for col in ["product_id", "objective", "price", "status", "predicted_units", "predicted_revenue", "predicted_margin", "reliability_score"] if col in comparisons.columns]],
            height=300,
        )


def _model_dump(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()
