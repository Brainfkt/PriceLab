from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.analytics.moments import best_moments
from pricelab.analytics.product import price_performance_bins, product_summary
from pricelab.analytics.promotions import promotion_depth_summary, promotion_summary, promotion_timing_effect
from pricelab.modeling.elasticity import fit_loglog_elasticity
from pricelab.modeling.reliability import compute_reliability
from pricelab.ui.components import (
    app_header,
    compact_number,
    dense_dataframe,
    price_bin_chart,
    price_units_chart,
    promotion_depth_chart,
    promotion_timing_chart,
    reliability_components_chart,
    reliability_gauge,
    revenue_margin_chart,
    section_header,
    status_pills,
    temporal_heatmap_chart,
)


def render_product_page(df: pd.DataFrame, product_id: str | None, product_metrics: pd.DataFrame | None) -> None:
    summary = product_summary(df, product_id)
    if not summary:
        app_header("Product deep dive", "Select a product in the left rail to inspect pricing evidence.", [("No product", "bad")])
        st.warning("No product selected.")
        return
    app_header(
        f"{summary.get('product_name', product_id)}",
        "Product-level pricing evidence: demand, promotion pressure, reliability, elasticity, and observed price performance.",
        [(f"Product {product_id}", ""), (str(summary.get("category", "Unknown")), "ok")],
    )
    cols = st.columns(5)
    cols[0].metric("Units", compact_number(summary["units"]))
    cols[1].metric("Revenue", compact_number(summary["revenue"]))
    cols[2].metric("Avg price", f"{summary['avg_price']:.2f}")
    cols[3].metric("Price points", f"{summary['price_points']}")
    cols[4].metric("Stockout rate", f"{summary['stockout_rate']:.1%}")

    section_header("Price and demand", "History with promotion and stock pressure markers.")
    st.plotly_chart(price_units_chart(df, product_id), width="stretch")
    section_header("Commercial history", "Revenue and gross margin trajectory.")
    st.plotly_chart(revenue_margin_chart(df, product_id), width="stretch")
    section_header("Seasonality and promotion depth", "Temporal pattern by source grain.")
    st.plotly_chart(temporal_heatmap_chart(df, product_id), width="stretch")

    section_header("Reliability", "Guardrails that determine whether PriceLab can recommend or only simulate.")
    reliability = compute_reliability(df, product_id, product_backtest_metrics=product_metrics)
    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(reliability_gauge(reliability.score), width="stretch")
    with right:
        st.plotly_chart(reliability_components_chart(reliability.components), width="stretch")
    if reliability.hard_blocks:
        st.error("Blocked: " + "; ".join(reliability.hard_blocks))
    if reliability.strengths:
        status_pills([(value, "ok") for value in reliability.strengths[:4]])
    if reliability.reasons and not reliability.hard_blocks:
        st.warning("; ".join(reliability.reasons))

    section_header("Elasticity", "Interpretable log-log elasticity evidence for this product.")
    elasticity = fit_loglog_elasticity(df, product_id)
    ci = "n/a"
    if elasticity.ci_low is not None and elasticity.ci_high is not None:
        ci = f"{elasticity.ci_low:.2f} to {elasticity.ci_high:.2f}"
    e1, e2, e3 = st.columns(3)
    e1.metric("Elasticity", f"{elasticity.elasticity:.2f}" if pd.notna(elasticity.elasticity) else "n/a")
    e2.metric("Interval", ci)
    e3.metric("Train R2", f"{elasticity.r2_train:.2f}" if pd.notna(elasticity.r2_train) else "n/a")
    if elasticity.warnings:
        st.warning("; ".join(elasticity.warnings))

    section_header("Price performance matrix", "Observed price bins and business outcomes.")
    bins = price_performance_bins(df, product_id)
    st.plotly_chart(price_bin_chart(bins), width="stretch")
    if not bins.empty:
        dense_dataframe(bins.drop(columns=["price_bin"], errors="ignore"), height=300)

    depth = promotion_depth_summary(df, product_id)
    timing = promotion_timing_effect(df, product_id)
    c1, c2 = st.columns(2)
    with c1:
        section_header("Best moments", "High-performing contexts observed in history.")
        dense_dataframe(best_moments(df, product_id), height=360)
    with c2:
        section_header("Promotion analyzer", "Depth and timing effects.")
        st.plotly_chart(promotion_depth_chart(depth), width="stretch")
        st.plotly_chart(promotion_timing_chart(timing), width="stretch")
    with st.expander("Promotion detail tables"):
        st.write("Promotion summary")
        dense_dataframe(promotion_summary(df, product_id), height=260)
        st.write("Discount depth")
        dense_dataframe(depth, height=260)
        st.write("Pre / post promotion")
        dense_dataframe(timing, height=260)
