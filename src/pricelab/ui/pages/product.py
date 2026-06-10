from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.analytics.moments import best_moments
from pricelab.analytics.product import price_performance_bins, product_summary
from pricelab.analytics.promotions import promotion_summary
from pricelab.modeling.elasticity import fit_loglog_elasticity
from pricelab.modeling.reliability import compute_reliability
from pricelab.ui.components import compact_number, price_bin_chart, price_units_chart, reliability_components_chart, reliability_gauge, revenue_margin_chart


def render_product_page(df: pd.DataFrame, product_id: str, product_metrics: pd.DataFrame | None) -> None:
    st.subheader("Product deep dive")
    summary = product_summary(df, product_id)
    if not summary:
        st.warning("No product selected.")
        return
    cols = st.columns(5)
    cols[0].metric("Units", compact_number(summary["units"]))
    cols[1].metric("Revenue", compact_number(summary["revenue"]))
    cols[2].metric("Avg price", f"{summary['avg_price']:.2f}")
    cols[3].metric("Price points", f"{summary['price_points']}")
    cols[4].metric("Stockout rate", f"{summary['stockout_rate']:.1%}")

    st.plotly_chart(price_units_chart(df, product_id), use_container_width=True)
    st.plotly_chart(revenue_margin_chart(df, product_id), use_container_width=True)

    reliability = compute_reliability(df, product_id, product_backtest_metrics=product_metrics)
    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(reliability_gauge(reliability.score), use_container_width=True)
    with right:
        st.plotly_chart(reliability_components_chart(reliability.components), use_container_width=True)
    if reliability.hard_blocks:
        st.error("Blocked: " + "; ".join(reliability.hard_blocks))
    elif reliability.reasons:
        st.warning("; ".join(reliability.reasons))

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

    st.write("Price performance matrix")
    st.plotly_chart(price_bin_chart(price_performance_bins(df, product_id)), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Best moments")
        st.dataframe(best_moments(df, product_id), use_container_width=True)
    with c2:
        st.write("Promotion analyzer")
        st.dataframe(promotion_summary(df, product_id), use_container_width=True)
