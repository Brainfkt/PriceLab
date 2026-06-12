from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.analytics.catalogue import catalogue_kpis, category_mix, portfolio_health, portfolio_timeseries, product_leaderboard
from pricelab.modeling.backtest import BacktestResult
from pricelab.ui.components import (
    app_header,
    category_mix_chart,
    compact_number,
    dense_dataframe,
    portfolio_health_chart,
    portfolio_scatter_chart,
    portfolio_trend_chart,
    section_header,
    status_pills,
)


def render_walkthrough_page(df: pd.DataFrame, backtest: BacktestResult | None) -> None:
    app_header(
        "Portfolio brief",
        "Executive-grade overview for trend, category mix, model readiness, and reference products.",
        [(f"{df['product_id'].nunique():,} products", ""), ("Portfolio scope", "ok")],
    )
    kpis = catalogue_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", f"{kpis['products']:,}")
    c2.metric("Revenue", compact_number(kpis["revenue"]))
    c3.metric("Gross margin", compact_number(kpis["gross_margin"]))
    c4.metric("Promo rate", f"{kpis['promo_rate']:.1%}")

    if backtest is not None and backtest.valid:
        b1, b2, b3 = st.columns(3)
        b1.metric("Model wMAPE", f"{backtest.metrics['wmape']:.1%}")
        b2.metric("Baseline wMAPE", f"{backtest.metrics['baseline_wmape']:.1%}")
        b3.metric("SMAPE", f"{backtest.metrics['smape']:.1%}")
    elif backtest is None:
        status_pills([("Run temporal backtest for validation evidence", "warn")])
    else:
        st.warning(backtest.message)

    section_header("Portfolio trend", "Revenue, margin, and units over time.")
    st.plotly_chart(portfolio_trend_chart(portfolio_timeseries(df)), width="stretch")

    left, right = st.columns(2)
    with left:
        section_header("Category mix", "Business contribution by category.")
        st.plotly_chart(category_mix_chart(category_mix(df)), width="stretch")
    with right:
        section_header("Portfolio health", "Reliability level distribution.")
        health = portfolio_health(df, backtest.product_metrics if backtest is not None and backtest.valid else None)
        st.plotly_chart(portfolio_health_chart(health), width="stretch")

    section_header("Revenue vs margin position", "Product scale and profitability posture.")
    all_products = product_leaderboard(df, metric="revenue", top_n=None)
    st.plotly_chart(portfolio_scatter_chart(all_products), width="stretch")

    top_n = st.selectbox("Reference products shown", [8, 15, 25, 50], index=0)
    leaderboard = product_leaderboard(df, metric="revenue", top_n=top_n)
    display_cols = ["product_id", "product_name", "category", "units", "revenue", "gross_margin", "avg_price"]
    section_header("Reference products", "Top products by revenue in the active scope.")
    dense_dataframe(leaderboard[[col for col in display_cols if col in leaderboard.columns]], height=360)
