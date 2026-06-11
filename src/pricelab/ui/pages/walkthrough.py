from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.analytics.catalogue import catalogue_kpis, category_mix, portfolio_health, portfolio_timeseries, product_leaderboard
from pricelab.modeling.backtest import BacktestResult
from pricelab.ui.components import category_mix_chart, compact_number, portfolio_health_chart, portfolio_scatter_chart, portfolio_trend_chart


def render_walkthrough_page(df: pd.DataFrame, backtest: BacktestResult | None) -> None:
    st.subheader("Portfolio brief")
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
        st.info("Run temporal backtest in the sidebar to include model validation evidence.")
    else:
        st.warning(backtest.message)

    st.write("Portfolio trend")
    st.plotly_chart(portfolio_trend_chart(portfolio_timeseries(df)), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.write("Category mix")
        st.plotly_chart(category_mix_chart(category_mix(df)), use_container_width=True)
    with right:
        st.write("Portfolio health")
        health = portfolio_health(df, backtest.product_metrics if backtest is not None and backtest.valid else None)
        st.plotly_chart(portfolio_health_chart(health), use_container_width=True)

    st.write("Revenue vs margin position")
    all_products = product_leaderboard(df, metric="revenue", top_n=None)
    st.plotly_chart(portfolio_scatter_chart(all_products), use_container_width=True)

    top_n = st.selectbox("Reference products shown", [8, 15, 25, 50], index=0)
    leaderboard = product_leaderboard(df, metric="revenue", top_n=top_n)
    display_cols = ["product_id", "product_name", "category", "units", "revenue", "gross_margin", "avg_price"]
    st.write("Reference products")
    st.dataframe(leaderboard[[col for col in display_cols if col in leaderboard.columns]], use_container_width=True)
