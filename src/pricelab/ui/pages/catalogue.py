from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pricelab.analytics.catalogue import catalogue_kpis, product_leaderboard
from pricelab.modeling.backtest import BacktestResult
from pricelab.ui.components import compact_number


def render_catalogue_page(df: pd.DataFrame, backtest: BacktestResult | None) -> None:
    st.subheader("Catalogue overview")
    kpis = catalogue_kpis(df)
    cols = st.columns(5)
    cols[0].metric("Products", f"{kpis['products']:,}")
    cols[1].metric("Units", compact_number(kpis["units"]))
    cols[2].metric("Revenue", compact_number(kpis["revenue"]))
    cols[3].metric("Gross margin", compact_number(kpis["gross_margin"]))
    cols[4].metric("Promo rate", f"{kpis['promo_rate']:.1%}")

    leaderboard = product_leaderboard(df, metric="revenue")
    fig = px.bar(leaderboard, x="product_id", y="revenue", color="category", hover_data=["product_name", "units", "avg_price"])
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(leaderboard, use_container_width=True)

    if backtest is None:
        st.info("Temporal backtest is disabled.")
    elif not backtest.valid:
        st.warning(backtest.message)
    else:
        st.write("Temporal backtest")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("wMAPE", f"{backtest.metrics['wmape']:.1%}")
        c2.metric("Baseline wMAPE", f"{backtest.metrics['baseline_wmape']:.1%}")
        c3.metric("MAE", f"{backtest.metrics['mae']:.2f}")
        c4.metric("Folds", f"{int(backtest.metrics['fold_count'])}")
        st.dataframe(backtest.fold_metrics, use_container_width=True)
