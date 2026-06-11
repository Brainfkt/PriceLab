from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pricelab.analytics.catalogue import catalogue_kpis, filter_catalogue, product_leaderboard, product_pareto
from pricelab.modeling.backtest import BacktestResult
from pricelab.ui.components import compact_number, pareto_chart, portfolio_scatter_chart, product_backtest_error_chart, product_leaderboard_chart


METRIC_OPTIONS = {
    "Revenue": "revenue",
    "Gross margin": "gross_margin",
    "Units": "units",
    "Average price": "avg_price",
    "Price points": "price_points",
}


def render_catalogue_page(df: pd.DataFrame, backtest: BacktestResult | None) -> None:
    st.subheader("Catalogue overview")
    f1, f2, f3 = st.columns(3)
    categories = f1.multiselect("Category", _options(df, "category"), placeholder="All categories")
    channels = f2.multiselect("Channel", _options(df, "channel"), placeholder="All channels")
    regions = f3.multiselect("Region", _options(df, "region"), placeholder="All regions")
    filtered = filter_catalogue(df, categories=categories, channels=channels, regions=regions)

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    kpis = catalogue_kpis(filtered)
    cols = st.columns(5)
    cols[0].metric("Products", f"{kpis['products']:,}")
    cols[1].metric("Units", compact_number(kpis["units"]))
    cols[2].metric("Revenue", compact_number(kpis["revenue"]))
    cols[3].metric("Gross margin", compact_number(kpis["gross_margin"]))
    cols[4].metric("Promo rate", f"{kpis['promo_rate']:.1%}")

    c1, c2 = st.columns([1, 1])
    metric_label = c1.selectbox("Leaderboard metric", list(METRIC_OPTIONS), index=0)
    top_choice = c2.selectbox("Products shown", ["Top 10", "Top 15", "Top 25", "Top 50", "All"], index=1)
    metric = METRIC_OPTIONS[metric_label]
    top_n = None if top_choice == "All" else int(top_choice.split()[-1])

    leaderboard = product_leaderboard(filtered, metric=metric, top_n=top_n)
    st.write(f"{top_choice} products by {metric_label.lower()}")
    st.plotly_chart(product_leaderboard_chart(leaderboard, metric=metric), use_container_width=True)

    st.write("Portfolio position")
    all_products = product_leaderboard(filtered, metric="revenue", top_n=None)
    st.plotly_chart(portfolio_scatter_chart(all_products), use_container_width=True)

    st.write("Pareto contribution")
    st.plotly_chart(pareto_chart(product_pareto(filtered, metric=metric, top_n=top_n), metric=metric), use_container_width=True)

    show_table = st.toggle("Show product table", value=top_choice == "All")
    if show_table:
        st.dataframe(product_leaderboard(filtered, metric=metric, top_n=None), use_container_width=True)

    if backtest is None:
        st.info("Temporal backtest is disabled.")
    elif not backtest.valid:
        st.warning(backtest.message)
    else:
        st.write("Temporal backtest")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("wMAPE", f"{backtest.metrics['wmape']:.1%}")
        c2.metric("Baseline wMAPE", f"{backtest.metrics['baseline_wmape']:.1%}")
        c3.metric("SMAPE", f"{backtest.metrics['smape']:.1%}")
        c4.metric("MAE", f"{backtest.metrics['mae']:.2f}")
        c5.metric("Folds", f"{int(backtest.metrics['fold_count'])}")
        if not backtest.predictions.empty:
            pred = backtest.predictions.copy()
            pred["date"] = pd.to_datetime(pred["date"])
            trend = pred.groupby("date", as_index=False)[["actual", "predicted", "baseline"]].sum()
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=trend["date"], y=trend["actual"], mode="lines+markers", name="Actual"))
            fig_bt.add_trace(go.Scatter(x=trend["date"], y=trend["predicted"], mode="lines+markers", name="Model"))
            fig_bt.add_trace(go.Scatter(x=trend["date"], y=trend["baseline"], mode="lines", name="Baseline"))
            fig_bt.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig_bt, use_container_width=True)
        if not backtest.product_metrics.empty:
            st.plotly_chart(product_backtest_error_chart(backtest.product_metrics), use_container_width=True)
        st.dataframe(backtest.fold_metrics, use_container_width=True)


def _options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())
