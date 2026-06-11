from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pricelab.analytics.opportunities import scan_catalogue_opportunities
from pricelab.ui.components import opportunity_matrix_chart


def render_opportunities_page(
    df: pd.DataFrame,
    objective: str,
    product_metrics: pd.DataFrame | None,
) -> None:
    st.subheader("Catalogue opportunity scanner")
    product_count = int(df["product_id"].nunique())
    if product_count == 0:
        st.warning("No products are available to scan.")
        return
    if product_count <= 5:
        limit = product_count
        st.caption(f"Scanning all {product_count} products.")
    else:
        limit = st.slider("Products to scan", min_value=5, max_value=min(50, product_count), value=min(20, product_count))
    if st.button("Scan opportunities", type="primary"):
        with st.spinner("Scanning catalogue..."):
            opportunities = scan_catalogue_opportunities(
                df,
                objective=objective,
                product_backtest_metrics=product_metrics,
                limit=limit,
            )
        if opportunities.empty:
            st.info("No opportunities were returned for the selected scope.")
            return
        f1, f2, f3 = st.columns(3)
        actions = f1.multiselect("Action", _options(opportunities, "action_category"), placeholder="All actions")
        statuses = f2.multiselect("Status", _options(opportunities, "status"), placeholder="All statuses")
        categories = f3.multiselect("Category", _options(opportunities, "category"), placeholder="All categories")
        filtered = _filter_opportunities(opportunities, actions=actions, statuses=statuses, categories=categories)
        st.write("Opportunity matrix")
        st.plotly_chart(opportunity_matrix_chart(filtered, objective=objective), use_container_width=True)
        if not filtered.empty and "action_category" in filtered.columns:
            counts = filtered["action_category"].value_counts().reset_index()
            counts.columns = ["action_category", "products"]
            fig = px.bar(counts, x="action_category", y="products", text="products")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="", yaxis_title="Products")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered, use_container_width=True)
    else:
        st.info("Click scan to compute guarded recommendations across the catalogue.")


def _options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def _filter_opportunities(
    opportunities: pd.DataFrame,
    actions: list[str],
    statuses: list[str],
    categories: list[str],
) -> pd.DataFrame:
    frame = opportunities.copy()
    for column, values in {"action_category": actions, "status": statuses, "category": categories}.items():
        if values and column in frame.columns:
            frame = frame[frame[column].astype(str).isin(values)]
    return frame
