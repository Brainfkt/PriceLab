from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.analytics.opportunities import scan_catalogue_opportunities


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
        st.dataframe(opportunities, use_container_width=True)
    else:
        st.info("Click scan to compute guarded recommendations across the catalogue.")
