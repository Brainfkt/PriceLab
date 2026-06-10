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
    limit = st.slider("Products to scan", min_value=5, max_value=min(50, int(df["product_id"].nunique())), value=min(20, int(df["product_id"].nunique())))
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

