from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pricelab.modeling.backtest import BacktestResult
from pricelab.modeling.demand_model import train_demand_model
from pricelab.modeling.elasticity import fit_loglog_elasticity
from pricelab.modeling.explainability import elasticity_coefficient_table, permutation_importance_table


def render_explainability_page(df: pd.DataFrame, product_id: str, backtest: BacktestResult | None) -> None:
    st.subheader("Model explainability")
    elasticity = fit_loglog_elasticity(df, product_id)
    coef_table = elasticity_coefficient_table(elasticity)
    st.write("Interpretable log-log model coefficients")
    st.dataframe(coef_table.head(25), use_container_width=True)
    if not coef_table.empty:
        fig = px.bar(coef_table.head(12), x="coefficient", y="feature", orientation="h")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Compute challenger permutation importance"):
        with st.spinner("Training challenger model for explanation..."):
            bundle = train_demand_model(df)
            importance = permutation_importance_table(bundle, df)
        st.dataframe(importance.head(25), use_container_width=True)

    if backtest is not None and backtest.valid:
        st.write("Product-level backtest metrics")
        product_row = backtest.product_metrics[backtest.product_metrics["product_id"].astype(str) == str(product_id)]
        st.dataframe(product_row, use_container_width=True)
    elif backtest is not None:
        st.warning(backtest.message)

