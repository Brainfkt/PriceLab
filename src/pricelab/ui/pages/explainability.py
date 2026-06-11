from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.modeling.backtest import BacktestResult
from pricelab.modeling.demand_model import train_demand_model
from pricelab.modeling.elasticity import fit_loglog_elasticity, fit_segment_elasticities
from pricelab.modeling.explainability import elasticity_coefficient_table, permutation_importance_table
from pricelab.ui.components import coefficient_chart, permutation_importance_chart, product_backtest_error_chart, segment_elasticity_chart


def render_explainability_page(df: pd.DataFrame, product_id: str, backtest: BacktestResult | None) -> None:
    st.subheader("Model explainability")
    elasticity = fit_loglog_elasticity(df, product_id)
    coef_table = elasticity_coefficient_table(elasticity)
    st.write("Interpretable elasticity model")
    if not coef_table.empty:
        st.plotly_chart(coefficient_chart(coef_table), use_container_width=True)
    with st.expander("Coefficient table"):
        st.dataframe(coef_table.head(25), use_container_width=True)

    s1, s2 = st.columns(2)
    with s1:
        st.write("Category elasticity")
        category_segments = fit_segment_elasticities(df, "category").head(20)
        st.plotly_chart(segment_elasticity_chart(category_segments, "Category elasticity"), use_container_width=True)
        st.dataframe(category_segments, use_container_width=True)
    with s2:
        st.write("Season elasticity")
        season_segments = fit_segment_elasticities(df, "season").head(20)
        st.plotly_chart(segment_elasticity_chart(season_segments, "Season elasticity"), use_container_width=True)
        st.dataframe(season_segments, use_container_width=True)

    if st.button("Compute challenger permutation importance"):
        with st.spinner("Training challenger model for explanation..."):
            bundle = train_demand_model(df)
            importance = permutation_importance_table(bundle, df)
        st.write("Challenger ML model")
        st.plotly_chart(permutation_importance_chart(importance), use_container_width=True)
        st.dataframe(importance.head(25), use_container_width=True)

    if backtest is not None and backtest.valid:
        st.write("Product-level backtest metrics")
        product_row = backtest.product_metrics[backtest.product_metrics["product_id"].astype(str) == str(product_id)]
        st.plotly_chart(product_backtest_error_chart(backtest.product_metrics), use_container_width=True)
        st.dataframe(product_row, use_container_width=True)
    elif backtest is not None:
        st.warning(backtest.message)
