from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.modeling.backtest import BacktestResult
from pricelab.modeling.demand_model import train_demand_model
from pricelab.modeling.elasticity import fit_loglog_elasticity, fit_segment_elasticities
from pricelab.modeling.explainability import elasticity_coefficient_table, permutation_importance_table
from pricelab.ui.components import (
    app_header,
    coefficient_chart,
    dense_dataframe,
    permutation_importance_chart,
    product_backtest_error_chart,
    section_header,
    segment_elasticity_chart,
    status_pills,
)


def render_explainability_page(df: pd.DataFrame, product_id: str | None, backtest: BacktestResult | None) -> None:
    app_header(
        "Model explainability",
        "Inspect interpretable elasticity, segment behavior, challenger feature importance, and backtest evidence.",
        [(f"Product {product_id}" if product_id else "No product", "" if product_id else "bad")],
    )
    elasticity = fit_loglog_elasticity(df, product_id)
    coef_table = elasticity_coefficient_table(elasticity)
    section_header("Interpretable elasticity model", "Top coefficients from the product-level log-log model.")
    if not coef_table.empty:
        st.plotly_chart(coefficient_chart(coef_table), width="stretch")
    with st.expander("Coefficient table"):
        dense_dataframe(coef_table.head(25), height=360)

    s1, s2 = st.columns(2)
    with s1:
        section_header("Category elasticity", "Segment elasticity by product category.")
        category_segments = fit_segment_elasticities(df, "category").head(20)
        st.plotly_chart(segment_elasticity_chart(category_segments, "Category elasticity"), width="stretch")
        dense_dataframe(category_segments, height=300)
    with s2:
        section_header("Season elasticity", "Segment elasticity by season.")
        season_segments = fit_segment_elasticities(df, "season").head(20)
        st.plotly_chart(segment_elasticity_chart(season_segments, "Season elasticity"), width="stretch")
        dense_dataframe(season_segments, height=300)

    if st.button("Compute challenger permutation importance"):
        with st.spinner("Training challenger model for explanation..."):
            bundle = train_demand_model(df)
            importance = permutation_importance_table(bundle, df)
        section_header("Challenger ML model", "Permutation importance for the Random Forest challenger.")
        st.plotly_chart(permutation_importance_chart(importance), width="stretch")
        dense_dataframe(importance.head(25), height=360)

    if backtest is not None and backtest.valid:
        section_header("Product-level backtest metrics", "Temporal validation evidence by product.")
        product_row = backtest.product_metrics[backtest.product_metrics["product_id"].astype(str) == str(product_id)]
        st.plotly_chart(product_backtest_error_chart(backtest.product_metrics), width="stretch")
        dense_dataframe(product_row, height=220)
    elif backtest is not None:
        st.warning(backtest.message)
    else:
        status_pills([("Backtest not run", "warn")])
