from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.modeling.elasticity import fit_loglog_elasticity
from pricelab.modeling.optimization import find_price_recommendation
from pricelab.modeling.reliability import compute_reliability
from pricelab.reporting.html import markdown_to_basic_html
from pricelab.reporting.markdown import build_product_markdown_report


def render_export_page(
    df: pd.DataFrame,
    product_id: str,
    objective: str,
    product_metrics: pd.DataFrame | None,
) -> None:
    st.subheader("Export product report")
    reliability = compute_reliability(df, product_id, product_backtest_metrics=product_metrics, objective=objective)
    recommendation = find_price_recommendation(df, product_id, objective=objective, product_backtest_metrics=product_metrics)
    elasticity = fit_loglog_elasticity(df, product_id)
    markdown_report = build_product_markdown_report(df, product_id, reliability, recommendation, elasticity)
    html_report = markdown_to_basic_html(markdown_report)
    st.download_button("Download Markdown report", markdown_report, file_name=f"pricelab_{product_id}.md", mime="text/markdown")
    st.download_button("Download HTML report", html_report, file_name=f"pricelab_{product_id}.html", mime="text/html")
    st.text_area("Report preview", markdown_report, height=520)

