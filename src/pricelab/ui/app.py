from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.config import OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from pricelab.data.mapping import missing_required_columns
from pricelab.data.quality import scan_quality
from pricelab.modeling.backtest import run_backtest
from pricelab.schemas import ColumnMapping
from pricelab.ui.pages.catalogue import render_catalogue_page
from pricelab.ui.pages.data import render_data_page
from pricelab.ui.pages.explainability import render_explainability_page
from pricelab.ui.pages.export import render_export_page
from pricelab.ui.pages.opportunities import render_opportunities_page
from pricelab.ui.pages.product import render_product_page
from pricelab.ui.pages.simulator import render_simulator_page
from pricelab.ui.pages.walkthrough import render_walkthrough_page
from pricelab.ui.state import default_mapping, demo_data, feature_frame_cached, standardize_cached


def run_app() -> None:
    st.set_page_config(page_title="PriceLab", page_icon="PL", layout="wide")
    st.title("PriceLab")
    st.caption("Local pricing intelligence simulator with guarded ML recommendations.")

    raw_df, mapping = _sidebar_data_source()
    missing = missing_required_columns(mapping)
    if missing:
        st.error("Missing required mappings: " + ", ".join(missing))
        st.stop()

    try:
        df = standardize_cached(raw_df, mapping.as_dict())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    quality_report = scan_quality(df)
    page = st.sidebar.radio(
        "Page",
        [
            "Data",
            "Portfolio",
            "Catalogue",
            "Product",
            "Simulator",
            "Opportunities",
            "Explainability",
            "Export",
        ],
        index=2,
    )
    if quality_report.error_count:
        st.error("Fix data quality errors before running modelling or recommendations.")
        render_data_page(raw_df, df, quality_report, mapping)
        st.stop()
    if page == "Data":
        render_data_page(raw_df, df, quality_report, mapping)
        return

    frame = feature_frame_cached(df)
    if frame.empty:
        st.error("No modelable rows are available after feature engineering.")
        st.stop()

    backtest_result = _maybe_backtest(frame)
    product_metrics = backtest_result.product_metrics if backtest_result and backtest_result.valid else None

    products = sorted(frame["product_id"].astype(str).unique())
    selected_product = None
    if page in {"Product", "Simulator", "Explainability", "Export"}:
        selected_product = st.sidebar.selectbox("Product", products, index=0 if products else None)
    objective = "revenue"
    if page in {"Simulator", "Opportunities", "Export"}:
        objective = st.sidebar.selectbox("Objective", ["revenue", "margin", "volume", "prudence"], index=0)

    if page == "Portfolio":
        render_walkthrough_page(frame, backtest_result)
    elif page == "Catalogue":
        render_catalogue_page(frame, backtest_result)
    elif page == "Product":
        render_product_page(frame, selected_product, product_metrics)
    elif page == "Simulator":
        render_simulator_page(frame, selected_product, objective, product_metrics)
    elif page == "Opportunities":
        render_opportunities_page(frame, objective, product_metrics)
    elif page == "Explainability":
        render_explainability_page(frame, selected_product, backtest_result)
    elif page == "Export":
        render_export_page(frame, selected_product, objective, product_metrics, backtest_result)


def _sidebar_data_source() -> tuple[pd.DataFrame, ColumnMapping]:
    st.sidebar.header("Data source")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        raw_df = demo_data()
        st.sidebar.info("Using generated demo dataset.")
    else:
        raw_df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {len(raw_df):,} rows.")

    inferred = default_mapping(raw_df)
    mapping_values: dict[str, str | None] = {}
    options = [""] + [str(column) for column in raw_df.columns]
    with st.sidebar.expander("Column mapper", expanded=uploaded is not None):
        st.write("Required columns")
        for canonical in REQUIRED_COLUMNS:
            inferred_value = getattr(inferred, canonical)
            index = options.index(inferred_value) if inferred_value in options else 0
            value = st.selectbox(canonical, options=options, index=index, key=f"map_{canonical}")
            mapping_values[canonical] = value or None
        st.write("Optional columns")
        for canonical in OPTIONAL_COLUMNS:
            inferred_value = getattr(inferred, canonical)
            index = options.index(inferred_value) if inferred_value in options else 0
            value = st.selectbox(canonical, options=options, index=index, key=f"map_{canonical}")
            mapping_values[canonical] = value or None
    return raw_df, ColumnMapping(**mapping_values)


@st.cache_data(show_spinner="Running temporal backtest...")
def _run_backtest_cached(frame: pd.DataFrame):
    return run_backtest(frame, n_splits=3)


def _maybe_backtest(frame: pd.DataFrame):
    run_it = st.sidebar.checkbox("Run temporal backtest", value=False)
    if not run_it:
        return None
    return _run_backtest_cached(frame)
