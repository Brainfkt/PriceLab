from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.config import OBJECTIVES, OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from pricelab.data.mapping import missing_required_columns
from pricelab.data.quality import quality_score, scan_quality
from pricelab.modeling.backtest import run_backtest
from pricelab.schemas import ColumnMapping
from pricelab.ui.components import (
    app_header,
    dense_dataframe,
    inject_app_css,
    mapping_editor_frame,
    panel_note,
    status_pills,
    workflow_rail,
)
from pricelab.ui.context import AppContext, GlobalFilters
from pricelab.ui.pages.catalogue import render_catalogue_page
from pricelab.ui.pages.cockpit import render_cockpit_page
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
    inject_app_css()
    context = _build_context()
    _render_page_shell(context)


def _render_page_shell(context: AppContext) -> None:
    pages = {
        "Setup": [
            st.Page(
                lambda: render_data_page(context.raw_df, context.standardized_df, context.quality_report, context.mapping, context.data_error),
                title="Data",
                icon=":material/database:",
                url_path="data",
                default=not context.modelling_ready,
            ),
        ],
    }
    if context.modelling_ready:
        frame = context.filtered_frame
        if frame is None or frame.empty:
            frame = context.frame
        pages.update(
            {
                "Analyse": [
                    st.Page(lambda: render_cockpit_page(context), title="Cockpit", icon=":material/monitoring:", url_path="cockpit", default=True),
                    st.Page(lambda: render_walkthrough_page(frame, context.backtest_result), title="Portfolio", icon=":material/dashboard:", url_path="portfolio"),
                    st.Page(lambda: render_catalogue_page(frame, context.backtest_result), title="Catalogue", icon=":material/table_chart:", url_path="catalogue"),
                    st.Page(lambda: render_product_page(frame, context.selected_product, context.product_metrics), title="Product", icon=":material/query_stats:", url_path="product"),
                ],
                "Decision": [
                    st.Page(lambda: render_simulator_page(frame, context.selected_product, context.objective, context.product_metrics), title="Simulator", icon=":material/tune:", url_path="simulator"),
                    st.Page(lambda: render_opportunities_page(frame, context.objective, context.product_metrics), title="Opportunities", icon=":material/priority_high:", url_path="opportunities"),
                ],
                "Evidence": [
                    st.Page(lambda: render_explainability_page(frame, context.selected_product, context.backtest_result), title="Explainability", icon=":material/schema:", url_path="explainability"),
                    st.Page(lambda: render_export_page(frame, context.selected_product, context.objective, context.product_metrics, context.backtest_result), title="Export", icon=":material/download:", url_path="export"),
                ],
            }
        )
    else:
        pages["Setup"].append(st.Page(lambda: _render_not_ready(context), title="Readiness", icon=":material/rule:", url_path="readiness"))

    page = st.navigation(pages, position="top", expanded=True)
    page.run()


def _build_context() -> AppContext:
    _ensure_session_defaults()
    with st.sidebar:
        st.markdown("### PriceLab")
        panel_note("Analyst workbench for guarded pricing recommendations.")
        raw_df, mapping = _sidebar_data_source()

    data_error = None
    standardized_df = None
    quality_report = None
    frame = None
    backtest_result = None
    selected_product = None
    objective = "revenue"
    filters = GlobalFilters()

    missing = missing_required_columns(mapping)
    if missing:
        data_error = "Missing required mappings: " + ", ".join(missing)
    else:
        try:
            standardized_df = standardize_cached(raw_df, mapping.as_dict())
            quality_report = scan_quality(standardized_df)
        except ValueError as exc:
            data_error = str(exc)

    if standardized_df is not None and quality_report is not None and quality_report.error_count == 0:
        frame = feature_frame_cached(standardized_df)
        if frame.empty:
            data_error = "No modelable rows are available after feature engineering."
        else:
            with st.sidebar:
                st.divider()
                _render_quality_summary(quality_report)
                filters = _sidebar_global_filters(frame)
                scoped = _apply_filters(frame, filters)
                product_options = sorted((scoped if not scoped.empty else frame)["product_id"].astype(str).unique())
                pending_product = st.session_state.pop("pending_selected_product", None)
                current_product = pending_product or st.session_state.get("selected_product")
                product_index = product_options.index(current_product) if current_product in product_options else 0
                selected_product = st.selectbox(
                    "Active product",
                    product_options,
                    index=product_index if product_options else None,
                    key="selected_product",
                )
                objective = st.selectbox("Objective", OBJECTIVES, index=OBJECTIVES.index(st.session_state.get("objective", "revenue")) if st.session_state.get("objective", "revenue") in OBJECTIVES else 1)
                st.session_state["objective"] = objective
                backtest_result = _maybe_backtest(frame)
                _render_workflow_summary(quality_report, frame, backtest_result)
    else:
        with st.sidebar:
            st.divider()
            _render_quality_summary(quality_report)
            _render_workflow_summary(quality_report, frame, backtest_result)

    return AppContext(
        raw_df=raw_df,
        mapping=mapping,
        standardized_df=standardized_df,
        quality_report=quality_report,
        frame=frame,
        backtest_result=backtest_result,
        selected_product=selected_product,
        objective=objective,
        filters=filters,
        data_error=data_error,
    )


def _sidebar_data_source() -> tuple[pd.DataFrame, ColumnMapping]:
    st.markdown("#### Source")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        raw_df = demo_data()
        status_pills([("Demo dataset", "ok"), (f"{len(raw_df):,} rows", "")])
    else:
        raw_df = pd.read_csv(uploaded)
        status_pills([("CSV loaded", "ok"), (f"{len(raw_df):,} rows", "")])

    inferred = default_mapping(raw_df)
    mapping_values: dict[str, str | None] = {}
    options = [""] + [str(column) for column in raw_df.columns]
    with st.expander("Column mapping", expanded=uploaded is not None):
        st.caption("Required")
        for canonical in REQUIRED_COLUMNS:
            inferred_value = getattr(inferred, canonical)
            index = options.index(inferred_value) if inferred_value in options else 0
            value = st.selectbox(canonical, options=options, index=index, key=f"map_{canonical}")
            mapping_values[canonical] = value or None
        st.caption("Optional")
        for canonical in OPTIONAL_COLUMNS:
            inferred_value = getattr(inferred, canonical)
            index = options.index(inferred_value) if inferred_value in options else 0
            value = st.selectbox(canonical, options=options, index=index, key=f"map_{canonical}")
            mapping_values[canonical] = value or None
        dense_dataframe(
            mapping_editor_frame(mapping_values, set(REQUIRED_COLUMNS)),
            height=260,
            column_order=["field", "source_column", "required"],
        )
    return raw_df, ColumnMapping(**mapping_values)


def _render_quality_summary(report) -> None:
    st.markdown("#### Readiness")
    if report is None:
        status_pills([("Mapping required", "bad")])
        return
    score = quality_score(report)
    tone = "ok" if report.error_count == 0 and score >= 75 else "warn" if report.error_count == 0 else "bad"
    status_pills(
        [
            (f"Quality {score:.0f}/100", tone),
            (f"{report.error_count} errors", "bad" if report.error_count else "ok"),
            (f"{report.warning_count} warnings", "warn" if report.warning_count else "ok"),
        ]
    )


def _render_workflow_summary(report, frame, backtest_result) -> None:
    data_state = "done" if report is not None and report.error_count == 0 else "blocked"
    model_state = "done" if frame is not None and not frame.empty else "blocked"
    backtest_state = "done" if backtest_result is not None and backtest_result.valid else "active"
    steps = [
        ("Data", "validated" if data_state == "done" else "required", data_state),
        ("Model frame", "ready" if model_state == "done" else "blocked", model_state),
        ("Backtest", "available" if backtest_state == "done" else "optional", backtest_state),
        ("Analyze", "portfolio scope", "done" if model_state == "done" else "blocked"),
        ("Decide", "simulate/export", "active" if model_state == "done" else "blocked"),
    ]
    workflow_rail(steps)


def _sidebar_global_filters(frame: pd.DataFrame) -> GlobalFilters:
    st.markdown("#### Scope")
    return GlobalFilters(
        categories=st.multiselect("Categories", _options(frame, "category"), placeholder="All categories"),
        channels=st.multiselect("Channels", _options(frame, "channel"), placeholder="All channels"),
        regions=st.multiselect("Regions", _options(frame, "region"), placeholder="All regions"),
    )


def _render_not_ready(context: AppContext) -> None:
    app_header(
        "Readiness",
        "Resolve import, mapping, and quality gates before running modelled pricing workflows.",
        [("Setup", "active"), ("Modelling blocked", "bad")],
    )
    if context.data_error:
        st.error(context.data_error)
    render_data_page(context.raw_df, context.standardized_df, context.quality_report, context.mapping, context.data_error)


def _apply_filters(frame: pd.DataFrame, filters: GlobalFilters) -> pd.DataFrame:
    out = frame.copy()
    for column, values in {
        "category": filters.categories,
        "channel": filters.channels,
        "region": filters.regions,
    }.items():
        if values and column in out.columns:
            out = out[out[column].astype(str).isin(values)]
    return out


def _options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def _ensure_session_defaults() -> None:
    st.session_state.setdefault("opportunity_results", None)
    st.session_state.setdefault("scenario_comparisons", [])
    st.session_state.setdefault("objective", "revenue")


@st.cache_data(show_spinner="Running temporal backtest...")
def _run_backtest_cached(frame: pd.DataFrame):
    return run_backtest(frame, n_splits=3)


def _maybe_backtest(frame: pd.DataFrame):
    run_it = st.checkbox("Run temporal backtest", value=False)
    if not run_it:
        return None
    with st.status("Temporal validation", expanded=False) as status:
        result = _run_backtest_cached(frame)
        if result.valid:
            status.update(label="Temporal validation ready", state="complete")
        else:
            status.update(label="Temporal validation unavailable", state="error")
        return result
