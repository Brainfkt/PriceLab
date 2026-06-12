from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.data.quality import quality_score
from pricelab.schemas import ColumnMapping, DataQualityReport
from pricelab.ui.components import app_header, dense_dataframe, mapping_editor_frame, section_header, status_pills, workflow_rail


def render_data_page(
    raw_df: pd.DataFrame,
    df: pd.DataFrame | None,
    report: DataQualityReport | None,
    mapping: ColumnMapping,
    data_error: str | None = None,
) -> None:
    ready = report is not None and report.error_count == 0 and df is not None
    app_header(
        "Data setup",
        "Validate source columns, quality gates, and model readiness before pricing workflows.",
        [("Ready" if ready else "Action required", "ok" if ready else "bad"), (f"{len(raw_df):,} raw rows", "")],
    )
    workflow_rail(
        [
            ("Source", f"{len(raw_df):,} rows", "done"),
            ("Mapping", "complete" if not data_error else "review", "done" if not data_error else "blocked"),
            ("Quality", "passed" if ready else "required", "done" if ready else "blocked"),
            ("Model frame", "available after validation", "done" if ready else "blocked"),
            ("Analysis", "enabled" if ready else "blocked", "active" if ready else "blocked"),
        ]
    )
    if data_error:
        st.error(data_error)
    if report is None:
        section_header("Mapped schema", "Review inferred source columns before continuing.")
        dense_dataframe(mapping_editor_frame(mapping.model_dump() if hasattr(mapping, "model_dump") else mapping.dict(), set()), height=320)
        with st.expander("Raw preview", expanded=True):
            dense_dataframe(raw_df.head(100), height=360)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{report.row_count:,}")
    c2.metric("Products", f"{report.product_count:,}")
    c3.metric("Quality score", f"{quality_score(report):.0f}/100")
    c4.metric("Issues", f"{report.error_count} errors / {report.warning_count} warnings")

    if report.error_count:
        status_pills([("Fix errors before modelling", "bad")])
    elif report.warning_count:
        status_pills([("Modelling enabled", "ok"), ("Warnings remain", "warn")])
    else:
        status_pills([("Modelling enabled", "ok"), ("No scanner findings", "ok")])

    section_header("Mapped schema", "Canonical fields used by PriceLab after import.")
    dense_dataframe(pd.DataFrame([mapping.as_dict()]).T.rename(columns={0: "source_column"}).reset_index(names="field"), height=300)

    if report.issues:
        section_header("Quality findings", "Errors block modelling; warnings lower recommendation confidence.")
        dense_dataframe(pd.DataFrame([_model_dump(issue) for issue in report.issues]), height=420)
    else:
        st.success("No quality issue detected by the scanner.")

    tabs = st.tabs(["Raw preview", "Standardized preview"])
    with tabs[0]:
        dense_dataframe(raw_df.head(100), height=420)
    with tabs[1]:
        if df is None:
            st.info("Standardized preview is available after mapping validation.")
        else:
            dense_dataframe(df.head(100), height=420)


def _model_dump(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()
