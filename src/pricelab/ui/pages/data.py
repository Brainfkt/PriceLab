from __future__ import annotations

import pandas as pd
import streamlit as st

from pricelab.data.quality import quality_score
from pricelab.schemas import ColumnMapping, DataQualityReport


def render_data_page(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    report: DataQualityReport,
    mapping: ColumnMapping,
) -> None:
    st.subheader("Data import and quality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{report.row_count:,}")
    c2.metric("Products", f"{report.product_count:,}")
    c3.metric("Quality score", f"{quality_score(report):.0f}/100")
    c4.metric("Issues", f"{report.error_count} errors / {report.warning_count} warnings")

    st.write("Mapped schema")
    st.dataframe(pd.DataFrame([mapping.as_dict()]).T.rename(columns={0: "source_column"}), use_container_width=True)

    if report.issues:
        st.write("Quality findings")
        st.dataframe(pd.DataFrame([issue.dict() for issue in report.issues]), use_container_width=True)
    else:
        st.success("No quality issue detected by the scanner.")

    with st.expander("Raw preview"):
        st.dataframe(raw_df.head(100), use_container_width=True)
    with st.expander("Standardized preview"):
        st.dataframe(df.head(100), use_container_width=True)
