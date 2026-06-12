from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pricelab.analytics.opportunities import scan_catalogue_opportunities
from pricelab.ui.components import app_header, dense_dataframe, opportunity_matrix_chart, section_header, status_pills, style_figure


def render_opportunities_page(
    df: pd.DataFrame,
    objective: str,
    product_metrics: pd.DataFrame | None,
) -> None:
    product_count = int(df["product_id"].nunique())
    app_header(
        "Opportunity scanner",
        "Batch guarded recommendations across the active catalogue scope, then filter and promote products into analysis.",
        [(f"{product_count:,} products", ""), (objective.title(), "ok")],
    )
    if product_count == 0:
        st.warning("No products are available to scan.")
        return

    section_header("Scan controls", "Run explicitly; results are stored in session for filtering and table review.")
    with st.form("opportunity_page_scan"):
        c1, c2, c3 = st.columns([1, 1, 1])
        if product_count <= 5:
            limit = product_count
            c1.caption(f"Scanning all {product_count} products.")
        else:
            limit = c1.slider("Products to scan", min_value=5, max_value=min(20, product_count), value=min(8, product_count))
        replace = c2.checkbox("Replace stored scan", value=True)
        submitted = c3.form_submit_button("Scan opportunities", type="primary", width="stretch")
    if submitted:
        with st.status("Scanning catalogue", expanded=False) as status:
            opportunities = scan_catalogue_opportunities(
                df,
                objective=objective,
                product_backtest_metrics=product_metrics,
                limit=limit,
            )
            if replace or not isinstance(st.session_state.get("opportunity_results"), pd.DataFrame):
                st.session_state["opportunity_results"] = opportunities
            else:
                st.session_state["opportunity_results"] = pd.concat([st.session_state["opportunity_results"], opportunities], ignore_index=True)
            status.update(label="Opportunity scan complete", state="complete")

    opportunities = st.session_state.get("opportunity_results")
    if not isinstance(opportunities, pd.DataFrame) or opportunities.empty:
        st.info("Run a scan to compute guarded recommendations across the catalogue.")
        return

    status_pills([(f"{len(opportunities):,} rows stored", "ok"), ("Session scoped", "")])
    f1, f2, f3 = st.columns(3)
    actions = f1.multiselect("Action", _options(opportunities, "action_category"), placeholder="All actions", key="opp_action_filter")
    statuses = f2.multiselect("Status", _options(opportunities, "status"), placeholder="All statuses", key="opp_status_filter")
    categories = f3.multiselect("Category", _options(opportunities, "category"), placeholder="All categories", key="opp_category_filter")
    filtered = _filter_opportunities(opportunities, actions=actions, statuses=statuses, categories=categories)

    section_header("Opportunity matrix", "Reliability versus impact, sized by expected business value.")
    st.plotly_chart(opportunity_matrix_chart(filtered, objective=objective), width="stretch")
    if not filtered.empty and "action_category" in filtered.columns:
        counts = filtered["action_category"].value_counts().reset_index()
        counts.columns = ["action_category", "products"]
        fig = px.bar(counts, x="action_category", y="products", text="products")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="", yaxis_title="Products")
        st.plotly_chart(style_figure(fig), width="stretch")

    section_header("Recommendation queue", "Select a row and set it as the active product for product or simulator pages.")
    display_cols = [
        "product_id",
        "product_name",
        "category",
        "action_category",
        "status",
        "current_price",
        "recommended_price",
        "price_delta_pct",
        "expected_revenue",
        "expected_margin",
        "opportunity_score",
        "reliability_score",
    ]
    event = dense_dataframe(filtered[[col for col in display_cols if col in filtered.columns]], height=460, key="opportunity_table", selectable=True)
    selected = _selected_row(event)
    if selected is not None and not filtered.empty:
        product_id = str(filtered.iloc[selected]["product_id"])
        c1, c2 = st.columns([1, 3])
        c1.caption(f"Selected: {product_id}")
        if c2.button("Set as active product", type="primary"):
            st.session_state["pending_selected_product"] = product_id
            st.rerun()


def _options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def _filter_opportunities(
    opportunities: pd.DataFrame,
    actions: list[str],
    statuses: list[str],
    categories: list[str],
) -> pd.DataFrame:
    frame = opportunities.copy()
    for column, values in {"action_category": actions, "status": statuses, "category": categories}.items():
        if values and column in frame.columns:
            frame = frame[frame[column].astype(str).isin(values)]
    return frame


def _selected_row(event) -> int | None:
    if event is None:
        return None
    try:
        rows = event.selection.rows
    except AttributeError:
        rows = event.get("selection", {}).get("rows", []) if isinstance(event, dict) else []
    return int(rows[0]) if rows else None
