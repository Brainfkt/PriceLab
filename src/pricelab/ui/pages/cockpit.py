from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pricelab.analytics.catalogue import catalogue_kpis, category_mix, portfolio_health, product_leaderboard
from pricelab.analytics.opportunities import scan_catalogue_opportunities
from pricelab.ui.components import (
    app_header,
    category_mix_chart,
    compact_number,
    dense_dataframe,
    opportunity_matrix_chart,
    portfolio_health_chart,
    portfolio_scatter_chart,
    section_header,
    status_pills,
    style_figure,
)
from pricelab.ui.context import AppContext


def render_cockpit_page(context: AppContext) -> None:
    frame = context.filtered_frame if context.filtered_frame is not None else context.frame
    if frame is None or frame.empty:
        app_header("Cockpit", "No modelable rows are available in the current scope.", [("Scope empty", "bad")])
        return

    backtest_label = "Backtest ready" if context.backtest_result is not None and context.backtest_result.valid else "Backtest optional"
    app_header(
        "Analyst cockpit",
        "Dense portfolio view for prioritizing pricing moves, inspecting reliability, and moving into simulation.",
        [
            (f"{frame['product_id'].nunique():,} products", ""),
            (context.objective.title(), "ok"),
            (backtest_label, "ok" if context.backtest_result is not None and context.backtest_result.valid else "warn"),
        ],
    )

    kpis = catalogue_kpis(frame)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Products", f"{kpis['products']:,}")
    c2.metric("Units", compact_number(kpis["units"]))
    c3.metric("Revenue", compact_number(kpis["revenue"]))
    c4.metric("Gross margin", compact_number(kpis["gross_margin"]))
    c5.metric("Promo rate", f"{kpis['promo_rate']:.1%}")

    section_header("Priority board", "Scan or reuse guarded recommendations for the filtered catalogue.")
    left, right = st.columns([1.55, 1])
    with left:
        opportunities = _opportunity_controls(frame, context)
        if opportunities is not None and not opportunities.empty:
            st.plotly_chart(opportunity_matrix_chart(opportunities, objective=context.objective), width="stretch")
        else:
            _empty_priority_view(frame)
    with right:
        _render_recommendation_queue(opportunities)

    section_header("Portfolio diagnostics", "Revenue, margin, category mix, and reliability at portfolio level.")
    a, b = st.columns([1.25, 1])
    with a:
        all_products = product_leaderboard(frame, metric="revenue", top_n=None)
        st.plotly_chart(portfolio_scatter_chart(all_products), width="stretch")
    with b:
        health = portfolio_health(frame, context.product_metrics)
        st.plotly_chart(portfolio_health_chart(health), width="stretch")
        st.plotly_chart(category_mix_chart(category_mix(frame)), width="stretch")

    section_header("Product worklist", "Sortable product-level facts for analyst triage.")
    leaderboard = product_leaderboard(frame, metric="revenue", top_n=None)
    display_cols = [
        "product_id",
        "product_name",
        "category",
        "units",
        "revenue",
        "gross_margin",
        "margin_rate",
        "avg_price",
        "price_points",
        "promo_rate",
        "stockout_rate",
    ]
    dense_dataframe(leaderboard[[col for col in display_cols if col in leaderboard.columns]], height=440)


def _opportunity_controls(frame: pd.DataFrame, context: AppContext) -> pd.DataFrame | None:
    product_count = int(frame["product_id"].nunique())
    stored = st.session_state.get("opportunity_results")
    with st.form("opportunity_scan_form"):
        c1, c2, c3 = st.columns([1, 1, 1])
        limit = c1.slider("Products to scan", min_value=5, max_value=max(5, min(20, product_count)), value=min(8, max(5, product_count)))
        clear = c2.checkbox("Replace previous scan", value=True)
        submitted = c3.form_submit_button("Scan opportunities", type="primary", width="stretch")
    if submitted:
        with st.status("Scanning recommendations", expanded=False) as status:
            opportunities = scan_catalogue_opportunities(
                frame,
                objective=context.objective,
                product_backtest_metrics=context.product_metrics,
                limit=limit,
            )
            st.session_state["opportunity_results"] = opportunities if clear or stored is None else pd.concat([stored, opportunities], ignore_index=True)
            status.update(label="Opportunity scan complete", state="complete")
            stored = st.session_state["opportunity_results"]
    if isinstance(stored, pd.DataFrame) and not stored.empty:
        status_pills([(f"{len(stored):,} recommendations", "ok"), ("Stored in session", "")])
        return _filter_opportunities(stored)
    status_pills([("No scan yet", "warn")])
    return None


def _filter_opportunities(opportunities: pd.DataFrame) -> pd.DataFrame:
    f1, f2, f3 = st.columns(3)
    actions = f1.multiselect("Action", _options(opportunities, "action_category"), placeholder="All actions", key="cockpit_action_filter")
    statuses = f2.multiselect("Status", _options(opportunities, "status"), placeholder="All statuses", key="cockpit_status_filter")
    categories = f3.multiselect("Category", _options(opportunities, "category"), placeholder="All categories", key="cockpit_category_filter")
    filtered = opportunities.copy()
    for column, values in {"action_category": actions, "status": statuses, "category": categories}.items():
        if values and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str).isin(values)]
    return filtered


def _render_recommendation_queue(opportunities: pd.DataFrame | None) -> None:
    if opportunities is None or opportunities.empty:
        st.markdown("<div class='pl-panel'>Run a scan to populate the recommendation queue.</div>", unsafe_allow_html=True)
        return
    sort_cols = [col for col in ["opportunity_score", "reliability_score"] if col in opportunities.columns]
    queue = opportunities.sort_values(sort_cols, ascending=False).head(15) if sort_cols else opportunities.head(15)
    cols = [
        "product_id",
        "product_name",
        "action_category",
        "status",
        "current_price",
        "recommended_price",
        "price_delta_pct",
        "opportunity_score",
        "reliability_score",
    ]
    event = dense_dataframe(queue[[col for col in cols if col in queue.columns]], height=430, key="cockpit_queue", selectable=True)
    selected = _selected_row(event)
    if selected is not None:
        product_id = str(queue.iloc[selected]["product_id"])
        st.caption(f"Selected product: {product_id}")
        if st.button("Set as active product", type="primary", width="stretch"):
            st.session_state["pending_selected_product"] = product_id
            st.rerun()


def _empty_priority_view(frame: pd.DataFrame) -> None:
    products = product_leaderboard(frame, metric="revenue", top_n=15)
    if products.empty:
        st.info("No products are available in the selected scope.")
        return
    fig = px.bar(
        products.sort_values("revenue"),
        x="revenue",
        y="product_id",
        orientation="h",
        color="category" if "category" in products.columns else None,
        labels={"revenue": "Revenue", "product_id": "Product"},
    )
    st.plotly_chart(style_figure(fig, height=430), width="stretch")


def _selected_row(event) -> int | None:
    if event is None:
        return None
    try:
        rows = event.selection.rows
    except AttributeError:
        rows = event.get("selection", {}).get("rows", []) if isinstance(event, dict) else []
    return int(rows[0]) if rows else None


def _options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())
