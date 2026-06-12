from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


PLOTLY_TEMPLATE = "plotly_white"
ACCENT = "#0f766e"
ACCENT_BLUE = "#2563eb"
BORDER = "#dbe3ef"
SURFACE = "#f8fafc"
TEXT = "#111827"
MUTED = "#64748b"

METRIC_LABELS = {
    "units": "Units",
    "revenue": "Revenue",
    "gross_margin": "Gross margin",
    "avg_price": "Average price",
    "price_points": "Price points",
    "margin_rate": "Margin rate",
    "opportunity_score": "Opportunity score",
    "reliability_score": "Reliability score",
}


def inject_app_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --pl-accent: #0f766e;
            --pl-blue: #2563eb;
            --pl-surface: #f8fafc;
            --pl-border: #dbe3ef;
            --pl-text: #111827;
            --pl-muted: #64748b;
            --pl-red: #dc2626;
            --pl-orange: #d97706;
            --pl-green: #15803d;
        }
        .stApp {
            background: #ffffff;
            color: var(--pl-text);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }
        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid var(--pl-border);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            font-size: 0.86rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--pl-border);
            border-radius: 8px;
            padding: 0.72rem 0.82rem;
            min-height: 86px;
        }
        [data-testid="stMetricLabel"] {
            color: var(--pl-muted);
            font-size: 0.76rem;
            letter-spacing: 0;
        }
        [data-testid="stMetricValue"] {
            color: var(--pl-text);
            font-size: 1.38rem;
            font-weight: 720;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--pl-border);
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stExpander"] {
            border-color: var(--pl-border);
            border-radius: 8px;
        }
        .pl-hero {
            border: 1px solid var(--pl-border);
            border-radius: 10px;
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 72%, #eef6f5 100%);
            padding: 1.0rem 1.1rem;
            margin: 0.25rem 0 1rem;
        }
        .pl-title {
            font-size: 1.42rem;
            line-height: 1.2;
            font-weight: 760;
            margin: 0;
            letter-spacing: 0;
        }
        .pl-subtitle {
            color: var(--pl-muted);
            margin: 0.25rem 0 0;
            font-size: 0.93rem;
            line-height: 1.45;
        }
        .pl-section {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1rem;
            border-bottom: 1px solid var(--pl-border);
            padding-bottom: 0.45rem;
            margin: 1.15rem 0 0.7rem;
        }
        .pl-section h2 {
            font-size: 1.0rem;
            margin: 0;
            font-weight: 720;
            letter-spacing: 0;
        }
        .pl-section p {
            color: var(--pl-muted);
            margin: 0.18rem 0 0;
            font-size: 0.82rem;
        }
        .pl-panel {
            background: #ffffff;
            border: 1px solid var(--pl-border);
            border-radius: 8px;
            padding: 0.86rem;
            min-height: 100%;
        }
        .pl-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.38rem;
            margin: 0.45rem 0 0.2rem;
        }
        .pl-pill {
            border: 1px solid var(--pl-border);
            border-radius: 999px;
            padding: 0.22rem 0.52rem;
            font-size: 0.76rem;
            color: var(--pl-muted);
            background: #ffffff;
            white-space: nowrap;
        }
        .pl-pill.ok { color: var(--pl-green); border-color: #bbf7d0; background: #f0fdf4; }
        .pl-pill.warn { color: var(--pl-orange); border-color: #fed7aa; background: #fff7ed; }
        .pl-pill.bad { color: var(--pl-red); border-color: #fecaca; background: #fef2f2; }
        .pl-workflow {
            display: grid;
            grid-template-columns: repeat(5, minmax(110px, 1fr));
            gap: 0.45rem;
            margin: 0.15rem 0 0.85rem;
        }
        .pl-step {
            border: 1px solid var(--pl-border);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.55rem 0.65rem;
        }
        .pl-step strong {
            display: block;
            font-size: 0.78rem;
            color: var(--pl-text);
        }
        .pl-step span {
            color: var(--pl-muted);
            font-size: 0.72rem;
        }
        .pl-step.done { border-color: #99f6e4; background: #f0fdfa; }
        .pl-step.active { border-color: #bfdbfe; background: #eff6ff; }
        .pl-step.blocked { border-color: #fecaca; background: #fef2f2; }
        .pl-note {
            color: var(--pl-muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }
        .stButton>button, .stDownloadButton>button {
            border-radius: 7px;
            font-weight: 650;
        }
        @media (max-width: 860px) {
            .block-container { padding-left: 0.7rem; padding-right: 0.7rem; }
            .pl-workflow { grid-template-columns: 1fr; }
            .pl-section { align-items: start; flex-direction: column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def app_header(title: str, subtitle: str, pills: list[tuple[str, str]] | None = None) -> None:
    pill_html = ""
    if pills:
        pill_html = "<div class='pl-pill-row'>" + "".join(
            f"<span class='pl-pill {tone}'>{label}</span>" for label, tone in pills
        ) + "</div>"
    st.markdown(
        f"""
        <div class="pl-hero">
            <h1 class="pl-title">{title}</h1>
            <p class="pl-subtitle">{subtitle}</p>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str = "", action: str = "") -> None:
    action_html = f"<span class='pl-note'>{action}</span>" if action else ""
    st.markdown(
        f"""
        <div class="pl-section">
            <div><h2>{title}</h2><p>{caption}</p></div>
            {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_rail(steps: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, detail, state in steps:
        cards.append(f"<div class='pl-step {state}'><strong>{label}</strong><span>{detail}</span></div>")
    st.markdown("<div class='pl-workflow'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def status_pills(items: list[tuple[str, str]]) -> None:
    st.markdown(
        "<div class='pl-pill-row'>" + "".join(f"<span class='pl-pill {tone}'>{label}</span>" for label, tone in items) + "</div>",
        unsafe_allow_html=True,
    )


def panel_note(text: str) -> None:
    st.markdown(f"<p class='pl-note'>{text}</p>", unsafe_allow_html=True)


def dense_dataframe(
    df: pd.DataFrame,
    *,
    height: int = 360,
    key: str | None = None,
    selectable: bool = False,
    column_order: list[str] | None = None,
):
    if df.empty:
        st.info("No rows to display for the current scope.")
        return None
    kwargs = {
        "width": "stretch",
        "height": height,
        "hide_index": True,
        "column_config": dataframe_column_config(df),
    }
    if column_order is not None:
        kwargs["column_order"] = [col for col in column_order if col in df.columns]
    if selectable:
        return st.dataframe(
            df,
            key=key,
            on_select="rerun",
            selection_mode="single-row",
            **kwargs,
        )
    return st.dataframe(df, **kwargs)


def mapping_editor_frame(mapping: dict[str, str | None], required: set[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field": field,
                "source_column": source or "",
                "required": field in required,
            }
            for field, source in mapping.items()
        ]
    )


def dataframe_column_config(df: pd.DataFrame) -> dict[str, object]:
    config: dict[str, object] = {}
    money_cols = {
        "revenue",
        "gross_margin",
        "avg_price",
        "price",
        "current_price",
        "recommended_price",
        "lower_price",
        "upper_price",
        "expected_revenue",
        "expected_margin",
        "predicted_revenue",
        "predicted_margin",
        "cost",
    }
    pct_cols = {
        "margin_rate",
        "promo_rate",
        "stockout_rate",
        "discount_rate",
        "price_delta_pct",
        "wmape",
        "baseline_wmape",
        "smape",
    }
    score_cols = {"reliability_score", "opportunity_score", "score"}
    for column in df.columns:
        if column in money_cols:
            config[column] = st.column_config.NumberColumn(METRIC_LABELS.get(column, column.replace("_", " ").title()), format="$%.2f")
        elif column in pct_cols:
            config[column] = st.column_config.NumberColumn(METRIC_LABELS.get(column, column.replace("_", " ").title()), format="%.1f%%")
        elif column in score_cols:
            config[column] = st.column_config.ProgressColumn(
                METRIC_LABELS.get(column, column.replace("_", " ").title()),
                min_value=0,
                max_value=100,
                format="%.0f",
            )
        elif column in {"units", "units_sold", "predicted_units", "expected_units"}:
            config[column] = st.column_config.NumberColumn(METRIC_LABELS.get(column, column.replace("_", " ").title()), format="%.0f")
    return config


def style_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color=TEXT, size=12),
        colorway=["#0f766e", "#2563eb", "#d97706", "#dc2626", "#7c3aed", "#475569"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zerolinecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zerolinecolor="#cbd5e1")
    if height is not None:
        fig.update_layout(height=height)
    return fig


def compact_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{sign}{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}{value / 1_000:.1f}K"
    return f"{sign}{value:.0f}"


def product_leaderboard_chart(leaderboard: pd.DataFrame, metric: str = "revenue") -> go.Figure:
    metric = metric if metric in leaderboard.columns else "revenue"
    if leaderboard.empty:
        return _empty_figure("No products match the selected filters.", height=360)
    frame = leaderboard.sort_values(metric, ascending=True).copy()
    height = int(min(1100, max(380, 28 * len(frame) + 120)))
    fig = px.bar(
        frame,
        x=metric,
        y="product_id",
        color="category" if "category" in frame.columns else None,
        orientation="h",
        hover_data=[col for col in ["product_name", "units", "revenue", "gross_margin", "margin_rate", "avg_price"] if col in frame.columns],
        labels={metric: METRIC_LABELS.get(metric, metric), "product_id": "Product"},
    )
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    return fig


def pareto_chart(pareto: pd.DataFrame, metric: str = "revenue") -> go.Figure:
    metric = metric if metric in pareto.columns else "revenue"
    if pareto.empty:
        return _empty_figure("No Pareto view is available.", height=360)
    labels = pareto["product_id"].astype(str)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=pareto[metric], name=METRIC_LABELS.get(metric, metric)), secondary_y=False)
    fig.add_trace(
        go.Scatter(x=labels, y=pareto["cumulative_share"] * 100, mode="lines+markers", name="Cumulative share"),
        secondary_y=True,
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    fig.update_yaxes(title_text=METRIC_LABELS.get(metric, metric), secondary_y=False)
    fig.update_yaxes(title_text="Cumulative share (%)", range=[0, 105], secondary_y=True)
    return fig


def portfolio_scatter_chart(products: pd.DataFrame) -> go.Figure:
    if products.empty:
        return _empty_figure("No portfolio scatter is available.", height=420)
    frame = products.copy()
    frame["margin_rate_pct"] = frame.get("margin_rate", 0.0) * 100
    fig = px.scatter(
        frame,
        x="revenue",
        y="margin_rate_pct",
        color="category" if "category" in frame.columns else None,
        size="units" if "units" in frame.columns else None,
        hover_name="product_name" if "product_name" in frame.columns else "product_id",
        hover_data=[col for col in ["product_id", "gross_margin", "avg_price", "promo_rate", "stockout_rate"] if col in frame.columns],
        labels={"revenue": "Revenue", "margin_rate_pct": "Margin rate (%)"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    return fig


def portfolio_trend_chart(trend: pd.DataFrame) -> go.Figure:
    if trend.empty:
        return _empty_figure("No time series is available.", height=360)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["revenue"], mode="lines", name="Revenue"), secondary_y=False)
    if "gross_margin" in trend.columns:
        fig.add_trace(go.Scatter(x=trend["date"], y=trend["gross_margin"], mode="lines", name="Gross margin"), secondary_y=False)
    fig.add_trace(go.Bar(x=trend["date"], y=trend["units"], name="Units", opacity=0.25), secondary_y=True)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Revenue / margin", secondary_y=False)
    fig.update_yaxes(title_text="Units", secondary_y=True)
    return fig


def category_mix_chart(mix: pd.DataFrame) -> go.Figure:
    if mix.empty:
        return _empty_figure("No category mix is available.", height=360)
    frame = mix.sort_values("revenue", ascending=False)
    fig = px.bar(
        frame,
        x="category",
        y=["revenue", "gross_margin"],
        barmode="group",
        hover_data=[col for col in ["products", "units", "margin_rate"] if col in frame.columns],
        labels={"value": "Amount", "variable": "Metric", "category": "Category"},
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    return fig


def portfolio_health_chart(health: pd.DataFrame) -> go.Figure:
    if health.empty:
        return _empty_figure("No reliability summary is available.", height=320)
    order = ["blocked", "simulation_only", "cautious", "normal"]
    counts = health["reliability_level"].value_counts().reindex(order, fill_value=0).reset_index()
    counts.columns = ["reliability_level", "products"]
    fig = px.bar(counts, x="reliability_level", y="products", text="products", labels={"reliability_level": "Reliability level"})
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=25, b=10), xaxis_title="", yaxis_title="Products")
    return fig


def price_units_chart(df: pd.DataFrame, product_id: str) -> go.Figure:
    product = _product_time_series(df, product_id)
    if product.empty:
        return _empty_figure("No product history is available.", height=420)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=product["date"], y=product["price"], mode="lines+markers", name="Price"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=product["date"], y=product["units_sold"], name="Units", opacity=0.35),
        secondary_y=True,
    )
    promo = product[product["promo_rate"] > 0] if "promo_rate" in product.columns else pd.DataFrame()
    if not promo.empty:
        fig.add_trace(
            go.Scatter(x=promo["date"], y=promo["price"], mode="markers", name="Promotion", marker=dict(symbol="diamond", size=10)),
            secondary_y=False,
        )
    stockout = product[product["stockout_rate"] > 0] if "stockout_rate" in product.columns else pd.DataFrame()
    if not stockout.empty:
        fig.add_trace(
            go.Scatter(x=stockout["date"], y=stockout["price"], mode="markers", name="Stock pressure", marker=dict(symbol="x", size=10)),
            secondary_y=False,
        )
    current_price = float(product["price"].iloc[-1])
    fig.add_hline(y=current_price, line_dash="dot", line_color="#64748b", annotation_text="Current price", annotation_position="top left")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Units sold", secondary_y=True)
    return fig


def revenue_margin_chart(df: pd.DataFrame, product_id: str) -> go.Figure:
    product = _product_time_series(df, product_id)
    if product.empty:
        return _empty_figure("No revenue history is available.", height=360)
    fig = px.line(product, x="date", y=["revenue", "gross_margin"], labels={"value": "Amount", "variable": "Metric"})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


def price_bin_chart(price_bins: pd.DataFrame) -> go.Figure:
    if price_bins.empty:
        return _empty_figure("Not enough price variation", height=320)
    labels = price_bins.apply(lambda r: f"{r['price_min']:.2f}-{r['price_max']:.2f}", axis=1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=price_bins["units"], name="Units"), secondary_y=False)
    fig.add_trace(go.Scatter(x=labels, y=price_bins["revenue"], mode="lines+markers", name="Revenue"), secondary_y=True)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Observed price bins")
    fig.update_yaxes(title_text="Units", secondary_y=False)
    fig.update_yaxes(title_text="Revenue", secondary_y=True)
    return fig


def reliability_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 35], "color": "#fee2e2"},
                    {"range": [35, 55], "color": "#ffedd5"},
                    {"range": [55, 75], "color": "#fef9c3"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def reliability_components_chart(components: dict[str, float]) -> go.Figure:
    frame = pd.DataFrame({"component": list(components.keys()), "score": [v * 100 for v in components.values()]})
    fig = px.bar(frame, x="score", y="component", orientation="h", range_x=[0, 100])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def scenario_curve_chart(
    rows: pd.DataFrame,
    current_price: float | None = None,
    selected_price: float | None = None,
    recommended_price: float | None = None,
    observed_low: float | None = None,
    observed_high: float | None = None,
) -> go.Figure:
    if rows.empty:
        return _empty_figure("No scenario curve available", height=360)
    rows = rows.sort_values("price").copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if {"low_units", "high_units"}.issubset(rows.columns):
        fig.add_trace(
            go.Scatter(
                x=pd.concat([rows["price"], rows["price"].iloc[::-1]]),
                y=pd.concat([rows["high_units"], rows["low_units"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.12)",
                line=dict(color="rgba(37, 99, 235, 0)"),
                name="Units uncertainty",
                hoverinfo="skip",
            ),
            secondary_y=False,
        )
    fig.add_trace(go.Scatter(x=rows["price"], y=rows["predicted_units"], mode="lines", name="Units"), secondary_y=False)
    fig.add_trace(go.Scatter(x=rows["price"], y=rows["predicted_revenue"], mode="lines", name="Revenue"), secondary_y=True)
    if "predicted_margin" in rows.columns and rows["predicted_margin"].notna().any():
        fig.add_trace(go.Scatter(x=rows["price"], y=rows["predicted_margin"], mode="lines", name="Margin"), secondary_y=True)
    if observed_low is not None and observed_high is not None and np.isfinite(observed_low) and np.isfinite(observed_high):
        fig.add_vrect(x0=observed_low, x1=observed_high, fillcolor="#e2e8f0", opacity=0.25, line_width=0, annotation_text="Observed range")
    for value, label, color, dash in [
        (current_price, "Current", "#64748b", "dot"),
        (selected_price, "Selected", "#2563eb", "dash"),
        (recommended_price, "Recommended", "#16a34a", "dashdot"),
    ]:
        if value is not None and np.isfinite(value):
            fig.add_vline(x=float(value), line_dash=dash, line_color=color, annotation_text=label)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Scenario price")
    fig.update_yaxes(title_text="Predicted units", secondary_y=False)
    fig.update_yaxes(title_text="Revenue / margin", secondary_y=True)
    return fig


def temporal_heatmap_chart(df: pd.DataFrame, product_id: str, metric: str = "revenue") -> go.Figure:
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty:
        return _empty_figure("No temporal data available", height=320)
    product["date"] = pd.to_datetime(product["date"])
    product["month"] = product.get("month", product["date"].dt.month)
    product["revenue"] = product["units_sold"] * product["price"]
    value_col = metric if metric in product.columns else "revenue"
    source_grain = str(product["source_grain"].dropna().iloc[0]) if "source_grain" in product.columns and product["source_grain"].notna().any() else ""
    if source_grain == "daily":
        product["day_of_week"] = product.get("day_of_week", product["date"].dt.dayofweek)
        heat = product.pivot_table(index="day_of_week", columns="month", values=value_col, aggfunc="mean", fill_value=0)
        y_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        x_labels = [str(i) for i in range(1, 13)]
        heat = heat.reindex(index=range(7), columns=range(1, 13), fill_value=0)
        labels = dict(x="Month", y="Day", color=value_col)
    else:
        if "season" not in product.columns:
            product["season"] = product["month"].map(_season)
        if "promo_depth_bucket" not in product.columns:
            product["promo_depth_bucket"] = "none"
        y_labels = ["Winter", "Spring", "Summer", "Autumn"]
        x_labels = ["none", "light", "medium", "deep", "extreme"]
        heat = product.pivot_table(index="season", columns="promo_depth_bucket", values=value_col, aggfunc="mean", fill_value=0)
        heat = heat.reindex(index=y_labels, columns=x_labels, fill_value=0)
        labels = dict(x="Promotion depth", y="Season", color=value_col)
    fig = px.imshow(heat, labels=labels, x=x_labels, y=y_labels, aspect="auto")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def promotion_depth_chart(depth: pd.DataFrame) -> go.Figure:
    if depth.empty:
        return _empty_figure("No promotion depth analysis is available.", height=320)
    frame = depth.copy()
    order = ["none", "light", "medium", "deep", "extreme"]
    frame["promo_depth_bucket"] = pd.Categorical(frame["promo_depth_bucket"], categories=order, ordered=True)
    frame = frame.sort_values("promo_depth_bucket")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=frame["promo_depth_bucket"].astype(str), y=frame["avg_units"], name="Avg units"), secondary_y=False)
    if "avg_margin_delta_vs_none" in frame.columns:
        fig.add_trace(
            go.Scatter(x=frame["promo_depth_bucket"].astype(str), y=frame["avg_margin_delta_vs_none"], mode="lines+markers", name="Margin delta"),
            secondary_y=True,
        )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Average units", secondary_y=False)
    fig.update_yaxes(title_text="Margin delta vs none", secondary_y=True)
    return fig


def promotion_timing_chart(timing: pd.DataFrame) -> go.Figure:
    if timing.empty:
        return _empty_figure("No promotion timing effect is available.", height=300)
    row = timing.iloc[0]
    frame = pd.DataFrame(
        {
            "period": ["Pre", "Promotion", "Post"],
            "units": [row.get("avg_pre_units", 0), row.get("avg_promo_units", 0), row.get("avg_post_units", 0)],
        }
    )
    fig = px.bar(frame, x="period", y="units", text="units", labels={"period": "", "units": "Average units"})
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def opportunity_matrix_chart(opportunities: pd.DataFrame, objective: str = "revenue") -> go.Figure:
    if opportunities.empty:
        return _empty_figure("No opportunities are available.", height=420)
    frame = opportunities.copy()
    size_source = "expected_margin" if objective == "margin" and "expected_margin" in frame.columns else "expected_revenue"
    if size_source in frame.columns:
        frame["size_metric"] = pd.to_numeric(frame[size_source], errors="coerce").fillna(0).clip(lower=0)
    else:
        frame["size_metric"] = 1.0
    if float(frame["size_metric"].sum()) <= 0:
        frame["size_metric"] = 1.0
    fig = px.scatter(
        frame,
        x="reliability_score",
        y="opportunity_score",
        color="action_category" if "action_category" in frame.columns else None,
        size="size_metric",
        hover_name="product_name" if "product_name" in frame.columns else "product_id",
        hover_data=[col for col in ["product_id", "category", "current_price", "recommended_price", "price_delta_pct", "status"] if col in frame.columns],
        labels={"reliability_score": "Reliability score", "opportunity_score": "Opportunity score"},
    )
    fig.add_vline(x=55, line_dash="dot", line_color="#94a3b8")
    fig.add_hline(y=50, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    return fig


def coefficient_chart(coef_table: pd.DataFrame) -> go.Figure:
    if coef_table.empty:
        return _empty_figure("No coefficient table is available.", height=340)
    frame = coef_table.head(12).sort_values("coefficient")
    fig = px.bar(frame, x="coefficient", y="feature", orientation="h", labels={"feature": "Feature", "coefficient": "Coefficient"})
    fig.add_vline(x=0, line_color="#94a3b8")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def segment_elasticity_chart(segments: pd.DataFrame, title: str = "Segment elasticity") -> go.Figure:
    if segments.empty:
        return _empty_figure("No segment elasticity is available.", height=320)
    frame = segments.sort_values("elasticity")
    fig = px.bar(frame, x="elasticity", y="segment", orientation="h", labels={"segment": "", "elasticity": "Elasticity"})
    fig.add_vline(x=0, line_color="#94a3b8")
    fig.update_layout(height=max(300, 40 * len(frame) + 120), margin=dict(l=10, r=10, t=25, b=10), title=title)
    return fig


def permutation_importance_chart(importance: pd.DataFrame) -> go.Figure:
    if importance.empty:
        return _empty_figure("No permutation importance is available.", height=340)
    frame = importance.head(15).sort_values("importance_mean")
    fig = px.bar(frame, x="importance_mean", y="feature", orientation="h", error_x="importance_std" if "importance_std" in frame.columns else None)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=25, b=10))
    return fig


def product_backtest_error_chart(product_metrics: pd.DataFrame) -> go.Figure:
    if product_metrics.empty or "wmape" not in product_metrics.columns:
        return _empty_figure("No product-level backtest metrics are available.", height=340)
    frame = product_metrics.sort_values("wmape", ascending=True).tail(25)
    fig = px.bar(frame, x="wmape", y="product_id", orientation="h", labels={"wmape": "wMAPE", "product_id": "Product"})
    fig.update_layout(height=max(340, 28 * len(frame) + 100), margin=dict(l=10, r=10, t=25, b=10))
    return fig


def _product_time_series(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty:
        return pd.DataFrame()
    product["date"] = pd.to_datetime(product["date"], errors="coerce")
    product["revenue"] = product["units_sold"] * product["price"]
    product["gross_margin"] = (product["price"] - product["cost"]) * product["units_sold"] if "cost" in product.columns else 0.0
    if "promotion_flag" not in product.columns:
        product["promotion_flag"] = False
    if "stockout_flag" not in product.columns:
        product["stockout_flag"] = product["stock_available"] <= 0 if "stock_available" in product.columns else False
    product["_price_weight"] = product["units_sold"].clip(lower=0) + 1
    product["_price_weighted"] = product["price"] * product["_price_weight"]
    grouped = (
        product.dropna(subset=["date"])
        .groupby("date", as_index=False)
        .agg(
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
            promo_rate=("promotion_flag", "mean"),
            stockout_rate=("stockout_flag", "mean"),
            price_weight=("_price_weight", "sum"),
            price_weighted=("_price_weighted", "sum"),
        )
    )
    grouped["price"] = grouped["price_weighted"] / grouped["price_weight"].replace(0, np.nan)
    return grouped.drop(columns=["price_weight", "price_weighted"]).sort_values("date").reset_index(drop=True)


def _empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=height, annotations=[{"text": message, "showarrow": False}], margin=dict(l=10, r=10, t=20, b=10))
    return fig


def _season(month: int) -> str:
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Spring"
    if month in {6, 7, 8}:
        return "Summer"
    return "Autumn"
