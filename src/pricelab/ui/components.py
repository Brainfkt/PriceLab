from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def price_units_chart(df: pd.DataFrame, product_id: str) -> go.Figure:
    product = df[df["product_id"].astype(str) == str(product_id)].sort_values("date")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=product["date"], y=product["price"], mode="lines+markers", name="Price"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=product["date"], y=product["units_sold"], name="Units", opacity=0.35),
        secondary_y=True,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Units sold", secondary_y=True)
    return fig


def revenue_margin_chart(df: pd.DataFrame, product_id: str) -> go.Figure:
    product = df[df["product_id"].astype(str) == str(product_id)].sort_values("date").copy()
    product["revenue"] = product["units_sold"] * product["price"]
    product["gross_margin"] = (product["price"] - product["cost"]) * product["units_sold"] if "cost" in product.columns else 0.0
    fig = px.line(product, x="date", y=["revenue", "gross_margin"], labels={"value": "Amount", "variable": "Metric"})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


def price_bin_chart(price_bins: pd.DataFrame) -> go.Figure:
    if price_bins.empty:
        fig = go.Figure()
        fig.update_layout(height=320, annotations=[{"text": "Not enough price variation", "showarrow": False}])
        return fig
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
