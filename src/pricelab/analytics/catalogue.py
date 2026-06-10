from __future__ import annotations

import pandas as pd


def catalogue_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    revenue = float((df["units_sold"] * df["price"]).sum())
    margin = float(((df["price"] - df["cost"]) * df["units_sold"]).sum()) if "cost" in df.columns else 0.0
    units = float(df["units_sold"].sum())
    return {
        "products": int(df["product_id"].nunique()),
        "rows": int(len(df)),
        "units": round(units, 2),
        "revenue": round(revenue, 2),
        "gross_margin": round(margin, 2),
        "avg_price": round(float(df["price"].mean()), 2),
        "promo_rate": round(float(df["promotion_flag"].astype(bool).mean()), 3) if "promotion_flag" in df.columns else 0.0,
    }


def product_leaderboard(df: pd.DataFrame, metric: str = "revenue", top_n: int = 15) -> pd.DataFrame:
    frame = df.copy()
    frame["revenue"] = frame["units_sold"] * frame["price"]
    frame["gross_margin"] = (frame["price"] - frame["cost"]) * frame["units_sold"] if "cost" in frame.columns else 0.0
    agg = (
        frame.groupby(["product_id", "product_name", "category"], dropna=False)
        .agg(
            units=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
            avg_price=("price", "mean"),
            price_points=("price", "nunique"),
        )
        .reset_index()
    )
    metric = metric if metric in agg.columns else "revenue"
    return agg.sort_values(metric, ascending=False).head(top_n).reset_index(drop=True)

