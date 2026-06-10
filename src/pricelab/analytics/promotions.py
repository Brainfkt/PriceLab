from __future__ import annotations

import pandas as pd


def promotion_summary(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty or "promotion_flag" not in product.columns:
        return pd.DataFrame()
    product["revenue"] = product["units_sold"] * product["price"]
    product["gross_margin"] = (product["price"] - product["cost"]) * product["units_sold"] if "cost" in product.columns else 0.0
    summary = (
        product.groupby("promotion_flag", dropna=False)
        .agg(
            observations=("date", "count"),
            avg_units=("units_sold", "mean"),
            avg_price=("price", "mean"),
            avg_discount=("discount_rate", "mean"),
            revenue=("revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )
    if len(summary) == 2:
        non_promo = summary[summary["promotion_flag"] == False]  # noqa: E712
        promo = summary[summary["promotion_flag"] == True]  # noqa: E712
        if not non_promo.empty and not promo.empty and float(non_promo["avg_units"].iloc[0]) > 0:
            uplift = float(promo["avg_units"].iloc[0] / non_promo["avg_units"].iloc[0] - 1)
            summary["unit_uplift_vs_non_promo"] = summary["promotion_flag"].map({True: uplift, False: 0.0})
    return summary

