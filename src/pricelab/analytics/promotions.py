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
            avg_margin=("gross_margin", "mean"),
        )
        .reset_index()
    )
    if len(summary) == 2:
        non_promo = summary[summary["promotion_flag"] == False]  # noqa: E712
        promo = summary[summary["promotion_flag"] == True]  # noqa: E712
        if not non_promo.empty and not promo.empty and float(non_promo["avg_units"].iloc[0]) > 0:
            uplift = float(promo["avg_units"].iloc[0] / non_promo["avg_units"].iloc[0] - 1)
            summary["unit_uplift_vs_non_promo"] = summary["promotion_flag"].map({True: uplift, False: 0.0})
            margin_delta = float(promo["avg_margin"].iloc[0] - non_promo["avg_margin"].iloc[0])
            summary["avg_margin_delta_vs_non_promo"] = summary["promotion_flag"].map({True: margin_delta, False: 0.0})
    return summary


def promotion_depth_summary(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    product = df[df["product_id"].astype(str) == str(product_id)].copy()
    if product.empty or "promo_depth_bucket" not in product.columns:
        return pd.DataFrame()
    product["revenue"] = product["units_sold"] * product["price"]
    product["gross_margin"] = (product["price"] - product["cost"]) * product["units_sold"] if "cost" in product.columns else 0.0
    result = (
        product.groupby("promo_depth_bucket", dropna=False)
        .agg(
            observations=("date", "count"),
            avg_discount=("discount_rate", "mean"),
            avg_units=("units_sold", "mean"),
            revenue=("revenue", "sum"),
            gross_margin=("gross_margin", "sum"),
            avg_margin=("gross_margin", "mean"),
        )
        .reset_index()
        .sort_values("avg_discount")
    )
    baseline = result[result["promo_depth_bucket"] == "none"]
    if not baseline.empty and float(baseline["avg_units"].iloc[0]) > 0:
        base_units = float(baseline["avg_units"].iloc[0])
        base_margin = float(baseline["avg_margin"].iloc[0])
        result["unit_uplift_vs_none"] = result["avg_units"] / base_units - 1
        result["avg_margin_delta_vs_none"] = result["avg_margin"] - base_margin
    return result


def promotion_timing_effect(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    product = df[df["product_id"].astype(str) == str(product_id)].sort_values(["channel", "region", "date"]).copy()
    if product.empty or "promotion_flag" not in product.columns:
        return pd.DataFrame()
    group_cols = [col for col in ["channel", "region"] if col in product.columns]
    group = product.groupby(group_cols, dropna=False) if group_cols else [(None, product)]
    frames = []
    if group_cols:
        product["pre_units"] = group["units_sold"].shift(1)
        product["post_units"] = group["units_sold"].shift(-1)
    else:
        product["pre_units"] = product["units_sold"].shift(1)
        product["post_units"] = product["units_sold"].shift(-1)
    promo = product[product["promotion_flag"].astype(bool)].copy()
    if promo.empty:
        return pd.DataFrame()
    pre = float(promo["pre_units"].dropna().mean()) if promo["pre_units"].notna().any() else 0.0
    during = float(promo["units_sold"].mean())
    post = float(promo["post_units"].dropna().mean()) if promo["post_units"].notna().any() else 0.0
    frames.append(
        {
            "promo_events": int(len(promo)),
            "avg_pre_units": round(pre, 2),
            "avg_promo_units": round(during, 2),
            "avg_post_units": round(post, 2),
            "promo_lift_vs_pre": round(during / pre - 1, 3) if pre > 0 else None,
            "post_promo_change_vs_pre": round(post / pre - 1, 3) if pre > 0 else None,
        }
    )
    return pd.DataFrame(frames)
