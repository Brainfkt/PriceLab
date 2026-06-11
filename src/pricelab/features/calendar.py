from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date = pd.to_datetime(out["date"])
    iso = date.dt.isocalendar()
    out["year"] = date.dt.year.astype(int)
    out["month"] = date.dt.month.astype(int)
    out["quarter"] = date.dt.quarter.astype(int)
    out["week_of_year"] = iso.week.astype(int)
    out["day_of_week"] = date.dt.dayofweek.astype(int)
    out["day_name"] = date.dt.day_name()
    out["season"] = out["month"].map(_season)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["week_sin"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["week_cos"] = np.cos(2 * np.pi * out["week_of_year"] / 52)
    return out


def _season(month: int) -> str:
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Spring"
    if month in {6, 7, 8}:
        return "Summer"
    return "Autumn"
