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
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["week_sin"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["week_cos"] = np.cos(2 * np.pi * out["week_of_year"] / 52)
    return out

