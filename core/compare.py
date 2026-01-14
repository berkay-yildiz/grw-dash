import numpy as np
import pandas as pd

def safe_pct(curr: float, prev: float | None) -> float | None:
    if prev is None or pd.isna(prev) or prev == 0 or pd.isna(curr):
        return None
    return (curr / prev - 1.0) * 100.0

def pick_current(agg_view: pd.DataFrame) -> pd.Series:
    agg_view = agg_view.sort_values("period_sort")
    return agg_view.iloc[-1]

def find_prev(agg_all: pd.DataFrame, current_row: pd.Series, granularity: str, compare_mode: str) -> pd.Series | None:
    agg_all = agg_all.sort_values("period_sort")

    if compare_mode == "WoW":
        idx = agg_all.index[agg_all["period_sort"] == current_row["period_sort"]]
        if len(idx) == 0:
            return None
        pos = agg_all.index.get_loc(idx[0])
        if pos - 1 < 0:
            return None
        return agg_all.iloc[pos - 1]

    if compare_mode == "MoM":
        if granularity == "Monthly":
            idx = agg_all.index[agg_all["period"] == current_row["period"]]
            if len(idx) == 0:
                return None
            pos = agg_all.index.get_loc(idx[0])
            if pos - 1 < 0:
                return None
            return agg_all.iloc[pos - 1]

        shift = 4 if granularity == "Weekly" else 30
        idx = agg_all.index[agg_all["period_sort"] == current_row["period_sort"]]
        if len(idx) == 0:
            return None
        pos = agg_all.index.get_loc(idx[0])
        if pos - shift < 0:
            return None
        return agg_all.iloc[pos - shift]

    if compare_mode == "YoY":
        if granularity == "Weekly":
            target_year = int(current_row["iso_year"]) - 1
            target_week = int(current_row["iso_week"])
            prev = agg_all.loc[(agg_all["iso_year"] == target_year) & (agg_all["iso_week"] == target_week)]
            return prev.iloc[-1] if len(prev) else None

        if granularity == "Monthly":
            ym = str(current_row["period"])
            year = int(ym.split("-")[0])
            month = ym.split("-")[1]
            target = f"{year-1}-{month}"
            prev = agg_all.loc[agg_all["period"] == target]
            return prev.iloc[-1] if len(prev) else None

        if granularity == "Daily":
            curr_date = pd.to_datetime(current_row["period_sort"])
            target = curr_date - pd.DateOffset(years=1)
            prev = agg_all.loc[agg_all["period_sort"] == target]
            return prev.iloc[-1] if len(prev) else None

    return None

