import numpy as np
import pandas as pd
from .config import CANONICAL_COLS, ALIASES, METRIC_COLS

def clean_colname(c: str) -> str:
    return str(c).strip()

def parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.strip()
    x = x.str.replace(" ", "", regex=False)
    x = x.str.replace(".", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(x, errors="coerce")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]
    rename_map = {k: v for k, v in ALIASES.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def validate_contract(df: pd.DataFrame) -> None:
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\nFound: {list(df.columns)}\nExpected: {CANONICAL_COLS}"
        )

def add_time_dims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["Date"]

    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)
    df["year_week"] = df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)

    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["month_start"] = df["date"].values.astype("datetime64[M]")
    df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    return df

def aggregate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    df = df.copy()

    if granularity == "Daily":
        out = df.groupby(["date"], as_index=False)[METRIC_COLS].sum()
        out = out.rename(columns={"date": "period"})
        out["period_label"] = out["period"].dt.strftime("%Y-%m-%d")
        out["period_sort"] = out["period"]
        return out.sort_values("period_sort")

    if granularity == "Weekly":
        out = df.groupby(["iso_year", "iso_week", "year_week"], as_index=False)[METRIC_COLS].sum()
        out["period"] = out["year_week"]
        out["period_label"] = out["year_week"]
        out["period_sort"] = out["iso_year"] * 100 + out["iso_week"]
        return out.sort_values("period_sort")

    if granularity == "Monthly":
        out = df.groupby(["month_start", "year_month"], as_index=False)[METRIC_COLS].sum()
        out["period"] = out["year_month"]
        out["period_label"] = out["year_month"]
        out["period_sort"] = out["month_start"]
        return out.sort_values("period_sort")

    raise ValueError("granularity must be Daily, Weekly, or Monthly")

def apply_default_range(agg: pd.DataFrame, granularity: str, weeks_back: int) -> pd.DataFrame:
    agg = agg.sort_values("period_sort")
    if granularity == "Weekly":
        return agg.tail(weeks_back)
    if granularity == "Daily":
        return agg.tail(weeks_back * 7)
    if granularity == "Monthly":
        # approx months
        return agg.tail(max(3, int(np.ceil(weeks_back / 4))))
    return agg

