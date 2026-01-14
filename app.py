import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================
SHEET_NAME = "Traffic"

CANONICAL_COLS = [
    "Date",
    "User Unique Session",
    "Sessions",
    "Organic&Direct",
    "Paid",
    "Paid-Search",
    "Paid-Display",
    "Influencer",
    "New User",
]

ALIASES = {
    "Inlfuencer": "Influencer",
    "Influencer ": "Influencer",
    " User Unique Session ": "User Unique Session",
    "User Unique Session ": "User Unique Session",
    " Sessions ": "Sessions",
    " Organic&Direct ": "Organic&Direct",
    " Paid ": "Paid",
    " Paid-Search ": "Paid-Search",
    " Paid-Display ": "Paid-Display",
    " New User ": "New User",
}

METRIC_COLS = [
    "User Unique Session",
    "Sessions",
    "Organic&Direct",
    "Paid",
    "Paid-Search",
    "Paid-Display",
    "Influencer",
    "New User",
]

# ============================================================
# HELPERS: parsing, normalization
# ============================================================
def clean_colname(c: str) -> str:
    return str(c).strip()

def parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def to_number(s: pd.Series) -> pd.Series:
    """
    Metrics are counts. Accepts:
      - numeric
      - "76.558" (TR thousands) -> 76558
      - "76,558" -> 76558
    """
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
            "Sheet column mismatch.\n"
            f"Missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n\n"
            "Beklenen kolonlar:\n"
            + "\n".join(CANONICAL_COLS)
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

@st.cache_data(show_spinner=False)
def load_from_excel(uploaded_file) -> pd.DataFrame:
    # Ensure openpyxl exists; fail fast with readable error
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError("openpyxl missing. requirements.txt içine openpyxl eklenmeli.")

    df = pd.read_excel(uploaded_file, sheet_name=SHEET_NAME)
    df = normalize_columns(df)
    validate_contract(df)

    df["Date"] = parse_date(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date")

    for c in METRIC_COLS:
        df[c] = to_number(df[c])

    # Optional: drop rows without core metrics
    df = df.dropna(subset=["Sessions", "User Unique Session"])

    df = add_time_dims(df)
    return df

# ============================================================
# AGGREGATION
# ============================================================
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

    raise ValueError("granularity must be one of: Daily, Weekly, Monthly")

def apply_default_range(agg: pd.DataFrame, granularity: str, weeks_back: int) -> pd.DataFrame:
    agg = agg.sort_values("period_sort")
    if granularity == "Weekly":
        return agg.tail(weeks_back)
    if granularity == "Daily":
        return agg.tail(weeks_back * 7)
    if granularity == "Monthly":
        # 8 hafta ~ 2 ay; biraz buffer ile 3 ay gösterelim
        return agg.tail(max(3, int(np.ceil(weeks_back / 4))))
    return agg

# ============================================================
# COMPARE ENGINE (WoW / MoM / YoY)
# ============================================================
def _safe_pct(curr: float, prev: float | None) -> float | None:
    if prev is None or pd.isna(prev) or prev == 0 or pd.isna(curr):
        return None
    return (curr / prev - 1.0) * 100.0

def _pick_current_period(agg_view: pd.DataFrame) -> pd.Series:
    agg_view = agg_view.sort_values("period_sort")
    return agg_view.iloc[-1]

def _find_prev_row(agg_all: pd.DataFrame, current_row: pd.Series, granularity: str, compare_mode: str) -> pd.Series | None:
    """
    Returns a single previous period row for KPI delta.
    compare_mode:
      - WoW: previous period (daily uses previous day; but we'll apply to aggregated daily series; KPI uses last day vs prev day)
      - MoM: previous month (monthly) or approximate (weekly/daily)
      - YoY: same week/month/day previous year where possible
    """
    agg_all = agg_all.sort_values("period_sort")

    if compare_mode == "WoW":
        # Previous period in the same granularity
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

        # For weekly/daily: map to "same period_sort minus ~4 weeks" (coarse but useful)
        # Weekly: subtract 4 weeks in sort-key space is not safe across year boundary; use position shift of 4 instead.
        # Daily: shift by ~30 days (position shift of 30) within daily aggregated series
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
            # match same iso_week in previous iso_year
            target_year = int(current_row["iso_year"]) - 1
            target_week = int(current_row["iso_week"])
            m = (agg_all["iso_year"] == target_year) & (agg_all["iso_week"] == target_week)
            prev = agg_all.loc[m]
            return prev.iloc[-1] if len(prev) else None

        if granularity == "Monthly":
            # match same month in previous year via year_month string
            ym = str(current_row["period"])
            year = int(ym.split("-")[0])
            month = ym.split("-")[1]
            target = f"{year-1}-{month}"
            prev = agg_all.loc[agg_all["period"] == target]
            return prev.iloc[-1] if len(prev) else None

        if granularity == "Daily":
            # match same calendar date previous year
            # period_sort is datetime for daily
            curr_date = pd.to_datetime(current_row["period_sort"])
            target_date = curr_date - pd.DateOffset(years=1)
            prev = agg_all.loc[agg_all["period_sort"] == target_date]
            return prev.iloc[-1] if len(prev) else None

    return None

# ============================================================
# KPI CALC + RENDER
# ============================================================
def calculate_kpis(current: pd.Series, previous: pd.Series | None) -> list[dict]:
    def prev_val(col: str):
        return previous[col] if previous is not None else None

    kpis = [
        {"name": "Total Unique Session", "value": float(current["User Unique Session"]), "prev": prev_val("User Unique Session")},
        {"name": "Sessions", "value": float(current["Sessions"]), "prev": prev_val("Sessions")},
        {"name": "Organic & Direct", "value": float(current["Organic&Direct"]), "prev": prev_val("Organic&Direct")},
        {"name": "Paid", "value": float(current["Paid"]), "prev": prev_val("Paid")},
        {"name": "Paid – Search", "value": float(current["Paid-Search"]), "prev": prev_val("Paid-Search")},
        {"name": "Paid – Display", "value": float(current["Paid-Display"]), "prev": prev_val("Paid-Display")},
        {"name": "Influencer", "value": float(current["Influencer"]), "prev": prev_val("Influencer")},
    ]

    # New User Rate
    curr_uus = float(current["User Unique Session"]) if pd.notna(current["User Unique Session"]) else 0.0
    curr_nu = float(current["New User"]) if pd.notna(current["New User"]) else 0.0
    curr_rate = (curr_nu / curr_uus) if curr_uus else np.nan

    if previous is not None:
        prev_uus = float(previous["User Unique Session"]) if pd.notna(previous["User Unique Session"]) else 0.0
        prev_nu = float(previous["New User"]) if pd.notna(previous["New User"]) else 0.0
        prev_rate = (prev_nu / prev_uus) if prev_uus else np.nan
    else:
        prev_rate = None

    kpis.append({"name": "New User Rate", "value": curr_rate, "prev": prev_rate})

    # compute deltas
    for k in kpis:
        k["delta_pct"] = _safe_pct(k["value"], k["prev"])

    return kpis

def _fmt_int(x):
    if x is None or pd.isna(x):
        return "–"
    return f"{int(round(x)):,}".replace(",", ".")

def _fmt_rate(x):
    if x is None or pd.isna(x):
        return "–"
    return f"{x*100:.1f}%"

def render_kpis(kpis: list[dict], compare_label: str):
    st.caption(f"Comparison: {compare_label}")

    cols = st.columns(4)
    for i, k in enumerate(kpis):
        name = k["name"]
        val = k["value"]
        delta = k["delta_pct"]

        if name == "New User Rate":
            value_str = _fmt_rate(val)
        else:
            value_str = _fmt_int(val)

        if delta is None:
            delta_str = "N/A"
            delta_color = "off"
        else:
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.1f}%"
            delta_color = "normal" if delta >= 0 else "inverse"

        with cols[i % 4]:
            st.metric(label=name, value=value_str, delta=delta_str, delta_color=delta_color)

st.divider()
st.subheader("Trafik — Trend & Kanal Dağılımı")

# ------------------------------------------------
# Prepare plotting dataframe (robust)
# ------------------------------------------------
plot_df = agg_view.copy()

# pick best x column available
if "period_label" in plot_df.columns:
    x_col = "period_label"
elif "period" in plot_df.columns:
    x_col = "period"
else:
    # last resort: create one
    plot_df = plot_df.reset_index(drop=True)
    plot_df["period_label"] = plot_df.index.astype(str)
    x_col = "period_label"

# Ensure expected metric columns exist (avoid KeyError/melt errors)
needed_metrics = [
    "User Unique Session",
    "Organic&Direct",
    "Paid",
    "Influencer",
    "Paid-Search",
    "Paid-Display",
]
missing_metrics = [c for c in needed_metrics if c not in plot_df.columns]
if missing_metrics:
    st.error(f"Grafikler için eksik kolonlar var: {missing_metrics}")
    st.info("Aggregated preview'de kolon isimlerini kontrol et.")
    st.stop()

# ------------------------------------------------
# 1) Total Unique Session Trend
# ------------------------------------------------
fig_total = px.line(
    plot_df,
    x=x_col,
    y="User Unique Session",
    markers=True,
    title="Total Unique Session Trend",
)
fig_total.update_layout(yaxis_title="User Unique Session", xaxis_title="", hovermode="x unified")
st.plotly_chart(fig_total, use_container_width=True)

# ------------------------------------------------
# 2) Trafik by Type (Multi-line)
# ------------------------------------------------
traffic_long = plot_df[[x_col, "Organic&Direct", "Paid", "Influencer"]].melt(
    id_vars=[x_col],
    var_name="Channel",
    value_name="Sessions",
)

fig_channel = px.line(
    traffic_long,
    x=x_col,
    y="Sessions",
    color="Channel",
    markers=True,
    title="Trafik by Type",
)
fig_channel.update_layout(yaxis_title="Sessions", xaxis_title="", hovermode="x unified")
st.plotly_chart(fig_channel, use_container_width=True)

# ------------------------------------------------
# 3) Paid Mix (Search vs Display)
# ------------------------------------------------
paid_long = plot_df[[x_col, "Paid-Search", "Paid-Display"]].melt(
    id_vars=[x_col],
    var_name="Paid Type",
    value_name="Sessions",
)

fig_paid = px.bar(
    paid_long,
    x=x_col,
    y="Sessions",
    color="Paid Type",
    title="Paid Mix — Search vs Display",
    barmode="stack",
)
fig_paid.update_layout(yaxis_title="Sessions", xaxis_title="", hovermode="x unified")
st.plotly_chart(fig_paid, use_container_width=True)

