import re
import numpy as np
import pandas as pd
import streamlit as st

SHEET_NAME = "Traffic"


# Canonical column names we want to end up with
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

# Aliases to fix common typos / trailing spaces
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
# HELPERS
# ============================================================
def clean_colname(c: str) -> str:
    return str(c).strip()

def parse_date(s: pd.Series) -> pd.Series:
    # Accepts DD.MM.YYYY and ISO; dayfirst handles Turkish format safely.
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def to_number(s: pd.Series) -> pd.Series:
    """
    Robust numeric parser for metrics that may look like:
      - 76558
      - "76.558" (TR thousands)
      - "76,558" (thousands or decimal)
      - " 76.558 "
    Assumption: metrics are integers (counts), so remove '.' and ',' thousands separators.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()
    x = x.str.replace(" ", "", regex=False)

    # Remove thousands separators (both '.' and ',')
    x = x.str.replace(".", "", regex=False).str.replace(",", "", regex=False)

    return pd.to_numeric(x, errors="coerce")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]
    # Apply aliases if present
    rename_map = {k: v for k, v in ALIASES.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def validate_contract(df: pd.DataFrame) -> None:
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Sheet column mismatch.\n"
            f"Missing columns: {missing}\n\n"
            f"Found columns: {list(df.columns)}"
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

    # month_start: first day of month
    df["month_start"] = df["date"].values.astype("datetime64[M]")
    df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)

    return df

@st.cache_data(show_spinner=False)
def load_from_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=SHEET_NAME)
    df = normalize_columns(df)
    validate_contract(df)

    # Parse date first
    df["Date"] = parse_date(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # Parse metrics
    for c in METRIC_COLS:
        df[c] = to_number(df[c])

    # Drop rows where core metrics are missing (optional: keep, but safer to drop)
    df = df.dropna(subset=["Sessions", "User Unique Session"])

    # Add time dimensions
    df = add_time_dims(df)

    return df

def aggregate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    granularity: 'Daily' | 'Weekly' | 'Monthly'
    Returns: period_label + period_sort + summed metrics
    """
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

def apply_default_range(agg: pd.DataFrame, granularity: str, weeks_back: int = 8) -> pd.DataFrame:
    agg = agg.copy()
    agg = agg.sort_values("period_sort")

    if granularity == "Weekly":
        return agg.tail(weeks_back)

    if granularity == "Daily":
        return agg.tail(weeks_back * 7)

    if granularity == "Monthly":
        return agg.tail(3)

    return agg

# ============================================================
# STREAMLIT UI (STEP 3 VALIDATION)
# ============================================================
st.set_page_config(page_title="Trafik Dashboard - Step 3", layout="wide")
st.title("Trafik Dashboard")

st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload grw_dash.xlsx", type=["xlsx"])

# --- Prevent NoneType crash ---
if uploaded is None:
    st.info("Sol menüden Excel dosyasını yükleyin. Dosya yüklenmeden uygulama çalışmaz.")
    st.stop()

# Load
try:
    df_raw = load_from_excel(uploaded)
except Exception as e:
    st.error("Dosya okunamadı veya sheet/kolon yapısı beklenen formatta değil.")
    st.exception(e)
    st.stop()

st.sidebar.header("Filters")
granularity = st.sidebar.radio("Granularity", ["Weekly", "Daily", "Monthly"], index=0)
weeks_back = st.sidebar.slider("Default range (weeks)", min_value=4, max_value=26, value=8, step=1)

agg = aggregate(df_raw, granularity=granularity)
agg_view = apply_default_range(agg, granularity=granularity, weeks_back=weeks_back)

# --- Output previews ---
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Raw preview (first 30)")
    st.dataframe(df_raw.head(30), use_container_width=True)
with c2:
    st.subheader(f"Aggregated preview ({granularity}) - default range")
    st.dataframe(agg_view, use_container_width=True)

st.divider()
st.subheader("Columns check")
st.write("Detected columns:", list(pd.read_excel(uploaded, sheet_name=SHEET_NAME).columns))
st.write("Normalized columns:", list(df_raw.columns))

# ===============================
# STEP 4 – Compare & KPI helpers
# ===============================
def get_current_and_previous(agg: pd.DataFrame) -> tuple[pd.Series, pd.Series | None]:
    """
    Returns:
      current_row, previous_row
    Assumes agg is already filtered to desired date range and sorted.
    """
    agg = agg.sort_values("period_sort")

    if len(agg) < 2:
        return agg.iloc[-1], None

    return agg.iloc[-1], agg.iloc[-2]


def pct_delta(curr: float, prev: float | None) -> float | None:
    if prev is None or pd.isna(prev) or prev == 0:
        return None
    return (curr / prev - 1.0) * 100.0

def calculate_kpis(current: pd.Series, previous: pd.Series | None) -> dict:
    kpis = {}

    def add_kpi(name, curr_val, prev_val):
        kpis[name] = {
            "value": curr_val,
            "delta": pct_delta(curr_val, prev_val)
        }

    add_kpi(
        "Total Unique Session",
        current["User Unique Session"],
        previous["User Unique Session"] if previous is not None else None,
    )

    add_kpi(
        "Sessions",
        current["Sessions"],
        previous["Sessions"] if previous is not None else None,
    )

    add_kpi(
        "Organic & Direct",
        current["Organic&Direct"],
        previous["Organic&Direct"] if previous is not None else None,
    )

    add_kpi(
        "Paid",
        current["Paid"],
        previous["Paid"] if previous is not None else None,
    )

    add_kpi(
        "Paid – Search",
        current["Paid-Search"],
        previous["Paid-Search"] if previous is not None else None,
    )

    add_kpi(
        "Paid – Display",
        current["Paid-Display"],
        previous["Paid-Display"] if previous is not None else None,
    )

    add_kpi(
        "Influencer",
        current["Influencer"],
        previous["Influencer"] if previous is not None else None,
    )

    # New User Rate
    curr_rate = current["New User"] / current["User Unique Session"] if current["User Unique Session"] else None
    prev_rate = (
        previous["New User"] / previous["User Unique Session"]
        if previous is not None and previous["User Unique Session"]
        else None
    )

    add_kpi("New User Rate", curr_rate, prev_rate)

    return kpis

def format_value(name, value):
    if value is None or pd.isna(value):
        return "–"

    if name == "New User Rate":
        return f"{value:.1%}"

    return f"{int(round(value)):,}".replace(",", ".")


def format_delta(delta):
    if delta is None:
        return "N/A", "neutral"

    sign = "+" if delta >= 0 else ""
    color = "green" if delta >= 0 else "red"
    return f"{sign}{delta:.1f}%", color


def render_kpis(kpis: dict):
    cols = st.columns(4)

    for i, (name, data) in enumerate(kpis.items()):
        with cols[i % 4]:
            delta_text, color = format_delta(data["delta"])
            st.metric(
                label=name,
                value=format_value(name, data["value"]),
                delta=delta_text,
                delta_color=color,
            )

current_row, previous_row = get_current_and_previous(agg_view)
kpis = calculate_kpis(current_row, previous_row)

st.subheader("Trafik – KPI Özeti")
render_kpis(kpis)

