import pandas as pd
import streamlit as st

from .config import SHEET_NAME
from .transform import normalize_columns, validate_contract, parse_date, to_number, add_time_dims
from .config import METRIC_COLS

@st.cache_data(show_spinner=False)
def load_traffic_from_excel(uploaded_file) -> pd.DataFrame:
    # Ensure openpyxl exists
    import openpyxl  # noqa: F401

    df = pd.read_excel(uploaded_file, sheet_name=SHEET_NAME)
    df = normalize_columns(df)
    validate_contract(df)

    df["Date"] = parse_date(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date")

    for c in METRIC_COLS:
        df[c] = to_number(df[c])

    df = df.dropna(subset=["Sessions", "User Unique Session"])
    df = add_time_dims(df)
    return df

