import numpy as np
import pandas as pd
from .compare import safe_pct

def calculate_traffic_kpis(current: pd.Series, previous: pd.Series | None) -> list[dict]:
    def pv(col: str):
        return previous[col] if previous is not None else None

    kpis = [
        {"name": "Total Unique Session", "value": float(current["User Unique Session"]), "prev": pv("User Unique Session")},
        {"name": "Sessions", "value": float(current["Sessions"]), "prev": pv("Sessions")},
        {"name": "Organic & Direct", "value": float(current["Organic&Direct"]), "prev": pv("Organic&Direct")},
        {"name": "Paid", "value": float(current["Paid"]), "prev": pv("Paid")},
        {"name": "Paid – Search", "value": float(current["Paid-Search"]), "prev": pv("Paid-Search")},
        {"name": "Paid – Display", "value": float(current["Paid-Display"]), "prev": pv("Paid-Display")},
        {"name": "Influencer", "value": float(current["Influencer"]), "prev": pv("Influencer")},
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

    for k in kpis:
        k["delta_pct"] = safe_pct(k["value"], k["prev"])

    return kpis

def fmt_int(x):
    if x is None or pd.isna(x):
        return "–"
    return f"{int(round(x)):,}".replace(",", ".")

def fmt_rate(x):
    if x is None or pd.isna(x):
        return "–"
    return f"{x*100:.1f}%"

