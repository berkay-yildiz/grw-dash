import streamlit as st

from core.io import load_traffic_from_excel
from core.transform import aggregate, apply_default_range
from core.compare import pick_current, find_prev
from core.kpi import calculate_traffic_kpis, fmt_int, fmt_rate
from core.charts import render_traffic_charts

def render_kpis(kpis: list[dict], compare_label: str):
    st.caption(f"Comparison: {compare_label}")
    cols = st.columns(4)

    for i, k in enumerate(kpis):
        name = k["name"]
        val = k["value"]
        delta = k["delta_pct"]

        value_str = fmt_rate(val) if name == "New User Rate" else fmt_int(val)

        if delta is None:
            delta_str = "N/A"
            delta_color = "off"
        else:
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.1f}%"
            delta_color = "normal" if delta >= 0 else "inverse"

        with cols[i % 4]:
            st.metric(label=name, value=value_str, delta=delta_str, delta_color=delta_color)

def page():
    st.header("Trafik")

    with st.sidebar:
        st.subheader("Data Source")
        uploaded = st.file_uploader("Upload grw_dash.xlsx", type=["xlsx"])

        if uploaded is None:
            st.info("Excel dosyasını yükleyin. (Sheet adı: Traffic)")
            st.stop()

    # Load
    try:
        df_raw = load_traffic_from_excel(uploaded)
    except Exception as e:
        st.error("Dosya okunamadı veya sheet/kolon yapısı beklenen formatta değil.")
        st.exception(e)
        st.stop()

    with st.sidebar:
        st.subheader("Filters")
        granularity = st.radio("Granularity", ["Weekly", "Daily", "Monthly"], index=0)
        weeks_back = st.slider("Default range (weeks)", 4, 26, 8, 1)
        compare_mode = st.selectbox("Compare mode", ["WoW", "MoM", "YoY"], index=0)

    agg_all = aggregate(df_raw, granularity)
    agg_view = apply_default_range(agg_all, granularity, weeks_back)

    current_row = pick_current(agg_view)
    previous_row = find_prev(agg_all, current_row, granularity, compare_mode)

    st.subheader("Trafik — KPI Özeti")
    kpis = calculate_traffic_kpis(current_row, previous_row)
    render_kpis(kpis, compare_label=f"{granularity} / {compare_mode}")

    st.divider()
    render_traffic_charts(agg_view)

    with st.expander("Aggregated preview (filtered)"):
        st.dataframe(agg_view, use_container_width=True)

    with st.expander("Raw preview (first 50)"):
        st.dataframe(df_raw.head(50), use_container_width=True)

