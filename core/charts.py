import pandas as pd
import streamlit as st
import plotly.express as px

def pick_xcol(df: pd.DataFrame) -> str:
    if "period_label" in df.columns:
        return "period_label"
    if "period" in df.columns:
        return "period"
    df = df.reset_index(drop=True)
    df["period_label"] = df.index.astype(str)
    return "period_label"

def render_traffic_charts(agg_view: pd.DataFrame):
    plot_df = agg_view.copy()
    x_col = pick_xcol(plot_df)

    needed = ["User Unique Session", "Organic&Direct", "Paid", "Influencer", "Paid-Search", "Paid-Display"]
    missing = [c for c in needed if c not in plot_df.columns]
    if missing:
        st.error(f"Grafikler için eksik kolonlar var: {missing}")
        st.stop()

    st.subheader("Trafik — Trend & Kanal Dağılımı")

    # 1) Total Unique Session
    fig_total = px.line(plot_df, x=x_col, y="User Unique Session", markers=True, title="Total Unique Session Trend")
    fig_total.update_layout(hovermode="x unified", xaxis_title="", yaxis_title="User Unique Session")
    st.plotly_chart(fig_total, use_container_width=True)

    # 2) Trafik by type
    traffic_long = plot_df[[x_col, "Organic&Direct", "Paid", "Influencer"]].melt(
        id_vars=[x_col],
        var_name="Channel",
        value_name="Sessions",
    )
    fig_channel = px.line(traffic_long, x=x_col, y="Sessions", color="Channel", markers=True, title="Trafik by Type")
    fig_channel.update_layout(hovermode="x unified", xaxis_title="", yaxis_title="Sessions")
    st.plotly_chart(fig_channel, use_container_width=True)

    # 3) Paid mix
    paid_long = plot_df[[x_col, "Paid-Search", "Paid-Display"]].melt(
        id_vars=[x_col],
        var_name="Paid Type",
        value_name="Sessions",
    )
    fig_paid = px.bar(paid_long, x=x_col, y="Sessions", color="Paid Type", barmode="stack",
                      title="Paid Mix — Search vs Display")
    fig_paid.update_layout(hovermode="x unified", xaxis_title="", yaxis_title="Sessions")
    st.plotly_chart(fig_paid, use_container_width=True)

