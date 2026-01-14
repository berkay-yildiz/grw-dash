import streamlit as st
from pages.traffic import page as traffic_page

st.set_page_config(page_title="GRW Dash", layout="wide")

PAGES = {
    "1- Trafik": traffic_page,
    # Sonraki adımlar:
    # "2- Sorgulama": ...
    # "3- Başvuru": ...
}

with st.sidebar:
    st.title("GRW Dash")
    choice = st.radio("Dashboard", list(PAGES.keys()), index=0)

PAGES[choice]()
