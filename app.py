"""
app.py — entry point. Two pages: the model **Builder** and the experiments **Compare** view.

Run:  streamlit run app.py
(Each page is a plain script; set_page_config lives here so it's called exactly once.)
"""
import streamlit as st

st.set_page_config(page_title="Tiny Model Builder", layout="wide")

st.navigation([
    st.Page("builder_app.py", title="Builder", icon="🛠", default=True),
    st.Page("compare_page.py", title="Compare", icon="📊"),
]).run()
