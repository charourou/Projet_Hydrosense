import streamlit as st
import os

html_path = os.path.join(os.path.dirname(__file__), "..", "assets", "Hydro-Sense-Dashboard-v2.html")

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=900, scrolling=True)
