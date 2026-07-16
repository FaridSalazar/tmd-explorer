#!/usr/bin/env python3
# =============================================================================
#  Small-x gluon TMD explorer -- multipage entry point.
#  Page 1 (default): MV dipole, 2 parameters (Qs0^2, C^2), emulator v7.
#  Page 2:           MVgamma dipole, 3 parameters (+ gamma), emulator v2.
# =============================================================================
import streamlit as st

st.set_page_config(page_title="small-x gluon TMD explorer", layout="wide")

pg = st.navigation([
    st.Page("mv_page.py", title="MV explorer (2 params)", default=True),
    st.Page("mvgamma_page.py", title="MV\u03b3 explorer (3 params)"),
])
pg.run()
