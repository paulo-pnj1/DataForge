"""
Demo autónoma do Previsor de Alunos (não precisa do resto do DataForge).
Correr:  streamlit run previsor_demo.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Previsor de Alunos", layout="centered")

from modules.previsor_alunos import render_previsor_alunos  # noqa: E402

render_previsor_alunos()
