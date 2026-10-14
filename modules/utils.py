"""
DataForge EDU - Utilitários partilhados
Paleta, CSS, componentes reutilizáveis
Design profissional: tema escuro consistente, WCAG AA/AAA
"""

import streamlit as st
import pandas as pd
import numpy as np
from tinydb import TinyDB, Query
import os
import datetime

# ══════════════════════════════════════════════════════
# PALETA DE CORES - Design System
# ══════════════════════════════════════════════════════
C_BG         = "#0D1117"   # fundo principal (GitHub-dark inspired)
C_SURFACE    = "#161B27"   # cards e painéis
C_SURFACE2   = "#1C2333"   # inputs e elementos secundários
C_SURFACE3   = "#21262D"   # hover states
C_BORDER     = "#30363D"   # bordas subtis
C_BORDER2    = "#3A4560"   # bordas mais visíveis
C_ACCENT     = "#58A6FF"   # azul primário
C_ACCENT_H   = "#79BAFF"   # hover do accent
C_ACCENT2    = "#BC8CFF"   # violeta secundário
C_GREEN      = "#3FB950"   # sucesso
C_AMBER      = "#D29922"   # aviso
C_RED        = "#F85149"   # erro
C_TEAL       = "#39D0C0"   # teal
C_TEXT       = "#E6EDF3"   # texto principal
C_TEXT_SEC   = "#8B949E"   # texto secundário
C_TEXT_MUTE  = "#484F58"   # texto muted

PALETTE = [C_ACCENT, C_TEAL, C_AMBER, C_RED, C_ACCENT2, C_GREEN]

# ══════════════════════════════════════════════════════
# CSS GLOBAL - Cobertura completa do Streamlit
# ══════════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
    /* ── GOOGLE FONTS ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ══════════════════════════════════════════════
       ROOT & RESET - força tema escuro em tudo
    ══════════════════════════════════════════════ */
    :root {{
        --bg: {C_BG};
        --surface: {C_SURFACE};
        --surface2: {C_SURFACE2};
        --border: {C_BORDER};
        --accent: {C_ACCENT};
        --text: {C_TEXT};
        --text-sec: {C_TEXT_SEC};
    }}

    html, body {{
        background-color: {C_BG} !important;
        color: {C_TEXT} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* Streamlit root containers */
    .stApp,
    .stApp > header,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main,
    .main .block-container,
    [data-testid="block-container"] {{
        background-color: {C_BG} !important;
        color: {C_TEXT} !important;
    }}

    /* ── TIPOGRAFIA BASE ── */
    p, div, span, label, li, td, th,
    .stMarkdown, .stMarkdown p, .stMarkdown div,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] div {{
        color: {C_TEXT} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* Ícones nativos do Streamlit (Material Symbols) - NUNCA usar Inter aqui,
       senão o glifo do ícone aparece como texto literal (ex: "keyboard_arrow_right") */
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"],
    [data-testid="stExpanderToggleIcon"] [data-testid="stIconMaterial"] {{
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        color: {C_TEXT_SEC} !important;
    }}

    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2,
    .stMarkdown h3, .stMarkdown h4 {{
        color: {C_TEXT} !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* Remove Streamlit default white backgrounds on text elements */
    [class*="css"] {{
        color: {C_TEXT};
    }}

    /* ── LAYOUT ── */
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }}

    /* ── FOCO VISÍVEL ── */
    *:focus-visible {{
        outline: 2px solid {C_ACCENT} !important;
        outline-offset: 2px !important;
        border-radius: 6px !important;
    }}

    /* ══════════════════════════════════════════════
       SIDEBAR - sempre visível, desktop e mobile
    ══════════════════════════════════════════════ */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div {{
        background: {C_SURFACE} !important;
        color: {C_TEXT} !important;
    }}
    [data-testid="stSidebar"] {{
        border-right: 1px solid {C_BORDER} !important;
        /* Garante que a sidebar nunca desaparece nem fica estreita demais */
        display: block !important;
        transform: none !important;
        width: 290px !important;
        min-width: 290px !important;
        max-width: 290px !important;
        flex: 0 0 290px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {{
        width: 100% !important;
        min-width: 0 !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        white-space: normal !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {{
        color: {C_TEXT} !important;
    }}

    /* Botão de colapso nativo (×) - torna-o num ícone de menu hamburger
       e impede que a sidebar fique permanentemente fechada */
    [data-testid="collapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: {C_SURFACE} !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 8px !important;
        color: {C_TEXT} !important;
        z-index: 9999 !important;
    }}

    /* Mobile - força sidebar visível e acessível */
    @media (max-width: 768px) {{
        section[data-testid="stSidebar"] {{
            display: block !important;
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            height: 100vh !important;
            z-index: 1000 !important;
            transform: none !important;
            width: 80vw !important;
            min-width: 260px !important;
            max-width: 80vw !important;
            overflow-y: auto !important;
            box-shadow: 4px 0 24px rgba(0,0,0,0.5) !important;
        }}
        /* Botão hamburger fixo no topo-esquerdo em mobile */
        [data-testid="collapsedControl"] {{
            position: fixed !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 1100 !important;
            background: {C_SURFACE} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 8px !important;
            padding: 6px 10px !important;
        }}
    }}

    /* ══════════════════════════════════════════════
       INPUTS - text, number, textarea, selectbox
    ══════════════════════════════════════════════ */
    /* Wrappers */
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: {C_SURFACE2} !important;
        border: 1px solid {C_BORDER2} !important;
        border-radius: 8px !important;
    }}

    /* Inputs nativos */
    .stTextInput input,
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stNumberInput > div > div > input,
    .stTextArea textarea,
    .stTextArea > div > div > textarea,
    input[type="text"], input[type="password"],
    input[type="number"], input[type="email"],
    textarea {{
        background-color: {C_SURFACE2} !important;
        color: {C_TEXT} !important;
        border: 1px solid {C_BORDER2} !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        padding: 10px 14px !important;
        font-family: 'Inter', sans-serif !important;
        caret-color: {C_ACCENT} !important;
    }}
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {{
        border-color: {C_ACCENT} !important;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.2) !important;
        outline: none !important;
    }}
    /* Placeholder text */
    input::placeholder, textarea::placeholder {{
        color: {C_TEXT_SEC} !important;
        opacity: 0.7 !important;
    }}

    /* Labels */
    .stTextInput label, .stTextArea label,
    .stSelectbox label, .stNumberInput label,
    .stMultiSelect label, .stSlider label,
    .stRadio label, .stCheckbox label,
    [data-testid="stWidgetLabel"] {{
        color: {C_TEXT} !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        margin-bottom: 6px !important;
    }}

    /* ── SELECTBOX dropdown ── */
    .stSelectbox > div > div > div,
    [data-baseweb="select"] > div,
    [data-baseweb="select"] span {{
        background-color: {C_SURFACE2} !important;
        color: {C_TEXT} !important;
        border-color: {C_BORDER2} !important;
    }}
    /* Dropdown list */
    [data-baseweb="popover"],
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"],
    [data-baseweb="menu"] ul {{
        background-color: {C_SURFACE} !important;
        border: 1px solid {C_BORDER2} !important;
        border-radius: 8px !important;
    }}
    [data-baseweb="option"],
    [data-baseweb="menu"] li {{
        background-color: {C_SURFACE} !important;
        color: {C_TEXT} !important;
    }}
    [data-baseweb="option"]:hover,
    [data-baseweb="menu"] li:hover {{
        background-color: {C_SURFACE3} !important;
    }}

    /* ── MULTISELECT tags ── */
    [data-baseweb="tag"] {{
        background-color: rgba(88,166,255,0.15) !important;
        border: 1px solid rgba(88,166,255,0.4) !important;
        color: {C_ACCENT} !important;
        border-radius: 6px !important;
    }}

    /* ══════════════════════════════════════════════
       BOTÕES
    ══════════════════════════════════════════════ */
    .stButton > button {{
        background: {C_ACCENT} !important;
        color: #0D1117 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 0.55rem 1.4rem !important;
        min-height: 42px !important;
        transition: all .18s ease !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.01em !important;
        cursor: pointer !important;
    }}
    .stButton > button:hover {{
        background: {C_ACCENT_H} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(88,166,255,0.35) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}
    /* Botões secundários (outline) */
    .stButton > button[kind="secondary"] {{
        background: transparent !important;
        color: {C_ACCENT} !important;
        border: 1px solid {C_ACCENT} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background: rgba(88,166,255,0.1) !important;
    }}

    /* ══════════════════════════════════════════════
       TABS
    ══════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {{
        background: {C_SURFACE} !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 2px !important;
        border: 1px solid {C_BORDER} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: 7px !important;
        color: {C_TEXT_SEC} !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 18px !important;
        min-height: 38px !important;
        transition: all .15s !important;
        border: none !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {C_TEXT} !important;
        background: {C_SURFACE2} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {C_ACCENT} !important;
        color: #0D1117 !important;
        font-weight: 700 !important;
    }}
    /* Tab content area */
    .stTabs [data-baseweb="tab-panel"] {{
        background: transparent !important;
        padding-top: 1rem !important;
    }}

    /* ══════════════════════════════════════════════
       EXPANDER
    ══════════════════════════════════════════════ */
    .streamlit-expanderHeader,
    [data-baseweb="accordion"] {{
        background: {C_SURFACE2} !important;
        border-radius: 8px !important;
        border: 1px solid {C_BORDER} !important;
        color: {C_TEXT} !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }}
    [data-testid="stExpanderToggleIcon"] {{
        color: {C_TEXT_SEC} !important;
    }}
    .streamlit-expanderHeader:hover {{
        border-color: {C_ACCENT} !important;
    }}
    .streamlit-expanderContent {{
        background: {C_SURFACE} !important;
        border: 1px solid {C_BORDER} !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        color: {C_TEXT} !important;
    }}
    .streamlit-expanderContent p,
    .streamlit-expanderContent div,
    .streamlit-expanderContent span {{
        color: {C_TEXT} !important;
    }}

    /* ══════════════════════════════════════════════
       MÉTRICAS
    ══════════════════════════════════════════════ */
    [data-testid="metric-container"] {{
        background: {C_SURFACE} !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="metric-container"] [data-testid="stMetricValue"] > div {{
        color: {C_TEXT} !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        font-family: 'Inter', sans-serif !important;
    }}
    [data-testid="stMetricLabel"],
    [data-testid="metric-container"] [data-testid="stMetricLabel"] > div {{
        color: {C_TEXT_SEC} !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}
    [data-testid="stMetricDelta"] {{
        color: {C_GREEN} !important;
    }}
    [data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Down"] {{
        color: {C_RED} !important;
    }}

    /* ══════════════════════════════════════════════
       DATAFRAME / TABLE
    ══════════════════════════════════════════════ */
    .stDataFrame,
    [data-testid="stDataFrameResizable"] {{
        background: {C_SURFACE} !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}

    /* ══════════════════════════════════════════════
       SLIDER
    ══════════════════════════════════════════════ */
    [data-baseweb="slider"] [data-testid="stThumbValue"],
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {{
        color: {C_TEXT_SEC} !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}
    [data-baseweb="slider"] div[role="slider"] {{
        background: {C_ACCENT} !important;
        border: 2px solid {C_BG} !important;
        box-shadow: 0 0 0 3px {C_ACCENT} !important;
    }}
    [data-baseweb="slider"] div[data-testid="stSlider-track"] {{
        background: {C_BORDER2} !important;
    }}

    /* ══════════════════════════════════════════════
       RADIO & CHECKBOX
    ══════════════════════════════════════════════ */
    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {{
        color: {C_TEXT} !important;
        font-size: 14px !important;
        padding: 4px 0 !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] {{
        gap: 4px !important;
    }}
    /* Radio / checkbox control circles */
    [data-baseweb="radio"] div[aria-checked="true"] div,
    [data-baseweb="checkbox"] div[aria-checked="true"] div {{
        background: {C_ACCENT} !important;
        border-color: {C_ACCENT} !important;
    }}

    /* ══════════════════════════════════════════════
       ALERTAS NATIVOS
    ══════════════════════════════════════════════ */
    [data-testid="stAlert"],
    .stAlert {{
        border-radius: 8px !important;
        font-size: 14px !important;
    }}
    /* Info */
    [data-testid="stAlert"][kind="info"],
    .stAlert[data-baseweb="notification"][kind="info"] {{
        background: rgba(88,166,255,0.1) !important;
        border: 1px solid rgba(88,166,255,0.3) !important;
        color: {C_TEXT} !important;
    }}
    /* Success */
    [data-testid="stAlert"][kind="success"] {{
        background: rgba(63,185,80,0.1) !important;
        border: 1px solid rgba(63,185,80,0.3) !important;
        color: {C_TEXT} !important;
    }}
    /* Warning */
    [data-testid="stAlert"][kind="warning"] {{
        background: rgba(210,153,34,0.1) !important;
        border: 1px solid rgba(210,153,34,0.3) !important;
        color: {C_TEXT} !important;
    }}
    /* Error */
    [data-testid="stAlert"][kind="error"] {{
        background: rgba(248,81,73,0.1) !important;
        border: 1px solid rgba(248,81,73,0.3) !important;
        color: {C_TEXT} !important;
    }}

    /* ══════════════════════════════════════════════
       PROGRESS BAR NATIVA
    ══════════════════════════════════════════════ */
    .stProgress > div > div > div {{
        background: {C_SURFACE2} !important;
        border-radius: 6px !important;
    }}
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {C_ACCENT}, {C_TEAL}) !important;
        border-radius: 6px !important;
    }}

    /* ══════════════════════════════════════════════
       DIVIDER
    ══════════════════════════════════════════════ */
    hr {{
        border: none !important;
        border-top: 1px solid {C_BORDER} !important;
        margin: 1.5rem 0 !important;
    }}

    /* ══════════════════════════════════════════════
       CODE / PRE
    ══════════════════════════════════════════════ */
    code, pre, .stCode {{
        background: {C_SURFACE2} !important;
        color: {C_TEAL} !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
    }}
    code {{
        padding: 2px 6px !important;
    }}

    /* ══════════════════════════════════════════════
       SCROLLBAR
    ══════════════════════════════════════════════ */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {C_BG}; }}
    ::-webkit-scrollbar-thumb {{
        background: {C_BORDER2};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {C_ACCENT}; }}

    /* ══════════════════════════════════════════════
       COMPONENTES CUSTOM - Design System
    ══════════════════════════════════════════════ */

    /* ── Cards ── */
    .df-card {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        transition: border-color .2s;
    }}
    .df-card:hover {{
        border-color: {C_BORDER2};
    }}
    .df-card-accent {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-left: 3px solid {C_ACCENT};
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }}
    .df-card p, .df-card div,
    .df-card-accent p, .df-card-accent div {{
        color: {C_TEXT} !important;
    }}

    /* ── Badges ── */
    .df-badge {{
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .03em;
        line-height: 1.5;
    }}
    .df-badge-blue   {{ background: rgba(88,166,255,0.15); color: {C_ACCENT}; border: 1px solid rgba(88,166,255,0.3); }}
    .df-badge-green  {{ background: rgba(63,185,80,0.15);  color: {C_GREEN};  border: 1px solid rgba(63,185,80,0.3); }}
    .df-badge-amber  {{ background: rgba(210,153,34,0.15); color: {C_AMBER};  border: 1px solid rgba(210,153,34,0.3); }}
    .df-badge-purple {{ background: rgba(188,140,255,0.15); color: {C_ACCENT2}; border: 1px solid rgba(188,140,255,0.3); }}
    .df-badge-red    {{ background: rgba(248,81,73,0.15);  color: {C_RED};    border: 1px solid rgba(248,81,73,0.3); }}
    .df-badge-teal   {{ background: rgba(57,208,192,0.15); color: {C_TEAL};   border: 1px solid rgba(57,208,192,0.3); }}

    /* ── Section title ── */
    .section-title {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: {C_TEXT_SEC} !important;
        padding-bottom: 8px;
        border-bottom: 1px solid {C_BORDER};
        margin: 1.6rem 0 1rem;
    }}

    /* ── Page header ── */
    .page-title {{
        font-size: 26px;
        font-weight: 800;
        color: {C_TEXT} !important;
        margin-bottom: 4px;
        line-height: 1.3;
        letter-spacing: -.3px;
    }}
    .page-subtitle {{
        font-size: 15px;
        color: {C_TEXT_SEC} !important;
        margin-bottom: 1.6rem;
        line-height: 1.6;
    }}

    /* ── Teoria box ── */
    .teoria-box {{
        background: rgba(88,166,255,0.06);
        border: 1px solid rgba(88,166,255,0.2);
        border-left: 3px solid {C_ACCENT};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
    }}
    .teoria-box h4 {{
        color: {C_ACCENT} !important;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 8px;
    }}
    .teoria-box p, .teoria-box li {{
        color: {C_TEXT} !important;
        font-size: 14px;
        line-height: 1.75;
    }}

    /* ── Info boxes ── */
    .df-info-box {{
        background: rgba(88,166,255,0.08);
        border: 1px solid rgba(88,166,255,0.25);
        border-left: 3px solid {C_ACCENT};
        border-radius: 8px;
        padding: .9rem 1.1rem;
        color: {C_TEXT} !important;
        font-size: 14px;
        margin-bottom: .9rem;
        line-height: 1.65;
    }}
    .df-success-box {{
        background: rgba(63,185,80,0.08);
        border: 1px solid rgba(63,185,80,0.25);
        border-left: 3px solid {C_GREEN};
        border-radius: 8px;
        padding: .9rem 1.1rem;
        color: {C_TEXT} !important;
        font-size: 14px;
        margin-bottom: .9rem;
        line-height: 1.65;
    }}
    .df-warning-box {{
        background: rgba(210,153,34,0.08);
        border: 1px solid rgba(210,153,34,0.25);
        border-left: 3px solid {C_AMBER};
        border-radius: 8px;
        padding: .9rem 1.1rem;
        color: {C_TEXT} !important;
        font-size: 14px;
        margin-bottom: .9rem;
        line-height: 1.65;
    }}
    .df-error-box {{
        background: rgba(248,81,73,0.08);
        border: 1px solid rgba(248,81,73,0.25);
        border-left: 3px solid {C_RED};
        border-radius: 8px;
        padding: .9rem 1.1rem;
        color: {C_TEXT} !important;
        font-size: 14px;
        margin-bottom: .9rem;
        line-height: 1.65;
    }}

    /* ── Steps do modo guiado ── */
    .step-indicator {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.2rem;
    }}
    .step-dot {{
        width: 30px; height: 30px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; font-weight: 700;
        flex-shrink: 0;
    }}
    .step-dot.active  {{ background: {C_ACCENT};  color: #0D1117; }}
    .step-dot.done    {{ background: {C_GREEN};   color: #0D1117; }}
    .step-dot.pending {{ background: {C_SURFACE2}; color: {C_TEXT_SEC}; border: 1px solid {C_BORDER2}; }}

    /* ── Barra de progresso custom ── */
    .progress-bar-wrap {{
        background: {C_SURFACE2};
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin: 6px 0;
        border: 1px solid {C_BORDER};
    }}
    .progress-bar-fill {{
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, {C_ACCENT}, {C_TEAL});
        transition: width .5s ease;
    }}

    /* ── Sidebar brand ── */
    .sidebar-brand {{
        display: flex; align-items: center; gap: 12px;
        padding: 6px 4px 16px;
        border-bottom: 1px solid {C_BORDER};
        margin-bottom: 16px;
    }}
    .sidebar-brand-icon {{
        width: 40px; height: 40px;
        background: linear-gradient(135deg, {C_ACCENT}, {C_ACCENT2});
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; flex-shrink: 0;
    }}
    .sidebar-brand-name {{
        font-size: 16px; font-weight: 800; color: {C_TEXT} !important;
    }}
    .sidebar-brand-sub {{
        font-size: 12px; color: {C_TEXT_SEC} !important;
    }}

    /* ── User pill ── */
    .user-pill {{
        display: flex; align-items: center; gap: 10px;
        background: {C_SURFACE2};
        border: 1px solid {C_BORDER};
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 16px;
    }}
    .user-avatar {{
        width: 32px; height: 32px;
        background: linear-gradient(135deg, {C_ACCENT}, {C_ACCENT2});
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; font-weight: 800; color: #0D1117;
        flex-shrink: 0;
    }}
    .user-name {{ font-size: 14px; font-weight: 700; color: {C_TEXT} !important; }}
    .user-role {{ font-size: 12px; color: {C_TEXT_SEC} !important; display: flex; align-items: center; gap: 4px; }}

    /* ── Selectbox streamlit override ── */
    .stSelectbox [data-testid="stWidgetLabel"] p {{
        color: {C_TEXT} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {C_TEXT} !important;
    }}
    div[data-baseweb="select"] input {{
        color: {C_TEXT} !important;
    }}

    /* ── Spinner / loader ── */
    [data-testid="stSpinner"] {{
        color: {C_ACCENT} !important;
    }}

    /* ── Number input arrows ── */
    .stNumberInput button {{
        background: {C_SURFACE2} !important;
        border: 1px solid {C_BORDER2} !important;
        color: {C_TEXT} !important;
    }}
    .stNumberInput button:hover {{
        background: {C_SURFACE3} !important;
        border-color: {C_ACCENT} !important;
    }}

    /* ── Sidebar nav items ── */
    section[data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        color: {C_TEXT_SEC} !important;
        border: none !important;
        border-radius: 8px !important;
        text-align: left !important;
        padding: 8px 12px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        width: 100% !important;
        box-shadow: none !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: {C_SURFACE2} !important;
        color: {C_TEXT} !important;
        transform: none !important;
        box-shadow: none !important;
    }}

    /* ── Fix para textos literais do Streamlit que ficam escuros ── */
    .stMarkdown code {{
        background: {C_SURFACE2} !important;
        color: {C_TEAL} !important;
        border: 1px solid {C_BORDER} !important;
    }}

    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# BASE DE DADOS LOCAL (TinyDB)
# ══════════════════════════════════════════════════════
def get_db():
    os.makedirs("data", exist_ok=True)
    return TinyDB("data/db.json")

def get_user_progress(username: str) -> dict:
    db = get_db()
    User = Query()
    result = db.table("progress").search(User.username == username)
    if result:
        return result[0]
    return {"username": username, "modulos": {}, "pontos": 0, "completados": []}

def save_user_progress(username: str, modulo: str, dados: dict):
    db = get_db()
    User = Query()
    progress = get_user_progress(username)
    progress["modulos"][modulo] = dados
    progress["ultima_actividade"] = datetime.datetime.now().isoformat()
    table = db.table("progress")
    if table.search(User.username == username):
        table.update(progress, User.username == username)
    else:
        table.insert(progress)

# Página do menu → módulo do percurso (para marcar como concluído)
PAGINA_PARA_MODULO = {
    "classificacao": "supervisionado_classificacao",
    "regressao":     "supervisionado_regressao",
    "clustering":    "nao_supervisionado_clustering",
    "reducao":       "nao_supervisionado_reducao",
    "anomalias":     "nao_supervisionado_anomalias",
    "mlp":           "redes_neurais_mlp",
    "visao":         "deep_learning_visao",
    "nlp":           "deep_learning_nlp",
    "series":        "deep_learning_series",
    "generativo":    "deep_learning_generativo",
    "reforco":       "reforco",
}


def ir_para(pagina: str):
    """Navega para outra página do menu (usado em botões 'Praticar agora')."""
    st.session_state.pagina = pagina
    if "guiado_step" in st.session_state:
        st.session_state.guiado_step = 1
    st.rerun()


def add_pontos(username: str, pontos: int, motivo: str = ""):
    db = get_db()
    User = Query()
    progress = get_user_progress(username)
    progress["pontos"] = progress.get("pontos", 0) + pontos
    # Treinar algo numa página conta como módulo concluído (desbloqueia o percurso)
    modulo_id = PAGINA_PARA_MODULO.get(st.session_state.get("pagina", ""))
    if modulo_id:
        completados = progress.setdefault("completados", [])
        if modulo_id not in completados:
            completados.append(modulo_id)
    if "historico_pontos" not in progress:
        progress["historico_pontos"] = []
    progress["historico_pontos"].append({
        "pontos": pontos,
        "motivo": motivo,
        "data": datetime.datetime.now().isoformat()
    })
    table = db.table("progress")
    if table.search(User.username == username):
        table.update(progress, User.username == username)
    else:
        table.insert(progress)

def get_licoes_concluidas(username: str) -> list:
    return get_user_progress(username).get("licoes", [])

def concluir_licao(username: str, licao_id: str, titulo: str, pontos: int = 5) -> bool:
    """Marca uma lição da Escola de ML como concluída. Devolve True se foi a 1ª vez."""
    progress = get_user_progress(username)
    licoes = progress.setdefault("licoes", [])
    if licao_id in licoes:
        return False
    licoes.append(licao_id)
    db = get_db()
    User = Query()
    table = db.table("progress")
    if table.search(User.username == username):
        table.update(progress, User.username == username)
    else:
        table.insert(progress)
    add_pontos(username, pontos, f"Lição concluída: {titulo}")
    return True

def save_historico_modelo(username: str, entrada: dict):
    db = get_db()
    entrada["username"] = username
    entrada["data"] = datetime.datetime.now().isoformat()
    db.table("historico_modelos").insert(entrada)

def get_historico_modelos(username: str) -> list:
    db = get_db()
    User = Query()
    return db.table("historico_modelos").search(User.username == username)


# ══════════════════════════════════════════════════════
# COMPONENTES UI
# ══════════════════════════════════════════════════════
def page_header(titulo: str, subtitulo: str = "", icone: str = ""):
    prefix = f"{icone} " if icone else ""
    st.markdown(f"""<div style="margin-bottom:1.6rem;"><div class="page-title">{prefix}{titulo}</div><div class="page-subtitle">{subtitulo}</div></div>""", unsafe_allow_html=True)

def section_title(titulo: str):
    st.markdown(f'<div class="section-title">{titulo}</div>', unsafe_allow_html=True)

def card(conteudo: str, accent: bool = False):
    classe = "df-card-accent" if accent else "df-card"
    st.markdown(f'<div class="{classe}">{conteudo}</div>', unsafe_allow_html=True)

def badge(texto: str, cor: str = "blue"):
    return f'<span class="df-badge df-badge-{cor}">{texto}</span>'

def teoria_box(titulo: str, conteudo: str):
    st.markdown(f"""<div class="teoria-box"><h4>{titulo}</h4><p>{conteudo}</p></div>""", unsafe_allow_html=True)

def progresso_bar(valor: float, label: str = ""):
    pct = int(valor * 100)
    st.markdown(f"""<div style="margin-bottom:.8rem;"><div style="display:flex;justify-content:space-between;font-size:13px; color:{C_TEXT_SEC};margin-bottom:5px;font-weight:600;"><span>{label}</span><span>{pct}%</span></div><div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{pct}%"></div></div></div>""", unsafe_allow_html=True)

def metricas_row(items: list):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(
                label=item.get("label", ""),
                value=item.get("valor", ""),
                delta=item.get("delta", None)
            )

def info_box(msg: str):
    st.markdown(f'<div class="df-info-box"><span style="font-weight:700;">Info</span> - {msg}</div>', unsafe_allow_html=True)

def sucesso_box(msg: str):
    st.markdown(f'<div class="df-success-box"><span style="font-weight:700;">OK</span> - {msg}</div>', unsafe_allow_html=True)

def aviso_box(msg: str):
    st.markdown(f'<div class="df-warning-box"><span style="font-weight:700;">Aviso</span> - {msg}</div>', unsafe_allow_html=True)

def erro_box(msg: str):
    st.markdown(f'<div class="df-error-box"><span style="font-weight:700;">Erro</span> - {msg}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# DATASETS EMBUTIDOS
# ══════════════════════════════════════════════════════
def _gerar_casas_luanda(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Dataset fictício, gerado localmente (sem internet), com preços de casas em Luanda."""
    rng = np.random.default_rng(seed)
    bairros = {  # bairro: (prémio de preço em milhões Kz, distância típica ao centro em km)
        "Ingombota": (25, 2), "Miramar": (30, 4), "Maianga": (10, 5), "Talatona": (20, 15),
        "Kilamba": (8, 25), "Benfica": (3, 18), "Camama": (0, 14), "Cazenga": (-5, 10),
        "Viana": (-6, 20), "Cacuaco": (-8, 22),
    }
    nomes = rng.choice(list(bairros), n)
    area = rng.uniform(45, 250, n).round()
    quartos = np.clip((area / 55 + rng.normal(0, .6, n)).round(), 1, 6).astype(int)
    premio = np.array([bairros[b][0] for b in nomes])
    dist = np.clip(np.array([bairros[b][1] for b in nomes]) + rng.normal(0, 1.5, n), .5, None).round(1)
    gerador = rng.integers(0, 2, n)
    quintal = rng.integers(0, 2, n)
    preco = 0.38 * area + 4 * quartos + premio + 6 * gerador + 4 * quintal + rng.normal(0, 6, n)
    return pd.DataFrame({
        "bairro": nomes, "quartos": quartos, "area_m2": area, "distancia_centro_km": dist,
        "tem_gerador": gerador, "tem_quintal": quintal,
        "preco_milhoes_kz": np.clip(preco, 8, None).round(1),
    })


def carregar_dataset_embutido(nome: str) -> tuple:
    """Retorna (df, descricao, tipo_problema)"""
    from sklearn import datasets
    import seaborn as sns

    catalogo = {
        "Iris - Classificação de flores":       ("iris",       "classificacao"),
        "Titanic - Sobrevivência":               ("titanic",    "classificacao"),
        "Vinho - Qualidade":                     ("wine",       "classificacao"),
        "Cancro - Diagnóstico":                 ("breast",     "classificacao"),
        "Casas em Luanda (fictício) - Preços":   ("luanda",     "regressao"),
        "Boston Housing - Preços":               ("boston",     "regressao"),
        "Diabetes - Progressão":                ("diabetes",   "regressao"),
        "Dígitos - Reconhecimento":              ("digits",     "classificacao"),
        "Meias-luas - Clustering":               ("moons",      "clustering"),
        "Círculos - Clustering":                 ("circles",    "clustering"),
        "Blobs - Clustering básico":             ("blobs",      "clustering"),
    }

    if nome not in catalogo:
        return None, "", ""

    chave, tipo = catalogo[nome]

    if chave == "iris":
        d = datasets.load_iris(as_frame=True)
        df = d.frame
        df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
        desc = "150 flores de 3 espécies. Clássico para classificação multiclasse. 4 features numéricas."
    elif chave == "titanic":
        df = sns.load_dataset("titanic")[["survived","pclass","sex","age","sibsp","parch","fare"]].dropna()
        desc = "Dados dos passageiros do Titanic. Prevê sobrevivência. Mix de features numéricas e categóricas."
    elif chave == "wine":
        d = datasets.load_wine(as_frame=True)
        df = d.frame
        desc = "178 vinhos com 13 propriedades químicas. 3 classes de qualidade."
    elif chave == "breast":
        d = datasets.load_breast_cancer(as_frame=True)
        df = d.frame
        df["target"] = df["target"].map({0: "maligno", 1: "benigno"})
        desc = "569 tumores com 30 features. Classifica tumores como malignos ou benignos."
    elif chave == "luanda":
        df = _gerar_casas_luanda()
        desc = ("400 casas fictícias de Luanda (bairro, quartos, área, distância ao centro, gerador, quintal). "
                "Prevê o preço em milhões de Kz. Funciona sem internet.")
    elif chave == "boston":
        try:
            d = datasets.fetch_california_housing(as_frame=True)
            df = d.frame
            desc = "Dados de habitação da Califórnia. Prevê o preço médio de casas. 8 features."
        except Exception:
            # Sem internet (o dataset é descarregado na 1ª utilização): usa o dataset local
            df = _gerar_casas_luanda()
            desc = ("Sem ligação à internet para descarregar os dados da Califórnia. "
                    "Foi carregado o dataset local de casas em Luanda (fictício), que serve o mesmo propósito.")
    elif chave == "diabetes":
        d = datasets.load_diabetes(as_frame=True)
        df = d.frame
        desc = "442 pacientes com 10 features clínicas. Prevê progressão da diabetes ao fim de 1 ano."
    elif chave == "digits":
        d = datasets.load_digits(as_frame=True)
        df = d.frame
        desc = "1797 imagens 8x8 de dígitos escritos à mão (0-9). 64 features de pixels."
    elif chave == "moons":
        X, y = datasets.make_moons(n_samples=300, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=["x1","x2"])
        df["target"] = y
        desc = "300 pontos em forma de meias-luas. Ideal para visualizar clustering não-linear."
    elif chave == "circles":
        X, y = datasets.make_circles(n_samples=300, noise=0.05, random_state=42)
        df = pd.DataFrame(X, columns=["x1","x2"])
        df["target"] = y
        desc = "300 pontos em dois círculos concêntricos. Bom para DBSCAN vs KMeans."
    elif chave == "blobs":
        X, y = datasets.make_blobs(n_samples=300, centers=4, random_state=42)
        df = pd.DataFrame(X, columns=["x1","x2"])
        df["target"] = y
        desc = "300 pontos em 4 grupos. Dataset simples para aprender clustering básico."
    else:
        return None, "", ""

    return df, desc, tipo


# ══════════════════════════════════════════════════════
# TEXTOS DE TEORIA POR ALGORITMO
# ══════════════════════════════════════════════════════
TEORIA = {
    "KNN": {
        "nome": "K-Nearest Neighbors (KNN)",
        "tipo": "Supervisionado - Classificação / Regressão",
        "analogia": "Imagina que chegas a uma cidade nova e queres saber se um bairro é seguro. Perguntas aos K vizinhos mais próximos. Se a maioria diz que é seguro, assumes que é seguro.",
        "como_funciona": "Para classificar um novo ponto, o KNN calcula a distância a todos os pontos de treino e escolhe os K mais próximos. A classe mais comum entre esses K vizinhos é a previsão.",
        "quando_usar": "Datasets pequenos/médios, quando os dados têm clusters naturais, quando precisas de um modelo simples e interpretável.",
        "cuidados": "Lento para datasets grandes. Muito afectado por features com escalas diferentes - normaliza sempre. K pequeno = overfitting, K grande = underfitting.",
        "hiperparametros": {"n_neighbors": "Número de vizinhos. Valores típicos: 3, 5, 7, 11", "metric": "Distância euclidiana (padrão), manhattan, minkowski"},
        "badge": "blue"
    },
    "Decision Tree": {
        "nome": "Árvore de Decisão",
        "tipo": "Supervisionado - Classificação / Regressão",
        "analogia": "Como um jogo de 20 perguntas. O modelo aprende a sequência de perguntas que melhor divide os dados em grupos puros.",
        "como_funciona": "Divide recursivamente os dados baseando-se na feature que mais reduz a impureza (Gini ou Entropia). Cada divisão cria um nó na árvore.",
        "quando_usar": "Quando precisas de interpretabilidade total. Quando os dados têm relações não-lineares. Bom ponto de partida antes de ensembles.",
        "cuidados": "Overfit facilmente sem poda (max_depth). Instável - pequenas mudanças nos dados mudam muito a árvore.",
        "hiperparametros": {"max_depth": "Profundidade máxima. None = overfitting. Começa com 3-5.", "min_samples_split": "Mínimo de amostras para dividir um nó"},
        "badge": "green"
    },
    "Random Forest": {
        "nome": "Random Forest",
        "tipo": "Supervisionado - Classificação / Regressão",
        "analogia": "Em vez de perguntar a um especialista, perguntas a 100 especialistas diferentes (cada um viu dados diferentes) e fazes uma votação. A decisão colectiva é mais robusta.",
        "como_funciona": "Treina N árvores de decisão, cada uma em subconjuntos aleatórios dos dados e das features (bagging). A previsão final é a votação (classificação) ou média (regressão).",
        "quando_usar": "Um dos melhores algoritmos para uso geral. Robusto a overfitting, lida bem com muitas features, dá importância das variáveis.",
        "cuidados": "Mais lento para treinar que uma árvore única. Menos interpretável (caixa negra comparado a uma árvore simples).",
        "hiperparametros": {"n_estimators": "Número de árvores. Mais = melhor (até certo ponto). Começa com 100.", "max_features": "Features por árvore. 'sqrt' para classificação, 'log2' para regressão"},
        "badge": "green"
    },
    "Gradient Boosting": {
        "nome": "Gradient Boosting",
        "tipo": "Supervisionado - Classificação / Regressão",
        "analogia": "Aprender com os erros. Treinas um modelo fraco, vês onde errou, treinas outro modelo focado nos erros, e assim sucessivamente. Cada modelo corrige o anterior.",
        "como_funciona": "Treina modelos sequencialmente, onde cada novo modelo tenta corrigir os erros residuais do anterior, usando o gradiente do erro como sinal de aprendizagem.",
        "quando_usar": "Quando precisas de alta performance. Competições de ML. Datasets tabulares estruturados.",
        "cuidados": "Propenso a overfitting com learning_rate alto. Mais lento que Random Forest. Sensível a outliers.",
        "hiperparametros": {"n_estimators": "Número de boosting rounds", "learning_rate": "Taxa de aprendizagem. Menor = mais lento mas melhor. Típico: 0.01-0.1", "max_depth": "Profundidade das árvores base. Típico: 3-5"},
        "badge": "amber"
    },
    "SVM": {
        "nome": "Support Vector Machine (SVM)",
        "tipo": "Supervisionado - Classificação / Regressão",
        "analogia": "Imagina separar laranjas de maçãs numa mesa. O SVM encontra a linha (ou plano) que separa as duas classes com a maior margem possível dos pontos mais próximos.",
        "como_funciona": "Encontra o hiperplano de separação com margem máxima entre classes. Com o kernel trick, consegue separar dados não-lineares mapeando para dimensões superiores.",
        "quando_usar": "Dados de alta dimensão (texto, imagens). Datasets pequenos a médios. Quando as classes são claramente separáveis.",
        "cuidados": "Lento para datasets grandes. Requer normalização. Difícil de interpretar com kernels não-lineares.",
        "hiperparametros": {"C": "Penalização do erro. Alto = menos margem, mais preciso no treino (overfitting). Baixo = margem maior (underfitting)", "kernel": "rbf (padrão, não-linear), linear, poly"},
        "badge": "purple"
    },
    "Logistic Regression": {
        "nome": "Regressão Logística",
        "tipo": "Supervisionado - Classificação",
        "analogia": "Apesar do nome, é um classificador. Estima a probabilidade de pertencer a uma classe usando uma função sigmoide - transforma qualquer número real em probabilidade entre 0 e 1.",
        "como_funciona": "Aprende pesos para cada feature que, combinados com a função sigmoide, produzem uma probabilidade. Usa máxima verossimilhança para optimizar os pesos.",
        "quando_usar": "Classificação binária simples. Quando precisas de probabilidades calibradas. Benchmark rápido antes de modelos complexos.",
        "cuidados": "Assume relação linear entre features e log-odds. Não captura relações não-lineares sem feature engineering.",
        "hiperparametros": {"C": "Inverso da regularização. C alto = menos regularização", "max_iter": "Iterações de optimização. Aumenta se não convergir"},
        "badge": "blue"
    },
    "Naive Bayes": {
        "nome": "Naive Bayes",
        "tipo": "Supervisionado - Classificação",
        "analogia": "'Naive' porque assume que todas as features são independentes entre si - o que raramente é verdade, mas funciona surpreendentemente bem para texto.",
        "como_funciona": "Calcula a probabilidade de cada classe dado os valores das features, usando o Teorema de Bayes e assumindo independência condicional entre features.",
        "quando_usar": "Classificação de texto (spam, sentimento). Datasets com muitas features. Quando velocidade é prioritária.",
        "cuidados": "A assunção de independência raramente é verdade. Pode ter problemas com features correlacionadas.",
        "hiperparametros": {"var_smoothing": "Estabilidade numérica. Ajusta se houver problemas de precisão"},
        "badge": "teal"
    },
    "KMeans": {
        "nome": "K-Means Clustering",
        "tipo": "Não Supervisionado - Clustering",
        "analogia": "Tens 100 pessoas numa sala e queres formar K grupos. Colocas K representantes aleatórios, cada pessoa junta-se ao mais próximo, depois os representantes movem-se para o centro do grupo. Repete até estabilizar.",
        "como_funciona": "Inicializa K centróides aleatoriamente. Assign cada ponto ao centróide mais próximo. Recalcula centróides como média do cluster. Repete até convergir.",
        "quando_usar": "Quando sabes o número de clusters. Dados com clusters esféricos e tamanhos similares. Boa performance em grandes datasets.",
        "cuidados": "Precisas definir K antes. Sensível a outliers. Assume clusters esféricos - não funciona bem com formas irregulares.",
        "hiperparametros": {"n_clusters": "Número de clusters K. Usa o método do cotovelo para escolher.", "init": "KMeans++ (padrão) dá melhores resultados que 'random'"},
        "badge": "teal"
    },
    "DBSCAN": {
        "nome": "DBSCAN",
        "tipo": "Não Supervisionado - Clustering",
        "analogia": "Imagina explorar uma floresta. Começas num ponto, expandes para todos os vizinhos próximos, e continuas a expansão. Regiões densas formam clusters, pontos isolados são ruído.",
        "como_funciona": "Define clusters como regiões de alta densidade separadas por regiões de baixa densidade. Não precisa de K definido. Detecta automaticamente outliers como ruído.",
        "quando_usar": "Quando não sabes o número de clusters. Clusters com formas arbitrárias. Quando há outliers nos dados.",
        "cuidados": "Difícil de ajustar eps e min_samples. Não funciona bem com clusters de densidades muito diferentes.",
        "hiperparametros": {"eps": "Raio da vizinhança. Usa gráfico kNN para escolher.", "min_samples": "Mínimo de pontos para ser core point. Típico: 5"},
        "badge": "purple"
    },
    "Linear Regression": {
        "nome": "Regressão Linear",
        "tipo": "Supervisionado - Regressão",
        "analogia": "A reta que melhor se ajusta a uma nuvem de pontos. Minimiza a soma dos quadrados das distâncias verticais entre os pontos e a reta.",
        "como_funciona": "Encontra os coeficientes da equação y = β₀ + β₁x₁ + ... + βₙxₙ que minimizam o erro quadrático médio (MSE).",
        "quando_usar": "Relação linear entre features e target. Baseline simples para qualquer problema de regressão. Interpretabilidade total.",
        "cuidados": "Assume linearidade e homocedasticidade. Sensível a outliers. Multicolinearidade causa instabilidade.",
        "hiperparametros": {"fit_intercept": "Inclui o intercepto β₀. Quase sempre True."},
        "badge": "blue"
    },
    "Ridge": {
        "nome": "Ridge Regression (L2)",
        "tipo": "Supervisionado - Regressão",
        "analogia": "Regressão Linear com penalização. Como a Linear mas com um imposto sobre coeficientes grandes - força o modelo a ser mais simples e menos propenso a overfitting.",
        "como_funciona": "Adiciona penalização L2 (soma dos quadrados dos coeficientes × alpha) à função de custo. Encolhe todos os coeficientes mas não os zera.",
        "quando_usar": "Quando tens multicolinearidade. Quando tens muitas features. Regularização suave.",
        "cuidados": "Alpha alto = mais regularização = underfitting. Não faz selecção de variáveis (todos os coeficientes permanecem).",
        "hiperparametros": {"alpha": "Força da regularização. 0 = Linear sem regularização. Típico: 0.1 a 100"},
        "badge": "blue"
    },
    "Lasso": {
        "nome": "Lasso Regression (L1)",
        "tipo": "Supervisionado - Regressão",
        "analogia": "Como Ridge mas mais agressivo - força alguns coeficientes a zero, fazendo selecção automática de variáveis. Útil quando suspeitas que poucas features são realmente importantes.",
        "como_funciona": "Adiciona penalização L1 (soma dos valores absolutos dos coeficientes × alpha). Produz coeficientes esparsos - muitos exactamente zero.",
        "quando_usar": "Feature selection automática. Quando suspeitas que poucas features são relevantes. Alta dimensionalidade.",
        "cuidados": "Pode descartar features correlacionadas de forma arbitrária. Instável com features altamente correlacionadas.",
        "hiperparametros": {"alpha": "Força da regularização L1. Valores maiores = mais coeficientes zerados"},
        "badge": "amber"
    },
    "AdaBoost": {
        "nome": "AdaBoost",
        "tipo": "Supervisionado - Classificação / Regressão",
        "analogia": "Aprende com os erros dando mais atenção aos casos difíceis. Cada novo classificador foca-se nos exemplos que os anteriores erraram mais.",
        "como_funciona": "Treina classificadores fracos sequencialmente. Os exemplos mal classificados recebem mais peso na iteração seguinte. A previsão final é uma combinação ponderada.",
        "quando_usar": "Quando o modelo base é fraco (stumps). Boa alternativa ao Gradient Boosting quando há menos overfitting.",
        "cuidados": "Sensível a outliers (recebem muito peso). Mais lento que Random Forest.",
        "hiperparametros": {"n_estimators": "Número de estimadores fracos", "learning_rate": "Peso de cada estimador"},
        "badge": "amber"
    },
    "XGBoost": {
        "nome": "XGBoost",
        "tipo": "Supervisionado - Classificação / Regressão",
        "analogia": "Gradient Boosting optimizado para velocidade e performance. Vencedor de dezenas de competições Kaggle. Regularização incorporada evita overfitting.",
        "como_funciona": "Gradient Boosting com optimizações: regularização L1/L2 incorporada, tratamento nativo de valores nulos, processamento paralelo, poda de árvores.",
        "quando_usar": "Datasets tabulares estruturados. Competições de ML. Quando precisas do melhor resultado possível.",
        "cuidados": "Muitos hiperparâmetros. Pode overfit em datasets pequenos sem tuning.",
        "hiperparametros": {"n_estimators": "Rounds de boosting", "max_depth": "Profundidade. Típico: 3-8", "learning_rate": "Típico: 0.01-0.3", "subsample": "Fracção de dados por árvore. Típico: 0.8"},
        "badge": "red"
    },
    "LightGBM": {
        "nome": "LightGBM",
        "tipo": "Supervisionado - Classificação / Regressão",
        "analogia": "XGBoost mais rápido. Usa crescimento de árvore leaf-wise (vertical) em vez de level-wise (horizontal), tornando-o muito mais eficiente em datasets grandes.",
        "como_funciona": "Gradient Boosting com GOSS (Gradient-based One-Side Sampling) e EFB (Exclusive Feature Bundling). Muito mais rápido que XGBoost em datasets grandes.",
        "quando_usar": "Datasets grandes (>10k linhas). Quando velocidade é importante. Resultados comparáveis ao XGBoost.",
        "cuidados": "Pode overfit em datasets pequenos. leaf-wise pode criar árvores muito profundas.",
        "hiperparametros": {"num_leaves": "Folhas por árvore. Típico: 31", "learning_rate": "Típico: 0.05-0.1", "n_estimators": "Número de árvores"},
        "badge": "red"
    },
}
