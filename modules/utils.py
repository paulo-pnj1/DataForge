"""
DataForge EDU - Utilitários partilhados
Paleta, CSS, componentes reutilizáveis
Design profissional: tema escuro consistente, WCAG AA/AAA
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import copy
import time
import datetime
from modules import db as _db

# ══════════════════════════════════════════════════════
# PALETA DE CORES - Design System
# ══════════════════════════════════════════════════════
TEMAS = {
    "escuro": dict(
        bg="#0D1117", surface="#161B27", surface2="#1C2333", surface3="#21262D",
        border="#30363D", border2="#3A4560",
        accent="#58A6FF", accent_h="#79BAFF", accent2="#BC8CFF",
        green="#3FB950", amber="#D29922", red="#F85149", teal="#39D0C0",
        text="#E6EDF3", text_sec="#8B949E", text_mute="#484F58", text2="#B8C4D8", text3="#7A8BA8",
        on_accent="#FFFFFF", overlay="rgba(255,255,255,.04)", shadow="rgba(0,0,0,.5)"),
    "claro": dict(
        bg="#FFFFFF", surface="#F6F8FA", surface2="#EAEEF2", surface3="#DDE3EA",
        border="#D0D7DE", border2="#AFB8C1",
        accent="#0969DA", accent_h="#218BFF", accent2="#8250DF",
        green="#1A7F37", amber="#9A6700", red="#CF222E", teal="#1B7C83",
        text="#1F2328", text_sec="#57606A", text_mute="#6E7781", text2="#424A53", text3="#6E7781",
        on_accent="#FFFFFF", overlay="rgba(0,0,0,.04)", shadow="rgba(31,35,40,.15)"),
}


class Cor(str):
    """Uma cor da paleta. Em Python (gráficos, matplotlib, plotly) é o hex do tema escuro;
    dentro de um f-string/HTML vira `var(--df-nome, #hex)`, por isso muda sozinha com o tema."""
    def __new__(cls, hex_: str, nome: str):
        o = super().__new__(cls, hex_)
        o.nome = nome
        return o

    def __format__(self, spec: str) -> str:
        if spec:
            return str.__format__(self, spec)
        return f"var(--df-{self.nome.replace('_', '-')}, {str.__str__(self)})"

    # copy/deepcopy/pickle: o __new__ pede 2 argumentos, por isso sem isto o Plotly (que faz deepcopy
    # das cores dos gráficos) falhava com "Cor.__new__() missing 1 required positional argument: 'nome'"
    def __reduce__(self):
        return (Cor, (str.__str__(self), self.nome))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _cor(nome: str) -> Cor:
    return Cor(TEMAS["escuro"][nome], nome)


C_BG         = _cor("bg")          # fundo principal
C_SURFACE    = _cor("surface")     # cards e painéis
C_SURFACE2   = _cor("surface2")    # inputs e elementos secundários
C_SURFACE3   = _cor("surface3")    # hover states
C_BORDER     = _cor("border")      # bordas subtis
C_BORDER2    = _cor("border2")     # bordas mais visíveis
C_ACCENT     = _cor("accent")      # azul primário
C_ACCENT_H   = _cor("accent_h")    # hover do accent
C_ACCENT2    = _cor("accent2")     # violeta secundário
C_GREEN      = _cor("green")       # sucesso
C_AMBER      = _cor("amber")       # aviso
C_RED        = _cor("red")         # erro
C_TEAL       = _cor("teal")        # teal
C_TEXT       = _cor("text")        # texto principal
C_TEXT_SEC   = _cor("text_sec")    # texto secundário
C_TEXT_MUTE  = _cor("text_mute")   # texto muted

PALETTE = [C_ACCENT, C_TEAL, C_AMBER, C_RED, C_ACCENT2, C_GREEN]

def tema_actual() -> str:
    """'claro' (por omissão) ou 'escuro', conforme a escolha guardada na sessão."""
    try:
        return "escuro" if st.session_state.get("tema") == "escuro" else "claro"
    except Exception:
        return "claro"


def paleta(tema: str = None) -> dict:
    return TEMAS.get(tema or tema_actual(), TEMAS["escuro"])


def _css_vars() -> str:
    return "\n        ".join(f"--df-{k.replace('_', '-')}: {v};" for k, v in paleta().items())


# ══════════════════════════════════════════════════════
# CSS GLOBAL - Cobertura completa do Streamlit
# ══════════════════════════════════════════════════════
def _css_global() -> str:
    return f"""
    <style>
    /* ── GOOGLE FONTS ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ══════════════════════════════════════════════
       ROOT & RESET - tema escuro ou claro (variáveis --df-*)
    ══════════════════════════════════════════════ */
    :root {{
        color-scheme: {'light' if tema_actual() == 'claro' else 'dark'};
        {_css_vars()}
        --bg: var(--df-bg);
        --surface: var(--df-surface);
        --surface2: var(--df-surface2);
        --border: var(--df-border);
        --accent: var(--df-accent);
        --text: var(--df-text);
        --text-sec: var(--df-text-sec);
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
            box-shadow: 4px 0 24px var(--df-shadow, rgba(0,0,0,.5)) !important;
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
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {{
        background: {C_ACCENT} !important;
        color: var(--df-on-accent, #FFFFFF) !important;
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
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover {{
        background: {C_ACCENT_H} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(88,166,255,0.35) !important;
    }}
    .stButton > button:active,
    .stDownloadButton > button:active,
    .stFormSubmitButton > button:active {{
        transform: translateY(0) !important;
    }}
    /* Botões secundários (outline) */
    .stButton > button[kind="secondary"],
    .stDownloadButton > button[kind="secondary"],
    .stFormSubmitButton > button[kind="secondary"] {{
        background: transparent !important;
        color: {C_ACCENT} !important;
        border: 1px solid {C_ACCENT} !important;
    }}
    .stButton > button[kind="secondary"]:hover,
    .stDownloadButton > button[kind="secondary"]:hover,
    .stFormSubmitButton > button[kind="secondary"]:hover {{
        background: rgba(88,166,255,0.1) !important;
    }}

    .stButton > button p, .stDownloadButton > button p, .stFormSubmitButton > button p {{
        color: inherit !important;
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
        color: var(--df-on-accent, #FFFFFF) !important;
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
    .step-dot.active  {{ background: {C_ACCENT};  color: var(--df-on-accent, #FFFFFF); }}
    .step-dot.done    {{ background: {C_GREEN};   color: var(--df-on-accent, #FFFFFF); }}
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
        font-size: 14px; font-weight: 800; color: var(--df-on-accent, #FFFFFF);
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
    """


def _minificar_css(css: str) -> str:
    """Tira comentários e espaços a mais: o CSS vai por rede a cada interacção, por isso quanto menor melhor."""
    import re
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"\s+", " ", css)
    css = re.sub(r"\s*([{};,])\s*", r"\1", css)
    return css.replace("</style><style>", "").strip()


_CSS_CACHE = {}


def inject_css():
    """Um único bloco de CSS por execução, já minificado e guardado em memória por tema (o resultado é igual em todas
    as execuções e sessões; só muda com o tema)."""
    tema = tema_actual()
    css = _CSS_CACHE.get(tema)
    if css is None:
        css = _minificar_css(_css_global() + _CSS_NATIVOS + (_CSS_ESCURO if tema == "escuro" else ""))
        _CSS_CACHE[tema] = css
    st.markdown(css, unsafe_allow_html=True)


# A base do Streamlit (.streamlit/config.toml) é clara. Aqui os elementos nativos seguem o tema escolhido
# (variáveis --df-*), seja claro ou escuro.
_CSS_NATIVOS = """
<style>
[data-testid="stHeader"] button, [data-testid="stToolbar"] * { color: var(--df-text) !important; }
[data-testid="stCode"] pre, .stCode pre { background: var(--df-surface2) !important; }
[data-testid="stCode"] code, [data-testid="stCode"] code span, .stCode code span { color: var(--df-text) !important; }

/* Botões do menu lateral */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] [data-testid^="stBaseButton"] {
    background: var(--df-surface2) !important;
    color: var(--df-text) !important;
    border: 1px solid var(--df-border) !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button *,
section[data-testid="stSidebar"] [data-testid^="stBaseButton"] * { color: var(--df-text) !important; }
section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] [data-testid^="stBaseButton"]:hover {
    background: var(--df-surface3) !important;
    border-color: var(--df-accent) !important;
}

/* Uma só barra de rolagem, no lado direito da janela: a página principal rola dentro da janela (sem rolagem lateral)
   e o cabeçalho do Streamlit não desenha linhas por cima do conteúdo */
html, body, .stApp, [data-testid="stApp"] { height: 100vh !important; overflow: hidden !important; }
[data-testid="stAppViewContainer"] { height: 100vh !important; overflow: hidden !important; }
[data-testid="stMain"], section.main { height: 100vh !important; overflow-x: hidden !important; overflow-y: scroll !important; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 100% !important; box-sizing: border-box !important; }
[data-testid="stHeader"], header[data-testid="stHeader"] { background: var(--df-bg) !important; border: none !important; box-shadow: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Barra de rolagem sempre visível na página principal, no menu lateral e nas listas (segue o tema) */
[data-testid="stMain"], [data-testid="stAppViewContainer"] > section.main, section.main,
[data-testid="stSidebarContent"], [data-testid="stSidebar"] > div:first-child {
    overflow-y: scroll !important; scrollbar-width: auto; scrollbar-color: var(--df-border2) var(--df-surface);
}
[data-testid="stMain"]::-webkit-scrollbar, section.main::-webkit-scrollbar,
[data-testid="stSidebarContent"]::-webkit-scrollbar, [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar,
.st-key-ia_msgs::-webkit-scrollbar { width: 12px; height: 12px; }
[data-testid="stMain"]::-webkit-scrollbar-track, section.main::-webkit-scrollbar-track,
[data-testid="stSidebarContent"]::-webkit-scrollbar-track, [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track,
.st-key-ia_msgs::-webkit-scrollbar-track { background: var(--df-surface); }
[data-testid="stMain"]::-webkit-scrollbar-thumb, section.main::-webkit-scrollbar-thumb,
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb, [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb,
.st-key-ia_msgs::-webkit-scrollbar-thumb { background: var(--df-border2); border-radius: 8px; border: 2px solid var(--df-surface); }
[data-testid="stMain"]::-webkit-scrollbar-thumb:hover, section.main::-webkit-scrollbar-thumb:hover,
.st-key-ia_msgs::-webkit-scrollbar-thumb:hover { background: var(--df-accent); }

/* Caixa de carregar ficheiros (upload): segue o tema, com texto legível nos dois */
[data-testid="stFileUploaderDropzone"] { background: var(--df-surface2) !important; border: 1px dashed var(--df-border2) !important; border-radius: 10px !important; }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--df-text-sec) !important; }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] [data-testid^="stBaseButton"] {
    background: var(--df-accent) !important; border: none !important; }
[data-testid="stFileUploaderDropzone"] button * { color: #FFFFFF !important; }
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] { color: var(--df-text) !important; }

/* Texto branco em tudo o que tem fundo azul (botões, separador activo, menu activo) */
.stButton > button:not([kind="secondary"]) *,
.stDownloadButton > button:not([kind="secondary"]) *,
.stFormSubmitButton > button:not([kind="secondary"]) *,
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * { color: #FFFFFF !important; }

/* Etiquetas dos selectores múltiplos (Colunas a usar, Algoritmos…), opções escolhidas em pílulas e controlos segmentados */
div[data-baseweb="select"] [data-baseweb="tag"], div[data-baseweb="select"] [data-baseweb="tag"] *,
[data-baseweb="tag"], [data-baseweb="tag"] *, [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
[data-testid="stBaseButton-pillsActive"], [data-testid="stBaseButton-pillsActive"] *,
[data-testid="stBaseButton-segmented_controlActive"], [data-testid="stBaseButton-segmented_controlActive"] * { color: #FFFFFF !important; }
div[data-baseweb="select"] [data-baseweb="tag"] svg, [data-baseweb="tag"] svg, [data-baseweb="tag"] [role="presentation"] { fill: #FFFFFF !important; color: #FFFFFF !important; }
[data-baseweb="tag"] { background-color: var(--df-accent) !important; border-color: var(--df-accent) !important; }

/* Menu lateral: botões com dica (tooltip) ocupam a largura toda */
section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"], section[data-testid="stSidebar"] .stButton { width: 100% !important; }
section[data-testid="stSidebar"] [data-testid^="stBaseButton"] { width: 100% !important; border-radius: 8px !important; }

/* Menu lateral: a página onde estás fica marcada a azul */
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: var(--df-accent) !important;
    border-color: var(--df-accent) !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] *,
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * { color: #FFFFFF !important; }

/* Interruptor (toggle): barra visível nos dois temas, azul quando ligado */
[data-testid="stToggle"] label:has(input[type="checkbox"]) > span:first-of-type,
[data-testid="stToggle"] label:has(input[type="checkbox"]) > div:first-of-type:not(:last-child) {
    background: var(--df-border2) !important;
}
[data-testid="stToggle"] label:has(input:checked) > span:first-of-type,
[data-testid="stToggle"] label:has(input:checked) > div:first-of-type:not(:last-child) {
    background: var(--df-accent) !important;
}

/* Dicas flutuantes (tooltips) */
[data-testid="stTooltipContent"], [role="tooltip"], [role="tooltip"] > div {
    background: var(--df-surface3) !important;
    color: var(--df-text) !important;
    border: 1px solid var(--df-border) !important;
}
[data-testid="stTooltipContent"] *, [role="tooltip"] * { color: var(--df-text) !important; }

/* Listas de escolha (selectbox, multiselect) e menus */
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul, [data-baseweb="menu"],
[data-baseweb="popover"] li, [data-baseweb="menu"] li, [role="listbox"], [role="option"] {
    background: var(--df-surface) !important;
    color: var(--df-text) !important;
}
[role="option"]:hover, [data-baseweb="menu"] li:hover { background: var(--df-surface3) !important; }
</style>
"""

# Só no tema escuro: tabelas em canvas e gráficos nativos são desenhados claros e convertem-se por filtro.
_CSS_ESCURO = """
<style>
/* Azul mais fundo no tema escuro, para o texto branco ler-se bem */
.stButton > button:not([kind="secondary"]), .stDownloadButton > button:not([kind="secondary"]),
.stFormSubmitButton > button:not([kind="secondary"]), .stTabs [aria-selected="true"], [data-baseweb="tag"],
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { background: #1F6FEB !important; }
[data-testid="stDataFrame"], [data-testid="stArrowVegaLiteChart"] { filter: invert(.91) hue-rotate(180deg); }   /* .91: fundo cinzento-escuro em vez de preto puro */
</style>
"""


# ══════════════════════════════════════════════════════
# GRÁFICOS: os gráficos são desenhados para o tema escuro; no claro adaptam-se ao mostrar
# ══════════════════════════════════════════════════════
def _luminancia(cor) -> float:
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(cor)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def tema_figura_matplotlib(fig, tema: str = None):
    """Adapta uma figura matplotlib ao tema claro (no escuro não faz nada). Nunca levanta erros."""
    if (tema or tema_actual()) != "claro" or fig is None:
        return fig
    P = TEMAS["claro"]
    try:
        fig.patch.set_facecolor(P["surface"])
        for t in list(fig.texts) + ([fig._suptitle] if getattr(fig, "_suptitle", None) else []):
            if _luminancia(t.get_color()) > 0.6:
                t.set_color(P["text"])
        for ax in fig.axes:
            ax.set_facecolor(P["surface"])
            for sp in ax.spines.values():
                sp.set_edgecolor(P["border2"])
            ax.tick_params(colors=P["text_sec"], which="both")
            rotulos = [ax.xaxis.label, ax.yaxis.label, ax.title]
            if hasattr(ax, "zaxis"):
                rotulos.append(ax.zaxis.label)
            for r in rotulos:
                r.set_color(P["text"])
            for ln in ax.get_xgridlines() + ax.get_ygridlines():
                ln.set_color(P["border"])
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(P["surface2"])
                leg.get_frame().set_edgecolor(P["border"])
                for t in leg.get_texts() + [leg.get_title()]:
                    t.set_color(P["text"])
            if not ax.images:          # em mapas de calor o texto das células tem a sua própria cor
                for t in ax.texts:
                    if _luminancia(t.get_color()) > 0.6:
                        t.set_color(P["text"])
    except Exception:
        pass
    return fig


def tema_figura_plotly(fig, tema: str = None):
    """Adapta uma figura plotly ao tema claro (no escuro não faz nada). Nunca levanta erros."""
    if (tema or tema_actual()) != "claro" or fig is None or not hasattr(fig, "update_layout"):
        return fig
    P = TEMAS["claro"]
    eixos = dict(color=P["text_sec"], gridcolor=P["border"], zerolinecolor=P["border2"], linecolor=P["border2"])
    passos = (
        lambda: fig.update_layout(paper_bgcolor=P["surface"], plot_bgcolor=P["surface"],
                                  font=dict(color=P["text"]), title_font_color=P["text"],
                                  legend=dict(font=dict(color=P["text"]))),
        lambda: fig.update_xaxes(**eixos),
        lambda: fig.update_yaxes(**eixos),
        lambda: fig.update_polars(bgcolor=P["surface"],
                                  radialaxis=dict(color=P["text_sec"], gridcolor=P["border"]),
                                  angularaxis=dict(color=P["text"], gridcolor=P["border"])),
        lambda: fig.update_annotations(font_color=P["text"]),
    )
    for passo in passos:          # cada passo é independente: um que não se aplica não impede os outros
        try:
            passo()
        except Exception:
            pass
    return fig


def instalar_tema_graficos():
    """Faz st.pyplot e st.plotly_chart adaptarem os gráficos ao tema escolhido (chamar uma vez por execução)."""
    if not getattr(st.pyplot, "_df_tema", False):
        _pyplot = st.pyplot

        def pyplot(fig=None, *args, **kwargs):
            if fig is None:
                try:
                    import matplotlib.pyplot as plt
                    fig = plt.gcf()
                except Exception:
                    pass
            tema_figura_matplotlib(fig)
            return _pyplot(fig, *args, **kwargs)
        pyplot._df_tema = True
        st.pyplot = pyplot
    if not getattr(st.plotly_chart, "_df_tema", False):
        _plotly = st.plotly_chart

        def plotly_chart(fig, *args, **kwargs):
            tema_figura_plotly(fig)
            return _plotly(fig, *args, **kwargs)
        plotly_chart._df_tema = True
        st.plotly_chart = plotly_chart


# ══════════════════════════════════════════════════════
# ESCOLHA DO TEMA (fica guardada na conta)
# ══════════════════════════════════════════════════════
def definir_tema(username: str, tema: str):
    tema = "escuro" if tema == "escuro" else "claro"
    st.session_state["tema"] = tema
    st.session_state["tema_toggle"] = tema == "claro"
    st.session_state["perfil_tema"] = "Claro" if tema == "claro" else "Escuro"
    try:
        progress = get_user_progress(username)
        progress["tema"] = tema
        _gravar_progress(username, progress)
    except Exception:
        pass          # sem base de dados a escolha vale só para esta sessão


def carregar_tema(username: str) -> bool:
    """Ao entrar, aplica o tema guardado na conta (claro se nunca escolheu). True se mudou: fazer st.rerun()."""
    chave = f"_tema_lido_{username}"
    if st.session_state.get(chave):
        return False
    st.session_state[chave] = True
    try:
        guardado = get_user_progress(username).get("tema")
    except Exception:
        guardado = None
    alvo = guardado if guardado in ("claro", "escuro") else "claro"
    st.session_state["tema_toggle"] = alvo == "claro"
    st.session_state["perfil_tema"] = "Claro" if alvo == "claro" else "Escuro"
    if alvo != tema_actual():
        st.session_state["tema"] = alvo
        return True
    return False


def _ao_mudar_toggle(username: str):
    definir_tema(username, "claro" if st.session_state.get("tema_toggle") else "escuro")


def _ao_mudar_radio(username: str):
    definir_tema(username, "claro" if st.session_state.get("perfil_tema") == "Claro" else "escuro")


def render_seletor_tema(username: str):
    """Interruptor compacto (menu lateral)."""
    st.session_state.setdefault("tema_toggle", tema_actual() == "claro")
    st.toggle("Tema claro", key="tema_toggle", on_change=_ao_mudar_toggle, args=(username,),
              help="Fundo branco em todo o DataForge. Fica guardado na tua conta.")


def render_seletor_tema_perfil(username: str):
    """Escolha explícita Escuro / Claro (página Perfil)."""
    st.session_state.setdefault("perfil_tema", "Claro" if tema_actual() == "claro" else "Escuro")
    st.radio("Tema", ["Escuro", "Claro"], key="perfil_tema", horizontal=True,
             on_change=_ao_mudar_radio, args=(username,))


# ══════════════════════════════════════════════════════
# ESCOLHA DO IDIOMA / LANGUAGE (fica guardada na conta, como o tema)
# ══════════════════════════════════════════════════════
def definir_idioma(username, lang: str):
    from modules.i18n import definir_idioma_sessao
    definir_idioma_sessao(lang)
    if not username:
        return
    try:
        progress = get_user_progress(username)
        progress["lang"] = "en" if lang == "en" else "pt"
        _gravar_progress(username, progress)
    except Exception:
        pass          # sem base de dados a escolha vale só para esta sessão


def carregar_idioma(username: str) -> bool:
    """Ao entrar, aplica o idioma guardado na conta. True se mudou: fazer st.rerun()."""
    from modules.i18n import idioma_actual, definir_idioma_sessao
    chave = f"_idioma_lido_{username}"
    if st.session_state.get(chave):
        return False
    st.session_state[chave] = True
    try:
        guardado = get_user_progress(username).get("lang")
    except Exception:
        guardado = None
    if guardado in ("pt", "en") and guardado != idioma_actual():
        definir_idioma_sessao(guardado)
        return True
    return False


def _ao_mudar_idioma(username):
    from modules.i18n import IDIOMAS
    escolhido = st.session_state.get("_idioma_radio", "Português")
    lang = "en" if escolhido == IDIOMAS["en"] else "pt"
    definir_idioma(username, lang)


def render_seletor_idioma(username=None):
    """Escolha Português / English (menu lateral e ecrã de login)."""
    from modules.i18n import IDIOMAS, idioma_actual
    st.session_state.setdefault("_idioma_radio", IDIOMAS[idioma_actual()])
    st.radio("Idioma / Language", list(IDIOMAS.values()), key="_idioma_radio", horizontal=True,
             on_change=_ao_mudar_idioma, args=(username,))


# ══════════════════════════════════════════════════════
# PROGRESSO E HISTORIAL (Supabase ou TinyDB local, via modules/db.py)
# ══════════════════════════════════════════════════════
_CACHE_TTL = 45  # segundos: evita 1 pedido à base de dados por cada interacção


def _cache() -> dict:
    return st.session_state.setdefault("_prog_cache", {})


def get_user_progress(username: str) -> dict:
    c = _cache()
    e = c.get(username)
    if e and time.time() - e[0] < _CACHE_TTL:
        return copy.deepcopy(e[1])
    data = _db.get_progress(username)
    data["username"] = username
    data.setdefault("modulos", {})
    data.setdefault("pontos", 0)
    data.setdefault("completados", [])
    c[username] = (time.time(), data)
    return copy.deepcopy(data)


def _gravar_progress(username: str, progress: dict):
    _db.save_progress(username, progress)
    _cache()[username] = (time.time(), copy.deepcopy(progress))


def save_user_progress(username: str, modulo: str, dados: dict):
    progress = get_user_progress(username)
    progress["modulos"][modulo] = dados
    progress["ultima_actividade"] = datetime.datetime.now().isoformat()
    _gravar_progress(username, progress)


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
    progress = get_user_progress(username)
    progress["pontos"] = progress.get("pontos", 0) + pontos
    # Treinar algo numa página conta como módulo concluído (desbloqueia o percurso)
    modulo_id = PAGINA_PARA_MODULO.get(st.session_state.get("pagina", ""))
    if modulo_id:
        completados = progress.setdefault("completados", [])
        if modulo_id not in completados:
            completados.append(modulo_id)
    hist = progress.setdefault("historico_pontos", [])
    hist.append({"pontos": pontos, "motivo": motivo, "data": datetime.datetime.now().isoformat()})
    progress["historico_pontos"] = hist[-300:]
    progress["ultima_actividade"] = datetime.datetime.now().isoformat()
    _gravar_progress(username, progress)


def marcar_marco(username: str, nome: str) -> bool:
    """Regista que um marco foi atingido (ex.: 'comparador'). True se foi a 1ª vez."""
    progress = get_user_progress(username)
    marcos = progress.setdefault("marcos", {})
    if nome in marcos:
        return False
    marcos[nome] = datetime.datetime.now().isoformat()
    _gravar_progress(username, progress)
    return True


def get_licoes_concluidas(username: str) -> list:
    return get_user_progress(username).get("licoes", [])


def concluir_licao(username: str, licao_id: str, titulo: str, pontos: int = 5) -> bool:
    """Marca uma lição da Escola de ML como concluída. Devolve True se foi a 1ª vez."""
    progress = get_user_progress(username)
    licoes = progress.setdefault("licoes", [])
    if licao_id in licoes:
        return False
    licoes.append(licao_id)
    _gravar_progress(username, progress)
    add_pontos(username, pontos, f"Lição concluída: {titulo}")
    return True


def registar_quiz(username: str, licao_id: str, erradas: list):
    """Guarda as perguntas falhadas (para o professor ver e para a revisão espaçada)."""
    if not erradas:
        return
    progress = get_user_progress(username)
    erros = progress.setdefault("quiz_erros", {})
    erros[licao_id] = erros.get(licao_id, 0) + len(erradas)
    rev = progress.setdefault("revisao", {})
    amanha = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    for q in erradas:
        rev.setdefault(f"{licao_id}:{q}", {"proxima": amanha, "intervalo": 1})
    _gravar_progress(username, progress)


def save_historico_modelo(username: str, entrada: dict):
    entrada = dict(entrada)
    entrada["username"] = username
    entrada["data"] = datetime.datetime.now().isoformat()
    _db.add_historico(username, entrada)


def get_historico_modelos(username: str) -> list:
    return _db.get_historico(username)


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


def _gerar_banca(n: int = 365, seed: int = 7) -> pd.DataFrame:
    """Vendas diárias de uma banca de rua (fictício, sem internet)."""
    rng = np.random.default_rng(seed)
    dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]
    idx = np.arange(n)
    dia = np.array([dias[i % 7] for i in idx])
    dia_mes = (idx % 30) + 1
    fim_mes = (dia_mes >= 25).astype(int)
    chuva = (rng.random(n) < 0.25).astype(int)
    temp = np.round(rng.normal(29, 3, n), 1)
    preco = rng.integers(200, 401, n)
    fds = np.isin(dia, ["Sab", "Dom"]).astype(int)
    vendas = 30000 + 9000 * fds + 6000 * fim_mes - 12000 * chuva + 900 * (temp - 28) - 55 * (preco - 300) + rng.normal(0, 3500, n)
    return pd.DataFrame({"dia_semana": dia, "chuva": chuva, "fim_de_mes": fim_mes, "temperatura_c": temp,
                         "preco_do_kg_kz": preco, "vendas_kz": np.clip(vendas, 5000, None).round(-2)})


def _gerar_alunos(n: int = 500, seed: int = 11) -> pd.DataFrame:
    """Alunos fictícios e se passaram de ano (sem internet)."""
    rng = np.random.default_rng(seed)
    horas = rng.uniform(0, 25, n).round(1)
    faltas = rng.integers(0, 21, n)
    nota_ant = np.clip(rng.normal(11, 3, n), 0, 20).round(1)
    trabalha = (rng.random(n) < 0.35).astype(int)
    internet = (rng.random(n) < 0.5).astype(int)
    score = 0.22 * horas - 0.17 * faltas + 0.35 * nota_ant - 0.9 * trabalha + 0.6 * internet + rng.normal(0, 1.3, n) - 4.6
    return pd.DataFrame({"horas_estudo_semana": horas, "faltas": faltas, "nota_anterior": nota_ant,
                         "trabalha": trabalha, "tem_internet": internet,
                         "passou_de_ano": np.where(score > 0, "sim", "nao")})


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
        "Banca de rua (fictício) - Vendas":      ("banca",      "regressao"),
        "Alunos em Angola (fictício) - Aprovação": ("alunos_ao", "classificacao"),
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
    elif chave == "banca":
        df = _gerar_banca()
        desc = "365 dias de vendas de uma banca de rua fictícia (dia da semana, chuva, fim de mês, temperatura, preço). Prevê as vendas do dia em Kz."
    elif chave == "alunos_ao":
        df = _gerar_alunos()
        desc = "500 alunos fictícios (horas de estudo, faltas, nota anterior, trabalha, internet em casa). Prevê se passa de ano."
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
