"""
DataForge EDU - Entry Point
Plataforma de Aprendizagem de Machine Learning para Universidades
"""

import streamlit as st
import os, sys, time

_T0 = time.perf_counter()

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="DataForge",
    page_icon="https://api.iconify.design/ph/brain-duotone.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.i18n import iniciar_idioma, idioma_actual
from modules.i18n_patch import instalar as _instalar_i18n
_instalar_i18n()      # texto, rótulos, tabelas e gráficos passam pela tradução (Português / English)
iniciar_idioma()

from modules.utils import inject_css, C_TEXT_MUTE, C_BORDER, C_SURFACE, C_ACCENT, C_TEXT_SEC, C_TEXT, C_BG, C_SURFACE2, \
    instalar_tema_graficos, carregar_tema, carregar_idioma
from modules.auth import render_login_page, render_sidebar_user, get_user_role

inject_css()
instalar_tema_graficos()      # gráficos seguem o tema escuro/claro

# ── PÁGINA PÚBLICA DE VERIFICAÇÃO DE CERTIFICADOS (?verificar=CODIGO) ──
if "verificar" in st.query_params:
    from modules.certificados import render_verificacao
    render_verificacao(st.query_params["verificar"])
    st.stop()

# ── ÍCONES SVG PROFISSIONAIS (Phosphor Icons) ────────
# Inline SVG - sem dependência externa, carrega instantaneamente
def icon(name: str, size: int = 18, color: str = "currentColor") -> str:
    icons = {
        "home":        '<path d="M218.83 103.77l-80-75.48a1.94 1.94 0 0 0-.06-.06 16 16 0 0 0-21.54 0l-.06.06-80 75.48A16 16 0 0 0 32 115.55V208a16 16 0 0 0 16 16H96a16 16 0 0 0 16-16V160h32v48a16 16 0 0 0 16 16h48a16 16 0 0 0 16-16V115.55a16 16 0 0 0-5.17-11.78z"/>',
        "chart-bar":   '<path d="M224 200h-8V40a8 8 0 0 0-8-8H152a8 8 0 0 0-8 8v56H104a8 8 0 0 0-8 8v40H64a8 8 0 0 0-8 8v48H40a8 8 0 0 0 0 16h184a8 8 0 0 0 0-16z"/>',
        "trend-up":    '<path d="M240 56a8 8 0 0 1-8 8h-51.06L155.7 98.35A8 8 0 0 1 152 101.4l-.1.09L104 138.84 50.35 97.76a8 8 0 1 1 9.3-13l48.35 36.29 46.83-35.78 1.68-1.57A8 8 0 0 1 160 82h72a8 8 0 0 1 8 8z" opacity=".2"/><path d="M240 56a8 8 0 0 1-8 8h-51.06L155.7 98.35A8 8 0 0 1 152 101.4l-.1.09L104 138.84 50.35 97.76a8 8 0 1 1 9.3-13l48.35 36.29 46.83-35.78.05-.05 1.62-1.54A8 8 0 0 1 160 82h72a8 8 0 0 1 8 8z"/>',
        "circles":     '<circle cx="128" cy="128" r="96" fill="none" stroke-width="16"/><circle cx="128" cy="128" r="48" fill="none" stroke-width="16"/>',
        "warning":     '<path d="M236.8 188.09 149.35 36.22a24.76 24.76 0 0 0-42.7 0L19.2 188.09a23.51 23.51 0 0 0 0 23.72A24.35 24.35 0 0 0 40.55 224h174.9a24.35 24.35 0 0 0 21.33-12.19 23.51 23.51 0 0 0 .02-23.72zM120 104a8 8 0 0 1 16 0v40a8 8 0 0 1-16 0zm8 88a12 12 0 1 1 12-12 12 12 0 0 1-12 12z"/>',
        "brain":       '<path d="M248 124a56.11 56.11 0 0 0-32-50.61V72a48 48 0 0 0-88-26.49A48 48 0 0 0 40 72v1.39A56 56 0 0 0 64 180.6V208a16 16 0 0 0 16 16h96a16 16 0 0 0 16-16v-27.4A56.09 56.09 0 0 0 248 124z" opacity=".2"/><path d="M248 124a56.11 56.11 0 0 0-32-50.61V72a48 48 0 0 0-88-26.49A48 48 0 0 0 40 72v1.39A56 56 0 0 0 64 180.6V208a16 16 0 0 0 16 16h96a16 16 0 0 0 16-16v-27.4A56.09 56.09 0 0 0 248 124z" fill="none" stroke-width="16"/>',
        "cpu":         '<rect x="80" y="80" width="96" height="96" rx="8" fill="none" stroke-width="16"/><path d="M160 80V48m-32 32V48m-32 32V48m0 160v32m32-32v32m32-32v32m48-80h32m-32 32h32m-32-64h32M48 160H16m32-32H16m32-64H16" fill="none" stroke-width="16"/>',
        "network":     '<circle cx="200" cy="56" r="24" fill="none" stroke-width="16"/><circle cx="56" cy="128" r="24" fill="none" stroke-width="16"/><circle cx="200" cy="200" r="24" fill="none" stroke-width="16"/><path d="M80 128h96m-76.26-17.74L176 79.55M79.74 145.74 176 176.45" fill="none" stroke-width="16"/>',
        "robot":       '<rect x="88" y="24" width="80" height="56" rx="16" fill="none" stroke-width="16"/><path d="M160 80h16a32 32 0 0 1 32 32v16H48V112A32 32 0 0 1 80 80h8" fill="none" stroke-width="16"/><path d="M48 128v72a8 8 0 0 0 8 8h144a8 8 0 0 0 8-8V128" fill="none" stroke-width="16"/><circle cx="104" cy="152" r="16"/><circle cx="152" cy="152" r="16"/>',
        "clock":       '<circle cx="128" cy="128" r="96" fill="none" stroke-width="16"/><polyline points="128 72 128 128 184 128" fill="none" stroke-width="16"/>',
        "sparkle":     '<path d="M208 144h-79.06l55.86-55.87A8 8 0 0 0 173.49 77l-96 40a8 8 0 0 0 0 14.78L152 159.06V208a8 8 0 0 0 14.78 4L211 116.52A8 8 0 0 0 208 104z" fill="none" stroke-width="16"/>',
        "trophy":      '<path d="M96 208h64m-32-48v48m-80-48h160a16 16 0 0 0 16-16V64H32v80a16 16 0 0 0 16 16zm-32-96h-16a16 16 0 0 0-16 16v8a40 40 0 0 0 40 40h8m160-64h16a16 16 0 0 1 16 16v8a40 40 0 0 1-40 40h-8" fill="none" stroke-width="16"/>',
        "compare":     '<path d="M128 48v160M88 88 48 128l40 40m80-80 40 40-40 40" fill="none" stroke-width="16"/>',
        "history":     '<path d="M40 128a88 88 0 1 0 88-88 88.1 88.1 0 0 0-88 88z" fill="none" stroke-width="16"/><polyline points="128 88 128 128 160 160" fill="none" stroke-width="16"/>',
        "book":        '<path d="M48 80h112v160H48a16 16 0 0 1-16-16V96a16 16 0 0 1 16-16zm112 0h48a16 16 0 0 1 16 16v128a16 16 0 0 1-16 16H160m-112-96h112m-112 40h112" fill="none" stroke-width="16"/>',
        "eye":         '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" fill="none" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="none" stroke-width="2"/>',
        "text":        '<path d="M48 48h160m-80 0v160M80 208h96" fill="none" stroke-width="16"/>',
        "game":        '<rect x="32" y="80" width="192" height="112" rx="32" fill="none" stroke-width="16"/><path d="M96 128h64m-32-32v64m56-32h.01M64 128h.01" fill="none" stroke-width="16"/>',
        "logout":      '<path d="M112 40H48a8 8 0 0 0-8 8v160a8 8 0 0 0 8 8h64m32-56 40-40-40-40m-104 40h144" fill="none" stroke-width="16"/>',
        "menu":        '<path d="M40 128h176M40 64h176M40 192h176" fill="none" stroke-width="16"/>',
        "down":        '<polyline points="208 96 128 176 48 96" fill="none" stroke-width="16"/>',
        "school":      '<path d="M16 96l112-56 112 56-112 56zM64 120v56c0 16 28 32 64 32s64-16 64-32v-56M240 96v64" fill="none" stroke-width="16"/>',
        "rocket":      '<path d="M96 160L64 192m32-32c-16-48 0-96 64-128 32 64 16 112-32 128zM96 160l-24-8-24 24 24 8M136 200l8 24 24-24-8-24" fill="none" stroke-width="16"/><circle cx="148" cy="92" r="10"/>',
        "help":        '<circle cx="128" cy="128" r="96" fill="none" stroke-width="16"/><path d="M100 104a28 28 0 1 1 40 25c-8 4-12 10-12 19m0 28h.01" fill="none" stroke-width="16"/>',
        "compass2":    '<circle cx="128" cy="128" r="96" fill="none" stroke-width="16"/><path d="M168 88l-24 56-56 24 24-56z" fill="none" stroke-width="16"/>',
    }
    path = icons.get(name, icons["menu"])
    stroke = f'stroke="{color}" stroke-linecap="round" stroke-linejoin="round"'
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 256 256" fill="{color}" {stroke} style="vertical-align:middle;flex-shrink:0;">{path}</svg>'


# ── PÁGINAS AUXILIARES ────────────────────────────────
def _render_em_breve(titulo: str, descricao: str, icone_svg: str = ""):
    from modules.utils import page_header
    st.markdown(f"""<div style="background:{C_SURFACE};border:2px solid {C_BORDER};border-radius:20px; padding:3.5rem 2rem;text-align:center;margin-top:2rem;"><div style="width:64px;height:64px;background:rgba(107,163,255,.15);border:2px solid {C_ACCENT}; border-radius:16px;display:flex;align-items:center;justify-content:center; margin:0 auto 1.4rem;">{icone_svg}</div><div style="font-size:22px;font-weight:800;color:var(--df-text,#FFFFFF);margin-bottom:.6rem;">{titulo}</div><div style="font-size:16px;color:var(--df-text2,#B8C4D8);margin-bottom:1.2rem;">{descricao}</div><div style="display:inline-block;background:rgba(107,163,255,.12);border:1px solid rgba(107,163,255,.3); border-radius:8px;padding:.4rem 1rem;font-size:13px;font-weight:700;color:{C_ACCENT};">Em desenvolvimento - Fase 2</div></div>""", unsafe_allow_html=True)


def _render_historico(username: str):
    from modules.utils import page_header, get_historico_modelos
    page_header("Historial de Modelos", "Todos os modelos que treinaste", "")
    historico = get_historico_modelos(username)
    if not historico:
        st.markdown(f"""<div style="text-align:center;padding:3rem;color:var(--df-text2,#B8C4D8);"><div style="font-size:48px;margin-bottom:1rem;">📂</div>Ainda sem modelos treinados. Começa por <b>O Meu Primeiro Modelo</b> no menu.</div>""", unsafe_allow_html=True)
        return
    for h in reversed(historico[-20:]):
        tipo  = h.get("tipo", "")
        algo  = h.get("algo", h.get("algoritmo", ""))
        data  = h.get("data", "")[:16].replace("T", " ")
        metricas = ""
        if tipo == "classificacao" and h.get("accuracy"):
            metricas = f"Accuracy: {h.get('accuracy',0):.1%} · F1: {h.get('f1',0):.1%}"
        elif tipo == "regressao" and h.get("r2"):
            metricas = f"R²: {h.get('r2',0):.4f} · RMSE: {h.get('rmse',0):.4f}"
        tipo_icon = icon("chart-bar", 16, "#6BA3FF") if tipo == "classificacao" else icon("trend-up", 16, "#4ADE80")
        st.markdown(f"""<div style="background:{C_SURFACE};border:2px solid {C_BORDER};border-radius:12px; padding:.9rem 1.2rem;margin-bottom:.7rem;display:flex;align-items:center;gap:14px;">{tipo_icon}<div style="flex:1;"><div style="font-size:15px;font-weight:700;color:var(--df-text,#FFFFFF);">{algo}</div><div style="font-size:13px;color:var(--df-text3,#7A8BA8);">{tipo.capitalize()} · {data}</div><div style="font-size:13px;color:var(--df-text2,#B8C4D8);margin-top:2px;">{metricas}</div></div></div>""", unsafe_allow_html=True)


# ── AUTH ─────────────────────────────────────────────
authenticator, auth_status, username, config = render_login_page()
_T_LOGIN = time.perf_counter()
if auth_status is not True:
    st.stop()

name = st.session_state.get("name", username)
role = get_user_role(username, config)

# Tema guardado na conta (claro por omissão): se for diferente do actual, volta a desenhar já com ele
if carregar_tema(username) | carregar_idioma(username):
    st.rerun()

# Aluno criado pelo professor: tem de trocar a palavra-passe temporária antes de usar a plataforma
if st.session_state.get("_pw_ok") != username:
    try:
        from modules import db as _db0
        _precisa_pw = bool((_db0.get_user(username) or {}).get("must_change_pw"))
    except Exception:
        _precisa_pw = False
    if _precisa_pw:
        from modules.auth import render_mudar_pw_obrigatoria
        render_mudar_pw_obrigatoria(username)
        st.stop()
    st.session_state["_pw_ok"] = username

_ZOOM = {"Pequeno": 90, "Normal": 100, "Grande": 115, "Muito grande": 130}
_z = _ZOOM.get(st.session_state.get("tamanho_texto", "Normal"), 100)
if _z != 100:
    st.markdown(f"<style>html {{ font-size: {_z}% !important; }}</style>", unsafe_allow_html=True)

# ── PAINEL (cada pessoa vê só o menu do seu papel) ────
from modules import db as _dbm
from modules.navegacao import (PAINEIS, PRODUTO_DO_PAINEL, INICIO, paineis_disponiveis, menu_do_painel,
                               paginas_permitidas, resolver_painel)


def _membros_do(user: str) -> list:
    """Pertenças a escolas/empresas (guardadas na sessão; renovam-se ao voltar a entrar)."""
    cache = st.session_state.get("_membros_cache")
    if not cache or cache[0] != user:
        try:
            lista = _dbm.membros_do_utilizador(user)
        except Exception:
            lista = []
        cache = (user, lista)
        st.session_state["_membros_cache"] = cache
    return cache[1]


_tem_turma = bool(config["credentials"]["usernames"].get(username, {}).get("turma"))
_disponiveis = paineis_disponiveis(role, _membros_do(username), _tem_turma)
_empresa_pendente = None
if role != "admin" and "business" not in _disponiveis:
    try:
        from modules import acesso as _acesso
        _empresa_pendente = _acesso.empresa_do_dono(username)
    except Exception:
        _empresa_pendente = None
if _empresa_pendente:      # dono de uma empresa ainda não aprovada: só vê o estado do pedido
    with st.sidebar:
        render_sidebar_user(name, username, role, authenticator, produto="DataForge Business",
                            papel_rotulo="Empresa", mostrar_pontos=False, subtitulo="Machine Learning para empresas")
    from modules.admin import render_empresa_pendente
    render_empresa_pendente(username, _empresa_pendente)
    st.stop()

if not _disponiveis:       # conta de empresa que já não pertence a nenhuma empresa activa
    with st.sidebar:
        render_sidebar_user(name, username, role, authenticator, produto="DataForge Business",
                            papel_rotulo="Empresa", mostrar_pontos=False, subtitulo="Machine Learning para empresas")
    from modules.business import render_sem_empresa
    render_sem_empresa(username)
    st.stop()

# Ao actualizar o navegador (F5) a sessão do Streamlit recomeça: o painel e a página ficam no endereço (?painel=…&p=…)
if "painel" not in st.session_state and st.query_params.get("painel"):
    st.session_state["painel"] = st.query_params.get("painel")
painel = resolver_painel(st.session_state.get("painel"), _disponiveis)
st.session_state["painel"] = painel

if "pagina" not in st.session_state:
    _p_url = st.query_params.get("p")
    if _p_url and _p_url in paginas_permitidas(painel):
        st.session_state.pagina = _p_url
if "pagina" not in st.session_state or st.session_state.pagina not in paginas_permitidas(painel):
    st.session_state.pagina = INICIO[painel]
if "escola_aberta" not in st.session_state and st.session_state.pagina == "escola" and st.query_params.get("licao"):
    st.session_state.escola_aberta = st.query_params.get("licao")

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    if painel == "business":
        from modules.business import papel_rotulo as _papel_emp, render_seletor_empresa
        render_sidebar_user(name, username, role, authenticator, produto=PRODUTO_DO_PAINEL[painel],
                            papel_rotulo=_papel_emp(username) or "Empresa", mostrar_pontos=False,
                            subtitulo="Machine Learning para empresas")
    else:
        render_sidebar_user(name, username, role, authenticator, produto=PRODUTO_DO_PAINEL[painel])

    if len(_disponiveis) > 1:
        escolhido = st.selectbox("Painel", _disponiveis, index=_disponiveis.index(painel),
                                 format_func=PAINEIS.get, key="painel_sel")
        if escolhido != painel:
            st.session_state["painel"] = escolhido
            st.session_state.pagina = INICIO[escolhido]
            st.rerun()

    if painel == "business":
        render_seletor_empresa(username)

    rotulo_rev = "Revisão do dia"
    if painel == "aluno":
        # Revisões pendentes (mostra o número no menu)
        try:
            from modules.revisao import pendentes
            from modules.utils import get_user_progress
            n_rev = len(pendentes(get_user_progress(username)))
        except Exception:
            n_rev = 0
        rotulo_rev = f"Revisão do dia ({n_rev})" if n_rev else "Revisão do dia"

    MENU = [(label, key, seccao, icon(ic, 16, cor))
            for label, key, seccao, ic, cor in menu_do_painel(painel, rotulo_rev)]

    from modules.pesquisa import render_pesquisa_sidebar
    render_pesquisa_sidebar([(l, k) for l, k, _, _ in MENU], so_paginas=(painel == "business"))

    from modules.navegacao import AJUDA_PAGINA, MODULOS_INFO

    def _botao_menu(label, key):
        if st.button(f"  {label}", key=f"nav_{key}", width='stretch', help=AJUDA_PAGINA.get(key) if painel == "aluno" else None,
                     type="primary" if st.session_state.pagina == key else "secondary"):
            st.session_state.pagina = key
            if "guiado_step" in st.session_state:
                st.session_state.guiado_step = 1
            st.rerun()

    from modules.navegacao import agrupar_menu
    if painel == "aluno":
        # Menu em módulos recolhíveis: abre o que contém a página actual e mostra quantas páginas do módulo já treinaste
        try:
            from modules.utils import PAGINA_PARA_MODULO, get_user_progress
            _feitos = set(get_user_progress(username).get("completados", []))
        except Exception:
            PAGINA_PARA_MODULO, _feitos = {}, set()
        _todas = [k for _, k, _, _ in MENU if k in PAGINA_PARA_MODULO]
        _feitas = sum(1 for k in _todas if PAGINA_PARA_MODULO[k] in _feitos)
        if _todas:
            st.progress(_feitas / len(_todas), text=f"O teu percurso: {_feitas} de {len(_todas)} práticas feitas" if idioma_actual() != "en" else f"Your path: {_feitas} of {len(_todas)} practices done")
        for g_idx, (seccao, itens) in enumerate(agrupar_menu(MENU)):
            contem_activa = any(k == st.session_state.pagina for _, k, _, _ in itens)
            com_modulo = [k for _, k, _, _ in itens if k in PAGINA_PARA_MODULO]
            progresso = ""
            if com_modulo:
                progresso = f"  ({sum(1 for k in com_modulo if PAGINA_PARA_MODULO[k] in _feitos)}/{len(com_modulo)})"
            emoji, descricao = MODULOS_INFO.get(seccao, ("", ""))
            with st.expander(f"{emoji} {seccao}{progresso}".strip(), expanded=contem_activa or g_idx == 0):
                if descricao:
                    st.caption(descricao)
                for label, key, _, _ in itens:
                    _botao_menu(label, key)
    else:
        ultima_seccao = "__start__"
        for label, key, seccao, icone_svg in MENU:
            if seccao and seccao != ultima_seccao:
                ultima_seccao = seccao
                st.markdown(f"""<div style="font-size:10px;font-weight:800;letter-spacing:.14em; color:var(--df-text3,#7A8BA8);padding:14px 0 4px;text-transform:uppercase; border-top:1px solid {C_BORDER};margin-top:4px;">{seccao}</div>""", unsafe_allow_html=True)
            _botao_menu(label, key)

    st.markdown("---")
    st.markdown(f'<div style="font-size:11px;font-weight:600;color:var(--df-text3,#7A8BA8);text-align:center;padding:4px;">{PRODUTO_DO_PAINEL[painel]} v2.0 · Angola 🇦🇴</div>', unsafe_allow_html=True)


# ── ROTEADOR ─────────────────────────────────────────
BUSINESS_PAGINAS = {"business_home": "render_inicio", "business_dados": "render_dados",
                    "business_analises": "render_analises", "business_previsoes": "render_previsoes",
                    "business_relatorios": "render_relatorios", "business_equipa": "render_equipa"}
TUTOR_PAGINAS = {"primeiro_modelo", "classificacao", "regressao", "clustering", "reducao", "anomalias", "mlp", "visao",
                 "nlp", "series", "generativo", "reforco", "comparador", "projeto"}
pagina = st.session_state.get("pagina", INICIO[painel])

# Guarda onde estás no endereço, para o refresh te deixar na mesma página (e na mesma lição)
_alvo_url = {"painel": painel, "p": pagina}
if pagina == "escola" and st.session_state.get("escola_aberta"):
    _alvo_url["licao"] = str(st.session_state.escola_aberta)
for _k, _v in _alvo_url.items():
    if st.query_params.get(_k) != _v:
        st.query_params[_k] = _v
if pagina != "escola" and "licao" in st.query_params:
    del st.query_params["licao"]

st.session_state.pop("_ia_pagina", None)      # cada página regista de novo o que o aluno está a ver (painel da IA)
try:

    if pagina == "dashboard":
        from modules.dashboard import render_dashboard
        render_dashboard(name, username)

    elif pagina == "escola":
        from modules.escola import render_escola
        render_escola(username)

    elif pagina == "primeiro_modelo":
        from modules.primeiro_modelo import render_primeiro_modelo
        render_primeiro_modelo(username)

    elif pagina == "assistente":
        from modules.ajuda import render_assistente
        render_assistente(username)

    elif pagina == "dicionario":
        from modules.ajuda import render_dicionario
        render_dicionario(username)

    elif pagina == "classificacao":
        from modules.supervisionado.classificacao import render_classificacao
        render_classificacao(username)

    elif pagina == "regressao":
        from modules.supervisionado.regressao import render_regressao
        render_regressao(username)

    elif pagina == "clustering":
        from modules.nao_supervisionado.clustering import render_clustering
        render_clustering(username)

    elif pagina == "reducao":
        from modules.nao_supervisionado.reducao import render_reducao
        render_reducao(username)

    elif pagina == "anomalias":
        from modules.nao_supervisionado.anomalias import render_anomalias
        render_anomalias(username)

    elif pagina == "mlp":
        from modules.redes_neurais.mlp import render_mlp
        render_mlp(username)

    elif pagina == "visao":
        from modules.deep_learning.visao import render_visao
        render_visao(username)

    elif pagina == "nlp":
        from modules.deep_learning.nlp import render_nlp
        render_nlp(username)

    elif pagina == "series":
        from modules.deep_learning.series_temporais import render_series_temporais
        render_series_temporais(username)

    elif pagina == "generativo":
        from modules.deep_learning.generativo import render_generativo
        render_generativo(username)

    elif pagina == "reforco":
        tab_q, tab_dqn = st.tabs(["Q-Learning (GridWorld)", "DQN (CartPole)"])
        with tab_q:
            from modules.reforco.q_learning import render_q_learning
            render_q_learning(username)
        with tab_dqn:
            try:
                from modules.reforco.dqn import render_dqn
                render_dqn(username)
            except ImportError:
                st.info("O DQN precisa do PyTorch. Instala com: pip install torch")

    elif pagina == "comparador":
        from modules.comparador import render_comparador
        render_comparador(username)

    elif pagina == "laboratorio":
        from modules.laboratorio import render_laboratorio
        render_laboratorio(username)

    elif pagina == "historico":
        _render_historico(username)

    elif pagina == "revisao":
        from modules.revisao import render_revisao
        render_revisao(username)

    elif pagina == "projeto":
        from modules.projeto import render_projeto
        render_projeto(username, name)

    elif pagina == "conquistas":
        from modules.gamificacao import render_conquistas
        render_conquistas(username)

    elif pagina == "certificados":
        from modules.certificados import render_certificados
        render_certificados(username, name)

    elif pagina == "perfil" and painel == "business" and role != "admin":
        from modules.business import render_perfil as render_perfil_business
        render_perfil_business(username, name)

    elif pagina == "perfil":
        from modules.auth import render_perfil
        render_perfil(username, name, role)

    elif pagina == "professor" and role == "professor" and painel == "professor":
        from modules.professor import render_painel_professor
        render_painel_professor(username, role)

    elif pagina == "admin_users" and role == "admin":
        from modules.professor import render_admin_utilizadores
        render_admin_utilizadores(username)

    elif pagina == "admin_home" and role == "admin":
        from modules.admin import render_admin_home
        render_admin_home(username)

    elif pagina == "admin_orgs" and role == "admin":
        from modules.admin import render_admin_orgs
        render_admin_orgs(username)

    elif pagina == "admin_convites" and role == "admin":
        from modules.admin import render_admin_convites
        render_admin_convites(username)

    elif pagina == "admin_email" and role == "admin":
        from modules.admin import render_admin_email
        render_admin_email(username)

    elif pagina == "admin_conteudos" and role == "admin":
        from modules.admin import render_admin_conteudos
        render_admin_conteudos(username)

    elif pagina == "admin_erros" and role == "admin":
        from modules.admin import render_admin_erros
        render_admin_erros(username)

    elif pagina in BUSINESS_PAGINAS and painel == "business" and role != "admin":
        from modules import business as _bn
        from modules.db import TabelaEmFalta
        try:
            getattr(_bn, BUSINESS_PAGINAS[pagina])(username, name)
        except TabelaEmFalta:
            st.warning("A base de dados ainda não tem as tabelas do painel Business. "
                       "No Supabase, abre o SQL Editor, cola o SQL abaixo e carrega em Run. Depois recarrega esta página.")
            try:
                st.code(open("schema_business.sql", encoding="utf8").read(), language="sql")
            except OSError:
                st.info("Corre o ficheiro schema.sql no SQL Editor do Supabase.")

    else:  # página desconhecida: volta ao início do painel actual
        st.session_state.pagina = INICIO[painel]
        st.rerun()

    # Tutor de IA depois de cada treino (só no painel do aluno; só aparece se houver um resultado para explicar)
    if painel == "aluno" and pagina in TUTOR_PAGINAS:
        from modules.tutor_ia import SUGESTOES_TREINO, _contexto_da_pagina, registar_painel
        _c = _contexto_da_pagina(pagina)
        if _c:
            registar_painel(f"pag_{pagina}", f"{next((l for l, k, _, _ in MENU if k == pagina), pagina)}: {_c[0]}", _c[1],
                            SUGESTOES_TREINO, padrao_primeiro=True)
    if painel == "aluno":
        from modules.tutor_ia import render_painel_ia
        render_painel_ia(username, pagina, next((l for l, k, _, _ in MENU if k == pagina), pagina))
except Exception as _e:  # nunca mostrar uma página partida ao aluno
    import traceback
    from modules import db as _db
    _db.log_erro(username, pagina, traceback.format_exc())
    st.error("Algo correu mal nesta página. O erro foi registado e não perdeste o teu progresso. "
             "Tenta voltar ao Início ou recarregar a página.")
    with st.expander("Detalhes técnicos (para o professor ou programador)"):
        st.code(traceback.format_exc())
    if st.button("Voltar ao Início", key="err_home"):
        st.session_state.pagina = INICIO.get(painel, "dashboard")
        st.rerun()

# Tempo do servidor nesta execução (aparece nos logs: "Gerenciar aplicativo"). Se for baixo e a página
# parecer lenta, a demora está na rede ou no navegador, não no código.
print(f"[tempo] pagina={pagina} login={_T_LOGIN - _T0:.2f}s total={time.perf_counter() - _T0:.2f}s", flush=True)
