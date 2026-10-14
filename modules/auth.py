"""
DataForge EDU - Módulo de Autenticação
Login, registo, perfil do utilizador
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from modules.utils import (
    inject_css, C_ACCENT, C_SURFACE, C_BORDER, C_TEXT, C_TEXT_SEC,
    C_TEXT_MUTE, C_BG, C_SURFACE2, C_GREEN, C_AMBER, get_user_progress, badge
)

CONFIG_PATH = "config.yaml"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        _create_default_config()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoader)


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _create_default_config():
    import bcrypt
    def hash_pw(pw):
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    config = {
        "credentials": {"usernames": {
            "admin":     {"email": "admin@dataforge.ao",   "name": "Administrador",  "password": hash_pw("admin123"),  "role": "admin"},
            "professor": {"email": "prof@dataforge.ao",    "name": "Professor Demo", "password": hash_pw("prof123"),   "role": "professor"},
            "aluno":     {"email": "aluno@dataforge.ao",   "name": "Aluno Demo",     "password": hash_pw("aluno123"),  "role": "aluno"},
        }},
        "cookie": {"expiry_days": 7, "key": "dataforge_edu_2025", "name": "dataforge_session"},
        "preauthorized": {"emails": []}
    }
    save_config(config)


def get_authenticator():
    if "authenticator" not in st.session_state or "auth_config" not in st.session_state:
        config = load_config()
        st.session_state.auth_config = config
        st.session_state.authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"]
        )
    return st.session_state.authenticator, st.session_state.auth_config


def render_login_page():
    inject_css()

    authenticator, config = get_authenticator()

    # Estado actual antes de qualquer tentativa de (re)autenticação.
    auth_status = st.session_state.get("authentication_status")

    if auth_status is not True:
        # Ainda não autenticado: desenha o ecrã de login completo (cabeçalho,
        # formulário - que também tenta o cookie guardado - e mensagens).
        col1, col2, col3 = st.columns([1, 1.6, 1])
        with col2:
            st.markdown(f"""<div style=" display:flex; flex-direction:column; align-items:center; padding: 3rem 1rem 2rem; "><div style=" width:72px; height:72px; background: linear-gradient(135deg, {C_ACCENT}, #BC8CFF); border-radius:20px; display:flex; align-items:center; justify-content:center; font-size:22px; font-weight:900; color:#0D1117; letter-spacing:-1px; margin-bottom:1.2rem; box-shadow: 0 8px 32px rgba(88,166,255,0.3); font-family: Inter, sans-serif; ">DF</div><div style=" font-size:32px; font-weight:800; color:{C_TEXT}; letter-spacing:-.5px; margin-bottom:.4rem; ">DataForge EDU</div><div style=" font-size:15px; color:{C_TEXT_SEC}; font-weight:500; margin-bottom:2.4rem; ">Plataforma de Aprendizagem de Machine Learning</div><div style=" font-size:15px; font-weight:600; color:{C_TEXT_SEC}; margin-bottom:.8rem; text-transform:uppercase; letter-spacing:.06em; font-size:12px; ">Entra na tua conta</div></div>""", unsafe_allow_html=True)
            try:
                authenticator.login(location="main", key="login_form")
            except Exception:
                authenticator.login(location="main")

            auth_status = st.session_state.get("authentication_status")

            if auth_status is False:
                st.markdown(f"""<div style=" background:rgba(248,81,73,0.1); border:1px solid rgba(248,81,73,0.3); border-left:3px solid #F85149; border-radius:8px; padding:.75rem 1rem; color:#F85149; font-size:14px; font-weight:600; margin-top:.8rem; ">Utilizador ou palavra-passe incorrectos.</div>""", unsafe_allow_html=True)

            if auth_status is None:
                st.markdown(f"""<div style=" background:rgba(88,166,255,0.08); border:1px solid rgba(88,166,255,0.2); border-radius:8px; padding:.65rem 1rem; font-size:13px; color:{C_TEXT_SEC}; text-align:center; margin-top:1rem; ">Conta demo: <code style=" background:rgba(88,166,255,0.15); color:{C_ACCENT}; padding:2px 6px; border-radius:4px; font-family:monospace; ">aluno</code>&nbsp;/&nbsp;<code style=" background:rgba(88,166,255,0.15); color:{C_ACCENT}; padding:2px 6px; border-radius:4px; font-family:monospace; ">aluno123</code></div>""", unsafe_allow_html=True)

        # Feature cards no rodapé (fora da coluna, largura própria)
        st.markdown(f"""<div style=" display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; max-width:420px; margin:2rem auto 0; "><div style=" background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:12px; padding:.9rem .8rem; text-align:center; "><div style="margin-bottom:8px;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#58A6FF" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg></div><div style="font-size:11px; font-weight:700; color:{C_TEXT_SEC}; line-height:1.4;">Teoria<br>+ Prática</div></div><div style=" background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:12px; padding:.9rem .8rem; text-align:center; "><div style="margin-bottom:8px;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D29922" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg></div><div style="font-size:11px; font-weight:700; color:{C_TEXT_SEC}; line-height:1.4;">Certificados<br>Verificáveis</div></div><div style=" background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:12px; padding:.9rem .8rem; text-align:center; "><div style="margin-bottom:8px;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#39D0C0" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg></div><div style="font-size:11px; font-weight:700; color:{C_TEXT_SEC}; line-height:1.4;">50+<br>Algoritmos</div></div></div><div style=" text-align:center; font-size:11px; color:{C_TEXT_MUTE}; margin-top:2rem; padding-bottom:1rem; ">DataForge EDU v1.0 · Feito em Angola 🇦🇴</div>""", unsafe_allow_html=True)
    else:
        # Já autenticado: apenas confirma/renova o estado via cookie, sem
        # desenhar absolutamente nada - assim não fica HTML de login
        # "escondido" no DOM por trás da sidebar (visível se ela colapsar).
        try:
            authenticator.login(location="unrendered")
        except Exception:
            pass

    auth_status = st.session_state.get("authentication_status")
    username    = st.session_state.get("username")
    return authenticator, auth_status, username, config


def render_sidebar_user(name: str, username: str, role: str, authenticator):
    initial = name[0].upper() if name else "U"
    role_labels = {"admin": ("Admin", "red"), "professor": ("Professor", "amber"), "aluno": ("Aluno", "blue")}
    role_label, role_color = role_labels.get(role, ("Utilizador", "blue"))
    progress = get_user_progress(username)
    pontos = progress.get("pontos", 0)
    st.markdown(f"""<div class="sidebar-brand"><div class="sidebar-brand-icon" style="font-size:15px;font-weight:900;color:#0D1117;letter-spacing:-1px;font-family:Inter,sans-serif;">DF</div><div><div class="sidebar-brand-name">DataForge EDU</div><div class="sidebar-brand-sub">Machine Learning</div></div></div><div class="user-pill"><div class="user-avatar">{initial}</div><div style="flex:1;min-width:0;"><div class="user-name">{name}</div><div class="user-role">{badge(role_label, role_color)} &nbsp;⭐ {pontos} pts</div></div></div>""", unsafe_allow_html=True)
    if st.button("Sair", key="logout_btn", use_container_width=True):
        authenticator.logout(location="unrendered")
        st.rerun()


def get_user_role(username: str, config: dict) -> str:
    try:
        return config["credentials"]["usernames"][username].get("role", "aluno")
    except Exception:
        return "aluno"
