"""
DataForge EDU - Definições centrais
Lê configuração de st.secrets (Streamlit Cloud) ou de variáveis de ambiente.
"""
import os
import streamlit as st

PONTOS_LICAO = 8
PONTOS_LICAO_FINAL = 10
PONTOS_PRIMEIRO_MODELO = 10
PONTOS_COMPARACAO = 20
PONTOS_PALPITE = 5
PONTOS_REVISAO = 2
PONTOS_PROJETO = 40


def secret(nome: str, default=None):
    """Valor de st.secrets[nome] ou da variável de ambiente NOME (por esta ordem)."""
    try:
        if nome in st.secrets:
            return st.secrets[nome]
    except Exception:
        pass
    return os.environ.get(nome.upper(), default)


def usando_supabase() -> bool:
    return bool(secret("supabase_url") and secret("supabase_key"))


def contas_demo() -> bool:
    """Contas demo (aluno/professor/admin). Por omissão ligadas só em modo local."""
    v = secret("contas_demo")
    if v is None:
        return not usando_supabase()
    return str(v).lower() in ("1", "true", "sim", "yes")


def app_url() -> str:
    return str(secret("app_url", "") or "").rstrip("/")
