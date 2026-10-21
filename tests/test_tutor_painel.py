"""Histórico do tutor de IA (guardado na conta) e CSS minificado."""
import time

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


SCRIPT = """
import sys
sys.path.insert(0, {root!r})
import streamlit as st
from modules import tutor_ia as t
doc = t._doc_chat("ana")
acao = st.session_state.get("acao")
if acao == "gravar":
    doc["c"]["pag_x~abc123"] = {{"titulo": "Lab", "t": 1.0, "m": [["p1", "r1"], ["p2", "r2"]]}}
    t._guardar_chat("ana", doc)
elif acao == "nova":
    t._nova_conversa("ana", "pag_x~abc123")
st.text("|".join(sorted(doc["c"])))
"""


def _app():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return AppTest.from_string(SCRIPT.format(root=root), default_timeout=30)


def test_conversa_fica_guardada_e_volta_numa_sessao_nova():
    at = _app()
    at.session_state["acao"] = "gravar"
    at.run()
    assert not at.exception
    at2 = _app()                      # sessão nova: lê da base de dados
    at2.run()
    assert at2.text[0].value == "pag_x~abc123"


def test_nova_conversa_arquiva_a_antiga():
    at = _app()
    at.session_state["acao"] = "gravar"
    at.run()
    at.session_state["acao"] = "nova"
    at.run()
    assert not at.exception
    chaves = at.text[0].value.split("|")
    assert len(chaves) == 1 and chaves[0].startswith("pag_x~abc123#")


def test_limites_de_historico():
    from modules import tutor_ia as t
    assert t._MAX_TROCAS >= 10 and t._MAX_CONVERSAS >= 5


def test_css_minificado_mantem_chavetas_equilibradas():
    from modules import utils
    bruto = utils._css_global() + utils._CSS_NATIVOS + utils._CSS_ESCURO
    mini = utils._minificar_css(bruto)
    assert mini.count("{") == mini.count("}")
    assert len(mini) < len(bruto) * 0.75
    assert "@import" not in mini                       # as fontes carregam sem bloquear a página (ver _carregar_fontes)
    assert "fonts.googleapis.com" in utils._FONTES_URL


def test_css_do_expansor_aberto_nao_deixa_o_rotulo_invisivel():
    """O cabeçalho aberto ficava claro sobre claro no tema escuro (menu lateral sem rótulo)."""
    from modules import utils
    mini = utils._minificar_css(utils._CSS_NATIVOS)
    assert "details[open]>summary" in mini.replace(" ", "")
    assert "background:transparent!important" in mini.replace(" ", "")
