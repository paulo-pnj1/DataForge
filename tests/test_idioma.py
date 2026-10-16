"""Testes do idioma (Português / English): correr com `pytest -q tests/test_idioma.py`."""
import os
import re
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(autouse=True)
def pasta_temporaria(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


def _app(pagina, lang="en", user="aluno"):
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=180)
    at.session_state["authentication_status"] = True
    at.session_state["username"] = user
    at.session_state["name"] = user.title()
    at.session_state["pagina"] = pagina
    at.session_state["lang"] = lang
    return at.run()


def _textos(at):
    out = []
    for grupo in (at.markdown, at.caption, at.info, at.success, at.warning, at.error):
        out += [e.value for e in grupo]
    out += [b.label or "" for b in at.button]
    return " ".join(re.sub(r"<[^>]+>", " ", t) for t in out)


# ── tradução de frases ────────────────────────────────────────────────────────
def test_frases_com_partes_dinamicas():
    from modules.i18n import tr_en
    assert tr_en("Fundamentos de Machine Learning") == "Machine Learning Fundamentals"
    assert tr_en("Emitido em 2026-10-16") == "Issued on 2026-10-16"
    assert tr_en("Concluir todas as lições da Escola de ML (0/14)") == "Complete all the ML School lessons (0/14)"
    r = tr_en("A coluna **idade** tem mais de metade dos valores em falta. Provavelmente é melhor não a usar.")
    assert "**idade**" in r and "has more than half" in r


def test_html_nao_e_alterado():
    from modules.i18n import tr_en
    r = tr_en('<div style="color:red;">Treinar Agente</div>')
    assert r == '<div style="color:red;">Train Agent</div>'


def test_texto_sem_traducao_fica_igual():
    from modules.i18n import tr_en
    assert tr_en("xyzzy qwerty") == "xyzzy qwerty"
    assert tr_en("") == ""


def test_ficheiro_de_traducoes_sem_vazios():
    import json
    d = json.load(open(os.path.join(RAIZ, "modules", "locales", "en.json"), encoding="utf8"))
    assert len(d["t"]) > 2000
    assert not [k for k, v in d["t"].items() if not v.strip()]


# ── a app inteira ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pagina", ["dashboard", "escola", "comparador", "reforco", "dicionario", "perfil"])
def test_paginas_em_ingles_sem_erros(pagina):
    at = _app(pagina)
    assert not at.exception, [e.value for e in at.exception]


def test_painel_em_ingles_e_em_portugues():
    en = _textos(_app("dashboard", "en"))
    pt = _textos(_app("dashboard", "pt"))
    assert "Welcome back" in en and "Bem-vindo de volta" not in en
    assert "Bem-vindo de volta" in pt


def test_seletor_de_idioma_muda_a_sessao():
    at = _app("dashboard", "pt")
    assert at.session_state["lang"] == "pt"
    at.radio(key="_idioma_radio").set_value("English").run()
    assert at.session_state["lang"] == "en"
    assert "Welcome back" in _textos(at)
    at.radio(key="_idioma_radio").set_value("Português").run()
    assert at.session_state["lang"] == "pt"


def test_valores_dos_widgets_continuam_em_portugues():
    """As opções são traduzidas só na apresentação: o valor devolvido não muda."""
    at = _app("perfil", "en")
    rs = [r for r in at.radio if r.key == "perfil_tema"]
    assert rs and rs[0].value in ("Escuro", "Claro")


# ── ortografia ────────────────────────────────────────────────────────────────
ERROS = ["accoes", "ultimas", "experiencias", "aleatorios", "atraves", "visualizacao", "customizavel",
         "Relatorio", "Comparacao", "fiavel", "Obstaculos", "Exploracao", "Hiperparametros", "previssemos"]


@pytest.mark.parametrize("ficheiro", ["modules/reforco/dqn.py", "modules/reforco/q_learning.py",
                                      "modules/comparador.py", "modules/projeto.py",
                                      "modules/certificados.py", "modules/primeiro_modelo.py"])
def test_sem_erros_ortograficos_conhecidos(ficheiro):
    src = open(os.path.join(RAIZ, ficheiro), encoding="utf8").read()
    achados = [e for e in ERROS if re.search(rf"\b{e}\b", src)]
    assert not achados, achados
