"""Testes rápidos: correr com `pytest -q` na pasta do projecto."""
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGINAS = ["dashboard", "escola", "primeiro_modelo", "assistente", "dicionario", "revisao", "projeto", "conquistas",
           "certificados", "perfil", "classificacao", "regressao", "clustering", "reducao", "anomalias", "mlp",
           "reforco", "comparador", "historico", "professor"]


@pytest.fixture(autouse=True)
def pasta_temporaria(tmp_path, monkeypatch):
    """Cada teste usa uma pasta vazia, para não sujar data/db.json."""
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


def _app(pagina, user="professor"):
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=180)
    at.session_state["authentication_status"] = True
    at.session_state["username"] = user
    at.session_state["name"] = user.title()
    at.session_state["pagina"] = pagina
    return at.run()


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_abre_sem_erros(pagina):
    at = _app(pagina)
    assert not at.exception, [e.value for e in at.exception]
    assert not [e for e in at.error if "correu mal" in e.value]


def test_licoes_todas_abrem():
    from modules.escola import LICOES
    for l in LICOES:
        at = _app("escola")
        at.session_state["escola_aberta"] = l["id"]
        at.run()
        assert not at.exception, (l["id"], [e.value for e in at.exception])


def test_base_de_dados_local():
    from modules import db
    db.create_user({"username": "t1", "name": "T", "password_hash": "x"})
    db.save_progress("t1", {"pontos": 3})
    db.add_historico("t1", {"algo": "a"})
    assert db.get_user("t1")["name"] == "T"
    assert db.get_progress("t1")["pontos"] == 3
    assert db.get_historico("t1")[0]["algo"] == "a"


def test_diagnostico_detecta_problemas():
    import numpy as np
    import pandas as pd
    from modules.diagnostico import diagnosticar
    df = pd.DataFrame({"id": [f"a{i}" for i in range(30)], "x": [1.0] * 29 + [np.nan], "y": [0, 1] * 15})
    textos = " ".join(t for _, t in diagnosticar(df))
    assert "identificador" in textos and "em falta" in textos


def test_certificado_pdf():
    from modules.certificados import gerar_pdf
    pdf = gerar_pdf({"codigo": "DF-TESTE", "nome": "Ana Teste", "tipo": "fundamentos", "emitido_em": "2026-01-01"})
    assert pdf[:4] == b"%PDF"
