"""Testes rápidos: correr com `pytest -q` na pasta do projecto."""
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGINAS = ["dashboard", "escola", "primeiro_modelo", "assistente", "dicionario", "revisao", "projeto", "conquistas",
           "certificados", "perfil", "classificacao", "regressao", "clustering", "reducao", "anomalias", "mlp",
           "reforco", "comparador", "historico"]                      # painel do aluno
PAGINAS_PROFESSOR = ["professor", "escola", "dicionario", "perfil"]
PAGINAS_ADMIN = ["admin_home", "admin_orgs", "admin_users", "admin_conteudos", "admin_erros", "perfil"]


@pytest.fixture(autouse=True)
def pasta_temporaria(tmp_path, monkeypatch):
    """Cada teste usa uma pasta vazia, para não sujar data/db.json."""
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


def _app(pagina, user="aluno"):
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=180)
    at.session_state["authentication_status"] = True
    at.session_state["username"] = user
    at.session_state["name"] = user.title()
    at.session_state["pagina"] = pagina
    return at.run()


def _sem_erros(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not [e for e in at.error if "correu mal" in e.value]


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_abre_sem_erros(pagina):
    _sem_erros(_app(pagina, "aluno"))


@pytest.mark.parametrize("pagina", PAGINAS_PROFESSOR)
def test_painel_professor(pagina):
    at = _app(pagina, "professor")
    _sem_erros(at)
    assert at.session_state["painel"] == "professor" and at.session_state["pagina"] == pagina


@pytest.mark.parametrize("pagina", PAGINAS_ADMIN)
def test_painel_admin(pagina):
    at = _app(pagina, "admin")
    _sem_erros(at)
    assert at.session_state["painel"] == "admin" and at.session_state["pagina"] == pagina


def test_cada_papel_cai_no_seu_painel():
    for user, painel, inicio in [("admin", "admin", "admin_home"), ("professor", "professor", "professor"),
                                 ("aluno", "aluno", "dashboard")]:
        at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=180)
        at.session_state["authentication_status"] = True
        at.session_state["username"] = user
        at.session_state["name"] = user.title()
        at.run()
        _sem_erros(at)
        assert at.session_state["painel"] == painel and at.session_state["pagina"] == inicio


def test_aluno_nao_acede_a_paginas_de_outros_paineis():
    for pagina in ("admin_users", "admin_orgs", "professor", "business_home"):
        at = _app(pagina, "aluno")
        _sem_erros(at)
        assert at.session_state["painel"] == "aluno" and at.session_state["pagina"] == "dashboard", pagina


def test_professor_nao_acede_ao_admin_nem_ao_menu_do_aluno():
    for pagina in ("admin_users", "admin_erros", "classificacao"):
        at = _app(pagina, "professor")
        _sem_erros(at)
        assert at.session_state["pagina"] == "professor", pagina


def test_membro_de_empresa_ganha_painel_business():
    from modules import db
    db.create_org("e1", "Empresa Teste", "empresa", "ABC123")
    db.create_user({"username": "ana", "name": "Ana Silva", "password_hash": "x"})
    db.add_membro("e1", "ana", "analista")
    at = _app("business_home", "ana")
    _sem_erros(at)
    assert at.session_state["painel"] == "business" and at.session_state["pagina"] == "business_home"


def test_organizacoes_e_membros():
    import pytest as _pt
    from modules import db
    from modules.admin import adicionar_membro_por_codigo
    db.create_org("s1", "Escola A", "escola", "AAA111")
    db.create_org("e1", "Empresa B", "empresa", "BBB222")
    db.create_user({"username": "u1", "name": "U Um", "password_hash": "x"})
    assert adicionar_membro_por_codigo("u1", "aaa111")[0]                        # código não distingue maiúsculas
    assert [m["papel"] for m in db.list_membros(username="u1")] == ["aluno"]
    assert adicionar_membro_por_codigo("u1", "BBB222")[0]
    assert {m["tipo"] for m in db.membros_do_utilizador("u1")} == {"escola", "empresa"}
    assert not adicionar_membro_por_codigo("u1", "ZZZ999")[0]
    with _pt.raises(db.DBError):
        db.add_membro("s1", "u1", "analista")                                    # papel de empresa numa escola
    db.update_org("e1", {"activa": False})
    assert {m["tipo"] for m in db.membros_do_utilizador("u1")} == {"escola"}      # org desactivada perde acesso
    db.remove_membro("s1", "u1")
    assert db.list_membros(username="u1")[0]["org_id"] == "e1"


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
