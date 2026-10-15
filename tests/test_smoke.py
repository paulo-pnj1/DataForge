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
PAGINAS_ADMIN = ["admin_home", "admin_orgs", "admin_convites", "admin_users", "admin_email", "admin_conteudos", "admin_erros", "perfil"]


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


def test_admin_so_tem_o_painel_de_administracao():
    from modules.navegacao import paineis_disponiveis, paginas_permitidas
    assert paineis_disponiveis("admin", [], False) == ["admin"]
    assert paineis_disponiveis("admin", [{"tipo": "empresa"}], True) == ["admin"]
    proibidas = {"professor", "business_home", "dashboard", "escola", "primeiro_modelo", "classificacao",
                 "regressao", "projeto", "certificados", "conquistas", "revisao", "historico", "dicionario"}
    assert not (paginas_permitidas("admin") & proibidas)


def test_admin_nao_entra_em_paginas_de_outros_paineis():
    for pagina in ("professor", "business_home", "dashboard", "escola", "classificacao"):
        at = _app(pagina, "admin")
        _sem_erros(at)
        assert at.session_state["painel"] == "admin" and at.session_state["pagina"] == "admin_home", pagina


def test_admin_nao_cria_turmas_nem_organizacoes_nem_entra_nelas():
    from modules import db
    from modules.admin import adicionar_membro_por_codigo
    db.create_org("s1", "Escola A", "escola", "AAA111")
    db.create_user({"username": "adm", "name": "Adm Teste", "password_hash": "x", "role": "admin"})
    assert not adicionar_membro_por_codigo("adm", "AAA111")[0]
    assert db.list_membros(username="adm") == []
    at = _app("admin_orgs", "admin")
    _sem_erros(at)
    assert not [b for b in at.button if "Criar" in (b.label or "") or "Adicionar" in (b.label or "")]
    assert not at.text_input                                   # sem formulário de criação


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


def test_formulario_criar_conta_abre_para_cada_papel():
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=180).run()
    _sem_erros(at)
    for rotulo in ("Sou aluno", "Sou professor", "Sou uma empresa (Business)"):
        at.radio(key="reg_papel").set_value(rotulo).run()
        _sem_erros(at)


def test_dono_de_empresa_pendente_so_ve_o_estado():
    from modules import acesso, db
    db.create_user({"username": "dono", "name": "Dono Teste", "password_hash": "x"})
    acesso.criar_empresa_pendente("dono", "Empresa Y")
    at = _app("dashboard", "dono")
    _sem_erros(at)
    assert any("aprovação" in i.value for i in at.info)
    assert "painel" not in at.session_state                      # nunca chega a nenhum painel


def test_aluno_com_palavra_passe_temporaria_tem_de_a_trocar():
    from modules import db
    db.create_user({"username": "novo", "name": "Novo Aluno", "password_hash": "x", "must_change_pw": True})
    at = _app("dashboard", "novo")
    _sem_erros(at)
    assert "painel" not in at.session_state


# ── Tema claro ────────────────────────────────────────
@pytest.mark.parametrize("pagina", ["dashboard", "perfil", "classificacao", "comparador"])
def test_tema_claro_desenha_as_paginas_sem_erros(pagina):
    from modules import db
    db.save_progress("aluno", {"tema": "claro"})          # tema guardado na conta: a app aplica-o ao entrar
    at = _app(pagina)
    _sem_erros(at)
    assert at.session_state["tema"] == "claro"


def test_interruptor_do_tema_muda_e_fica_guardado():
    from modules import db
    at = _app("dashboard")
    assert at.toggle(key="tema_toggle").value is False     # escuro por omissão
    at.toggle(key="tema_toggle").set_value(True).run()
    _sem_erros(at)
    assert at.session_state["tema"] == "claro"
    assert db.get_progress("aluno").get("tema") == "claro"
