"""Regras de acesso (convites, escolas, empresas, turmas, alunos, repor palavra-passe). Correr com `pytest -q`."""
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)


@pytest.fixture(autouse=True)
def ambiente(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)
    for k, v in {"SMTP_HOST": "h", "SMTP_USER": "u", "SMTP_PASSWORD": "p", "EMAIL_FROM": "f@x.com",
                 "ADMIN_CODE": "ADM-SECRETO", "APP_URL": "https://app.test"}.items():
        monkeypatch.setenv(k, v)
    from modules import email_service
    enviados = []
    monkeypatch.setattr(email_service, "_smtp", lambda para, ass, txt: (enviados.append((para, ass, txt)) or (True, "")))
    return enviados


def _professor(acesso, db, user="prof", email="prof@x.com"):
    cod, estado, _ = acesso.criar_convite_professor(email, "adm")
    assert estado == "enviado"
    ok, m = acesso.registar("professor", "Prof Teste", user, email, "123456", "123456", cod_convite=cod)
    assert ok, m
    return cod


def test_convite_de_professor_uso_unico_e_ligado_ao_email(ambiente):
    from modules import acesso, db
    cod, _, _ = acesso.criar_convite_professor("prof@x.com", "adm")
    assert "convite=" in ambiente[-1][2]
    ok, m = acesso.registar("professor", "Prof A", "profa", "outro@x.com", "123456", "123456", cod_convite=cod)
    assert not ok and "outro email" in m
    ok, m = acesso.registar("professor", "Prof A", "profa", "prof@x.com", "123456", "123456", cod_convite=cod)
    assert ok and db.get_user("profa")["role"] == "professor"
    ok, m = acesso.registar("professor", "Prof B", "profb", "prof@x.com", "123456", "123456", cod_convite=cod)
    assert not ok and "usado" in m


def test_convite_cancelado_ou_expirado_nao_funciona():
    from modules import acesso, db
    c1, *_ = acesso.criar_convite_professor("a@x.com", "adm")
    acesso.cancelar_convite(c1)
    assert not acesso.registar("professor", "Prof A", "profa", "a@x.com", "123456", "123456", cod_convite=c1)[0]
    c2, *_ = acesso.criar_convite_professor("b@x.com", "adm")
    db.update_convite(c2, {"expira_em": "2000-01-01T00:00:00+00:00"})
    assert acesso.estado_convite(db.get_convite(c2)) == "Expirado"
    assert not acesso.registar("professor", "Prof B", "profb", "b@x.com", "123456", "123456", cod_convite=c2)[0]


def test_admin_code_so_cria_o_primeiro_admin():
    from modules import acesso, db
    ok, m = acesso.registar("professor", "Admin Um", "adm1", "a1@x.com", "123456", "123456", cod_convite="ADM-SECRETO")
    assert ok and db.get_user("adm1")["role"] == "admin"
    ok, m = acesso.registar("professor", "Admin Dois", "adm2", "a2@x.com", "123456", "123456", cod_convite="ADM-SECRETO")
    assert not ok


def test_escola_criada_pelo_professor_e_turma_com_aprovacao(ambiente):
    from modules import acesso, db
    _professor(acesso, db)
    esc = acesso.criar_escola("prof", "Escola Teste")
    assert [m["papel"] for m in db.list_membros(username="prof")] == ["gestor"]
    db.create_turma("TURMA1", "Turma 1", "prof", esc["id"])
    ok, m = acesso.registar("aluno", "Aluno Um", "al1", "", "123456", "123456", turma="turma1", cod_org=esc["codigo_convite"])
    assert ok and db.get_user("al1")["turma"] is None and len(db.list_pedidos_turma(codigo="TURMA1")) == 1
    assert any(e[0] == "prof@x.com" for e in ambiente)                    # professor avisado
    acesso.aceitar_pedido("TURMA1", "al1")
    assert db.get_user("al1")["turma"] == "TURMA1"
    db.update_turma("TURMA1", {"aprovacao": False})
    assert acesso.registar("aluno", "Aluno Dois", "al2", "", "123456", "123456", turma="TURMA1")[0]
    assert db.get_user("al2")["turma"] == "TURMA1"


def test_admin_nao_cria_escola_nem_entra_em_organizacoes():
    from modules import acesso, db
    from modules.admin import adicionar_membro_por_codigo
    acesso.registar("professor", "Admin Um", "adm1", "a1@x.com", "123456", "123456", cod_convite="ADM-SECRETO")
    with pytest.raises(db.DBError):
        acesso.criar_escola("adm1", "Escola do Admin")
    db.create_org("s1", "Escola A", "escola", "AAA111")
    assert not adicionar_membro_por_codigo("adm1", "AAA111")[0]


def test_alunos_em_lote_com_palavra_passe_temporaria(ambiente):
    from modules import acesso, db
    _professor(acesso, db)
    db.create_turma("TURMA1", "Turma 1", "prof")
    res = acesso.criar_alunos("prof", "TURMA1", ["Ana Silva", "Ana Silva, ana@x.com", "x", "Rui Costa, mau"], enviar_email=True)
    assert [r["utilizador"] for r in res[:2]] == ["ana_silva", "ana_silva2"]
    assert res[2]["erro"] and res[3]["erro"]
    assert db.get_user("ana_silva")["must_change_pw"] is True and db.get_user("ana_silva2")["turma"] == "TURMA1"
    assert any(e[0] == "ana@x.com" and res[1]["palavra_passe"] in e[2] for e in ambiente)
    with pytest.raises(db.DBError):
        acesso.criar_alunos("outro", "TURMA1", ["Ana Nova"])


def test_empresa_pendente_ate_o_admin_aprovar(ambiente):
    from modules import acesso, db
    acesso.registar("professor", "Admin Um", "adm1", "adm@x.com", "123456", "123456", cod_convite="ADM-SECRETO")
    assert not acesso.registar("empresa", "Dono", "dono", "", "123456", "123456", empresa_nome="Empresa X")[0]   # email obrigatório
    ok, m = acesso.registar("empresa", "Dono", "dono", "dono@x.com", "123456", "123456", empresa_modo="nova", empresa_nome="Empresa X")
    assert ok, m
    org = acesso.empresa_do_dono("dono")
    assert org["estado"] == "pendente" and org["activa"] is False
    assert db.membros_do_utilizador("dono") == []                          # ainda sem painel Business
    assert any(e[0] == "adm@x.com" for e in ambiente)                      # admin avisado
    acesso.recusar_org(org["id"], "Dados incompletos")
    assert acesso.empresa_do_dono("dono")["estado"] == "recusada"
    novo = acesso.novo_pedido_empresa("dono", "Empresa X Lda")
    acesso.aprovar_org(novo["id"])
    mem = db.membros_do_utilizador("dono")
    assert mem[0]["papel"] == "gestor" and acesso.empresa_do_dono("dono") is None
    assert any(e[0] == "dono@x.com" and mem[0]["codigo_convite"] in e[2] for e in ambiente)
    ok, m = acesso.registar("empresa", "Colega", "colega", "c@x.com", "123456", "123456", empresa_modo="existente",
                            empresa_cod=mem[0]["codigo_convite"])
    assert ok and db.list_membros(username="colega")[0]["papel"] == "membro"


def test_repor_palavra_passe_por_email(ambiente):
    from modules import acesso, db
    _professor(acesso, db)
    ambiente.clear()
    acesso.pedir_reposicao("PROF@X.COM")
    token = ambiente[-1][2].split("?reset=")[1].split()[0]
    assert acesso.validar_reposicao(token) == "prof"
    assert acesso.aplicar_reposicao(token, "novapass") and not acesso.validar_reposicao(token)   # uso único
    n = len(ambiente)
    acesso.pedir_reposicao("naoexiste")
    assert len(ambiente) == n                                              # resposta neutra: nada enviado


def test_limite_diario_poe_emails_em_fila(monkeypatch):
    from modules import email_service
    monkeypatch.setenv("EMAIL_LIMITE_DIARIO", "1")
    assert email_service.enviar("a@x.com", "A", "t")[0] == "enviado"
    assert email_service.enviar("b@x.com", "B", "t")[0] == "fila"
    monkeypatch.setenv("EMAIL_LIMITE_DIARIO", "100")
    assert email_service.processar_fila()["restam"] == 0


def test_professor_pede_convite_e_admin_aprova(ambiente, monkeypatch):
    from modules import acesso, db
    monkeypatch.setenv("ADMIN_EMAIL", "admin@x.com")
    ok, m = acesso.pedir_convite("Prof Novo", "Novo@X.com", "ISPKZ", "Dou ML")
    assert ok, m
    assert any(e[0] == "admin@x.com" and "Pedido de convite" in e[1] for e in ambiente)   # avisa o admin
    # um pedido pendente por email
    ok, m = acesso.pedir_convite("Prof Novo", "novo@x.com", "ISPKZ")
    assert not ok and "aguardar" in m
    pend = acesso.pedidos_convite_pendentes()
    assert len(pend) == 1 and pend[0]["email"] == "novo@x.com"
    # o admin aprova: convite criado, ligado ao email do pedido e enviado
    codigo, estado, _ = acesso.aprovar_pedido_convite(pend[0]["id"], "adm")
    assert estado == "enviado" and ambiente[-1][0] == "novo@x.com" and codigo in ambiente[-1][2]
    assert not acesso.pedidos_convite_pendentes()
    assert db.get_pedido_convite(pend[0]["id"])["estado"] == "aprovado"
    # o professor regista-se com esse convite
    ok, m = acesso.registar("professor", "Prof Novo", "profnovo", "novo@x.com", "123456", "123456", cod_convite=codigo)
    assert ok and db.get_user("profnovo")["role"] == "professor"
    # não se trata o mesmo pedido duas vezes
    with pytest.raises(db.DBError):
        acesso.aprovar_pedido_convite(pend[0]["id"], "adm")


def test_pedido_de_convite_recusado_avisa_e_valida_campos(ambiente):
    from modules import acesso, db
    assert not acesso.pedir_convite("Ab", "a@x.com", "Escola")[0]               # nome curto
    assert not acesso.pedir_convite("Prof Teste", "sem-arroba", "Escola")[0]    # email inválido
    assert not acesso.pedir_convite("Prof Teste", "a@x.com", "")[0]             # escola em falta
    assert acesso.pedir_convite("Prof Teste", "a@x.com", "Escola X")[0]
    pid = acesso.pedidos_convite_pendentes()[0]["id"]
    acesso.recusar_pedido_convite(pid, "adm", "Não conseguimos confirmar a escola")
    assert db.get_pedido_convite(pid)["estado"] == "recusado"
    assert ambiente[-1][0] == "a@x.com" and "confirmar a escola" in ambiente[-1][2]
    assert not acesso.pedidos_convite_pendentes()
    # depois de recusado, pode voltar a pedir
    assert acesso.pedir_convite("Prof Teste", "a@x.com", "Escola X")[0]
