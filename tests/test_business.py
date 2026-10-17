"""Painel Business: separação do painel de aluno, lógica de dados/modelos, permissões e «Sair». Correr com `pytest -q`."""
import os
import sys
import time

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)


@pytest.fixture(autouse=True)
def pasta_temporaria(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


# ── Painel só de empresa (nada de aluno) ──────────────
def test_conta_de_empresa_so_tem_o_painel_business():
    from modules.navegacao import menu_do_painel, paineis_disponiveis, paginas_permitidas
    emp = [{"tipo": "empresa", "papel": "gestor"}]
    assert paineis_disponiveis("empresa", emp, False) == ["business"]
    assert paineis_disponiveis("empresa", [], False) == []                     # sem empresa activa não há painel
    assert paineis_disponiveis("aluno", emp, False) == ["business"]           # conta antiga: aluno sem turma nem escola
    assert paineis_disponiveis("aluno", emp, True) == ["aluno", "business"]   # aluno com turma mantém os dois
    permitidas = paginas_permitidas("business")
    assert {"business_home", "business_dados", "business_analises", "business_previsoes",
            "business_relatorios", "business_equipa", "perfil"} <= permitidas
    assert not (permitidas & {"dashboard", "escola", "primeiro_modelo", "classificacao", "projeto", "conquistas",
                              "certificados", "revisao", "historico", "dicionario"})
    assert len(menu_do_painel("business")) == 7


def test_registo_de_empresa_cria_conta_com_papel_empresa():
    from modules import acesso, db
    acesso.email_service.enviar_a_admins = lambda *a, **k: None
    ok, m = acesso.registar("empresa", "Maria Silva", "maria", "m@x.ao", "abc123", "abc123", empresa_modo="nova", empresa_nome="Loja Maria")
    assert ok, m
    assert db.get_user("maria")["role"] == "empresa"


# ── Lógica de dados e modelos ─────────────────────────
def test_ler_csv_com_virgula_decimal_e_datas():
    from modules import business_core as bc
    raw = "data;valor;loja\n01/02/2024;1 234,5;A\n02/02/2024;2 000,0;B\n03/02/2024;1500,25;A\n".encode("latin-1")
    df = bc.ler_tabela("a.csv", raw)
    assert str(df["valor"].dtype).startswith("float") and df["valor"].iloc[0] == pytest.approx(1234.5)
    assert "datetime" in str(df["data"].dtype)
    with pytest.raises(ValueError):
        bc.ler_tabela("vazio.csv", b"  ")


def test_guardar_e_ler_conjunto_de_dados():
    from modules import business_core as bc
    df = bc.exemplo("clientes")
    volta = bc.texto_para_tabela(bc.tabela_para_texto(df))
    assert volta.shape == df.shape


@pytest.mark.parametrize("exemplo,problema", [("clientes", "churn"), ("vendas", "vendas"), ("stock", "stock")])
def test_treinar_e_prever(exemplo, problema):
    from modules import business_core as bc
    df = bc.exemplo(exemplo)
    alvo = bc.sugerir_alvo(df, problema)
    assert alvo
    r = bc.treinar(df, alvo, bc.PROBLEMAS[problema]["tipo"], por_data=bool(bc.colunas_data(df)))
    res = r["resultado"]
    assert res["metricas"] and res["importancias"] and bc.manchete(res)
    modelo = bc.texto_para_modelo(bc.modelo_para_texto(r["modelo"]))
    novo = df.drop(columns=[alvo]).head(5)
    saida = bc.aplicar_modelo(r["meta"], modelo, novo)
    assert len(saida) == 5 and ("previsao" in saida.columns)
    if problema == "churn":
        assert set(saida["risco"]) <= {"alto", "médio", "baixo"}
    pdf = bc.relatorio_pdf("Empresa", {"titulo": "T", "problema": problema, "dataset_nome": exemplo, "resumo": res, "criado_por": "u"})
    assert pdf.startswith(b"%PDF")


def test_fraude_apanha_a_maioria_das_anomalias():
    from modules import business_core as bc
    df = bc.exemplo("transaccoes")
    a = bc.detectar_anomalias(df, excluir=["fraude"], taxa=0.02)
    top = [t["linha"] - 2 for t in a["resultado"]["top"]]
    assert df.iloc[top]["fraude"].eq("sim").sum() >= 30
    assert a["resultado"]["top"][0]["motivos"]


def test_segmentacao_e_aplicacao_a_clientes_novos():
    from modules import business_core as bc
    df = bc.exemplo("clientes")
    s = bc.segmentar(df)
    assert 2 <= s["resultado"]["k"] <= 7
    seg = bc.aplicar_modelo(s["meta"], s["modelo"], df.head(4))["segmento"]
    assert len(seg) == 4


def test_dados_sem_coluna_a_prever_dao_erro_claro():
    from modules import business_core as bc
    df = bc.exemplo("clientes")
    with pytest.raises(ValueError):
        bc.treinar(df, "nao_existe", "classificacao")


# ── Base de dados e permissões ────────────────────────
def test_guardar_analise_e_dados_por_empresa():
    from modules import business_core as bc, db
    db.create_org("e1", "Loja", "empresa", "ABC123")
    db.create_org("e2", "Outra", "empresa", "DEF456")
    df = bc.exemplo("clientes")
    _, texto, _ = bc.reduzir_para_guardar(df)
    did = db.bn_criar_dataset("e1", "Clientes", texto, len(df), df.shape[1], "ana")
    assert [d["nome"] for d in db.bn_datasets("e1")] == ["Clientes"]
    assert db.bn_datasets("e2") == []                                         # cada empresa só vê os seus dados
    assert "conteudo" not in db.bn_datasets("e1")[0]                          # a lista não traz o conteúdo pesado
    assert db.bn_get_dataset(did)["conteudo"] == texto
    aid = db.bn_criar_analise("e1", did, "Clientes", "ana", "churn", "T", "saiu", {"tipo": "classificacao"}, {}, "xx")
    assert db.bn_analises("e1")[0]["modelo_guardado"] is True and "modelo" not in db.bn_analises("e1")[0]
    db.bn_apagar_analise(aid)
    db.bn_apagar_dataset(did)
    assert db.bn_datasets("e1") == [] and db.bn_analises("e1") == []


def test_permissoes_por_papel():
    from modules.business import pode
    g, a, m = ({"papel": p} for p in ("gestor", "analista", "membro"))
    assert pode(g, "carregar") and pode(g, "equipa") and pode(a, "treinar") and not pode(a, "equipa")
    assert not pode(m, "carregar") and not pode(m, "treinar") and pode(m, "prever")


# ── «Sair» não deixa a sessão voltar ao actualizar ────
def test_cookie_antigo_fica_revogado_depois_de_sair():
    import jwt
    from modules import auth
    cfg = {"cookie": {"name": "dataforge_edu_cookie", "key": "k", "expiry_days": 7}}
    raw = jwt.encode({"username": "ana", "exp_date": time.time() - 100 + 7 * 86400}, "k", algorithm="HS256")
    tok = auth._descodificar_cookie(raw, cfg)
    assert not auth._cookie_revogado(tok, cfg)
    auth._revogar_sessoes("ana")
    assert auth._cookie_revogado(tok, cfg)                                    # o cookie velho já não serve
    time.sleep(0.05)
    novo = auth._descodificar_cookie(jwt.encode({"username": "ana", "exp_date": time.time() + 7 * 86400}, "k", algorithm="HS256"), cfg)
    assert not auth._cookie_revogado(novo, cfg)                               # um login novo continua a funcionar
    assert not auth._cookie_revogado({"username": "rui", "exp_date": time.time() + 86400}, cfg)


# ── Inglês ────────────────────────────────────────────
def test_modelos_com_valores_traduzem_para_ingles():
    from modules import i18n
    i18n.st.session_state["lang"] = "en"
    try:
        assert i18n.tr("Previsões para **120** linhas.") == "Predictions for **120** rows."
        assert i18n.tr("Passo 2 de 5: Escolher o modelo").startswith("Step 2 of 5")
        assert i18n.tr("Criar uma empresa nova") == "Create a new company"
    finally:
        i18n.st.session_state["lang"] = "pt"


def test_todos_os_modelos_com_chavetas_tem_o_mesmo_numero_de_valores():
    import json
    t = json.load(open(os.path.join(RAIZ, "modules", "locales", "en.json"), encoding="utf8"))["t"]
    assert [k for k, v in t.items() if k.count("{}") != v.count("{}")] == []
