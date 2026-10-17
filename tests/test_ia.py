"""IA (Grok) no painel Business: a IA só propõe, o sistema valida; limite diário; nada de dados enviados. Sem chamadas reais."""
import json
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)


class _Resp:
    def __init__(self, conteudo=None, status=200):
        self.status_code = status
        self._c = conteudo

    def json(self):
        return {"choices": [{"message": {"content": self._c}}]}


@pytest.fixture(autouse=True)
def ambiente(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-teste")
    monkeypatch.setenv("IA_LIMITE_DIARIO", "3")


def _responder(monkeypatch, conteudo, status=200, enviados=None):
    from modules import ia

    def falso(url, json=None, headers=None, timeout=None):
        if enviados is not None:
            enviados.append((url, json, headers))
        return _Resp(conteudo, status)
    monkeypatch.setattr(ia.requests, "post", falso)


def test_ia_desligada_sem_chave(monkeypatch):
    from modules import ia
    monkeypatch.delenv("XAI_API_KEY")
    assert not ia.configurada()
    with pytest.raises(ia.IAErro):
        ia.chamar([{"role": "user", "content": "x"}], "e1")


def test_assistente_valida_o_que_a_ia_sugere(monkeypatch):
    from modules import business_core as bc, ia
    df = bc.exemplo("clientes")
    enviados = []
    _responder(monkeypatch, json.dumps({"problema": "churn", "alvo": "saiu", "ignorar": ["id_cliente", "coluna_inventada"],
                                        "porque": "A coluna «saiu» diz se o cliente cancelou."}), enviados=enviados)
    plano = ia.sugerir_analise("quero saber que clientes vão cancelar", df, "e1")
    assert plano["problema"] == "churn" and plano["alvo"] == "saiu"
    assert plano["ignorar"] == ["id_cliente"]                                   # colunas inventadas são descartadas
    url, corpo, cab = enviados[0]
    assert url.endswith("/chat/completions") and cab["Authorization"] == "Bearer xai-teste"
    enviado = json.dumps(corpo, ensure_ascii=False)
    assert "C00001" not in enviado and "gasto_mensal_kz" in enviado            # nomes de colunas sim, linhas de dados não


def test_assistente_rejeita_problema_ou_alvo_inventados(monkeypatch):
    from modules import business_core as bc, ia
    df = bc.exemplo("clientes")
    _responder(monkeypatch, json.dumps({"problema": "apagar_tudo", "alvo": None}))
    with pytest.raises(ia.IAErro):
        ia.sugerir_analise("quero saber que clientes vão cancelar", df, "e1")
    _responder(monkeypatch, json.dumps({"problema": "churn", "alvo": "coluna_que_nao_existe"}))
    plano = ia.sugerir_analise("quero saber que clientes vão cancelar", df, "e1")
    assert plano["alvo"] == "saiu"                                              # cai na sugestão automática do sistema
    _responder(monkeypatch, "isto não é json")
    with pytest.raises(ia.IAErro):
        ia.sugerir_analise("quero saber que clientes vão cancelar", df, "e1")


def test_limite_diario_por_empresa(monkeypatch):
    from modules import ia
    _responder(monkeypatch, "ok")
    for _ in range(3):
        assert ia.chamar([{"role": "user", "content": "x"}], "e1") == "ok"
    assert ia.pedidos_restantes("e1") == 0
    with pytest.raises(ia.IAErro):
        ia.chamar([{"role": "user", "content": "x"}], "e1")
    assert ia.pedidos_restantes("e2") == 3                                      # outra empresa não é afectada


@pytest.mark.parametrize("status", [401, 429, 404, 500])
def test_erros_da_api_nao_mostram_a_chave(monkeypatch, status):
    from modules import ia
    _responder(monkeypatch, "x", status=status)
    with pytest.raises(ia.IAErro) as e:
        ia.chamar([{"role": "user", "content": "x"}], "e1")
    assert "xai-teste" not in str(e.value)
    assert ia.usos_hoje("e1") == 0                                              # pedidos falhados não gastam o limite


def test_comentario_usa_so_numeros_e_entra_no_pdf(monkeypatch):
    from modules import business_core as bc, ia
    df = bc.exemplo("clientes")
    r = bc.treinar(df, "saiu", "classificacao")
    analise = {"titulo": "Churn", "problema": "churn", "dataset_nome": "Clientes", "resumo": r["resultado"], "criado_por": "u"}
    enviados = []
    _responder(monkeypatch, "**Os clientes de risco alto** merecem contacto. " * 3, enviados=enviados)
    texto = ia.escrever_comentario(analise, "Loja", "e1")
    assert "*" not in texto and len(texto) > 40
    assert "C00001" not in json.dumps(enviados[0][1], ensure_ascii=False)
    analise["resumo"] = dict(analise["resumo"], texto_ia={"texto": texto, "modelo": "m", "em": "2026-01-01"})
    assert bc.relatorio_pdf("Loja", analise).startswith(b"%PDF")


def test_provedor_gemini_gratuito_usa_o_endereco_e_a_chave_certos(monkeypatch):
    from modules import ia
    monkeypatch.delenv("XAI_API_KEY")
    monkeypatch.setenv("IA_API_KEY", "chave-gemini")
    monkeypatch.setenv("IA_PROVEDOR", "gemini")
    enviados = []
    _responder(monkeypatch, "ok", enviados=enviados)
    assert ia.configurada() and ia.provedor() == "gemini"
    assert ia.chamar([{"role": "user", "content": "x"}], "e1") == "ok"
    url, corpo, cab = enviados[0]
    assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert corpo["model"] == "gemini-3.8-flash" and cab["Authorization"] == "Bearer chave-gemini"
    monkeypatch.setenv("IA_MODEL", "outro-modelo")                              # o segredo ia_model manda
    ia.chamar([{"role": "user", "content": "x"}], "e1")
    assert enviados[1][1]["model"] == "outro-modelo"


def test_repete_sem_response_format_quando_o_modelo_nao_o_aceita(monkeypatch):
    from modules import ia
    corpos = []

    def falso(url, json=None, headers=None, timeout=None):
        corpos.append(dict(json))
        return _Resp("{}", 400) if "response_format" in json else _Resp("{\"ok\": 1}", 200)
    monkeypatch.setattr(ia.requests, "post", falso)
    assert ia.chamar([{"role": "user", "content": "x"}], "e1", json_modo=True) == "{\"ok\": 1}"
    assert "response_format" in corpos[0] and "response_format" not in corpos[1]
    assert ia.usos_hoje("e1") == 1                                              # só o pedido que resultou conta
