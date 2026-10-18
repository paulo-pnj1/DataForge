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
    monkeypatch.setenv("IA_INTERVALO", "0")                 # sem espera entre pedidos (os testes do intervalo ligam-no)


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


def test_recusa_mostra_o_motivo_do_servico_sem_a_chave(monkeypatch):
    from modules import ia

    class Recusa:
        status_code = 400

        def json(self):
            return [{"error": {"message": "API key not valid. Please pass a valid API key. xai-teste"}}]
    monkeypatch.setattr(ia.requests, "post", lambda *a, **k: Recusa())
    with pytest.raises(ia.IAErro) as e:
        ia.chamar([{"role": "user", "content": "x"}], "e1")
    assert "API key not valid" in str(e.value) and "xai-teste" not in str(e.value)
    assert ia.testar.__name__ == "testar"


# ── Tutor de ML (lições, testes, treinos) ─────────────
def test_tutor_envia_o_contexto_e_conta_por_aluno(monkeypatch):
    from modules import ia
    enviados = []
    _responder(monkeypatch, "Overfitting é decorar em vez de aprender.", enviados=enviados)
    r = ia.perguntar_tutor("O que é overfitting?", "Lição 7", "Resumo: o modelo decorou os dados.", [], "ana", "pt")
    assert "decorar" in r
    msgs = enviados[0][1]["messages"]
    assert "Resumo: o modelo decorou os dados." in msgs[0]["content"] and msgs[-1]["content"] == "O que é overfitting?"
    assert "ana" not in json.dumps(msgs)                                        # o nome do aluno não segue para a IA
    assert ia.usos_hoje("u:ana") == 1 and ia.usos_hoje("u:rui") == 0            # cada aluno tem o seu limite


def test_tutor_nao_revela_resposta_de_pergunta_por_responder(monkeypatch):
    from modules import ia
    enviados = []
    _responder(monkeypatch, "Pensa no que o modelo faz com dados novos.", enviados=enviados)
    ia.perguntar_tutor("Ajuda-me", "Teste", "Pergunta: ...", [], "ana", "pt", nao_revelar=True)
    assert "NÃO digas qual é a resposta certa" in enviados[0][1]["messages"][0]["content"]
    ia.perguntar_tutor("Ajuda-me", "Teste", "Pergunta: ...", [], "ana", "pt", nao_revelar=False)
    assert "NÃO digas qual é a resposta certa" not in enviados[1][1]["messages"][0]["content"]


def test_tutor_historico_e_limite_do_aluno(monkeypatch):
    from modules import ia
    monkeypatch.setenv("IA_LIMITE_ALUNO", "2")
    enviados = []
    _responder(monkeypatch, "resposta curta ok", enviados=enviados)
    hist = [("p1", "r1"), ("p2", "r2"), ("p3", "r3"), ("p4", "r4")]
    ia.perguntar_tutor("e agora?", "t", "c", hist, "ana")
    msgs = enviados[0][1]["messages"]
    assert [m["content"] for m in msgs[1:-1]] == ["p2", "r2", "p3", "r3", "p4", "r4"]   # só as últimas 3 trocas
    ia.perguntar_tutor("outra", "t", "c", [], "ana")
    with pytest.raises(ia.IAErro):
        ia.perguntar_tutor("mais uma", "t", "c", [], "ana")
    with pytest.raises(ia.IAErro):
        ia.perguntar_tutor("", "t", "c", [], "rui")


def test_resumo_de_resultado_so_leva_numeros_curtos():
    import numpy as np
    import pandas as pd
    from modules.tutor_ia import resumir_resultado
    res = {"acc": 0.91, "modelo": "Random Forest", "cv": np.array([0.9, 0.92, 0.88]), "df": pd.DataFrame({"a": [1]}),
           "fig": object(), "_interno": 5, "loss": list(np.linspace(1, 0.1, 50)), "texto_longo": "x" * 300}
    t = resumir_resultado(res)
    assert "acc: 0.91" in t and "modelo: Random Forest" in t and "cv: [0.9, 0.92, 0.88]" in t
    assert "loss: 50 valores" in t
    assert "df" not in t and "fig" not in t and "_interno" not in t and "xxx" not in t


# ── Poupar o limite da chave: cache, tecto global, intervalo, modelo leve, chave de reserva ──
def _tutor(ia, user="ana", pergunta="Explica de outra forma", **kw):
    return ia.perguntar_tutor(pergunta, "Lição 7", "Resumo: o modelo decorou os dados.", [], user, "pt", usar_cache=True, **kw)


def test_cache_partilhada_entre_alunos_e_nao_gasta_limite(monkeypatch):
    from modules import ia
    enviados = []
    _responder(monkeypatch, "Overfitting é decorar em vez de aprender.", enviados=enviados)
    assert _tutor(ia, "ana") == _tutor(ia, "rui") == _tutor(ia, "eva", "  EXPLICA de outra forma!! ")   # mesma pergunta, mesma lição
    assert len(enviados) == 1                                                   # só a primeira foi à IA
    assert ia.usos_hoje("u:ana") == 1 and ia.usos_hoje("u:rui") == 0            # quem aproveitou a cache não gasta limite
    assert ia.uso_global()["dia"] == 1
    _tutor(ia, "ana", "Dá-me um exemplo do dia-a-dia")                          # outra pergunta: vai à IA
    assert len(enviados) == 2


def test_cache_nao_guarda_pergunta_nem_nome_e_so_vale_quando_deve(monkeypatch):
    from modules import db, ia
    enviados = []
    _responder(monkeypatch, "Resposta de teste ok.", enviados=enviados)
    ia.perguntar_tutor("Explica de outra forma", "L", "c", [], "ana", "pt", usar_cache=True)
    guardado = json.dumps([dict(r) for r in db._local("progress").all() if str(r.get("username", "")).startswith("iacache:")], ensure_ascii=False)
    assert "Resposta de teste ok." in guardado and "Explica de outra forma" not in guardado and "ana" not in guardado.replace("Resposta", "")
    for _ in range(2):                                                          # sem usar_cache, com histórico, com email ou número longo: nunca da cache
        ia.perguntar_tutor("Explica de outra forma", "L", "c", [], "ana", "pt")
        ia.perguntar_tutor("Explica de outra forma", "L", "c", [("p", "r")], "ana", "pt", usar_cache=True)
        ia.perguntar_tutor("o meu email é ana@exemplo.com", "L", "c", [], "ana", "pt", usar_cache=True)
        ia.perguntar_tutor("o meu número é 923456789", "L", "c", [], "ana", "pt", usar_cache=True)
    assert len(enviados) == 1 + 8


def test_cache_distingue_idioma_contexto_e_expira(monkeypatch):
    import datetime
    from modules import db, ia
    enviados = []
    _responder(monkeypatch, "Resposta de teste ok.", enviados=enviados)
    ia.perguntar_tutor("Explica", "L", "contexto A", [], "ana", "pt", usar_cache=True)
    ia.perguntar_tutor("Explica", "L", "contexto B", [], "ana", "pt", usar_cache=True)
    ia.perguntar_tutor("Explica", "L", "contexto A", [], "ana", "en", usar_cache=True)
    assert len(enviados) == 3
    antigo = str(datetime.date.today() - datetime.timedelta(days=ia.CACHE_DIAS + 1))
    chave = ia._chave_cache("tutor", "pt", "L", "contexto A", "explica", False)
    db.save_progress(chave, {"t": "velha", "em": antigo})
    assert ia._cache_ler(chave) is None                                         # passou o prazo: pergunta-se outra vez


def test_tecto_global_por_dia_com_reserva_para_empresas_e_professores(monkeypatch):
    from modules import ia
    monkeypatch.setenv("IA_LIMITE_GLOBAL_DIA", "10")
    monkeypatch.setenv("IA_RESERVA_PCT", "30")                                  # alunos: até 7; o resto é de empresas e professores
    monkeypatch.setenv("IA_LIMITE_ALUNO", "50")
    _responder(monkeypatch, "resposta curta ok")
    for i in range(7):
        ia.perguntar_tutor(f"pergunta {i}", "t", "c", [], f"aluno{i}")
    with pytest.raises(ia.IAErro) as e:
        ia.perguntar_tutor("mais uma", "t", "c", [], "aluno9")
    assert "descansar" in str(e.value)
    assert ia.chamar([{"role": "user", "content": "x"}], "empresa1") == "resposta curta ok"            # empresa ainda passa
    for _ in range(2):
        ia.chamar([{"role": "user", "content": "x"}], "empresa2")
    assert ia.uso_global()["dia"] == 10
    with pytest.raises(ia.IAErro) as e:
        ia.chamar([{"role": "user", "content": "x"}], "empresa3")                                      # a chave acabou para todos
    assert "muito ocupada" in str(e.value)


def test_tecto_global_por_minuto_e_zero_desliga(monkeypatch):
    from modules import ia
    monkeypatch.setenv("IA_LIMITE_GLOBAL_MINUTO", "2")
    monkeypatch.setenv("IA_LIMITE_DIARIO", "50")
    _responder(monkeypatch, "ok")
    ia.chamar([{"role": "user", "content": "x"}], "e1")
    ia.chamar([{"role": "user", "content": "x"}], "e2")
    with pytest.raises(ia.IAErro) as e:
        ia.chamar([{"role": "user", "content": "x"}], "e3")
    assert "minuto" in str(e.value)
    monkeypatch.setenv("IA_LIMITE_GLOBAL_MINUTO", "0")                          # 0 = sem tecto
    assert ia.chamar([{"role": "user", "content": "x"}], "e3") == "ok"


def test_intervalo_minimo_entre_perguntas_do_mesmo_aluno(monkeypatch):
    from modules import ia
    monkeypatch.setenv("IA_INTERVALO", "5")
    _responder(monkeypatch, "resposta curta ok")
    ia.perguntar_tutor("primeira", "t", "c", [], "ana")
    with pytest.raises(ia.IAErro) as e:
        ia.perguntar_tutor("segunda", "t", "c", [], "ana")
    assert "segundos" in str(e.value)
    assert ia.perguntar_tutor("outra pessoa", "t", "c", [], "rui")             # só vale para o mesmo aluno
    assert ia.chamar([{"role": "user", "content": "x"}], "e1") and ia.chamar([{"role": "user", "content": "x"}], "e1")   # empresas não têm intervalo
    assert ia.usos_hoje("u:ana") == 1                                           # o pedido recusado não gastou limite


def test_modelo_leve_so_para_o_tutor(monkeypatch):
    from modules import ia
    monkeypatch.delenv("XAI_API_KEY")
    monkeypatch.setenv("IA_API_KEY", "k")
    monkeypatch.setenv("IA_PROVEDOR", "gemini")
    enviados = []
    _responder(monkeypatch, "resposta curta ok", enviados=enviados)
    ia.perguntar_tutor("pergunta", "t", "c", [], "ana")
    ia.chamar([{"role": "user", "content": "x"}], "e1")
    assert enviados[0][1]["model"] == "gemini-3.8-flash" and enviados[1][1]["model"] == "gemini-3.8-flash"
    monkeypatch.setenv("IA_MODEL_TUTOR", "modelo-leve")
    ia.perguntar_tutor("pergunta 2", "t", "c", [], "bia")
    assert enviados[2][1]["model"] == "modelo-leve"
    monkeypatch.delenv("IA_MODEL_TUTOR")
    del enviados[2:]
    monkeypatch.setenv("IA_MODEL_TUTOR", "meu-modelo")
    ia.perguntar_tutor("outra", "t", "c", [], "rui")
    assert enviados[2][1]["model"] == "meu-modelo"


def test_chave_de_reserva_so_entra_quando_a_principal_esgota(monkeypatch):
    from modules import ia
    monkeypatch.setenv("IA_API_KEY_2", "reserva")
    monkeypatch.setenv("IA_PROVEDOR_2", "gemini")
    chaves = []

    def falso(url, json=None, headers=None, timeout=None):
        chaves.append(headers["Authorization"])
        return _Resp("x", 429) if headers["Authorization"] == "Bearer xai-teste" else _Resp("da reserva", 200)
    monkeypatch.setattr(ia.requests, "post", falso)
    assert ia.chamar([{"role": "user", "content": "x"}], "e1") == "da reserva"
    assert chaves == ["Bearer xai-teste", "Bearer reserva"]
    monkeypatch.setattr(ia.requests, "post", lambda *a, **k: _Resp("x", 401))   # chave recusada não vai para a reserva
    with pytest.raises(ia.IAErro) as e:
        ia.chamar([{"role": "user", "content": "x"}], "e2")
    assert "recusada" in str(e.value)
    monkeypatch.setattr(ia.requests, "post", lambda *a, **k: _Resp("x", 429))   # as duas esgotadas
    with pytest.raises(ia.IAErro) as e:
        ia.chamar([{"role": "user", "content": "x"}], "e3")
    assert "reserva" not in str(e.value) and ia.usos_hoje("e3") == 0


# ── IA para o professor ───────────────────────────────
def test_professor_tem_o_seu_limite_e_nao_gasta_o_dos_alunos(monkeypatch):
    from modules import ia
    monkeypatch.setenv("IA_LIMITE_PROFESSOR", "2")
    _responder(monkeypatch, "A turma está bem no geral, mas falha em overfitting. Revê esse conceito.")
    dados = {"alunos": 20, "activos": 12}
    ia.resumo_turma(dados, "prof1")
    ia.resumo_turma(dados, "prof1")
    with pytest.raises(ia.IAErro):
        ia.resumo_turma(dados, "prof1")
    assert ia.usos_hoje("p:prof1") == 2 and ia.usos_hoje("u:prof1") == 0 and ia.usos_hoje("prof1") == 0


def test_resumo_e_aula_so_levam_numeros_e_texto_das_licoes(monkeypatch):
    import pandas as pd
    from modules.professor import _dados_turma
    from modules import ia
    df = pd.DataFrame([{"Nome": "Ana Silva", "Utilizador": "ana", "Pontos": 10, "_licoes": 3, "Estado": "Activo"},
                       {"Nome": "Rui Costa", "Utilizador": "rui", "Pontos": 0, "_licoes": 0, "Estado": "Nunca entrou"},
                       {"Nome": "Eva Neto", "Utilizador": "eva", "Pontos": 5, "_licoes": 1, "Estado": "Parado há 9 dias"}])
    dados = _dados_turma(df, [("overfitting", 7)], 14)
    assert dados["alunos"] == 3 and dados["activos"] == 1 and dados["nunca_entraram"] == 1 and dados["parados_ha_mais_de_7_dias"] == 1
    enviados = []
    _responder(monkeypatch, "A turma precisa de reforço em overfitting. Faz um exemplo no quadro.", enviados=enviados)
    ia.resumo_turma(dados, "prof1")
    ia.sugerir_aula({"titulo": "Overfitting", "ideia": "decorar", "resumo": ["a", "b"]}, 7, 3, "prof1")
    enviado = json.dumps([e[1] for e in enviados], ensure_ascii=False)
    for nome in ("Ana", "Rui", "Eva", "Silva", "Costa", "prof1"):
        assert nome not in enviado


def test_exercicios_so_ficam_os_validos(monkeypatch):
    from modules import ia
    bons = {"pergunta": "O que é overfitting?", "opcoes": ["Decorar os dados", "Aprender bem", "Apagar dados", "Mudar de modelo"],
            "certa": 0, "explicacao": "O modelo decora."}
    maus = [{"pergunta": "curta", "opcoes": ["a", "b", "c", "d"], "certa": 0},
            dict(bons, opcoes=["a", "a", "b", "c"]),                              # opções repetidas
            dict(bons, opcoes=["a", "b", "c"]),                                   # só 3 opções
            dict(bons, certa=4), dict(bons, certa=None), dict(bons, certa=True), "texto solto"]
    _responder(monkeypatch, json.dumps({"exercicios": maus + [dict(bons, certa="2")]}))
    lista = ia.gerar_exercicios("overfitting", 5, "prof1")
    assert len(lista) == 1 and lista[0]["certa"] == 2                             # «2» em texto é aceite; o resto é descartado
    _responder(monkeypatch, json.dumps({"exercicios": maus}))
    with pytest.raises(ia.IAErro):
        ia.gerar_exercicios("overfitting", 3, "prof1")
    with pytest.raises(ia.IAErro):
        ia.gerar_exercicios("", 3, "prof1")


def test_mensagem_nao_leva_o_nome_e_o_servidor_preenche_ou_retira():
    from modules import ia
    assert ia.aplicar_nome("Olá {nome}, tens lições por fazer.", "Ana") == "Olá Ana, tens lições por fazer."
    assert ia.aplicar_nome("Olá {nome}, tens lições por fazer.") == "Olá, tens lições por fazer."
    r = ia.aplicar_nome("{nome}, estás atrasado.")
    assert r == "Estás atrasado." and "{nome}" not in ia.aplicar_nome("Boa, {nome}!")


def test_mensagem_envia_so_estado_e_limpa_a_nota(monkeypatch):
    from modules import ia
    enviados = []
    _responder(monkeypatch, "Olá {nome}, já fizeste 3 lições. Continua assim e fala comigo se precisares!", enviados=enviados)
    texto = ia.redigir_mensagem("aluno", "encorajar", {"estado": "Activo", "licoes_concluidas": 3}, "contacta ana@exemplo.com ou 923456789", "prof1")
    assert "{nome}" in texto
    enviado = json.dumps(enviados[0][1], ensure_ascii=False)
    assert "ana@exemplo.com" not in enviado and "923456789" not in enviado and "[removido]" in enviado
    with pytest.raises(ia.IAErro):
        ia.redigir_mensagem("aluno", "tom_inventado", {}, "", "prof1")
    with pytest.raises(ia.IAErro):
        ia.redigir_mensagem("todos", "encorajar", {}, "", "prof1")


def test_feedback_retira_emails_e_telefones_e_usa_so_criterios_conhecidos(monkeypatch):
    from modules import ia
    enviados = []
    _responder(monkeypatch, "Percebeu o problema: bom trabalho a descrever os dados. Avaliação e conclusão: falta uma conclusão clara.", enviados=enviados)
    texto = "O meu projecto prevê preços de casas com regressão. Contacto: ana@exemplo.com, tel +244 923 456 789. " * 2
    r = ia.rascunho_feedback(texto, ["Avaliação e conclusão", "critério inventado"], "prof1")
    assert len(r) > 20
    corpo = json.loads(enviados[0][1]["messages"][1]["content"])
    assert corpo["criterios"] == ["Avaliação e conclusão"]
    assert "ana@exemplo.com" not in corpo["texto_do_aluno"] and "923 456 789" not in corpo["texto_do_aluno"]
    with pytest.raises(ia.IAErro):
        ia.rascunho_feedback("curto", [], "prof1")


# ── Explicações padrão (sem IA) ───────────────────────
def test_explicacoes_padrao_vem_do_glossario_e_nao_usam_ia():
    from modules.tutor_ia import explicacoes_padrao
    nomes = [n for n, _ in explicacoes_padrao("Accuracy (taxa de acerto): 91.0%\nF1: 90.0%\nValidação cruzada (5 provas): média 90.0%, variação ±1.0%")]
    assert {"Accuracy", "F1-Score", "Média CV", "Desvio Padrão"} <= set(nomes) and "RMSE" not in nomes
    assert [n for n, _ in explicacoes_padrao("R²: 0.812\nRMSE: 3.2")] == ["R²", "RMSE"]
    assert explicacoes_padrao("sem métricas aqui") == []


def test_resposta_cortada_acaba_na_ultima_frase_completa():
    from modules import ia
    t = "Classificar é descobrir a que grupo algo pertence. O algoritmo KNN pergunta aos vizinhos. O algoritmo **"
    assert ia._cortar_na_frase(t).endswith("vizinhos.")
    assert ia._cortar_na_frase("Texto completo.") == "Texto completo."


def test_gemini_pede_pouco_pensamento_e_repete_sem_o_parametro(monkeypatch):
    from modules import ia
    monkeypatch.delenv("XAI_API_KEY")
    monkeypatch.setenv("IA_API_KEY", "k")
    monkeypatch.setenv("IA_PROVEDOR", "gemini")
    corpos = []

    def falso(url, json=None, headers=None, timeout=None):
        corpos.append(dict(json))
        return _Resp("{}", 400) if "reasoning_effort" in json else _Resp("ok", 200)
    monkeypatch.setattr(ia.requests, "post", falso)
    assert ia.chamar([{"role": "user", "content": "x"}], "e1") == "ok"
    assert corpos[0]["reasoning_effort"] == "minimal" and "reasoning_effort" not in corpos[1]


def test_tutor_recebe_o_estado_do_grafico(monkeypatch):
    from modules import ia
    enviados = []
    _responder(monkeypatch, "resposta curta ok", enviados=enviados)
    ia.perguntar_tutor("explica o gráfico", "t", "c", [], "ana", estado="K=5, 4 dizem Compra")
    assert "K=5, 4 dizem Compra" in enviados[0][1]["messages"][0]["content"]
