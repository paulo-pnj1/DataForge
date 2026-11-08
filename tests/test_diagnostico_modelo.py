"""Secção «Diagnóstico do Modelo»: lógica (Overfitting / Underfitting / Boa Generalização), modelos reais e página."""
import os
import re
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.diagnostico_modelo import (BOA, INCONCLUSIVO, OVERFITTING, UNDERFITTING, diagnosticar_classificacao,  # noqa: E402
                                        diagnosticar_modelo, metricas_treino_teste, metricas_treino_teste_clf)


# ── regras de decisão ─────────────────────────────────────────────────────────
def test_boa_generalizacao():
    d = diagnosticar_modelo(0.92, 0.90, 1.0, 1.1, 800, 200)
    assert d["estado"] == BOA and d["rotulo"] == "Boa Generalização" and d["nivel"] == "ok"
    assert d["dif_r2"] == pytest.approx(0.02)
    assert d["dif_mae"] == pytest.approx(0.1) and d["dif_mae_pct"] == pytest.approx(0.1)


def test_overfitting_por_diferenca_de_r2():
    d = diagnosticar_modelo(0.99, 0.70, 0.5, 2.0, 800, 200)
    assert d["estado"] == OVERFITTING and d["rotulo"] == "Overfitting"
    assert "memoriz" in d["titulo"].lower()
    assert any("MAE" in p for p in d["partes"])                 # o erro sobe de 0.5 para 2.0 e isso é dito


def test_overfitting_pela_regra_do_mae():
    # R² desce 0,12 (abaixo de 0,15), mas o erro de teste é 2× o de treino
    d = diagnosticar_modelo(0.95, 0.83, 1.0, 2.0, 800, 200)
    assert d["estado"] == OVERFITTING


def test_erro_de_treino_minusculo_com_bom_r2_de_teste_nao_e_overfitting():
    # caso real: Random Forest nas casas de Luanda (R² 0,989 -> 0,916; MAE 2,5 -> 6,8) é um bom modelo, não memorização
    d = diagnosticar_modelo(0.9891, 0.9156, 2.5358, 6.7643, 320, 80)
    assert d["estado"] == BOA
    assert any("MAE" in p and "acompanhar" in p for p in d["partes"])        # e a subida do erro é explicada


def test_mae_a_subir_sozinho_nao_chega():
    d = diagnosticar_modelo(0.90, 0.89, 1.0, 2.0, 800, 200)    # R² quase igual: não é overfitting
    assert d["estado"] == BOA


def test_underfitting():
    d = diagnosticar_modelo(0.20, 0.18, 5.0, 5.2, 800, 200)
    assert d["estado"] == UNDERFITTING and d["rotulo"] == "Underfitting"


def test_overfitting_tem_prioridade_sobre_underfitting():
    d = diagnosticar_modelo(0.45, 0.05, 1.0, 3.0, 800, 200)    # fraco no treino mas muito pior no teste
    assert d["estado"] == OVERFITTING


def test_r2_de_teste_negativo_e_overfitting_se_treino_bom():
    assert diagnosticar_modelo(0.9, -0.5, 1.0, 4.0, 800, 200)["estado"] == OVERFITTING


@pytest.mark.parametrize("args", [(None, 0.5, 1, 1), (0.5, float("nan"), 1, 1), (0.5, 0.5, float("inf"), 1), ("x", 0.5, 1, 1)])
def test_valores_invalidos_dao_inconclusivo(args):
    d = diagnosticar_modelo(*args)
    assert d["estado"] == INCONCLUSIVO and d["partes"] and d["dif_r2"] is None


def test_mae_de_treino_zero_nao_rebenta():
    d = diagnosticar_modelo(1.0, 0.9, 0.0, 0.5, 800, 200)
    assert d["dif_mae_pct"] is None and d["estado"] in (BOA, OVERFITTING)


def test_poucos_dados_geram_aviso_mas_continuam_a_diagnosticar():
    d = diagnosticar_modelo(0.9, 0.88, 1.0, 1.1, 20, 10)
    assert d["estado"] == BOA and len(d["avisos"]) == 2


# ── modelos reais ─────────────────────────────────────────────────────────────
def _dividir(X, y):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.25, random_state=42)


def _diag(modelo, X, y):
    Xtr, Xte, ytr, yte = _dividir(X, y)
    modelo.fit(Xtr, ytr)
    m = metricas_treino_teste(ytr, modelo.predict(Xtr), yte, modelo.predict(Xte))
    return m, diagnosticar_modelo(m["r2_treino"], m["r2_teste"], m["mae_treino"], m["mae_teste"], m["n_treino"], m["n_teste"])


def test_arvore_sem_limite_em_dados_com_ruido_faz_overfitting():
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X[:, 0] + rng.normal(scale=1.5, size=300)               # muito ruído: uma árvore profunda decora-o
    m, d = _diag(DecisionTreeRegressor(random_state=0), X, y)
    assert m["r2_treino"] > 0.99 and d["estado"] == OVERFITTING


def test_regressao_linear_em_dados_lineares_generaliza():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.3, size=400)
    assert _diag(LinearRegression(), X, y)[1]["estado"] == BOA


def test_regressao_linear_em_relacao_nao_linear_faz_underfitting():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(2)
    X = rng.uniform(-3, 3, size=(400, 1))
    y = X[:, 0] ** 2 + rng.normal(scale=0.2, size=400)          # parábola: uma recta não a apanha
    m, d = _diag(LinearRegression(), X, y)
    assert m["r2_treino"] < 0.5 and d["estado"] == UNDERFITTING


# ── página de Regressão e idioma ─────────────────────────────────────────────
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(autouse=True)
def pasta_temporaria(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


def _treinar_na_pagina(pagina, lang):
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=180)
    at.session_state["authentication_status"] = True
    at.session_state["username"] = "aluno"
    at.session_state["name"] = "Aluno"
    at.session_state["pagina"] = pagina
    at.session_state["lang"] = lang
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    botao = next(b for b in at.button if (b.label or "").startswith(("Treinar", "Train")))
    botao.click()
    return at.run()


def _textos(at):
    out = []
    for grupo in (at.markdown, at.caption, at.info, at.success, at.warning, at.error):
        out += [e.value for e in grupo]
    out += [m.label or "" for m in at.metric] + [str(m.value) for m in at.metric]
    return " ".join(re.sub(r"<[^>]+>", " ", t) for t in out)


def test_pagina_de_regressao_mostra_a_seccao_em_portugues():
    at = _treinar_na_pagina("regressao", "pt")
    assert not at.exception, [e.value for e in at.exception]
    txt = _textos(at)
    for esperado in ("Diagnóstico do Modelo", "R² do treino", "R² do teste", "MAE do treino", "MAE do teste",
                     "Diferença de R²", "Diferença de MAE"):
        assert esperado in txt, esperado
    valores = {m.label: m.value for m in at.metric}
    assert valores["Diagnóstico"] in ("Overfitting", "Underfitting", "Boa Generalização")


def _secao_em(lang, r2_tr, r2_te, mae_tr, mae_te):
    """Desenha só a secção (sem radios do resto da página: o AppTest não consegue voltar a correr páginas em inglês com radios)."""
    def script():
        import streamlit as st
        from modules.i18n import iniciar_idioma
        from modules.i18n_patch import instalar
        instalar()
        iniciar_idioma()
        from modules.diagnostico_modelo import mostrar_diagnostico_modelo
        a = st.session_state["_args"]
        mostrar_diagnostico_modelo(*a, 800, 200, alvo="price")

    at = AppTest.from_function(script, default_timeout=60)
    at.session_state["lang"] = lang
    at.session_state["_args"] = (r2_tr, r2_te, mae_tr, mae_te)
    return at.run()


@pytest.mark.parametrize("args,rotulo", [((0.99, 0.60, 0.5, 2.0), "Overfitting"), ((0.20, 0.18, 5.0, 5.2), "Underfitting"),
                                          ((0.92, 0.90, 1.0, 1.1), "Good Generalization")])
def test_seccao_em_ingles_nao_deixa_portugues(args, rotulo):
    at = _secao_em("en", *args)
    assert not at.exception, [e.value for e in at.exception]
    txt = _textos(at)
    for esperado in ("Model Diagnosis", "R² (train)", "R² (test)", "MAE (train)", "MAE (test)", "R² difference", "MAE difference"):
        assert esperado in txt, esperado
    assert {m.label: m.value for m in at.metric}["Diagnosis"] == rotulo
    for portugues in ("Diagnóstico", "do treino", "do teste", "memoriz", "modelo", "dados de", "Diferença"):
        assert portugues not in txt, (portugues, txt)


def test_seccao_em_portugues_com_todos_os_indicadores():
    at = _secao_em("pt", 0.99, 0.60, 0.5, 2.0)
    assert not at.exception, [e.value for e in at.exception]
    rot = {m.label: m.value for m in at.metric}
    assert rot["R² do treino"] == "0.9900" and rot["R² do teste"] == "0.6000"
    assert rot["MAE do treino"] == "0.5000" and rot["MAE do teste"] == "2.0000"
    assert rot["Diferença de R² (treino − teste)"] == "+0.3900" and rot["Diagnóstico"] == "Overfitting"


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO (Accuracy e F1) e COMPARADOR
# ══════════════════════════════════════════════════════════════════════════════
def test_clf_boa_generalizacao():
    d = diagnosticar_classificacao(0.92, 0.90, 0.92, 0.90, 800, 200)
    assert d["estado"] == BOA and d["rotulo"] == "Boa Generalização" and d["nivel"] == "ok"
    assert d["dif_acc"] == pytest.approx(0.02) and d["dif_f1"] == pytest.approx(0.02)


def test_clf_overfitting_pela_accuracy():
    d = diagnosticar_classificacao(1.0, 0.70, 1.0, 0.69, 800, 200)
    assert d["estado"] == OVERFITTING and "memoriz" in d["titulo"].lower()
    assert any("F1" in p for p in d["partes"])                    # o F1 também desce e isso é dito


def test_clf_overfitting_so_pelo_f1():
    # Accuracy quase igual, mas o F1 cai 19 pontos (classes desequilibradas)
    assert diagnosticar_classificacao(0.90, 0.88, 0.99, 0.80, 800, 200)["estado"] == OVERFITTING


def test_clf_limite_do_overfitting_e_15_pontos():
    d14 = diagnosticar_classificacao(0.95, 0.81, 0.95, 0.81, 800, 200)       # perde 14 pontos: ainda dentro do limite
    assert d14["estado"] == BOA and d14["dif_acc"] == pytest.approx(0.14)
    assert diagnosticar_classificacao(0.95, 0.79, 0.95, 0.79, 800, 200)["estado"] == OVERFITTING    # perde 16 pontos


def test_clf_underfitting():
    d = diagnosticar_classificacao(0.50, 0.48, 0.45, 0.44, 800, 200)
    assert d["estado"] == UNDERFITTING and d["rotulo"] == "Underfitting"


def test_clf_overfitting_tem_prioridade_sobre_underfitting():
    assert diagnosticar_classificacao(0.55, 0.30, 0.55, 0.28, 800, 200)["estado"] == OVERFITTING


def test_clf_accuracy_muito_acima_do_f1_avisa_de_classes_desequilibradas():
    d = diagnosticar_classificacao(0.93, 0.92, 0.80, 0.78, 800, 200)
    assert d["estado"] == BOA and any("desequilibradas" in a for a in d["avisos"])
    assert not diagnosticar_classificacao(0.93, 0.92, 0.93, 0.91, 800, 200)["avisos"]


@pytest.mark.parametrize("args", [(None, 0.5, 0.5, 0.5), (0.5, float("nan"), 0.5, 0.5), (0.5, 0.5, float("inf"), 0.5), ("x", 0.5, 0.5, 0.5)])
def test_clf_valores_invalidos_dao_inconclusivo(args):
    d = diagnosticar_classificacao(*args)
    assert d["estado"] == INCONCLUSIVO and d["partes"] and d["dif_acc"] is None and d["acc_treino"] is None


def test_clf_poucos_dados_geram_aviso_mas_continuam_a_diagnosticar():
    d = diagnosticar_classificacao(0.9, 0.88, 0.9, 0.88, 20, 10)
    assert d["estado"] == BOA and len(d["avisos"]) == 2


def _diag_clf(modelo, X, y):
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    modelo.fit(Xtr, ytr)
    m = metricas_treino_teste_clf(ytr, modelo.predict(Xtr), yte, modelo.predict(Xte))
    return m, diagnosticar_classificacao(m["acc_treino"], m["acc_teste"], m["f1_treino"], m["f1_teste"], m["n_treino"], m["n_teste"])


def test_clf_metricas_treino_teste_tem_os_seis_campos():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 2))
    m, _ = _diag_clf(LogisticRegression(), X, (X[:, 0] > 0).astype(int))
    assert set(m) == {"acc_treino", "acc_teste", "f1_treino", "f1_teste", "n_treino", "n_teste"}
    assert m["n_treino"] == 150 and m["n_teste"] == 50 and all(0 <= m[k] <= 1 for k in ("acc_treino", "acc_teste", "f1_treino", "f1_teste"))


def test_clf_arvore_sem_limite_em_dados_com_ruido_faz_overfitting():
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = ((X[:, 0] + rng.normal(scale=1.5, size=300)) > 0).astype(int)      # muito ruído: uma árvore profunda decora-o
    m, d = _diag_clf(DecisionTreeClassifier(random_state=0), X, y)
    assert m["acc_treino"] == 1.0 and d["estado"] == OVERFITTING


def test_clf_regressao_logistica_em_dados_separaveis_generaliza():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    y = (3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.3, size=400) > 0).astype(int)
    assert _diag_clf(LogisticRegression(), X, y)[1]["estado"] == BOA


def test_clf_modelo_linear_em_dados_xor_faz_underfitting():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(2)
    X = rng.uniform(-3, 3, size=(400, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)             # XOR: uma recta não separa as classes
    m, d = _diag_clf(LogisticRegression(), X, y)
    assert m["acc_treino"] < 0.6 and d["estado"] == UNDERFITTING


# ── página de Classificação, Comparador e idioma ─────────────────────────────
def test_pagina_de_classificacao_mostra_a_seccao_em_portugues():
    at = _treinar_na_pagina("classificacao", "pt")
    assert not at.exception, [e.value for e in at.exception]
    txt = _textos(at)
    for esperado in ("Diagnóstico do Modelo", "Accuracy do treino", "Accuracy do teste", "F1 do treino", "F1 do teste",
                     "Diferença de Accuracy", "Diferença de F1"):
        assert esperado in txt, esperado
    valores = {m.label: m.value for m in at.metric}
    assert valores["Diagnóstico"] in ("Overfitting", "Underfitting", "Boa Generalização")
    assert valores["Accuracy do treino"].endswith("%") and valores["Diferença de Accuracy (treino − teste)"].endswith("p.p.")


def _comparar_na_pagina():
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=240)
    at.session_state["authentication_status"] = True
    at.session_state["username"] = "aluno"
    at.session_state["name"] = "Aluno"
    at.session_state["pagina"] = "comparador"
    at.session_state["lang"] = "pt"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    next(b for b in at.button if (b.label or "").startswith("Comparar")).click()
    return at.run()


def test_comparador_mostra_a_seccao_para_cada_algoritmo():
    at = _comparar_na_pagina()
    assert not at.exception, [e.value for e in at.exception]
    res = at.session_state["cmp_result"]
    assert set(res["diag"]) == set(res["algos"]) and len(res["algos"]) >= 2
    assert set(res["diag"][res["algos"][0]]) == {"acc_treino", "acc_teste", "f1_treino", "f1_teste", "n_treino", "n_teste"}
    assert "Diagnóstico do Modelo" in _textos(at)
    rotulos = [m.value for m in at.metric if m.label == "Diagnóstico"]
    assert len(rotulos) == len(res["algos"]) and all(r in ("Overfitting", "Underfitting", "Boa Generalização") for r in rotulos)
    colunas = {c for d in at.dataframe for c in d.value.columns}
    assert {"Accuracy do treino", "Accuracy do teste", "F1 do treino", "F1 do teste", "Diagnóstico"} <= colunas


def test_comparador_sem_diag_guardado_nao_rebenta():
    """Comparações guardadas antes desta secção existir não têm «diag»: a página mostra o resto na mesma."""
    def script():
        import streamlit as st
        from modules.comparador import _diagnosticos, _secao_diagnostico
        res = {"tipo": "Classificação", "algos": ["KNN"], "alvo": "y"}
        _secao_diagnostico(res, _diagnosticos(res))
        st.write("fim")
    at = AppTest.from_function(script, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Diagnóstico do Modelo" not in _textos(at)


def _clf_em(lang, acc_tr, acc_te, f1_tr, f1_te):
    def script():
        import streamlit as st
        from modules.i18n import iniciar_idioma
        from modules.i18n_patch import instalar
        instalar()
        iniciar_idioma()
        from modules.diagnostico_modelo import mostrar_diagnostico_classificacao
        mostrar_diagnostico_classificacao(*st.session_state["_args"], 800, 200)

    at = AppTest.from_function(script, default_timeout=60)
    at.session_state["lang"] = lang
    at.session_state["_args"] = (acc_tr, acc_te, f1_tr, f1_te)
    return at.run()


@pytest.mark.parametrize("args,rotulo", [((0.99, 0.60, 0.99, 0.58), "Overfitting"), ((0.50, 0.48, 0.45, 0.44), "Underfitting"),
                                          ((0.92, 0.90, 0.92, 0.90), "Good Generalization"), ((0.93, 0.92, 0.80, 0.78), "Good Generalization")])
def test_clf_seccao_em_ingles_nao_deixa_portugues(args, rotulo):
    at = _clf_em("en", *args)
    assert not at.exception, [e.value for e in at.exception]
    txt = _textos(at)
    for esperado in ("Model Diagnosis", "Accuracy (train)", "Accuracy (test)", "F1 (train)", "F1 (test)",
                     "Accuracy difference", "F1 difference"):
        assert esperado in txt, esperado
    assert {m.label: m.value for m in at.metric}["Diagnosis"] == rotulo
    for portugues in ("Diagnóstico", "do treino", "do teste", "memoriz", "modelo", "dados de", "Diferença", "pontos percentuais", "desequilibr"):
        assert portugues not in txt, (portugues, txt)


@pytest.mark.parametrize("tipo,diag", [
    ("Classificação", {"KNN": {"acc_treino": .88, "acc_teste": .76, "f1_treino": .88, "f1_teste": .76, "n_treino": 400, "n_teste": 100},
                       "Random Forest": {"acc_treino": 1., "acc_teste": .78, "f1_treino": 1., "f1_teste": .78, "n_treino": 400, "n_teste": 100}}),
    ("Regressão", {"Ridge": {"r2_treino": .9, "r2_teste": .88, "mae_treino": 1., "mae_teste": 1.1, "n_treino": 400, "n_teste": 100},
                   "Random Forest": {"r2_treino": .99, "r2_teste": .7, "mae_treino": .5, "mae_teste": 2., "n_treino": 400, "n_teste": 100}})])
def test_comparador_seccao_em_ingles_nao_deixa_portugues(tipo, diag):
    def script():
        import streamlit as st
        from modules.i18n import iniciar_idioma
        from modules.i18n_patch import instalar
        instalar()
        iniciar_idioma()
        from modules.comparador import _diagnosticos, _secao_diagnostico
        res = st.session_state["_res"]
        _secao_diagnostico(res, _diagnosticos(res))

    at = AppTest.from_function(script, default_timeout=60)
    at.session_state["lang"] = "en"
    at.session_state["_res"] = {"tipo": tipo, "algos": list(diag), "alvo": "y", "diag": diag}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    txt = _textos(at) + " " + " ".join(str(c) for d in at.dataframe for c in d.value.columns) \
        + " " + " ".join(str(v) for d in at.dataframe for v in d.value.to_numpy().ravel())
    assert "Model Diagnosis" in txt and "Pick an algorithm" in txt
    assert [m.value for m in at.metric if m.label == "Diagnosis"] == ["Good Generalization", "Overfitting"]
    for portugues in ("Diagnóstico", "do treino", "do teste", "memoriz", "Diferença", "Escolhe", "Algoritmo", "Compara "):
        assert portugues not in txt, (portugues, txt)
