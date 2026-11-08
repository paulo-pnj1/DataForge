"""Ciclo de ML (laboratório de avaliação): avaliação, diagnóstico, optimização, exportação, previsão e monitorização."""
import ast
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import ciclo_ml as cl  # noqa: E402
from modules.ml_engine import model_persistence as mp  # noqa: E402
from modules.ml_engine import model_registry as reg  # noqa: E402
from modules.ml_engine.model_comparison import comparar_modelos  # noqa: E402
from modules.ml_engine.model_training import treinar_modelo  # noqa: E402
from modules.ml_engine.preprocessing import preparar_dados  # noqa: E402
from _ml_dados import tabela_classificacao, tabela_multiclasse, tabela_regressao  # noqa: E402


@pytest.fixture
def clf():
    d = preparar_dados(tabela_classificacao(300), "classe", "classificacao")
    return d, treinar_modelo("rf_clf", d, cv_folds=3)


@pytest.fixture
def rg():
    d = preparar_dados(tabela_regressao(300), "y", "regressao")
    return d, treinar_modelo("ridge", d, cv_folds=3)


# ── Etapa 1 ──
def test_cartoes_e_interpretacao_com_numeros_reais(clf, rg):
    d, r = clf
    for k, _ in cl.metricas_cartoes("classificacao"):
        assert r.metricas[k] is not None
        assert cl.interpretar_metrica("classificacao", k, r.metricas[k], d)
    assert [k for k, _ in cl.metricas_cartoes("regressao")] == ["mae", "mse", "rmse", "r2"]
    d2, r2 = rg
    assert "explica" in cl.interpretar_metrica("regressao", "r2", r2.metricas["r2"], d2)
    assert "pior" in cl.interpretar_metrica("regressao", "r2", -0.3, d2)
    assert "não foi possível" in cl.interpretar_metrica("regressao", "r2", None, d2).lower()


def test_accuracy_avisa_quando_a_classe_dominante_inflaciona():
    df = tabela_classificacao(300)
    df["classe"] = np.where(np.arange(len(df)) % 10 == 0, "nao", "sim")
    d = preparar_dados(df, "classe", "classificacao")
    r = treinar_modelo("logistic_reg", d) if "logistic_reg" in [m.id for m in reg.listar("classificacao")] else treinar_modelo("rf_clf", d)
    assert "Cuidado" in cl.interpretar_metrica("classificacao", "accuracy", r.metricas["accuracy"], d)


def test_relatorio_de_classificacao_multiclasse():
    d = preparar_dados(tabela_multiclasse(), "classe", "classificacao")
    r = treinar_modelo("rf_clf", d)
    rep = cl.relatorio_classificacao(d, r)
    classes = [str(c) for c in d.classes]
    assert set(classes) <= set(rep["Classe"])
    assert int(rep[rep["Classe"].isin(classes)]["Exemplos"].sum()) == d.n_teste
    assert "accuracy" in set(rep["Classe"])


def test_relatorio_so_para_classificacao(rg):
    with pytest.raises(cl.ErroCiclo):
        cl.relatorio_classificacao(*rg)


def test_metricas_de_treino_tem_as_mesmas_chaves(clf, rg):
    assert set(cl.avaliar_treino(clf[1], clf[0])) == {"accuracy", "precision", "recall", "f1"}
    assert set(cl.avaliar_treino(rg[1], rg[0])) == {"r2", "mae", "mse", "rmse"}


# ── Etapa 2 ──
def test_diagnostico_sobreajuste_numa_arvore_sem_limite():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    df["y"] = df.a + rng.normal(scale=2.0, size=200)             # muito ruído: uma árvore profunda decora
    d = preparar_dados(df, "y", "regressao")
    r = treinar_modelo("tree_reg", d, {"max_depth": 0, "min_samples_leaf": 1}, cv_folds=3)
    g = cl.diagnosticar_generalizacao(r, d, cl.obter_referencia(d))
    assert g.estado == "sobreajuste" and g.sugestoes
    assert g.numeros["diferenca"] > cl.LIMITE_SOBREAJUSTE


def test_diagnostico_subajuste_quando_o_alvo_e_ruido():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "y": rng.normal(size=300)})
    d = preparar_dados(df, "y", "regressao")
    r = treinar_modelo("linear_reg", d, cv_folds=3)
    g = cl.diagnosticar_generalizacao(r, d, cl.obter_referencia(d))
    assert g.estado == "subajuste"


def test_diagnostico_bom_ajustamento(rg):
    d, r = rg
    g = cl.diagnosticar_generalizacao(r, d, cl.obter_referencia(d))
    assert g.estado == "bom"
    assert g.numeros["acima_da_referencia_teste"] > 0.05


def test_diagnostico_inconclusivo_com_poucos_dados_e_sem_referencia(rg):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=40)})
    df["y"] = df.a + rng.normal(scale=.1, size=40)
    d = preparar_dados(df, "y", "regressao")                    # 8 exemplos de teste
    r = treinar_modelo("linear_reg", d)
    g = cl.diagnosticar_generalizacao(r, d, cl.obter_referencia(d))
    assert g.estado == "inconclusivo" and g.avisos
    d2, r2 = rg
    assert cl.diagnosticar_generalizacao(r2, d2, None).estado == "inconclusivo"     # sem referência não afirma «bom»


def test_curva_de_aprendizagem_nao_usa_o_teste(rg):
    d, r = rg
    c = cl.curva_aprendizagem(r, d, folds=3, pontos=4)
    assert list(c.columns[:3]) == ["exemplos", "treino", "validacao"]
    assert c["exemplos"].max() <= d.n_treino * 2 / 3 + 1       # cada bloco treina com uma parte do TREINO
    assert c["exemplos"].is_monotonic_increasing


# ── Etapa 3 ──
def test_melhor_por_metrica_e_grafico():
    d = preparar_dados(tabela_classificacao(300), "classe", "classificacao")
    comp = comparar_modelos(["rf_clf", "tree_clf", "knn_clf"], d, cv_folds=3)
    t = cl.melhor_por_metrica(comp)
    assert set(t["Critério"]) >= {"F1-score", "Accuracy"}
    assert isinstance(cl.divergem(t), bool)
    g = cl.dados_grafico_comparacao(comp, "f1")
    assert len(g) == len(comp.validos()) and g["tempo"].notna().all()


# ── Etapa 4 ──
def test_espaco_padrao_respeita_limites():
    for info in reg.listar("regressao") + reg.listar("classificacao"):
        for nome, vals in cl.espaco_padrao(info).items():
            s = info.parametros[nome]
            assert vals
            if s["tipo"] != "choice":
                assert all((s["min"] is None or v >= s["min"]) and (s["max"] is None or v <= s["max"]) for v in vals), (info.id, nome)


def test_validar_espaco_aceita_e_recusa():
    info = reg.obter("ridge")
    e, erros = cl.validar_espaco(info, {"alpha": "0.1, 1; 10"})
    assert e == {"alpha": [0.1, 1.0, 10.0]} and not erros
    e, erros = cl.validar_espaco(info, {"alpha": "abc, 99999"})
    assert not e and len(erros) == 2
    e, erros = cl.validar_espaco(reg.obter("rf_reg"), {"n_estimators": "10.5"})
    assert erros
    e, erros = cl.validar_espaco(reg.obter("svr"), {"kernel": "rbf, linear, banana"})
    assert e == {"kernel": ["rbf", "linear"]} and len(erros) == 1
    assert cl.validar_espaco(info, {"alpha": ""}) == ({}, [])


def test_grid_search_so_usa_o_treino_e_compara_antes_e_depois(rg, monkeypatch):
    d, _ = rg
    vistos = []
    import sklearn.model_selection._search as busca
    original = busca.GridSearchCV.fit

    def fit_espiao(self, X, y=None, **kw):
        vistos.append(len(X))
        return original(self, X, y, **kw)
    monkeypatch.setattr(busca.GridSearchCV, "fit", fit_espiao)
    o = cl.otimizar("ridge", d, {"alpha": [0.01, 1.0, 100.0]}, "grid", cv_folds=3)
    assert vistos == [d.n_treino]                                       # a procura só viu o treino
    assert o.n_testadas == 3 and o.melhores_params["alpha"] in (0.01, 1.0, 100.0)
    assert o.original.ok and o.otimizado.ok and o.conclusao
    assert o.otimizado.pipeline is not o.original.pipeline
    assert set(o.tabela.columns) >= {"alpha", "Nota média (CV)", "Posição"}
    assert o.tabela["Posição"].iloc[0] == 1


def test_random_search_classificacao_e_pipeline_completo(clf):
    d, _ = clf
    o = cl.otimizar("rf_clf", d, {"n_estimators": [10, 30, 60], "max_depth": [0, 3, 6]}, "random", n_iter=4, cv_folds=3)
    assert o.n_combinacoes == 9 and o.n_testadas == 4
    assert o.otimizado.pipeline.steps[0][0] == "prep"                  # pré-processamento dentro do pipeline
    # o modelo final usa os parâmetros pelo mesmo caminho do catálogo (0 = sem limite, etc.)
    assert o.otimizado.params["max_depth"] == o.melhores_params["max_depth"]


def test_otimizar_recusa_pedidos_maus(rg):
    d, _ = rg
    with pytest.raises(cl.ErroCiclo):
        cl.otimizar("ridge", d, {}, "grid")
    with pytest.raises(cl.ErroCiclo):
        cl.otimizar("ridge", d, {"banana": [1]}, "grid")
    with pytest.raises(cl.ErroCiclo):
        cl.otimizar("linear_reg", d, {"x": [1]}, "grid")
    with pytest.raises(cl.ErroCiclo):
        cl.otimizar("ridge", d, {"alpha": list(np.linspace(0.1, 50, 80))}, "grid")          # combinações a mais
    with pytest.raises(cl.ErroCiclo):
        cl.otimizar("ridge", d, {"alpha": [1.0, 2.0]}, "grid", criterio="f1")
    with pytest.raises(cl.ErroCiclo, match="demoraria"):
        cl.otimizar("rf_reg", d, {"n_estimators": [200, 400]}, "grid", orcamento_s=0.001)


# ── Etapa 5 ──
def test_meta_ciclo_tem_os_oito_itens_e_e_json(clf, tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda nome: None)
    d, r = clf
    meta = cl.meta_ciclo(d, r, "Meu modelo")
    json.dumps(meta)                                                    # sem objectos estranhos
    info = mp.guardar_modelo("ana", r.pipeline, meta, raiz=tmp_path)
    legivel = dict(cl.metadados_legiveis(info))
    assert set(legivel) == {"Nome do modelo", "Algoritmo", "Data do treino", "Métricas de avaliação (teste)", "Hiperparâmetros",
                            "Características utilizadas", "Tipo de problema", "Versão do modelo"}
    assert legivel["Versão do modelo"] == 1 and legivel["Características utilizadas"] == d.features
    assert "matriz_confusao" not in legivel["Métricas de avaliação (teste)"]
    info2 = mp.guardar_modelo("ana", r.pipeline, cl.meta_ciclo(d, r, "Meu modelo"), raiz=tmp_path)
    assert info2["versao"] == 2


def test_exportar_pacote_zip_reabre_e_preve_igual(clf):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "Exp")
    z = zipfile.ZipFile(io.BytesIO(cl.exportar_pacote(r.pipeline, meta)))
    assert set(z.namelist()) == {"modelo.joblib", "metadados.json", "usar_modelo.py", "LEEME.txt"}
    import joblib
    pipe = joblib.load(io.BytesIO(z.read("modelo.joblib")))
    assert (pipe.predict(d.X_test) == r.pipeline.predict(d.X_test)).all()
    assert json.loads(z.read("metadados.json"))["features"] == d.features


# ── Etapa 6 ──
def test_playground_csv_modelo_e_validacao(clf):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "P")
    modelo_csv = pd.read_csv(io.BytesIO(cl.csv_modelo(meta)))
    assert list(modelo_csv.columns) == d.features
    saida, rel = cl.prever_ficheiro(r.pipeline, meta, modelo_csv)
    assert f"previsao_{d.alvo}" in saida.columns and len(saida) == 1
    with pytest.raises(cl.ErroCiclo, match="Faltam colunas"):
        cl.prever_ficheiro(r.pipeline, meta, modelo_csv.drop(columns=[d.features[0]]))
    duplicado = pd.concat([modelo_csv, modelo_csv[[d.features[0]]]], axis=1)
    with pytest.raises(cl.ErroCiclo, match="mesmo nome"):
        cl.prever_ficheiro(r.pipeline, meta, duplicado)
    texto = pd.concat([modelo_csv, modelo_csv], ignore_index=True)
    texto[d.num_cols[0]] = texto[d.num_cols[0]].astype(object)
    texto.loc[0, d.num_cols[0]] = "abc"
    _, rel = cl.prever_ficheiro(r.pipeline, meta, texto)
    assert any("não são números" in a for a in rel["avisos"])


# ── Etapa 7: o código gerado compila E funciona ──
def test_codigo_gerado_compila(clf):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "C")
    for tipo in ("uso", "fastapi", "streamlit", "cliente"):
        ast.parse(cl.gerar_codigo(meta, tipo))
    assert "@@" not in cl.gerar_codigo(meta, "fastapi")
    with pytest.raises(cl.ErroCiclo):
        cl.gerar_codigo({**meta, "tipo": "serie"}, "uso")


def test_codigo_de_uso_da_as_mesmas_previsoes_que_a_plataforma(tmp_path):
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"a": rng.normal(size=n), "cat": rng.choice(["x", "y", "z"], n),
                       "data": pd.date_range("2023-01-01", periods=n, freq="D")})
    df["classe"] = np.where(df.a + (df.cat == "x") + rng.normal(scale=.3, size=n) > .5, "sim", "nao")
    d = preparar_dados(df, "classe", "classificacao")
    r = treinar_modelo("rf_clf", d)
    meta = cl.meta_ciclo(d, r, "Gerado")
    (tmp_path / "modelo.joblib").write_bytes(cl.exportar_joblib(r.pipeline))
    (tmp_path / "usar.py").write_text(cl.gerar_codigo(meta, "uso") + "\n", encoding="utf-8")
    novos = df.drop(columns=["classe"]).head(25)
    novos.to_csv(tmp_path / "novos.csv", index=False)
    teste = ("import pandas as pd, json, runpy\nns = runpy.run_path('usar.py', run_name='x')\n"
             "print(json.dumps(ns['prever'](pd.read_csv('novos.csv'))))\n")
    (tmp_path / "t.py").write_text(teste)
    saida = subprocess.run([sys.executable, "t.py"], cwd=tmp_path, capture_output=True, text=True)
    assert saida.returncode == 0, saida.stderr[-800:]
    de_fora = json.loads(saida.stdout.strip().splitlines()[-1])
    import joblib
    esperado, _ = cl.prever_ficheiro(joblib.load(tmp_path / "modelo.joblib"), meta, pd.read_csv(tmp_path / "novos.csv"))
    assert de_fora == esperado[f"previsao_{d.alvo}"].tolist()


# ── Etapa 8 ──
def test_psi_e_niveis():
    p = [.25, .25, .25, .25]
    assert cl.psi(p, p) == 0 or abs(cl.psi(p, p)) < 1e-9
    assert cl.nivel_deriva(0.05) == "estável" and cl.nivel_deriva(0.15) == "moderada" and cl.nivel_deriva(0.5) == "forte"
    assert cl.psi(p, [.7, .1, .1, .1]) > 0.25


def test_deriva_distingue_dados_iguais_de_dados_deslocados(rg):
    d, r = rg
    meta = cl.meta_ciclo(d, r, "M")
    iguais = cl.analisar_deriva(meta, d.X_train.sample(150, random_state=0, replace=True))
    assert (iguais["variaveis"]["Nível"] == "estável").all()
    desloc = d.X_train.copy()
    desloc["a"] = desloc["a"] + 5
    desloc["cat"] = "nunca_vista"
    res = cl.analisar_deriva(meta, desloc, previsoes=r.pipeline.predict(desloc))
    niveis = dict(zip(res["variaveis"]["Variável"], res["variaveis"]["Nível"]))
    assert niveis["a"] == "forte" and niveis["cat"] == "forte"
    assert res["previsoes"]["nivel"] == "forte"
    assert not cl.analisar_deriva(meta, desloc.head(10))["variaveis"].empty
    assert any("10 linhas" in a for a in cl.analisar_deriva(meta, desloc.head(10))["avisos"])


def test_deriva_sem_perfil_guardado_diz_o_que_fazer(rg):
    d, r = rg
    meta = cl.meta_ciclo(d, r, "M")
    meta.pop("referencia_monitorizacao")
    with pytest.raises(cl.ErroCiclo, match="guardá-lo"):
        cl.analisar_deriva(meta, d.X_test)


def test_deriva_de_previsoes_em_classificacao(clf):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "M")
    sempre = ["sim"] * 100
    res = cl.analisar_deriva(meta, d.X_test, previsoes=sempre)
    assert res["previsoes"]["nivel"] in ("moderada", "forte")
    assert set(res["previsoes"]["antes"]) == set(d.classes)


def test_metricas_com_valores_reais(clf, rg):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "M")
    reais = [d.classes[i] for i in d.y_test]
    pred = [d.classes[i] for i in r.pipeline.predict(d.X_test)]
    m = cl.metricas_com_valores_reais(meta, pred, reais)
    assert m["metricas"]["accuracy"] == pytest.approx(r.metricas["accuracy"])
    with pytest.raises(cl.ErroCiclo):
        cl.metricas_com_valores_reais(meta, pred, ["classe_estranha"] * len(pred))
    d2, r2 = rg
    meta2 = cl.meta_ciclo(d2, r2, "R")
    m3 = cl.metricas_com_valores_reais(meta2, r2.pipeline.predict(d2.X_test), d2.y_test)
    assert m3["metricas"]["r2"] == pytest.approx(r2.metricas["r2"])
    assert cl.metricas_com_valores_reais(meta2, [1.0, 2.0, 3.0], [1.0, None, 3.5])["n"] == 2


# ── Retreino, alerta de deriva e pacote para publicar ──
def test_retreinar_compara_antigo_e_novo_na_mesma_parte(clf):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "Base")
    assert meta["test_size"] == d.test_size
    out = cl.retreinar(meta, tabela_classificacao(300, seed=7), r.pipeline)
    assert out["resultado"].estado == "ok" and out["principal"] == "f1"
    assert out["antigo"] and set(out["antigo"]) == set(out["novo"])
    assert out["meta"]["origem"] == "retreino com dados novos" and out["meta"]["algoritmo_id"] == meta["algoritmo_id"]
    assert out["meta"]["parametros"] == meta["parametros"]               # mesmos hiperparâmetros


def test_retreinar_pede_a_resposta_real_e_avisa_de_poucos_dados(clf):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "Base")
    with pytest.raises(cl.ErroCiclo, match="Faltam"):
        cl.retreinar(meta, tabela_classificacao(300, seed=7).drop(columns=["classe"]), r.pipeline)
    pequeno = cl.retreinar(meta, tabela_classificacao(60, seed=7), r.pipeline)
    assert any("menos" in a or "linhas utilizáveis" in a for a in pequeno["avisos"])


def test_retreinar_regressao(rg):
    d, r = rg
    out = cl.retreinar(cl.meta_ciclo(d, r, "Reg"), tabela_regressao(300, seed=3), r.pipeline)
    assert out["principal"] == "r2" and out["novo"]["r2"] > 0.8 and out["melhorou"] in (True, False)


def test_alerta_de_deriva_estavel_e_forte(clf):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "Base")
    parecido = tabela_classificacao(300, seed=11).drop(columns=["classe"])
    assert cl.alerta_deriva(meta, parecido)["nivel"] == "estável"
    mudado = parecido.copy()
    mudado["a"] = mudado["a"] + 5
    alerta = cl.alerta_deriva(meta, mudado)
    assert alerta["nivel"] == "forte" and "a" in alerta["fortes"] and "Deriva forte" in alerta["mensagem"]


def test_alerta_de_deriva_sem_perfil_devolve_none(clf):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "Base")
    meta.pop("referencia_monitorizacao")
    assert cl.alerta_deriva(meta, tabela_classificacao(50).drop(columns=["classe"])) is None


def test_pacote_para_publicar_tem_docker_e_versoes_fixas(clf):
    d, r = clf
    meta = cl.meta_ciclo(d, r, "Pub")
    z = zipfile.ZipFile(io.BytesIO(cl.exportar_pacote_deploy(r.pipeline, meta)))
    assert {"modelo.joblib", "metadados.json", "usar_modelo.py", "LEEME.txt", "api.py", "Dockerfile", "requirements.txt",
            ".dockerignore", "PUBLICAR.md"} == set(z.namelist())
    req = z.read("requirements.txt").decode()
    assert f"scikit-learn=={__import__('sklearn').__version__}" in req and "fastapi" in req
    docker = z.read("Dockerfile").decode()
    assert docker.startswith("FROM python:") and "modelo.joblib" in docker and "USER app" in docker and "${PORT}" in docker
    ast.parse(z.read("api.py").decode())                                  # o código gerado é Python válido
    assert "DATAFORGE_API_KEY" in z.read("api.py").decode()
    assert "DATAFORGE_API_KEY" in z.read("PUBLICAR.md").decode()
    assert set(cl.exportar_pacote(r.pipeline, meta) and zipfile.ZipFile(io.BytesIO(cl.exportar_pacote(r.pipeline, meta))).namelist()) \
        == {"modelo.joblib", "metadados.json", "usar_modelo.py", "LEEME.txt"}   # o pacote simples não mudou


def test_api_publicada_exige_a_chave_quando_definida(clf, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    d, r = clf
    meta = cl.meta_ciclo(d, r, "Pub")
    zipfile.ZipFile(io.BytesIO(cl.exportar_pacote_deploy(r.pipeline, meta))).extractall(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("DATAFORGE_API_KEY", "segredo")
    sys.modules.pop("api", None)
    api = __import__("api")
    c = TestClient(api.app)
    caso = cl._exemplo_linha(meta)
    assert c.get("/saude").status_code == 200
    assert c.post("/prever", json={"casos": [caso]}).status_code == 401
    assert c.post("/prever", json={"casos": [caso]}, headers={"X-API-Key": "errada"}).status_code == 401
    ok = c.post("/prever", json={"casos": [caso]}, headers={"X-API-Key": "segredo"})
    assert ok.status_code == 200 and len(ok.json()["previsoes"]) == 1
    sys.modules.pop("api", None)
