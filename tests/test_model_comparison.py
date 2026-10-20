"""Comparação de modelos: tabela real, ordenação, falhas isoladas, limites e recomendação justificada."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402

from modules.ml_engine import model_registry as reg  # noqa: E402
from modules.ml_engine.model_comparison import comparar_modelos, recomendar  # noqa: E402
from modules.ml_engine.preprocessing import preparar_dados  # noqa: E402
from _ml_dados import tabela_classificacao, tabela_regressao  # noqa: E402


class _Partido(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        raise RuntimeError("falha de propósito")

    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture
def modelo_que_falha():
    reg.registar(reg.ModeloInfo("teste_falha", "Modelo que falha", "Teste", "Falha sempre.", ("regressao",), lambda p, rs: _Partido()))
    reg.registar(reg.ModeloInfo("teste_sem_lib", "Precisa de biblioteca", "Teste", "Sem biblioteca.", ("regressao",),
                                lambda p, rs: _Partido(), dependencia="biblioteca_que_nao_existe_xyz"))
    yield
    reg.remover("teste_falha")
    reg.remover("teste_sem_lib")


@pytest.fixture(scope="module")
def dados():
    return preparar_dados(tabela_regressao(), "y", "regressao")


def test_tabela_com_metricas_reais_e_referencia(dados):
    comp = comparar_modelos(["linear_reg", "tree_reg", "ridge"], dados, cv_folds=3)
    tab = comp.tabela_visivel("rmse")
    assert {"Modelo", "R²", "MAE", "RMSE", "Tempo de treino (s)", "Estado"} <= set(tab.columns)
    validos = tab[tab["Estado"] == "OK"]["RMSE"].tolist()
    assert validos == sorted(validos)                                        # ordenada pela métrica escolhida
    assert any("Referência" in e for e in tab["Estado"])
    t = comp.tabela_visivel("r2")["R²"].dropna().tolist()
    assert t[: len(t) - 1] == sorted(t[: len(t) - 1], reverse=True) or t == sorted(t, reverse=True)


def test_falha_de_um_modelo_nao_aborta_os_outros(dados, modelo_que_falha):
    comp = comparar_modelos(["linear_reg", "teste_falha", "teste_sem_lib", "ridge"], dados, cv_folds=0)
    estados = {r.id: r.estado for r in comp.resultados}
    assert estados["linear_reg"] == "ok" and estados["ridge"] == "ok"
    assert estados["teste_falha"] == "falhou" and "falha de propósito" in comp.por_id("teste_falha").motivo
    assert estados["teste_sem_lib"] == "indisponivel"
    assert len(comp.falhas()) == 2


def test_limite_de_modelos_e_de_tempo(dados):
    comp = comparar_modelos(["linear_reg", "ridge", "tree_reg"], dados, cv_folds=0, max_modelos=1)
    assert [r.estado for r in comp.resultados] == ["ok", "ignorado", "ignorado"]
    comp = comparar_modelos(["linear_reg", "ridge"], dados, cv_folds=0, orcamento_s=-1)
    assert comp.interrompido and all(r.estado == "ignorado" for r in comp.resultados)


def test_cancelamento_entre_modelos(dados):
    chamadas = []
    comp = comparar_modelos(["linear_reg", "ridge", "tree_reg"], dados, cv_folds=0, cancelar=lambda: len(chamadas) >= 1,
                            progresso=lambda i, n, nome: chamadas.append(nome))
    assert comp.interrompido and comp.resultados[0].ok and comp.resultados[1].estado == "ignorado"


def test_recomendacao_justificada_e_prudente(dados):
    comp = comparar_modelos(["linear_reg", "tree_reg", "rf_reg", "knn_reg"], dados, cv_folds=3)
    rec = recomendar(comp)
    assert rec.modelo_id and rec.justificacao and rec.base.startswith("validação cruzada")
    assert "melhor" not in rec.justificacao.lower() or "avaliad" in rec.justificacao.lower()
    assert all(r.metricas for r in comp.validos())


def test_sem_modelos_validos_nao_inventa_recomendacao(dados, modelo_que_falha):
    comp = comparar_modelos(["teste_falha"], dados, cv_folds=0)
    rec = recomendar(comp)
    assert rec.modelo_id is None


def test_classificacao_compara_com_classe_mais_frequente():
    d = preparar_dados(tabela_classificacao(), "classe", "classificacao")
    comp = comparar_modelos(["logistic", "tree_clf"], d, cv_folds=3)
    assert {"Accuracy", "Precision", "Recall", "F1-score"} <= set(comp.tabela_visivel().columns)
    assert recomendar(comp).modelo_id
