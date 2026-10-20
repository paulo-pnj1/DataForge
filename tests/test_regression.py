"""Regressão: os 16 algoritmos existem, os disponíveis treinam e as métricas são coerentes."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import model_registry as reg  # noqa: E402
from modules.ml_engine.model_training import INDISPONIVEL, treinar_modelo, treinar_referencia  # noqa: E402
from modules.ml_engine.preprocessing import preparar_dados  # noqa: E402
from _ml_dados import tabela_regressao  # noqa: E402

IDS_REGRESSAO = ["linear_reg", "poly_reg", "ridge", "lasso", "elasticnet", "bayesian_ridge", "tree_reg", "rf_reg", "extra_trees_reg",
                 "gb_reg", "hist_gb_reg", "svr", "knn_reg", "mlp_reg", "xgb_reg", "lgbm_reg"]


@pytest.fixture(scope="module")
def dados():
    return preparar_dados(tabela_regressao(), "y", "regressao")


def test_catalogo_tem_os_16_algoritmos():
    assert sorted(m.id for m in reg.listar("regressao")) == sorted(IDS_REGRESSAO)


@pytest.mark.parametrize("mid", IDS_REGRESSAO)
def test_cada_algoritmo_treina_ou_diz_porque_nao(dados, mid):
    r = treinar_modelo(mid, dados)
    if not reg.obter(mid).disponivel():
        assert r.estado == INDISPONIVEL and r.motivo and not r.ok          # dependência opcional em falta: degrada, não parte
        return
    assert r.ok, r.motivo
    assert set(["r2", "mae", "rmse"]) <= set(r.metricas)
    assert r.metricas["rmse"] >= r.metricas["mae"] >= 0
    assert r.metricas["r2"] > 0.5                                           # os dados têm sinal forte


def test_lineares_acertam_quase_perfeito(dados):
    assert treinar_modelo("linear_reg", dados).metricas["r2"] > 0.95


def test_referencia_nao_explica_nada(dados):
    ref = treinar_referencia(dados)
    assert ref.metricas["r2"] <= 0.05


def test_parametros_sao_limitados():
    info = reg.obter("rf_reg")
    p = reg.completar_parametros(info, {"n_estimators": 10 ** 9, "desconhecido": 1, "max_depth": "lixo"})
    assert p["n_estimators"] == info.parametros["n_estimators"]["max"] and "desconhecido" not in p


def test_semente_repete_resultados(dados):
    a = treinar_modelo("rf_reg", dados, random_state=3).metricas["rmse"]
    b = treinar_modelo("rf_reg", dados, random_state=3).metricas["rmse"]
    assert a == b
