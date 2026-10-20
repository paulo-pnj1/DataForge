"""Classificação: os 13 algoritmos, métricas, matriz de confusão e curvas ROC/PR."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import model_evaluation as ev, model_registry as reg  # noqa: E402
from modules.ml_engine.model_training import INCOMPATIVEL, INDISPONIVEL, treinar_modelo  # noqa: E402
from modules.ml_engine.preprocessing import preparar_dados  # noqa: E402
from _ml_dados import tabela_classificacao, tabela_multiclasse  # noqa: E402

IDS = ["logistic", "tree_clf", "rf_clf", "extra_trees_clf", "gb_clf", "hist_gb_clf", "svc", "knn_clf", "gaussian_nb", "multinomial_nb",
       "mlp_clf", "xgb_clf", "lgbm_clf"]


@pytest.fixture(scope="module")
def dados():
    return preparar_dados(tabela_classificacao(), "classe", "classificacao")


def test_catalogo_tem_os_13_algoritmos():
    assert sorted(m.id for m in reg.listar("classificacao")) == sorted(IDS)


@pytest.mark.parametrize("mid", IDS)
def test_cada_algoritmo(dados, mid):
    r = treinar_modelo(mid, dados)
    if not reg.obter(mid).disponivel():
        assert r.estado == INDISPONIVEL and r.motivo
        return
    assert r.ok or r.estado == INCOMPATIVEL, r.motivo
    if r.ok:
        for k in ("accuracy", "precision", "recall", "f1"):
            assert 0 <= r.metricas[k] <= 1
        assert r.metricas["accuracy"] > 0.6


def test_matriz_de_confusao_e_curvas(dados):
    r = treinar_modelo("logistic", dados)
    cm = ev.matriz_confusao(dados.y_test, r.predicoes["y_pred"], len(dados.classes))
    assert cm.shape == (2, 2) and cm.sum() == dados.n_teste
    c = ev.curvas_binarias(dados.y_test, r.predicoes["score"])
    assert set(c) == {"roc", "pr"} and r.metricas["roc_auc"] > 0.7


def test_multiclasse_sem_curvas_binarias():
    d = preparar_dados(tabela_multiclasse(), "classe", "classificacao")
    r = treinar_modelo("logistic", d)
    assert r.ok and len(d.classes) == 3
    cm = ev.matriz_confusao(d.y_test, r.predicoes["y_pred"], 3)
    assert cm.shape == (3, 3)


def test_desequilibrio_e_avisado():
    df = tabela_classificacao()
    df["classe"] = np.where(np.arange(len(df)) % 10 == 0, "raro", "comum")
    d = preparar_dados(df, "classe", "classificacao")
    assert ev.aviso_desequilibrio(d.y_train)


def test_naive_bayes_multinomial_nao_parte_com_negativos(dados):
    r = treinar_modelo("multinomial_nb", dados)
    assert r.ok or r.estado == INCOMPATIVEL
