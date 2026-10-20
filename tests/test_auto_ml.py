"""AutoML: fluxo completo, limites, sem usar o teste para escolher e erros claros."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import auto_ml  # noqa: E402
from modules.ml_engine.data_loader import ErroDados  # noqa: E402
from _ml_dados import tabela_classificacao, tabela_regressao  # noqa: E402


def test_automl_regressao_completo():
    res = auto_ml.executar_automl(tabela_regressao(), "y", max_modelos=4, orcamento_s=120)
    assert res.tipo == "regressao" and res.passos
    assert 1 <= len([r for r in res.comparacao.resultados if r.ok]) <= 4
    assert res.recomendacao.modelo_id and res.recomendacao.base.startswith("validação cruzada")
    assert res.comparacao.referencia is not None


def test_automl_classificacao_e_tipo_sugerido():
    res = auto_ml.executar_automl(tabela_classificacao(), "classe", max_modelos=3)
    assert res.tipo == "classificacao" and res.recomendacao.modelo_id


def test_limite_de_modelos_respeitado():
    res = auto_ml.executar_automl(tabela_regressao(), "y", max_modelos=2)
    assert len([r for r in res.comparacao.resultados if r.ok]) <= 2


def test_candidatos_dizem_porque_ficaram_de_fora():
    from modules.ml_engine.preprocessing import preparar_dados
    ids, fora = auto_ml.candidatos(preparar_dados(tabela_regressao(), "y", "regressao"))
    assert "linear_reg" in ids
    for nome, motivo in fora:
        assert nome and motivo


def test_erros_claros_sem_traceback():
    with pytest.raises(ErroDados):
        auto_ml.executar_automl(tabela_regressao(), "nao_existe")
    df = tabela_regressao().assign(y=1.0)
    with pytest.raises(ErroDados):
        auto_ml.executar_automl(df, "y", tipo="regressao")


def test_nao_usa_o_teste_para_escolher():
    """A recomendação tem de ser igual se o conjunto de teste for corrompido depois do treino (só CV decide)."""
    from modules.ml_engine.model_comparison import recomendar
    res = auto_ml.executar_automl(tabela_regressao(), "y", max_modelos=4, random_state=7)
    esperado = res.recomendacao.modelo_id
    for r in res.comparacao.resultados:
        r.metricas = {k: (v * 1000 if isinstance(v, float) else v) for k, v in r.metricas.items()}
    assert recomendar(res.comparacao).modelo_id == esperado
