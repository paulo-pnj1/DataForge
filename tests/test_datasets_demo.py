"""Datasets fictícios do Laboratório: geração, reprodutibilidade e que o motor consegue aprender com eles."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from modules.ml_engine import auto_ml, datasets_demo as demo, forecasting as fc  # noqa: E402
from modules.ml_engine.data_validation import sugerir_tipo_problema  # noqa: E402

IDS = list(demo.DATASETS)


@pytest.mark.parametrize("i", IDS)
def test_gera_com_alvo_e_sem_alvo_em_falta(i):
    d = demo.DATASETS[i]
    df = demo.gerar(i)
    assert len(df) == d.n_padrao and d.alvo in df.columns
    assert df[d.alvo].notna().all()
    assert df.columns[-1] == d.alvo


@pytest.mark.parametrize("i", IDS)
def test_reprodutivel_e_semente_muda_dados(i):
    pd.testing.assert_frame_equal(demo.gerar(i, semente=1), demo.gerar(i, semente=1))
    assert not demo.gerar(i, semente=1).equals(demo.gerar(i, semente=2))


def test_limites_e_id_invalido():
    with pytest.raises(ValueError):
        demo.gerar("imoveis", 5)
    with pytest.raises(KeyError):
        demo.gerar("nao_existe")


def test_csv_ida_e_volta():
    df = demo.gerar("credito", 200)
    lido = pd.read_csv(pd.io.common.BytesIO(demo.para_csv(df)))
    assert lido.shape == df.shape


@pytest.mark.parametrize("i", [d.id for d in demo.listar("regressao")])
def test_regressao_aprende_acima_da_referencia(i):
    d = demo.DATASETS[i]
    res = auto_ml.executar_automl(demo.gerar(i), d.alvo, tipo="regressao", max_modelos=3, orcamento_s=90)
    melhor = [r for r in res.comparacao.resultados if r.ok]
    assert melhor
    assert max(r.metricas["r2"] for r in melhor) > 0.6


@pytest.mark.parametrize("i", [d.id for d in demo.listar("classificacao")])
def test_classificacao_corre(i):
    d = demo.DATASETS[i]
    assert sugerir_tipo_problema(demo.gerar(i), d.alvo)[0] == "classificacao"
    res = auto_ml.executar_automl(demo.gerar(i), d.alvo, tipo="classificacao", max_modelos=3, orcamento_s=90)
    assert res.recomendacao.modelo_id


@pytest.mark.parametrize("i", [d.id for d in demo.listar("serie")])
def test_serie_e_preparavel(i):
    d = demo.DATASETS[i]
    df = demo.gerar(i)
    cd = fc.detectar_coluna_data(df)
    assert cd == "data"
    s = fc.preparar_serie(df, cd, d.alvo)
    assert s.n >= 30
