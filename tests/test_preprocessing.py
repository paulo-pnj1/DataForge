"""Pré-processamento: valores em falta, categorias, divisão e prevenção de fuga de informação."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import model_registry as reg  # noqa: E402
from modules.ml_engine.model_training import treinar_modelo  # noqa: E402
from modules.ml_engine.preprocessing import ErroDados, construir_preprocessador, preparar_dados  # noqa: E402
from _ml_dados import tabela_classificacao, tabela_regressao  # noqa: E402


def test_divisao_e_reprodutibilidade():
    df = tabela_regressao()
    a = preparar_dados(df, "y", "regressao", test_size=0.25, random_state=1)
    b = preparar_dados(df, "y", "regressao", test_size=0.25, random_state=1)
    assert a.n_teste == 50 and a.n_treino == 150
    assert list(a.X_test["a"]) == list(b.X_test["a"])
    assert not set(a.X_train["a"]) & set(a.X_test["a"])          # nenhuma linha está em treino e teste


def test_classificacao_estratificada():
    d = preparar_dados(tabela_classificacao(), "classe", "classificacao")
    assert d.estratificado and set(d.classes) == {"nao", "sim"}
    for y in (d.y_train, d.y_test):
        assert 0.3 < np.mean(y) < 0.7


def test_valores_em_falta_nao_partem_o_treino():
    df = tabela_regressao()
    df.loc[::7, "a"] = np.nan
    df.loc[::5, "cat"] = None
    df.loc[3, "y"] = np.nan                                # linha sem resposta é retirada, com nota
    d = preparar_dados(df, "y", "regressao")
    assert d.n_usadas == len(df) - 1 and any("alvo" in n.lower() or "sem valor" in n.lower() for n in d.notas)
    r = treinar_modelo("linear_reg", d)
    assert r.ok and np.isfinite(r.metricas["rmse"])


def test_estatisticas_so_do_treino_e_categoria_nova():
    df = tabela_regressao()
    d = preparar_dados(df, "y", "regressao")
    pre = construir_preprocessador(d.num_cols, d.cat_cols)
    pre.fit(d.X_train)
    media_treino = d.X_train["a"].mean()
    escalado = pre.transform(d.X_test)
    assert np.isfinite(escalado).all()
    scaler = pre.named_transformers_["num"].named_steps["escala"]
    assert scaler.mean_[d.num_cols.index("a")] == pytest.approx(media_treino)      # aprendeu só com o treino
    novo = d.X_test.head(2).copy()
    novo["cat"] = "categoria_nunca_vista"
    assert np.isfinite(pre.transform(novo)).all()                                  # handle_unknown="ignore"


def test_fuga_de_informacao_e_avisada():
    df = tabela_regressao()
    df["copia"] = df["y"] * 2
    d = preparar_dados(df, "y", "regressao", features=["a", "copia"])
    assert any("copia" in a for a in d.avisos)


def test_erros_claros():
    df = tabela_regressao()
    with pytest.raises(ErroDados):
        preparar_dados(df, "nao_existe", "regressao")
    with pytest.raises(ErroDados):
        preparar_dados(df, "y", "regressao", test_size=0.9)
    with pytest.raises(ErroDados):
        preparar_dados(df, "y", "outra_coisa")
