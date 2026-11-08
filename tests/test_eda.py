"""Exploração gráfica automática: resumo, distribuições, correlações e relação com o alvo (sem inventar números)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import eda  # noqa: E402
from _ml_dados import tabela_classificacao, tabela_regressao  # noqa: E402


def test_resumo_tem_uma_linha_por_coluna_e_numeros_reais():
    df = tabela_regressao()
    df.loc[:9, "b"] = np.nan
    r = eda.resumo_colunas(df)
    assert list(r["Coluna"]) == list(df.columns)
    linha = r[r["Coluna"] == "b"].iloc[0]
    assert linha["Tipo"] == "número" and linha["Em falta"].startswith("10 ")
    assert r[r["Coluna"] == "cat"].iloc[0]["Tipo"] == "categoria/texto"


def test_distribuicoes_ignoram_colunas_inuteis():
    df = tabela_regressao()
    df["constante"] = 1
    df["id"] = [f"id{i}" for i in range(len(df))]
    nomes = [c for c, _ in eda.figuras_distribuicao(df)]
    assert "constante" not in nomes and "id" not in nomes and {"a", "b", "cat"} <= set(nomes)
    assert len(eda.figuras_distribuicao(df, max_colunas=2)) == 2


def test_correlacao_e_pares_muito_correlacionados():
    df = tabela_regressao()
    df["a2"] = df["a"] * 2 + 1
    corr = eda.matriz_correlacao(df)
    assert corr.loc["a", "a"] == pytest.approx(1.0)
    pares = eda.pares_muito_correlacionados(corr)
    assert pares and {pares[0][0], pares[0][1]} == {"a", "a2"}
    assert eda.matriz_correlacao(pd.DataFrame({"x": [1, 2, 3], "t": ["a", "b", "c"]})).empty
    assert eda.figura_correlacao(corr) is not None


def test_forca_regressao_ordena_pela_ligacao_real():
    df = tabela_regressao(400)
    tab = eda.forca_associacao(df, "y", "regressao")
    assert tab.iloc[0]["Coluna"] == "a"                                  # y = 3a + 0.5b + ...: «a» domina
    assert set(tab["Coluna"]) == {"a", "b", "cat"}
    assert tab["Força"].between(0, 1).all()


def test_forca_classificacao_usa_eta_e_cramer():
    df = tabela_classificacao(400)
    tab = eda.forca_associacao(df, "classe", "classificacao")
    medidas = dict(zip(tab["Coluna"], tab["Medida"]))
    assert medidas["a"].startswith("η") and medidas["cat"] == "V de Cramér"
    assert tab.iloc[0]["Coluna"] in ("a", "b")


def test_fuga_de_informacao_e_assinalada():
    df = tabela_regressao(300)
    df["copia"] = df["y"] * 2
    tab = eda.forca_associacao(df, "y", "regressao")
    linha = tab[tab["Coluna"] == "copia"].iloc[0]
    assert linha["Força"] > 0.95 and "fuga de informação" in linha["Observação"]


def test_alvo_inexistente_da_erro_claro():
    with pytest.raises(ValueError):
        eda.forca_associacao(tabela_regressao(), "nao_existe")


def test_amostra_so_em_tabelas_grandes():
    df = pd.DataFrame({"x": range(50)})
    assert eda.amostra_para_graficos(df)[1] is None
    d, aviso = eda.amostra_para_graficos(df, max_linhas=10)
    assert len(d) == 10 and "amostra" in aviso


@pytest.mark.parametrize("tipo,coluna", [("regressao", "a"), ("regressao", "cat"), ("classificacao", "a"), ("classificacao", "cat")])
def test_figura_alvo_por_coluna_para_todas_as_combinacoes(tipo, coluna):
    df = tabela_regressao() if tipo == "regressao" else tabela_classificacao()
    alvo = "y" if tipo == "regressao" else "classe"
    if tipo == "classificacao" and coluna == "a":
        pass
    assert eda.figura_alvo_por_coluna(df, coluna, alvo, tipo).data
