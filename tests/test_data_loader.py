"""Carregamento e validação de dados: CSV, Excel, ficheiros inválidos e colunas problemáticas."""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine.data_loader import ErroDados, amostrar, carregar_ficheiro, folhas_excel  # noqa: E402
from modules.ml_engine.data_validation import sugerir_tipo_problema, validar_dados  # noqa: E402
from _ml_dados import tabela_regressao  # noqa: E402


def test_csv_carrega():
    df = carregar_ficheiro("d.csv", b"a,b\n1,2\n3,4\n")
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_excel_carrega_e_lista_folhas():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"x": [1, 2, 3], "z": [7, 8, 9]}).to_excel(w, sheet_name="Um", index=False)
        pd.DataFrame({"y": [4, 5], "w": [1, 2]}).to_excel(w, sheet_name="Dois", index=False)
    dados = buf.getvalue()
    assert folhas_excel("a.xlsx", dados) == ["Um", "Dois"]
    assert list(carregar_ficheiro("a.xlsx", dados, "Dois").columns) == ["y", "w"]


@pytest.mark.parametrize("nome,dados", [("a.csv", b""), ("a.xls", b"x"), ("a.exe", b"x"), ("a", b"x")])
def test_ficheiros_invalidos(nome, dados):
    with pytest.raises(ErroDados):
        carregar_ficheiro(nome, dados)


def test_ficheiro_grande_demais():
    with pytest.raises(ErroDados):
        carregar_ficheiro("a.csv", b"a\n1\n" * 10, max_bytes=5)


def test_amostragem_avisa():
    df = pd.DataFrame({"a": range(100)})
    out, aviso = amostrar(df, 10)
    assert len(out) == 10 and aviso
    assert amostrar(df, 1000)[1] is None


def test_validacao_deteta_problemas():
    df = pd.DataFrame({"id": range(30), "const": 1, "vazia": np.nan, "x": np.arange(30.0), "y": np.arange(30.0) ** 2})
    df.loc[0, "x"] = np.inf
    rel = validar_dados(df, "y", "regressao")
    assert rel.ok
    assert "const" in rel.colunas_constantes and "vazia" in rel.colunas_vazias and "id" in rel.colunas_id
    assert any("infinitos" in a for a in rel.avisos)


def test_validacao_alvo_invalido():
    df = pd.DataFrame({"x": range(30), "y": ["a"] * 30})
    assert not validar_dados(df, "y", "classificacao").ok            # alvo constante
    assert not validar_dados(df.assign(y=["a", "b"] * 15), "y", "regressao").ok   # alvo não numérico
    assert not validar_dados(df, "nao_existe", "regressao").ok
    assert not validar_dados(df.head(3).assign(y=[1, 2, 3]), "y", "regressao").ok    # poucas linhas
    assert not validar_dados(pd.DataFrame(), None).ok


def test_sugestao_de_tipo():
    df = tabela_regressao()
    assert sugerir_tipo_problema(df, "y")[0] == "regressao"
    assert sugerir_tipo_problema(df, "cat")[0] == "classificacao"
