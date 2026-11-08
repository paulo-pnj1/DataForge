"""Propostas de limpeza: só correções sem fuga de informação, descritas e escolhidas pelo estudante."""
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import limpeza  # noqa: E402
from modules.ml_engine.preprocessing import preparar_dados  # noqa: E402


def _tabela(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "cidade": rng.choice(["Luanda", "luanda ", "Uíge", "Huambo"], n),
        "salario": [f"{int(x):,}".replace(",", " ") + " Kz" for x in rng.integers(50_000, 500_000, n)],
        "data": [f"{rng.integers(13, 28):02d}/{rng.integers(1, 12):02d}/2024" for _ in range(n)],
        "codigo": [f"{i:05d}" for i in range(n)],
        "y": rng.normal(size=n),
    })
    return pd.concat([df, df.head(4)], ignore_index=True)


def test_propoe_as_quatro_correcoes_pela_ordem():
    ids = [p.id for p in limpeza.propor_limpeza(_tabela(), "y")]
    assert ids == ["texto_limpo", "numeros_em_texto", "datas_em_texto", "duplicados"]


def test_nada_a_propor_em_dados_arrumados():
    df = pd.DataFrame({"a": np.arange(30.0), "b": ["x", "y", "z"] * 10, "y": np.arange(30)})
    assert limpeza.propor_limpeza(df, "y") == []


def test_codigos_com_zeros_a_esquerda_nao_viram_numeros():
    nomes = [c for p in limpeza.propor_limpeza(_tabela(), "y") for c in p.colunas]
    assert "codigo" not in nomes and "salario" in nomes and "data" in nomes


def test_aplicar_converte_tipos_e_remove_duplicados():
    df = _tabela()
    ps = limpeza.propor_limpeza(df, "y")
    novo, notas = limpeza.aplicar_propostas(df, ps, [p.id for p in ps])
    assert pd.api.types.is_float_dtype(novo["salario"]) and pd.api.types.is_datetime64_any_dtype(novo["data"])
    assert novo["cidade"].nunique() == 3 and len(novo) == len(df) - 4
    assert any("repetidas" in n for n in notas)
    assert len(df) == 124                                                 # a tabela original não é alterada


def test_so_aplica_o_que_o_estudante_escolheu():
    df = _tabela()
    ps = limpeza.propor_limpeza(df, "y")
    novo, _ = limpeza.aplicar_propostas(df, ps, ["duplicados"])
    assert len(novo) == len(df) - 4 and not pd.api.types.is_float_dtype(novo["salario"])


def test_leitura_de_numeros_em_portugues_e_ingles():
    f = limpeza._para_numero
    assert f("1 250,50") == 1250.5 and f("12 %") == 12 and f("1,234.5") == 1234.5
    assert f("1.234.567") == 1234567 and f("7") == 7 and np.isnan(f("abc"))


def test_datas_dia_mes_ano_sao_lidas_com_o_dia_primeiro():
    s = pd.Series(["25/12/2024", "03/04/2024", "01/02/2024"] * 5)
    d = limpeza._converter_datas(s)
    assert d.iloc[0] == pd.Timestamp("2024-12-25") and d.iloc[1] == pd.Timestamp("2024-04-03")


def test_dados_limpos_seguem_para_o_motor_sem_erro():
    df = _tabela()
    ps = limpeza.propor_limpeza(df, "y")
    novo, _ = limpeza.aplicar_propostas(df, ps, [p.id for p in ps])
    d = preparar_dados(novo, "y", "regressao")
    assert "salario" in d.num_cols and "data" in d.colunas_data


def test_exemplos_mostram_antes_e_depois():
    p = next(p for p in limpeza.propor_limpeza(_tabela(), "y") if p.id == "numeros_em_texto")
    assert p.exemplos and all(len(par) == 2 for par in p.exemplos)
