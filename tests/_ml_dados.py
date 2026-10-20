"""Conjuntos de dados pequenos e determinísticos para os testes do motor de ML."""
import numpy as np
import pandas as pd


def tabela_regressao(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(0, 10, n), "cat": rng.choice(["x", "y", "z"], n)})
    df["y"] = 3 * df.a + 0.5 * df.b + (df.cat == "x") * 2 + rng.normal(scale=0.3, size=n)
    return df


def tabela_classificacao(n=240, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "cat": rng.choice(["u", "v"], n)})
    df["classe"] = np.where(df.a + 0.5 * df.b + rng.normal(scale=0.5, size=n) > 0, "sim", "nao")
    return df


def tabela_multiclasse(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["classe"] = pd.cut(df.a + df.b, [-99, -1, 1, 99], labels=["baixa", "media", "alta"]).astype(str)
    return df


def serie_mensal(n=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame({"data": pd.date_range("2010-01-01", periods=n, freq="MS"),
                         "vendas": 100 + 0.8 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(scale=1.0, size=n)})
