"""Serviço de previsões: verificação dos dados novos e previsões com modelos guardados."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import model_persistence as mp, prediction_service as ps  # noqa: E402
from modules.ml_engine.model_training import treinar_modelo  # noqa: E402
from modules.ml_engine.preprocessing import preparar_dados  # noqa: E402
from _ml_dados import tabela_classificacao, tabela_regressao  # noqa: E402


@pytest.fixture
def reg_guardado(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda n: None)
    df = tabela_regressao()
    d = preparar_dados(df, "y", "regressao")
    r = treinar_modelo("ridge", d)
    info = mp.guardar_modelo("ana", r.pipeline, mp.meta_tabular(d, r, "m"), tmp_path / "m")
    modelo, meta = mp.carregar_modelo("ana", info["id"], tmp_path / "m")
    return modelo, meta, df


def test_prever_tabela_e_um_caso(reg_guardado):
    modelo, meta, df = reg_guardado
    novos = df.drop(columns="y").head(5)
    out, rel = ps.prever_tabela(modelo, meta, novos)
    assert len(out) == 5 and "previsao_y" in out and not rel["erros"]
    p, det, _ = ps.prever_um(modelo, meta, novos.iloc[0].to_dict())
    assert p == pytest.approx(out["previsao_y"].iloc[0])


def test_colunas_em_falta_bloqueiam_e_extra_avisa(reg_guardado):
    modelo, meta, df = reg_guardado
    with pytest.raises(ps.ErroPrevisao, match="Faltam"):
        ps.prever_tabela(modelo, meta, df.drop(columns=["y", "a"]))
    rel = ps.verificar_entrada(df.assign(lixo=1), meta)
    assert any("lixo" in a for a in rel["avisos"])


def test_categoria_nova_texto_em_numero_e_fora_do_intervalo_avisam(reg_guardado):
    modelo, meta, df = reg_guardado
    novos = df.drop(columns="y").head(4).copy()
    novos["cat"] = "nunca_vista"
    novos["a"] = novos["a"].astype(object)
    novos.loc[novos.index[0], "a"] = "abc"
    novos.loc[novos.index[1], "b"] = 10 ** 6
    out, rel = ps.prever_tabela(modelo, meta, novos)
    texto = " ".join(rel["avisos"])
    assert "nunca vistas" in texto and "não são números" in texto and "fora do intervalo" in texto
    assert len(out) == 4 and np.isfinite(out["previsao_y"]).all()


def test_classificacao_devolve_classes_e_probabilidades(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda n: None)
    df = tabela_classificacao()
    d = preparar_dados(df, "classe", "classificacao")
    r = treinar_modelo("logistic", d)
    info = mp.guardar_modelo("ana", r.pipeline, mp.meta_tabular(d, r, "c"), tmp_path / "m")
    modelo, meta = mp.carregar_modelo("ana", info["id"], tmp_path / "m")
    out, _ = ps.prever_tabela(modelo, meta, df.drop(columns="classe").head(6))
    assert set(out["previsao_classe"]) <= {"sim", "nao"}
    assert {"prob_sim", "prob_nao"} <= set(out.columns)
    assert np.allclose(out[["prob_sim", "prob_nao"]].sum(axis=1), 1, atol=1e-3)


def test_data_nas_features_e_tratada(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda n: None)
    df = tabela_regressao()
    df["quando"] = pd.date_range("2020-01-01", periods=len(df))
    d = preparar_dados(df, "y", "regressao")
    r = treinar_modelo("linear_reg", d)
    info = mp.guardar_modelo("ana", r.pipeline, mp.meta_tabular(d, r, "d"), tmp_path / "m")
    modelo, meta = mp.carregar_modelo("ana", info["id"], tmp_path / "m")
    out, _ = ps.prever_tabela(modelo, meta, df.drop(columns="y").head(3))
    assert np.isfinite(out["previsao_y"]).all()


def test_modelo_de_serie_nao_prevê_por_tabela(reg_guardado):
    modelo, meta, df = reg_guardado
    with pytest.raises(ps.ErroPrevisao):
        ps.prever_futuro(modelo, meta, 3)
    with pytest.raises(ps.ErroPrevisao):
        ps.prever_tabela(modelo, dict(meta, tipo="serie"), df)
