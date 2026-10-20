"""Séries temporais: ordem cronológica, divisão temporal sem fuga, referências, métricas e previsão guardada."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import forecasting as fc, model_evaluation as ev, model_persistence as mp, model_registry as reg  # noqa: E402
from modules.ml_engine import prediction_service as ps  # noqa: E402
from modules.ml_engine.data_loader import ErroDados  # noqa: E402
from _ml_dados import serie_mensal  # noqa: E402


def test_ordena_cronologicamente_e_avisa():
    df = serie_mensal().sample(frac=1, random_state=1)
    s = fc.preparar_serie(df, "data", "vendas")
    assert s.datas.is_monotonic_increasing and s.periodo == 12 and s.freq
    assert any("ordem" in n.lower() or "ordenad" in n.lower() for n in s.notas)


def test_datas_invalidas_e_serie_curta():
    df = serie_mensal()
    df["data"] = df["data"].astype(str)
    df.loc[3, "data"] = "isto não é data"
    nota = fc.preparar_serie(df, "data", "vendas").notas                 # 1 linha má entre 120: retirada, com aviso
    assert any("inválida" in n for n in nota)
    df.loc[:40, "data"] = "lixo"                                          # mais de 20% inválidas: recusa
    with pytest.raises(ErroDados):
        fc.preparar_serie(df, "data", "vendas")
    with pytest.raises(ErroDados):
        fc.preparar_serie(serie_mensal(5), "data", "vendas")


def test_datas_repetidas_e_buracos():
    df = pd.concat([serie_mensal(60), serie_mensal(60).head(3)])
    s = fc.preparar_serie(df, "data", "vendas", duplicados="media")
    assert s.n == 60
    buraco = serie_mensal(60).drop(index=[10, 11])
    s2 = fc.preparar_serie(buraco, "data", "vendas")
    assert s2.n == 60 and s2.notas


def test_divisao_temporal_nunca_mistura_o_futuro():
    d = fc.dividir_temporal(160, 12, 12, 3)
    assert d.fim_validacao + d.horizonte == 160 and d.fim_treino <= d.fim_validacao
    assert all(o + d.horizonte <= d.fim_validacao + d.horizonte for o in d.origens_cv)
    assert max(d.origens_cv) == d.fim_treino or max(d.origens_cv) <= d.fim_validacao
    with pytest.raises(ErroDados):
        fc.dividir_temporal(10, 12, 1, 3)


def test_modelo_nunca_ve_o_futuro_durante_a_avaliacao():
    """Um modelo espião regista o que recebe: nas avaliações nunca aparece o pico colocado nos últimos h pontos."""
    vistos = []

    class Espiao(fc.ModeloSerie):
        def ajustar(self, y, periodo=1):
            vistos.append((len(y), float(np.max(y))))
            self.u = float(y[-1])
            return self

        def prever(self, h):
            return fc.PrevisaoSerie(np.full(h, self.u))

    reg.registar(reg.ModeloInfo("serie_espiao", "Espião", "Teste", "x", ("serie",), lambda p, rs: Espiao(p, rs), tipo_modelo="serie"))
    try:
        df = serie_mensal(100)
        df.loc[95:, "vendas"] = 10 ** 6                                   # pico só nos últimos 12 pontos
        s = fc.preparar_serie(df, "data", "vendas")
        comp = fc.comparar_series(s, ["serie_espiao"], 12, incluir_referencias=False)
        assert comp.por_id("serie_espiao").ok
        avaliacao, final = vistos[:-1], vistos[-1]
        assert all(m < 10 ** 6 for _, m in avaliacao)                      # validação e teste só viram o passado
        assert all(n <= 100 - 12 for n, _ in avaliacao) and final[0] == 100   # só o ajuste final usa tudo
    finally:
        reg.remover("serie_espiao")


def test_comparacao_com_referencias_e_metricas():
    s = fc.preparar_serie(serie_mensal(), "data", "vendas")
    comp = fc.comparar_series(s, ["serie_regressao", "serie_rf"], 12)
    tab = comp.tabela()
    assert {"MAE", "RMSE", "MAPE (%)", "MASE", "Ganho vs. referência"} <= set(tab.columns)
    assert any("referência" in n for n in tab["Modelo"])
    r = comp.por_id("serie_regressao")
    assert r.ok and r.metricas_teste["mae"] < comp.por_id("serie_naive").metricas_teste["mae"]
    assert r.previsao_futura.inferior is None                              # sem intervalos inventados
    assert len(r.previsao_futura.media) == 12


def test_mape_so_quando_valido_e_mase():
    m = ev.metricas_serie([0, 1, 2], [0.5, 1, 2], y_treino=[1, 2, 3, 4], periodo=1)
    assert m["mape"] is None and m["mase"] is not None                    # zero nos reais: MAPE não existe
    m = ev.metricas_serie([1, 2, 4], [1, 2, 2], y_treino=[1, 2, 3, 4], periodo=1)
    assert m["mape"] is not None and m["mae"] == pytest.approx(2 / 3)


def test_modelos_indisponiveis_ou_incompativeis_nao_partem():
    s = fc.preparar_serie(serie_mensal(40), "data", "vendas")
    comp = fc.comparar_series(s, ["serie_lstm", "serie_arima", "serie_rf"], 6)
    for r in comp.resultados:
        assert r.ok or r.motivo
    assert comp.por_id("serie_lstm").estado in ("incompativel", "indisponivel")


def test_exportar_previsoes_e_prever_com_modelo_guardado(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda n: None)
    s = fc.preparar_serie(serie_mensal(), "data", "vendas")
    comp = fc.comparar_series(s, ["serie_regressao"], 12)
    pv = comp.previsoes_dataframe("serie_regressao")
    assert set(pv["fase"]) == {"historico", "teste", "futuro"} and pv["data"].is_monotonic_increasing
    r = comp.por_id("serie_regressao")
    info = mp.guardar_modelo("ana", r.modelo_final, mp.meta_serie(s, r, "s", 12), tmp_path / "m")
    modelo, meta = mp.carregar_modelo("ana", info["id"], tmp_path / "m")
    fut = ps.prever_futuro(modelo, meta, 6)
    assert len(fut) == 6 and fut["data"].iloc[0] > s.datas[-1]
    assert np.allclose(fut["previsao"], r.previsao_futura.media[:6])
    with pytest.raises(ps.ErroPrevisao):
        ps.prever_futuro(modelo, meta, 0)
