"""Caches de velocidade: datasets embutidos e comparações partilhadas entre sessões."""
import time

import pytest

from modules import laboratorio as lab
from modules.ml_engine import datasets_demo as dd
from modules.ml_engine.model_comparison import comparar_modelos, recomendar
from modules.ml_engine.preprocessing import preparar_dados
from modules.utils import carregar_dataset_embutido


@pytest.fixture(scope="module")
def comparacao():
    d = dd.listar("classificacao")[0]
    df = dd.gerar(d.id, 400)
    dados = preparar_dados(df, d.alvo, "classificacao")
    comp = comparar_modelos(["logistic", "knn_clf"], dados, None, 0, 42)
    return df, d, comp, recomendar(comp)


def test_dataset_embutido_devolve_copias_independentes():
    a, desc, tipo = carregar_dataset_embutido("Iris - Classificação de flores")
    a.iloc[0, 0] = -999                       # estragar a cópia de um aluno...
    b, _, _ = carregar_dataset_embutido("Iris - Classificação de flores")
    assert b.iloc[0, 0] != -999               # ...não afecta o seguinte
    assert tipo == "classificacao" and desc


def test_dataset_desconhecido_continua_a_devolver_vazio():
    assert carregar_dataset_embutido("não existe") == (None, "", "")


def test_comparacao_guardada_e_devolvida_como_copia(comparacao):
    df, d, comp, rec = comparacao
    lab._COMPARACOES.clear()
    lab._guardar_comparacao("k1", comp, rec)
    c1, _ = lab._comparacao_guardada("k1")
    c2, _ = lab._comparacao_guardada("k1")
    assert c1 is not c2 and c1 is not comp
    assert [r.id for r in c1.resultados] == [r.id for r in comp.resultados]
    assert lab._comparacao_guardada("outra") is None


def test_comparacao_incompleta_nao_se_guarda(comparacao):
    df, d, comp, rec = comparacao
    lab._COMPARACOES.clear()
    import copy
    parcial = copy.deepcopy(comp)
    parcial.interrompido = True
    lab._guardar_comparacao("k2", parcial, rec)
    assert lab._comparacao_guardada("k2") is None


def test_cache_de_comparacoes_tem_limite(comparacao):
    df, d, comp, rec = comparacao
    lab._COMPARACOES.clear()
    for i in range(lab._COMPARACOES_MAX + 5):
        lab._guardar_comparacao(f"x{i}", comp, rec)
    assert len(lab._COMPARACOES) == lab._COMPARACOES_MAX
    assert lab._comparacao_guardada("x0") is None and lab._comparacao_guardada(f"x{lab._COMPARACOES_MAX + 4}")


def test_chave_muda_com_os_dados_e_com_as_escolhas(comparacao):
    df, d, comp, rec = comparacao
    e = dict(alvo=d.alvo, tipo="classificacao", feats=[c for c in df.columns if c != d.alvo][:3], teste=.2, seed=42)
    k1 = lab._chave_comparacao(df, e, ["logistic"], {}, 3, 6, 180)
    assert k1 == lab._chave_comparacao(df.copy(), e, ["logistic"], {}, 3, 6, 180)
    assert k1 != lab._chave_comparacao(df, e, ["logistic", "knn_clf"], {}, 3, 6, 180)
    df2 = df.copy(); df2.iloc[0, 0] = df2.iloc[0, 0] + 1 if df2.iloc[0, 0] == df2.iloc[0, 0] else 0
    assert k1 != lab._chave_comparacao(df2, e, ["logistic"], {}, 3, 6, 180)


def test_reaproveitar_e_mais_rapido_do_que_treinar(comparacao):
    df, d, comp, rec = comparacao
    lab._COMPARACOES.clear(); lab._guardar_comparacao("t", comp, rec)
    t0 = time.perf_counter(); lab._comparacao_guardada("t"); copia = time.perf_counter() - t0
    assert copia < max(0.5, comp.tempo_total * 5)
