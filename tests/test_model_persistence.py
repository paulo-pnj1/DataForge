"""Persistência: guardar/abrir, versões, isolamento por utilizador, assinatura e caminhos."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json  # noqa: E402

from modules.ml_engine import model_persistence as mp  # noqa: E402
from modules.ml_engine.model_training import treinar_modelo  # noqa: E402
from modules.ml_engine.preprocessing import preparar_dados  # noqa: E402
from _ml_dados import tabela_regressao  # noqa: E402


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda nome: None)
    d = preparar_dados(tabela_regressao(), "y", "regressao")
    r = treinar_modelo("linear_reg", d)
    return tmp_path / "modelos", d, r


def _guardar(utilizador, ambiente, nome="m1"):
    raiz, d, r = ambiente
    return mp.guardar_modelo(utilizador, r.pipeline, mp.meta_tabular(d, r, nome), raiz)


def test_guardar_e_abrir_da_as_mesmas_previsoes(ambiente):
    raiz, d, r = ambiente
    info = _guardar("ana", ambiente)
    mod, meta = mp.carregar_modelo("ana", info["id"], raiz)
    assert np.allclose(mod.predict(d.X_test), r.pipeline.predict(d.X_test))
    assert meta["alvo"] == "y" and meta["versao"] == 1 and meta["sha256"]


def test_versoes_incrementam_por_nome(ambiente):
    a, b = _guardar("ana", ambiente, "x"), _guardar("ana", ambiente, "x")
    assert (a["versao"], b["versao"]) == (1, 2)
    assert _guardar("ana", ambiente, "outro")["versao"] == 1


def test_utilizadores_nao_veem_modelos_uns_dos_outros(ambiente):
    raiz = ambiente[0]
    info = _guardar("ana", ambiente)
    assert mp.listar_modelos("bob", raiz) == []
    with pytest.raises(mp.ErroPersistencia):
        mp.carregar_modelo("bob", info["id"], raiz)
    assert not mp.apagar_modelo("bob", info["id"], raiz)
    assert len(mp.listar_modelos("ana", raiz)) == 1


def test_ficheiro_alterado_nao_e_aberto(ambiente):
    raiz = ambiente[0]
    info = _guardar("ana", ambiente)
    f = next(raiz.glob(f"*/{info['id']}.joblib"))
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(mp.ErroPersistencia, match="assinatura"):
        mp.carregar_modelo("ana", info["id"], raiz)


def test_ficheiro_sem_assinatura_nao_e_aberto(ambiente):
    raiz = ambiente[0]
    info = _guardar("ana", ambiente)
    next(raiz.glob(f"*/{info['id']}.sig")).unlink()
    with pytest.raises(mp.ErroPersistencia):
        mp.carregar_modelo("ana", info["id"], raiz)


def test_pickle_malicioso_assinado_com_outra_chave_e_recusado(ambiente):
    """Um joblib forjado com outra chave (ou sem chave) nunca chega a ser carregado."""
    import pickle
    raiz = ambiente[0]
    info = _guardar("ana", ambiente)
    marca = raiz.parent / "executou.txt"

    class Mau:
        def __reduce__(self):
            return (open, (str(marca), "w"))

    f = next(raiz.glob(f"*/{info['id']}.joblib"))
    f.write_bytes(pickle.dumps(Mau()))
    with pytest.raises(mp.ErroPersistencia):
        mp.carregar_modelo("ana", info["id"], raiz)
    assert not marca.exists()


@pytest.mark.parametrize("mau", ["../x", "..", "/etc/passwd", "0123456789abcdeg", "", "0123456789abcdef/../x", None])
def test_identificadores_e_caminhos_invalidos(ambiente, mau):
    _guardar("ana", ambiente)
    with pytest.raises(mp.ErroPersistencia):
        mp.carregar_modelo("ana", mau, ambiente[0])
    with pytest.raises(mp.ErroPersistencia):
        mp.apagar_modelo("ana", mau, ambiente[0])


def test_utilizador_vazio_e_recusado(ambiente):
    with pytest.raises(mp.ErroPersistencia):
        mp.listar_modelos("", ambiente[0])


def test_nome_do_utilizador_nao_escapa_da_pasta(ambiente):
    raiz = ambiente[0]
    _guardar("../../etc", ambiente)
    for p in raiz.rglob("*.joblib"):
        assert raiz.resolve() in p.resolve().parents


def test_limite_de_modelos(ambiente, monkeypatch):
    monkeypatch.setattr(mp, "MAX_MODELOS_POR_UTILIZADOR", 2)
    _guardar("ana", ambiente, "a")
    _guardar("ana", ambiente, "b")
    with pytest.raises(mp.ErroPersistencia, match="limite"):
        _guardar("ana", ambiente, "c")


def test_metadados_sem_credenciais_e_renomear_apagar(ambiente):
    raiz = ambiente[0]
    info = _guardar("ana", ambiente)
    texto = json.dumps(info).lower()
    assert "password" not in texto and "secret" not in texto and "key" not in texto.replace("keys", "")
    mp.renomear_modelo("ana", info["id"], "novo nome", raiz)
    assert mp.ler_meta("ana", info["id"], raiz)["nome"] == "novo nome"
    assert mp.apagar_modelo("ana", info["id"], raiz) and mp.listar_modelos("ana", raiz) == []
