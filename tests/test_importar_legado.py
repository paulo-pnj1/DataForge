"""Importar .pkl antigos das páginas (leitura restrita): aceita modelos legítimos e recusa ficheiros que executariam código."""
import io
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joblib import numpy_pickle  # noqa: E402
from sklearn.ensemble import AdaBoostClassifier, BaggingRegressor  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

from modules.ml_engine import importar_legado as il  # noqa: E402
from modules.ml_engine import model_persistence as mp  # noqa: E402
from modules.ml_engine import prediction_service as ps  # noqa: E402
from modules.ml_engine.modelo_legado import ErroLegado  # noqa: E402
from test_modelo_legado import _como_a_pagina_classificacao, _dados  # noqa: E402


def _bytes(obj) -> bytes:
    b = io.BytesIO()
    joblib.dump(obj, b)
    return b.getvalue()


def test_classificacao_antiga_importa_e_preve_como_na_pagina(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda nome: None)
    df = _dados()
    r, le_t, enc, sc = _como_a_pagina_classificacao(df, AdaBoostClassifier(n_estimators=15, random_state=0))
    pkl = _bytes({"model": r["model"], "scaler": sc, "encoders": enc, "le_target": le_t, "features": r["feature_names"]})
    info = il.inspeccionar(pkl)
    assert info["tipo"] == "classificacao" and info["features"] == ["idade", "cidade", "horas"] and info["tem_codificadores"]
    obj, meta = il.construir(info, "antigo", "classe")
    guardado = mp.guardar_modelo("ana", obj, meta, raiz=tmp_path)
    aberto, meta2 = mp.carregar_modelo("ana", guardado["id"], raiz=tmp_path)
    novos = df.drop(columns=["classe", "preco"]).head(20)
    saida, _ = ps.prever_tabela(aberto, meta2, novos)
    X = novos.copy(); X["cidade"] = enc["cidade"].transform(X["cidade"])
    esperado = le_t.inverse_transform(r["model"].predict(sc.transform(X[["idade", "cidade", "horas"]].to_numpy(float))))
    assert saida["previsao_classe"].tolist() == esperado.tolist()


def test_regressao_antiga_e_mlp_sem_colunas(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda nome: None)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3)); y = X @ [1., 2., 3.] + rng.normal(0, .1, 150)
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    m = BaggingRegressor(n_estimators=10, random_state=0).fit(Xs, y)
    info = il.inspeccionar(_bytes({"model": m, "scaler": sc, "encoders": {}, "features": ["a", "b", "c"]}))
    assert info["tipo"] == "regressao"
    obj, meta = il.construir(info, "reg", "alvo")
    assert np.allclose(obj.predict(pd.DataFrame(X[:5], columns=["a", "b", "c"])), m.predict(Xs[:5]))
    # MLP: o ficheiro antigo só tem model + scaler, por isso as colunas têm de ser indicadas
    yc = (y > 0).astype(int)
    mlp = MLPClassifier(hidden_layer_sizes=(6,), max_iter=300, random_state=0).fit(Xs, yc)
    info2 = il.inspeccionar(_bytes({"model": mlp, "scaler": sc}))
    with pytest.raises(ErroLegado, match="colunas"):
        il.construir(info2, "mlp", "alvo")
    with pytest.raises(ErroLegado, match="3 colunas"):
        il.construir(info2, "mlp", "alvo", ["a", "b"])
    with pytest.raises(ErroLegado, match="prevê"):
        il.construir(info2, "mlp", " ", ["a", "b", "c"])
    obj2, meta2 = il.construir(info2, "mlp", "alvo", ["a", "b", "c"])
    assert (obj2.predict(pd.DataFrame(X[:20], columns=["a", "b", "c"])) == mlp.predict(Xs[:20])).all()


class _Mau:
    def __reduce__(self):
        return (os.system, ("touch " + MARCA,))


MARCA = "/tmp/dataforge_pwned_marker"


def test_ficheiros_que_executam_codigo_sao_recusados():
    if os.path.exists(MARCA):
        os.remove(MARCA)
    for payload in (_bytes({"model": _Mau()}), _bytes(_Mau())):
        with pytest.raises(ErroLegado, match="segurança"):
            il.inspeccionar(payload)
    assert not os.path.exists(MARCA), "o código do ficheiro foi executado!"


def test_truques_com_nomes_com_ponto_e_builtins_perigosos():
    def pickle_global(modulo: str, nome: str, arg: str) -> bytes:
        m, n, a = modulo.encode(), nome.encode(), arg.encode()
        return (b"\x80\x04" + b"\x8c" + bytes([len(m)]) + m + b"\x94" + b"\x8c" + bytes([len(n)]) + n + b"\x94" + b"\x93\x94"
                + b"\x8c" + bytes([len(a)]) + a + b"\x94\x85\x94R.")
    for modulo, nome, arg in (("sklearn.utils.fixes", "os.system", "touch " + MARCA), ("builtins", "eval", "print(1)"), ("builtins", "__import__", "os"),
                              ("os", "system", "touch " + MARCA), ("subprocess", "getoutput", "id"), ("sklearn.utils.fixes", "os", "x")):
        with pytest.raises(ErroLegado):
            il.ler_pkl(pickle_global(modulo, nome, arg))
    assert not os.path.exists(MARCA)


def test_lixo_dicionario_errado_e_ficheiro_gigante():
    with pytest.raises(ErroLegado, match="segurança|Não consegui ler"):
        il.ler_pkl(b"isto nao e um modelo")
    with pytest.raises(ErroLegado, match="não parece"):
        il.inspeccionar(_bytes({"outra": 1}))
    with pytest.raises(ErroLegado, match="grande"):
        il.ler_pkl(b"0" * (il.MAX_BYTES + 1))


def test_protecao_activa_e_leitor_normal_restaurado():
    original = numpy_pickle.NumpyUnpickler
    assert il._canario_ok() is True
    il.inspeccionar(_bytes({"model": BaggingRegressor(n_estimators=2).fit([[1.], [2.], [3.]], [1., 2., 3.])}))
    assert numpy_pickle.NumpyUnpickler is original            # o joblib normal não fica alterado para o resto da aplicação
    assert joblib.load(io.BytesIO(_bytes({"a": 1}))) == {"a": 1}
