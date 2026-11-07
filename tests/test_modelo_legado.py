"""Guardar modelos das páginas Classificação/Regressão/Comparador (qualquer algoritmo) e usá-los com dados novos."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io  # noqa: E402

import joblib  # noqa: E402
from sklearn.ensemble import AdaBoostClassifier, BaggingRegressor  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

from modules.ml_engine import model_persistence as mp  # noqa: E402
from modules.ml_engine import prediction_service as ps  # noqa: E402
from modules.ml_engine.modelo_legado import ErroLegado, ModeloLegado, de_classificacao, de_regressao, do_comparador  # noqa: E402


def _dados(n=240, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"idade": rng.integers(18, 70, n).astype(float), "cidade": rng.choice(["Luanda", "Huambo", "Lobito"], n),
                       "horas": rng.normal(10, 3, n)})
    df["classe"] = np.where(df.horas + (df.cidade == "Luanda") * 3 + rng.normal(0, 1, n) > 12, "sim", "nao")
    df["preco"] = df.horas * 4 + df.idade * .5 + (df.cidade == "Luanda") * 10 + rng.normal(0, 2, n)
    return df


def _como_a_pagina_classificacao(df, modelo, scaler_cls=StandardScaler):
    """Reproduz `_executar_treino` da página Classificação (encoders, scaler, divisão) e devolve o dicionário de resultado."""
    feats, alvo = ["idade", "cidade", "horas"], "classe"
    d = df[feats + [alvo]].copy()
    enc = {}
    for c in d[feats].select_dtypes(include="object").columns:
        le = LabelEncoder(); d[c] = le.fit_transform(d[c].astype(str)); enc[c] = le
    le_t = LabelEncoder(); d[alvo] = le_t.fit_transform(d[alvo].astype(str))
    X, y = d[feats].to_numpy(dtype=float), d[alvo].to_numpy()
    sc = scaler_cls(); X = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    modelo.fit(Xtr, ytr)
    return {"model": modelo, "scaler": sc, "encoders": enc, "le_target": le_t, "feature_names": feats, "classes": le_t.classes_, "df_features": d[feats],
            "y": y, "X_train": Xtr, "X_test": Xte, "y_test": yte, "acc": .9, "prec": .9, "rec": .9, "f1": .9, "auc": None,
            "cv_scores": np.array([.8, .9, .85]), "setup": {"target": alvo}}, le_t, enc, sc


def test_classificacao_qualquer_algoritmo_preve_como_na_pagina(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda nome: None)
    df = _dados()
    for modelo in (AdaBoostClassifier(n_estimators=20, random_state=0), SGDClassifier(random_state=0), LinearDiscriminantAnalysis()):
        r, le_t, enc, sc = _como_a_pagina_classificacao(df, modelo)
        obj, meta = de_classificacao(r, type(modelo).__name__, "meu modelo")
        info = mp.guardar_modelo("ana", obj, meta, raiz=tmp_path)
        aberto, meta2 = mp.carregar_modelo("ana", info["id"], raiz=tmp_path)
        novos = df.drop(columns=["classe", "preco"]).head(30)
        saida, rel = ps.prever_tabela(aberto, meta2, novos)
        # «à mão», como a página faz
        X = novos.copy(); X["cidade"] = enc["cidade"].transform(X["cidade"])
        esperado = le_t.inverse_transform(modelo.predict(sc.transform(X[["idade", "cidade", "horas"]].to_numpy(float))))
        assert saida["previsao_classe"].tolist() == esperado.tolist(), type(modelo).__name__
        assert meta2["classes"] == ["nao", "sim"] and meta2["cat_cols"] == ["cidade"]


def test_categoria_nova_e_valores_em_falta_nao_rebentam(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda nome: None)
    df = _dados()
    r, *_ = _como_a_pagina_classificacao(df, AdaBoostClassifier(n_estimators=10, random_state=0))
    obj, meta = de_classificacao(r, "AdaBoost", "m")
    novos = pd.DataFrame({"idade": [30.0, None], "cidade": ["Benguela", "Luanda"], "horas": [None, 9.0]})
    saida, rel = ps.prever_tabela(obj, meta, novos)
    assert len(saida) == 2 and saida["previsao_classe"].notna().all()
    assert any("nunca vistas" in a for a in rel["avisos"])
    assert "confianca" in saida.columns


def test_probabilidades_alinhadas_quando_falta_uma_classe():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2)); y = np.array([0] * 30 + [2] * 30)           # a classe 1 não apareceu no treino
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression().fit(X, y)
    obj = ModeloLegado(["a", "b"], [], {}, None, m, "classificacao", np.array([0, 1, 2]), {"a": 0.0, "b": 0.0}, {})
    p = obj.predict_proba(pd.DataFrame(X[:5], columns=["a", "b"]))
    assert p.shape == (5, 3) and (p[:, 1] == 0).all() and np.allclose(p.sum(axis=1), 1)
    assert set(obj.predict(pd.DataFrame(X, columns=["a", "b"]))) <= {0, 2}


def test_regressao_qualquer_algoritmo(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda nome: None)
    df = _dados()
    feats = ["idade", "cidade", "horas"]
    d = df[feats + ["preco"]].copy()
    le = LabelEncoder(); d["cidade"] = le.fit_transform(d["cidade"])
    X, y = d[feats].to_numpy(float), d["preco"].to_numpy(float)
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=.2, random_state=42)
    m = BaggingRegressor(n_estimators=15, random_state=0).fit(Xtr, ytr)
    buf = io.BytesIO(); joblib.dump({"model": m, "scaler": sc, "encoders": {"cidade": le}, "features": feats}, buf)
    res = {"algo": "Bagging", "target": "preco", "features": feats, "n_linhas": len(d), "y_test": yte, "r2": .9, "rmse": 2., "mae": 1.5, "mse": 4.,
           "cv_scores": np.array([.8, .9]), "modelo_pkl": buf.getvalue(), "df_features": d[feats]}
    obj, meta = de_regressao(res, "preços")
    info = mp.guardar_modelo("ana", obj, meta, raiz=tmp_path)
    aberto, meta2 = mp.carregar_modelo("ana", info["id"], raiz=tmp_path)
    novos = df[feats].head(20)
    saida, _ = ps.prever_tabela(aberto, meta2, novos)
    Z = novos.copy(); Z["cidade"] = le.transform(Z["cidade"])
    assert np.allclose(saida["previsao_preco"], m.predict(sc.transform(Z[feats].to_numpy(float))))
    p, det, rel = ps.prever_um(aberto, meta2, {"idade": 40, "cidade": "Luanda", "horas": 10})
    assert isinstance(float(p), float) and not det
    assert meta2["intervalos_numericos"]["idade"][0] >= 18


def test_resultado_antigo_pede_para_treinar_outra_vez():
    with pytest.raises(ErroLegado, match="outra vez"):
        de_classificacao({}, "x", "y")


def test_comparador_so_com_colunas_numericas(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda nome: None)
    df = _dados()
    X = df[["idade", "horas"]]
    y = LabelEncoder().fit_transform(df["classe"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    m = make_pipeline(StandardScaler(), SGDClassifier(random_state=0, loss="log_loss")).fit(Xtr, ytr)
    res = {"tipo": "Classificação", "alvo": "classe", "classes": ["nao", "sim"], "X": X, "X_tr": Xtr, "X_te": Xte, "modelos": {"SGD": m},
           "resultados": [{"Algoritmo": "SGD", "Accuracy": .9, "F1 Score": .9, "CV Accuracy": .88, "Treino": .9}], "cat_cols": []}
    obj, meta = do_comparador(res, "SGD", "cmp")
    aberto, meta2 = mp.carregar_modelo("ana", mp.guardar_modelo("ana", obj, meta, raiz=tmp_path)["id"], raiz=tmp_path)
    saida, _ = ps.prever_tabela(aberto, meta2, X.head(10))
    assert saida["previsao_classe"].tolist() == np.array(["nao", "sim"])[m.predict(X.head(10))].tolist()
    assert "prob_sim" in saida.columns
    with pytest.raises(ErroLegado, match="texto"):
        do_comparador({**res, "cat_cols": ["cidade"]}, "SGD", "x")
    with pytest.raises(ErroLegado):
        do_comparador(res, "Inexistente", "x")
