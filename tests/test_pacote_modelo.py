"""Guardar qualquer modelo tabular e testá-lo com dados novos. `pytest -q tests/test_pacote_modelo.py`."""
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from modules.ml_engine import model_persistence as mp, pacote_modelo as pm, prediction_service as ps


@pytest.fixture()
def dados():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"idade": rng.integers(18, 70, 200).astype(float), "cidade": rng.choice(["Luanda", "Huambo", "Benguela"], 200),
                       "renda": rng.normal(100, 20, 200)})
    enc = {"cidade": LabelEncoder().fit(df.cidade)}
    Xe = df.copy()
    Xe["cidade"] = enc["cidade"].transform(df.cidade)
    return df, Xe, enc


def _guardar_abrir(tmp_path, pacote, meta):
    info = mp.guardar_modelo("paulo", pacote, meta, raiz=tmp_path)
    return mp.carregar_modelo("paulo", info["id"], raiz=tmp_path)


def test_classificacao_com_categorias_e_scaler(tmp_path, dados):
    df, Xe, enc = dados
    y = np.where(df.renda + (df.cidade == "Luanda") * 30 > 120, "alto", "baixo")
    le = LabelEncoder().fit(y)
    sc = StandardScaler().fit(Xe.to_numpy(float))
    m = RandomForestClassifier(50, random_state=0).fit(sc.transform(Xe.to_numpy(float)), le.transform(y))
    pac, meta = pm.construir(m, tarefa="classificacao", features=list(df.columns), alvo="nivel", algoritmo="RF", nome="t",
                             origem="teste", X_ref=Xe, scaler=sc, encoders=enc, le_target=le)
    modelo, mt = _guardar_abrir(tmp_path, pac, meta)
    saida, _ = ps.prever_tabela(modelo, mt, df)
    assert (saida["previsao_nivel"].values == y).mean() > 0.95
    assert set(mt["classes"]) == {"alto", "baixo"}
    p, det, rel = ps.prever_um(modelo, mt, {"idade": 30, "cidade": "Namibe", "renda": 100.0})      # cidade nunca vista
    assert p in ("alto", "baixo") and any("Namibe" in a for a in rel["avisos"])


def test_regressao(tmp_path, dados):
    df, Xe, enc = dados
    sc = StandardScaler().fit(Xe.to_numpy(float))
    reg = LinearRegression().fit(sc.transform(Xe.to_numpy(float)), df.renda)
    pac, meta = pm.construir(reg, tarefa="regressao", features=list(df.columns), alvo="renda", algoritmo="LR", nome="r",
                             origem="t", X_ref=Xe, scaler=sc, encoders=enc)
    modelo, mt = _guardar_abrir(tmp_path, pac, meta)
    p, _, _ = ps.prever_um(modelo, mt, {"idade": 30, "cidade": "Luanda", "renda": 100.0})
    assert abs(p - 100.0) < 1e-6


def test_clustering_e_anomalias(tmp_path, dados):
    _, Xe, _ = dados
    X = Xe[["idade", "renda"]]
    sc = StandardScaler().fit(X.to_numpy(float))
    km = KMeans(3, n_init=5, random_state=0).fit(sc.transform(X.to_numpy(float)))
    pac, meta = pm.construir(km, tarefa="agrupamento", features=["idade", "renda"], alvo="grupo", algoritmo="KMeans", nome="k",
                             origem="t", X_ref=X, scaler=sc)
    modelo, mt = _guardar_abrir(tmp_path, pac, meta)
    saida, _ = ps.prever_tabela(modelo, mt, X.head(5))
    assert saida["grupo"].str.startswith("Grupo").all()
    iso = IsolationForest(random_state=0).fit(sc.transform(X.to_numpy(float)))
    pac, meta = pm.construir(iso, tarefa="anomalias", features=["idade", "renda"], alvo="a", algoritmo="IF", nome="a",
                             origem="t", X_ref=X, scaler=sc)
    modelo, mt = _guardar_abrir(tmp_path, pac, meta)
    saida, _ = ps.prever_tabela(modelo, mt, pd.DataFrame({"idade": [40, 900], "renda": [100, 99999]}))
    assert list(saida["resultado"]) == ["Normal", "Anomalia"]


def test_algoritmo_sem_predict_e_recusado():
    assert "novos" in pm.pode_guardar(DBSCAN(), "agrupamento")
    with pytest.raises(pm.ErroEmpacotar):
        pm.construir(DBSCAN(), tarefa="agrupamento", features=["a"], alvo="g", algoritmo="DBSCAN", nome="d", origem="t",
                     X_ref=pd.DataFrame({"a": [1.0, 2.0]}))


def test_modo_get_dummies(tmp_path):
    rng = np.random.default_rng(1)
    raw = pd.DataFrame({"sexo": rng.choice(["m", "f"], 300), "idade": rng.integers(1, 80, 300).astype(float)})
    y = pd.Series(np.where(raw.sexo == "f", "sim", "nao"))
    X = pd.get_dummies(raw).astype(float)
    m = RandomForestClassifier(30, random_state=0).fit(X, y)
    pac, meta = pm.construir(m, tarefa="classificacao", features=list(raw.columns), alvo="r", algoritmo="RF", nome="d",
                             origem="t", X_ref=raw, colunas_dummy=list(X.columns))
    modelo, mt = _guardar_abrir(tmp_path, pac, meta)
    assert ps.prever_um(modelo, mt, {"sexo": "f", "idade": 20})[0] == "sim"


def test_descarregar_inclui_algoritmos_que_nao_se_guardam():
    import io
    import joblib
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0], "b": [1.0, 2.0, 1.5, 9.0, 10.0, 11.0]})
    db = DBSCAN(eps=3, min_samples=2).fit(X.to_numpy())
    pac, meta = pm.construir(db, tarefa="agrupamento", features=["a", "b"], alvo="g", algoritmo="DBSCAN", nome="d",
                             origem="t", X_ref=X, verificar=False)
    d = joblib.load(io.BytesIO(pm.bytes_para_descarregar(pac, meta)))
    assert d["formato"] == "dataforge-pacote-v1" and d["meta"]["algoritmo"] == "DBSCAN"


def test_generico_texto_reducao_e_reforco(tmp_path):
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from modules.ml_engine import pacote_generico as pg

    textos = ["o jogo de futebol foi bom", "golo no estadio", "o banco subiu os juros", "a bolsa caiu hoje"] * 5
    y = [0, 0, 1, 1] * 5
    vec = TfidfVectorizer().fit(textos)
    clf = LogisticRegression().fit(vec.transform(textos), y)
    pac, meta = pg.construir("texto", {"vectorizer": vec, "model": clf, "class_names": ["desporto", "economia"]},
                             nome="t", algoritmo="TF-IDF + LR", origem="t")
    modelo, mt = _guardar_abrir(tmp_path, pac, meta)
    assert mt["tipo"] == "texto"
    classe, probas = modelo.prever_texto("golo no futebol")
    assert classe == "desporto" and abs(sum(probas.values()) - 1) < 1e-9

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=["a", "b", "c"])
    sc = StandardScaler().fit(X.to_numpy())
    pca = PCA(2).fit(sc.transform(X.to_numpy()))
    pac, meta = pg.construir("reducao", {"model": pca, "scaler": sc, "features": ["a", "b", "c"]}, nome="p", algoritmo="PCA", origem="t")
    modelo, _ = _guardar_abrir(tmp_path, pac, meta)
    assert modelo.transformar(X.head(4)).shape == (4, 2)
    with pytest.raises(pg.ErroPacote):
        modelo.transformar(X[["a"]])

    Q = np.zeros((9, 4))
    Q[:, 3] = 1.0
    pac, meta = pg.construir("reforco_q", {"Q": Q, "env_params": (3, [], [])}, nome="q", algoritmo="Q", origem="t")
    modelo, _ = _guardar_abrir(tmp_path, pac, meta)
    pol = modelo.politica()
    assert pol.shape == (3, 3) and pol.iloc[0, 0] == "→" and pol.iloc[2, 2] == "★"
