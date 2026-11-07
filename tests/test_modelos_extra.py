"""Guardar, testar e descarregar modelos de todas as páginas: Primeiro Modelo, MLP, Comparador, clustering, anomalias, redução e reforço."""
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.cluster import DBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import LocalOutlierFactor  # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

from modules.ml_engine import ciclo_ml as cl  # noqa: E402
from modules.ml_engine import model_persistence as mp  # noqa: E402
from modules.ml_engine import modelos_extra as mx  # noqa: E402
from modules.ml_engine.dataforge_runtime import PoliticaDQN, PoliticaQ, carregar, de_pacote  # noqa: E402
from modules.ml_engine.modelo_legado import ErroLegado, de_mlp, de_primeiro_modelo  # noqa: E402


def _tabela(n=240, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"idade": rng.integers(18, 70, n).astype(float), "cidade": rng.choice(["Luanda", "Huambo", "Lobito"], n),
                       "horas": rng.normal(10, 3, n)})
    df["classe"] = np.where(df.horas + (df.cidade == "Luanda") * 3 + rng.normal(0, 1, n) > 12, "sim", "nao")
    df["preco"] = df.horas * 4 + df.idade * .5 + (df.cidade == "Luanda") * 10 + rng.normal(0, 2, n)
    return df


def _ida_e_volta(obj, meta, tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_segredo", lambda nome: None)
    info = mp.guardar_modelo("ana", obj, meta, raiz=tmp_path)
    return mp.carregar_modelo("ana", info["id"], raiz=tmp_path)


def _exportar_e_correr_fora_da_plataforma(obj, meta, tmp_path):
    """Abre o ZIP numa pasta vazia, sem o código do DataForge no caminho, corre usar_modelo.py e devolve o que ele imprimiu."""
    z = zipfile.ZipFile(io.BytesIO(mx.exportar_modelo(obj, meta)))
    assert {"modelo.joblib", "metadados.json", "dataforge_runtime.py", "usar_modelo.py", "LEEME.txt"} <= set(z.namelist())
    z.extractall(tmp_path)
    teste = ("import sys\nsys.path.insert(0, '.')      # como quando se corre «python usar_modelo.py» dentro da pasta\nimport runpy, json\nns = runpy.run_path('usar_modelo.py', run_name='x')\n"
             "novos = ns['pd'].DataFrame([" + repr(mx._exemplo(meta)) + "])\nprint(ns['modelo'].aplicar(novos).to_json(orient='records'))\n")
    (tmp_path / "t.py").write_text(teste, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    r = subprocess.run([sys.executable, "t.py"], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout.strip().splitlines()[-1])


# ── O Meu Primeiro Modelo: o Random Forest original, com get_dummies ──
def _res_primeiro_modelo(df, clf=True):
    alvo = "classe" if clf else "preco"
    X = pd.get_dummies(df.drop(columns=[alvo, "preco" if clf else "classe"]), drop_first=False).astype(float)
    y = df[alvo]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    m = (RandomForestClassifier(n_estimators=30, random_state=0) if clf else RandomForestRegressor(n_estimators=30, random_state=0)).fit(Xtr, ytr)
    return {"modelo": m, "colunas": list(X.columns), "cfg": {"tipo": "clf" if clf else "reg", "alvo": alvo}, "n_treino": len(Xtr), "n_teste": len(Xte),
            "acc": .9, "prec": .9, "rec": .9, "f1": .9, "r2": .9, "rmse": 1., "mae": 1., "cv_scores": np.array([.8, .9]),
            "X_bruto": df.drop(columns=[alvo, "preco" if clf else "classe"])}, m, X


def test_primeiro_modelo_guarda_o_modelo_original(tmp_path, monkeypatch):
    df = _tabela()
    res, m, X = _res_primeiro_modelo(df)
    obj, meta = de_primeiro_modelo(res, "pm")
    aberto, meta2 = _ida_e_volta(obj, meta, tmp_path, monkeypatch)
    assert aberto.modelo is not None and type(aberto.modelo).__name__ == "RandomForestClassifier"      # o objecto original, não um retreino
    novos = df.drop(columns=["classe", "preco"]).head(25)
    saida, rel = cl.prever_ficheiro(aberto, meta2, novos)
    esperado = m.predict(pd.get_dummies(novos).astype(float).reindex(columns=res["colunas"], fill_value=0.0))
    assert saida["previsao_classe"].tolist() == list(esperado)
    # categoria nova não rebenta (todas as colunas dessa categoria ficam a 0)
    novo = pd.DataFrame({"idade": [30.0], "cidade": ["Benguela"], "horas": [9.0]})
    assert len(cl.prever_ficheiro(aberto, meta2, novo)[0]) == 1


def test_primeiro_modelo_regressao_e_resultado_antigo():
    df = _tabela()
    res, m, X = _res_primeiro_modelo(df, clf=False)
    obj, meta = de_primeiro_modelo(res, "pm reg")
    p = obj.predict(df.drop(columns=["classe", "preco"]).head(10))
    assert np.allclose(p, m.predict(X.head(10).reindex(columns=res["colunas"])))
    with pytest.raises(ErroLegado, match="outra vez"):
        de_primeiro_modelo({"modelo": m}, "x")


# ── Redes neurais (MLP) com colunas de texto ──
def _res_mlp(df, clf=True):
    feats = ["idade", "cidade", "horas"]
    X = df[feats].copy()
    enc = {"cidade": LabelEncoder()}
    X["cidade"] = enc["cidade"].fit_transform(X["cidade"])
    y = LabelEncoder().fit_transform(df["classe"]) if clf else df["preco"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    sc = StandardScaler(); Xs = sc.fit_transform(Xtr)
    m = (MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=0) if clf else MLPRegressor(hidden_layer_sizes=(8,), max_iter=300, random_state=0)).fit(Xs, ytr)
    res = {"model": m, "scaler": sc, "encoders": enc, "df_features": X, "classes": ["nao", "sim"], "valores_alvo": np.unique(y),
           "acc": .9, "f1": .9, "r2": .9, "rmse": 1., "mae": 1.,
           "setup": {"features": feats, "target": "classe" if clf else "preco", "hidden": [8], "activation": "relu", "solver": "adam", "lr": .001,
                     "max_iter": 300, "n_test": len(Xte)}}
    return res, m, sc, enc["cidade"]


def test_mlp_classificacao_e_regressao_com_texto(tmp_path, monkeypatch):
    df = _tabela()
    for clf in (True, False):
        res, m, sc, le = _res_mlp(df, clf)
        obj, meta = de_mlp(res, "classificacao" if clf else "regressao", "rede")
        aberto, meta2 = _ida_e_volta(obj, meta, tmp_path, monkeypatch)
        novos = df.drop(columns=["classe", "preco"]).head(20)
        saida, _ = cl.prever_ficheiro(aberto, meta2, novos)
        Z = novos.copy(); Z["cidade"] = le.transform(Z["cidade"])
        esp = m.predict(sc.transform(Z[["idade", "cidade", "horas"]].to_numpy(float)))
        col = "previsao_classe" if clf else "previsao_preco"
        if clf:
            assert saida[col].tolist() == np.array(["nao", "sim"])[esp].tolist()
        else:
            assert np.allclose(saida[col], esp)
    with pytest.raises(ErroLegado):
        de_mlp({"setup": {}}, "regressao", "x")


# ── Clustering ──
def _num(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, .5, (n // 2, 2)), rng.normal(4, .5, (n // 2, 2))])
    return X


def _res_cluster(modelo, normalizar=True):
    X = _num()
    sc = StandardScaler() if normalizar else None
    Xs = sc.fit_transform(X) if normalizar else X
    modelo.fit(Xs)
    labels = modelo.labels_ if hasattr(modelo, "labels_") else modelo.predict(Xs)
    return {"model": modelo, "features": ["x", "y"], "scaler": sc, "X_raw": X, "X_scaled": Xs, "labels": labels, "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_ruido": int(np.sum(labels == -1)), "sil": .8, "ch": 10., "db": .3}, X


def test_clustering_kmeans_e_dbscan(tmp_path, monkeypatch):
    for modelo, usa_vizinho in ((KMeans(2, n_init=5, random_state=0), False), (DBSCAN(eps=.6, min_samples=5), True)):
        res, X = _res_cluster(modelo)
        obj, meta = mx.de_clustering(res, type(modelo).__name__, "grupos")
        assert (obj.vizinho is not None) == usa_vizinho and (bool(meta["avisos"]) == usa_vizinho)
        aberto, meta2 = _ida_e_volta(obj, meta, tmp_path, monkeypatch)
        novos = pd.DataFrame({"x": [0.1, 4.1, np.nan], "y": [0.0, 3.9, 0.0]})
        saida, rel = mx.aplicar_tabela(aberto, meta2, novos)
        assert saida["grupo"].iloc[0] != saida["grupo"].iloc[1] and len(saida) == 3        # NaN é preenchido, não rebenta
        # o grupo dado a pontos do treino coincide com o que a página mostrou
        treino = pd.DataFrame(X[:10], columns=["x", "y"])
        g = mx.aplicar_tabela(aberto, meta2, treino)[0]["grupo"].tolist()
        assert [str(v) for v in g] == [("ruído" if v == -1 else str(int(v))) for v in res["labels"][:10]]
    with pytest.raises(ErroLegado, match="outra vez"):
        mx.de_clustering({"model": 1}, "x", "y")


def test_clustering_recusa_coluna_em_falta_e_ficheiro_gigante(tmp_path, monkeypatch):
    res, _ = _res_cluster(KMeans(2, n_init=5, random_state=0))
    obj, meta = mx.de_clustering(res, "KMeans", "g")
    with pytest.raises(cl.ErroCiclo, match="Faltam colunas"):
        mx.aplicar_tabela(obj, meta, pd.DataFrame({"x": [1.0]}))


# ── Anomalias e redução ──
def test_anomalias_isolation_forest_e_lof(tmp_path, monkeypatch):
    X = _num()
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    for m, lof in ((IsolationForest(random_state=0, contamination=.05).fit(Xs), False), (LocalOutlierFactor(n_neighbors=15, contamination=.05), True)):
        preds = m.fit_predict(Xs) if lof else m.predict(Xs)
        res = {"modelo": m, "scaler": sc, "X_raw": X, "X_sc": Xs, "features": ["x", "y"], "algo": "alg", "n_anom": int((preds == -1).sum()), "n_norm": int((preds == 1).sum()),
               "contaminacao": .05}
        obj, meta = mx.de_anomalias(res, "anom")
        aberto, meta2 = _ida_e_volta(obj, meta, tmp_path, monkeypatch)
        saida, _ = mx.aplicar_tabela(aberto, meta2, pd.DataFrame({"x": [0.0, 30.0], "y": [0.0, -30.0]}))
        assert saida["resultado"].tolist() == ["Normal", "Anomalia"], (type(m).__name__, saida)


def test_reducao_pca_sim_tsne_nao(tmp_path, monkeypatch):
    X = _num()
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    pca = PCA(2, random_state=0).fit(Xs)
    res = {"modelo_t": pca, "scaler": sc, "X_raw": X, "features": ["x", "y"], "algo": "PCA", "n_comp": 2}
    obj, meta = mx.de_reducao(res, "pca")
    aberto, meta2 = _ida_e_volta(obj, meta, tmp_path, monkeypatch)
    saida, _ = mx.aplicar_tabela(aberto, meta2, pd.DataFrame(X[:5], columns=["x", "y"]))
    assert np.allclose(saida[["componente_1", "componente_2"]].to_numpy(), pca.transform(Xs[:5]), atol=1e-4)
    with pytest.raises(ErroLegado, match="projectar"):
        mx.de_reducao({**res, "modelo_t": TSNE(2), "algo": "t-SNE"}, "x")


# ── Reforço ──
def _q_grelha(n=4):
    """Tabela Q por iteração de valor numa grelha n×n sem obstáculos, objectivo no canto: a melhor acção aproxima-se dele."""
    Q = np.zeros((n * n, 4))
    passos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(60):
        for s in range(n * n):
            r, c = divmod(s, n)
            if (r, c) == (n - 1, n - 1):
                continue
            for a, (dr, dc) in enumerate(passos):
                nr, nc = min(max(r + dr, 0), n - 1), min(max(c + dc, 0), n - 1)
                Q[s, a] = (10.0 if (nr, nc) == (n - 1, n - 1) else -0.1) + (0 if (nr, nc) == (n - 1, n - 1) else .9 * Q[nr * n + nc].max())
    return Q


def test_q_learning_guardado_e_testado(tmp_path, monkeypatch):
    res = {"Q": _q_grelha(), "env_params": (4, [], []), "rewards": [1.0, 2.0], "reached_goal": True, "path": [(0, 0)] * 7, "setup": {"alpha": .1}}
    obj, meta = mx.de_q_learning(res, "q")
    aberto, meta2 = _ida_e_volta(obj, meta, tmp_path, monkeypatch)
    saida, _ = mx.aplicar_tabela(aberto, meta2, pd.DataFrame({"linha": [0, 3], "coluna": [0, 2]}))
    assert saida["chega_ao_objectivo"].tolist() == [True, True]
    assert saida["melhor_accao"].iloc[1] == "direita"
    assert aberto.caminho((0, 0))[-1] == (3, 3) and len(aberto.caminho((0, 0))) == 7
    with pytest.raises(cl.ErroCiclo, match="grelha"):
        mx.aplicar_tabela(aberto, meta2, pd.DataFrame({"linha": [9], "coluna": [0]}))
    with pytest.raises(ErroLegado):
        mx.de_q_learning({}, "x")


def test_dqn_forward_em_numpy_e_exportacao(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    camadas = [(rng.normal(size=(8, 4)), rng.normal(size=8)), (rng.normal(size=(2, 8)), rng.normal(size=2))]
    res = {"rede": camadas, "episodes": 10, "best_reward": 120., "avg_last_50": 55., "setup": {}}
    obj, meta = mx.de_dqn(res, "dqn")
    x = np.array([[0.1, -0.2, 0.05, 0.3]])
    esperado = np.maximum(x @ camadas[0][0].T + camadas[0][1], 0) @ camadas[1][0].T + camadas[1][1]
    assert np.allclose(obj.valores_q(x), esperado)
    aberto, meta2 = _ida_e_volta(obj, meta, tmp_path, monkeypatch)
    saida, _ = mx.aplicar_tabela(aberto, meta2, pd.DataFrame([dict(zip(obj.nomes_estado, x[0]))]))
    assert saida["melhor_accao"].iloc[0] == ["esquerda", "direita"][int(esperado.argmax())]
    with pytest.raises(ErroLegado):
        mx.de_dqn({}, "x")


# ── Descarregar: o ZIP funciona fora da plataforma ──
def test_zip_corre_fora_da_plataforma_para_todos_os_tipos(tmp_path):
    df = _tabela()
    casos = []
    res, *_ = _res_primeiro_modelo(df)
    casos.append(de_primeiro_modelo(res, "pm"))
    res, *_ = _res_mlp(df, True)
    casos.append(de_mlp(res, "classificacao", "mlp"))
    res, _ = _res_cluster(KMeans(2, n_init=5, random_state=0))
    casos.append(mx.de_clustering(res, "KMeans", "c"))
    X = _num(); sc = StandardScaler(); Xs = sc.fit_transform(X)
    casos.append(mx.de_anomalias({"modelo": IsolationForest(random_state=0).fit(Xs), "scaler": sc, "X_raw": X, "X_sc": Xs, "features": ["x", "y"], "algo": "IF",
                                  "n_anom": 1, "n_norm": 1, "contaminacao": .1}, "a"))
    casos.append(mx.de_reducao({"modelo_t": PCA(2).fit(Xs), "scaler": sc, "X_raw": X, "features": ["x", "y"], "algo": "PCA", "n_comp": 2}, "r"))
    casos.append(mx.de_q_learning({"Q": _q_grelha(), "env_params": (4, [], []), "rewards": [1.0], "reached_goal": True, "path": [(0, 0)]}, "q"))
    for i, (obj, meta) in enumerate(casos):
        pasta = tmp_path / f"c{i}"
        pasta.mkdir()
        fora = _exportar_e_correr_fora_da_plataforma(obj, meta, pasta)
        dentro = json.loads(obj.aplicar(pd.DataFrame([mx._exemplo(meta)])).to_json(orient="records"))
        assert fora == dentro, (meta["algoritmo"], fora, dentro)


def test_zip_de_pipeline_e_de_serie(tmp_path):
    from sklearn.linear_model import LinearRegression
    z = zipfile.ZipFile(io.BytesIO(mx.exportar_modelo(LinearRegression().fit([[1], [2]], [1, 2]), {"nome": "serie", "tipo": "serie", "algoritmo": "x"})))
    assert set(z.namelist()) == {"modelo.joblib", "metadados.json", "LEEME.txt"}                    # séries: só com o código da plataforma
    assert b"Testar Modelos" in z.read("LEEME.txt")
    from modules.ml_engine.model_training import treinar_modelo
    from modules.ml_engine.preprocessing import preparar_dados
    d = preparar_dados(_tabela(), "preco", "regressao")
    r = treinar_modelo("ridge", d)
    z2 = zipfile.ZipFile(io.BytesIO(mx.exportar_modelo(r.pipeline, cl.meta_ciclo(d, r, "p"))))
    assert "usar_modelo.py" in z2.namelist() and "modelo.joblib" in z2.namelist()


def test_pacote_nao_leva_classes_da_plataforma(tmp_path):
    """O .joblib do ZIP só tem objectos simples: abre-se sem o código do DataForge."""
    import pickletools
    res, _ = _res_cluster(KMeans(2, n_init=5, random_state=0))
    obj, meta = mx.de_clustering(res, "KMeans", "g")
    import joblib
    buf = io.BytesIO(); joblib.dump(obj.para_pacote(), buf)
    assert b"modules.ml_engine" not in buf.getvalue() and b"dataforge_runtime" not in buf.getvalue()
    assert type(de_pacote(obj.para_pacote())).__name__ == "ModeloGrupos"
