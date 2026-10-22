"""O tutor de IA tem de saber o que o aluno vê nos gráficos (eixos, features, números), não só «algo: PCA»."""
import os

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


def _correr(corpo: str):
    script = f"import sys\nsys.path.insert(0, {ROOT!r})\nimport streamlit as st\nimport numpy as np\n" + corpo
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_registar_ecra_e_acrescentar_chegam_ao_contexto_da_pagina():
    at = _correr('''
from modules import tutor_ia as t
st.session_state["pagina"] = "classificacao"
t.registar("Resultado de classificação", "Accuracy: 90%")
t.acrescentar_ecra("x", ["GRÁFICO «Matriz de Confusão»: linhas = Real"], "Vê 4 caixas")
tit, txt = t._contexto_da_pagina("classificacao")
st.text(txt + "||" + t._estado_da_pagina("classificacao"))
''')
    saida = at.text[0].value
    assert "Accuracy: 90%" in saida and "Matriz de Confusão" in saida and "Vê 4 caixas" in saida


def test_estado_so_vale_para_a_pagina_certa():
    at = _correr('''
from modules import tutor_ia as t
st.session_state["pagina"] = "reducao"
t.registar_ecra("R", ["linha"], "estado da redução")
st.text("[" + t._estado_da_pagina("clustering") + "]")
''')
    assert at.text[0].value == "[]"


def test_reducao_descreve_eixos_features_e_variancia():
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from modules.nao_supervisionado import reducao
    d = load_iris(as_frame=True).frame
    f = ["sepal length (cm)", "sepal width (cm)"]
    m = PCA(2, random_state=42)
    Xr = m.fit_transform(StandardScaler().fit_transform(d[f]))
    txt, est = reducao._descrever_para_tutor(dict(
        X_red=Xr, algo="PCA", n_comp=2, dataset="Iris", features=f, normalizar=True, label_col="", classes=[],
        var_ratio=m.explained_variance_ratio_.tolist(), loadings=m.components_.tolist()))
    for esperado in ("Componente 1", "Componente 2", "sepal length (cm)", "Variância Explicada", "Iris", "mesma cor"):
        assert esperado in txt
    assert "dois gráficos" in est and len(txt) < 3000 and len(est) <= 580


def test_clustering_descreve_graficos_e_tamanhos():
    from modules.nao_supervisionado import clustering
    X = np.random.RandomState(0).randn(60, 2)
    lab = np.array([0] * 30 + [1] * 25 + [-1] * 5)
    linhas = clustering._descrever_para_tutor("K-Means", lab, X, 0.55, 0.9, ["a", "b"], "Iris", True)
    txt = "\n".join(linhas)
    assert "PC1" in txt and "Dimensão dos Clusters" in txt and "Ruído: 5" in txt and "boa separação" in txt


def test_anomalias_e_outros_modulos_importam_sem_erro():
    import importlib
    for m in ("modules.nao_supervisionado.anomalias", "modules.nao_supervisionado.clustering",
              "modules.redes_neurais.mlp", "modules.supervisionado.classificacao",
              "modules.supervisionado.regressao", "modules.primeiro_modelo"):
        importlib.import_module(m)
