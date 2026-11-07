"""Guardar, descarregar e testar NLP (TF-IDF), e a verificação dos modelos antigos, sem PyTorch."""
import io
import zipfile

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from modules.ml_engine import model_persistence as mp, modelos_dl as dl
from modules.ml_engine.modelo_legado import ErroLegado
from modules.ml_engine.modelos_extra import exportar_modelo
from modules.ml_engine.pacote_generico import ErroPacote


def _res_bow():
    textos = ["o jogo de futebol foi bom", "golo no estadio", "o banco subiu os juros", "a bolsa caiu hoje"] * 5
    vec = TfidfVectorizer().fit(textos)
    clf = LogisticRegression().fit(vec.transform(textos), [0, 1, 2, 3] * 5 if False else [0, 0, 1, 1] * 5)
    return {"vectorizer": vec, "model": clf, "class_names": ["desporto", "economia"], "acc": 1.0, "f1": 1.0,
            "setup": {"vec": "TF-IDF", "clf": "Logistic Regression"}}


def test_nlp_guardar_testar_e_descarregar(tmp_path):
    obj, meta = dl.de_nlp_bow(_res_bow(), "noticias")
    info = mp.guardar_modelo("paulo", obj, meta, raiz=tmp_path)
    modelo, mt = mp.carregar_modelo("paulo", info["id"], raiz=tmp_path)
    assert mt["tipo"] == "texto"
    classe, probas = modelo.prever_texto("golo no futebol")
    assert classe == "desporto" and abs(sum(probas.values()) - 1) < 1e-9
    with pytest.raises(ErroPacote):
        modelo.prever_texto("123 !!!")
    z = zipfile.ZipFile(io.BytesIO(exportar_modelo(obj, meta)))
    assert {"modelo.joblib", "metadados.json", "LEEME.txt"} <= set(z.namelist())


def test_resultado_antigo_pede_novo_treino():
    with pytest.raises(ErroLegado):
        dl.de_visao({}, "x")
    with pytest.raises(ErroLegado):
        dl.de_generativo({}, "gan", "x")
    with pytest.raises(ErroLegado):
        dl.de_serie_dl({}, "x")
