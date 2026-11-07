"""Importar um modelo que descarregaste antes (.pkl das páginas Classificação, Regressão e Redes Neurais) para o poderes guardar e testar.

Abrir um .pkl qualquer executa código do ficheiro, por isso aqui a leitura é RESTRITA: só se reconstroem classes do scikit-learn,
numpy e scipy (mais uns tipos básicos). Qualquer outra coisa (os.system, eval, funções arbitrárias) é recusada e o ficheiro não é aberto.
É uma protecção boa mas não infalível: importa só ficheiros que tu mesmo descarregaste da plataforma.
"""
import io
import threading

import joblib
import numpy as np
from joblib import numpy_pickle

from modules.ml_engine import model_persistence as mp
from modules.ml_engine.dataforge_runtime import ModeloLegado
from modules.ml_engine.modelo_legado import ErroLegado

MAX_BYTES = 30 * 1024 * 1024
_PREFIXOS = ("sklearn.", "numpy.", "scipy.")
_FUNCOES_OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"), ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
               ("numpy.random._pickle", "__randomstate_ctor"), ("numpy.random._pickle", "__generator_ctor"), ("numpy.random._pickle", "__bit_generator_ctor"),
               ("copyreg", "_reconstructor"), ("joblib.numpy_pickle", "NumpyArrayWrapper"), ("joblib.numpy_pickle", "NDArrayWrapper"),
               ("collections", "OrderedDict"), ("collections", "defaultdict")}
_BUILTINS_OK = {"object", "set", "frozenset", "dict", "list", "tuple", "int", "float", "str", "bytes", "bytearray", "complex", "bool", "slice", "range"}
_LOCK = threading.Lock()
_CANARIO = None


class _Restrito(numpy_pickle.NumpyUnpickler):
    def find_class(self, module, name):
        if "." in name:                                    # «sklearn.x» + «os.system»: o truque dos nomes com ponto
            raise ErroLegado(f"Objecto não permitido: {module}.{name}")
        if module == "builtins":
            if name in _BUILTINS_OK:
                return super().find_class(module, name)
        elif (module, name) in _FUNCOES_OK:
            return super().find_class(module, name)
        elif module.startswith(_PREFIXOS) or module in ("numpy", "scipy"):
            obj = super().find_class(module, name)
            if isinstance(obj, type) and str(getattr(obj, "__module__", "")).startswith(("sklearn", "numpy", "scipy")):
                return obj
        raise ErroLegado(f"Objecto não permitido: {module}.{name}")


def _ler(dados: bytes):
    with _LOCK:
        original = numpy_pickle.NumpyUnpickler
        numpy_pickle.NumpyUnpickler = _Restrito
        try:
            return joblib.load(io.BytesIO(dados))
        finally:
            numpy_pickle.NumpyUnpickler = original


def _canario_ok() -> bool:
    """Confirma que a leitura restrita está mesmo activa (se uma versão futura do joblib a contornar, recusamos importar)."""
    global _CANARIO
    if _CANARIO is None:
        import os
        buf = io.BytesIO()
        joblib.dump(os.getcwd, buf)
        try:
            _ler(buf.getvalue())
            _CANARIO = False
        except ErroLegado:
            _CANARIO = True
        except Exception:
            _CANARIO = False
    return _CANARIO


def ler_pkl(dados: bytes):
    if len(dados) > MAX_BYTES:
        raise ErroLegado("O ficheiro é demasiado grande para importar.")
    if not _canario_ok():
        raise ErroLegado("A leitura segura de modelos não está disponível neste servidor, por isso não importo ficheiros.")
    try:
        return _ler(dados)
    except ErroLegado as e:
        raise ErroLegado(f"Por segurança não abri este ficheiro: tem objectos que o DataForge não reconhece ({e}). "
                         "Só se importam modelos descarregados desta plataforma.")
    except Exception:
        raise ErroLegado("Não consegui ler este ficheiro. É mesmo um .pkl descarregado de uma página de modelos?")


def inspeccionar(dados: bytes) -> dict:
    """Abre (em modo restrito) e diz o que o ficheiro tem, para a página saber o que perguntar ao estudante."""
    obj = ler_pkl(dados)
    if not isinstance(obj, dict) or "model" not in obj or not hasattr(obj["model"], "predict"):
        raise ErroLegado("Este ficheiro não parece um modelo exportado de uma página do DataForge.")
    modelo = obj["model"]
    clf = "le_target" in obj or hasattr(modelo, "classes_")
    n = int(getattr(modelo, "n_features_in_", 0) or 0)
    return {"objecto": obj, "tipo": "classificacao" if clf else "regressao", "features": list(obj.get("features") or []), "n_features": n,
            "tem_codificadores": bool(obj.get("encoders")), "algoritmo": type(modelo).__name__}


def construir(info: dict, nome: str, alvo: str, features=None) -> tuple:
    """(objecto, metadados) a partir de `inspeccionar`. `features` só é preciso se o ficheiro não as trouxer (redes neurais)."""
    obj, tipo = info["objecto"], info["tipo"]
    modelo, scaler, enc = obj["model"], obj.get("scaler"), obj.get("encoders") or {}
    feats = [str(f) for f in (info["features"] or features or [])]
    if not feats:
        raise ErroLegado("Diz quais são as colunas (pela ordem do treino).")
    if info["n_features"] and len(feats) != info["n_features"]:
        raise ErroLegado(f"O modelo foi treinado com {info['n_features']} colunas e indicaste {len(feats)}.")
    if not alvo.strip():
        raise ErroLegado("Diz o que o modelo prevê.")
    media = getattr(scaler, "mean_", None)
    preench = {c: float(m) for c, m in zip(feats, media)} if media is not None and len(media) == len(feats) else {}
    classes = None
    valores = None
    if tipo == "classificacao":
        valores = np.asarray(modelo.classes_)
        le_t = obj.get("le_target")
        classes = [str(c) for c in (le_t.classes_ if le_t is not None else valores)]
        if len(classes) != len(valores):
            raise ErroLegado("As classes do ficheiro não batem certo com as do modelo.")
    obj_final = ModeloLegado(feats, list(enc), enc, scaler, modelo, tipo, valores, preench, {}, nomes_classes=classes)
    cats = {c: [str(x) for x in le.classes_][:50] for c, le in enc.items()}
    meta = mp._simples({"tipo": tipo, "nome": nome, "algoritmo_id": "legado", "algoritmo": info["algoritmo"], "alvo": alvo.strip(), "features": feats,
                        "colunas_data": [], "colunas_modelo": feats, "num_cols": [c for c in feats if c not in enc], "cat_cols": list(enc), "classes": classes,
                        "categorias_conhecidas": cats, "intervalos_numericos": {}, "parametros": {}, "metricas_teste": {}, "cv": {},
                        "origem": "ficheiro .pkl importado", "avisos": ["Modelo importado de um ficheiro: não há métricas nem intervalos de referência."],
                        "pipeline_completo": True})
    return obj_final, meta
