"""Treino de UM modelo: pipeline = pré-processamento + estimador, com tratamento de erros por modelo.

Um algoritmo que falha, não existe ou não é compatível devolve um `ResultadoModelo` com o estado e o motivo;
nunca interrompe a comparação dos outros.
"""
import logging
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from modules.ml_engine import model_evaluation as ev
from modules.ml_engine import model_registry as reg
from modules.ml_engine.preprocessing import DadosPreparados, construir_preprocessador

log = logging.getLogger("dataforge.ml")

MAX_TREINOS_SIMULTANEOS = 2
_SLOTS = threading.BoundedSemaphore(MAX_TREINOS_SIMULTANEOS)

OK, FALHOU, INDISPONIVEL, INCOMPATIVEL, IGNORADO, REFERENCIA = "ok", "falhou", "indisponivel", "incompativel", "ignorado", "referencia"


@dataclass
class ResultadoModelo:
    id: str
    nome: str
    tipo: str
    estado: str = OK
    motivo: str = ""
    pipeline: object = None
    params: dict = field(default_factory=dict)
    metricas: dict = field(default_factory=dict)          # no conjunto de teste
    metricas_treino: dict = field(default_factory=dict)   # só a métrica principal, para detectar sobreajuste
    cv: dict = field(default_factory=dict)                # {metrica: (media, desvio)} na validação cruzada do treino
    tempo_treino: float = 0.0
    predicoes: dict = field(default_factory=dict)         # y_pred, score (para gráficos)
    avisos: list = field(default_factory=list)
    complexidade: int = 1

    @property
    def ok(self) -> bool:
        return self.estado in (OK, REFERENCIA)


@contextmanager
def _vaga(espera: float = 120.0):
    """Limita quantos treinos correm ao mesmo tempo no servidor (vários utilizadores)."""
    if not _SLOTS.acquire(timeout=espera):
        raise TimeoutError("O servidor está ocupado com outros treinos. Tenta daqui a pouco.")
    try:
        yield
    finally:
        _SLOTS.release()


def _scorers(tipo: str) -> dict:
    if tipo == "regressao":
        return {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    return {"accuracy": "accuracy",
            "precision": make_scorer(precision_score, average="macro", zero_division=0),
            "recall": make_scorer(recall_score, average="macro", zero_division=0),
            "f1": "f1_macro"}


def _validacao_cruzada(pipe, dados: DadosPreparados, folds: int):
    if dados.tipo == "classificacao":
        minimo = int(np.bincount(dados.y_train).min())
        folds = min(folds, minimo)
        if folds < 2:
            return None, "Há uma classe com menos de 2 exemplos de treino: sem validação cruzada."
        cv = StratifiedKFold(folds, shuffle=True, random_state=dados.random_state)
    else:
        folds = min(folds, dados.n_treino // 2)
        if folds < 2:
            return None, "Poucos exemplos de treino para validação cruzada."
        cv = KFold(folds, shuffle=True, random_state=dados.random_state)
    res = cross_validate(clone(pipe), dados.X_train, dados.y_train, cv=cv, scoring=_scorers(dados.tipo), n_jobs=1, error_score="raise")
    out = {}
    for k in _scorers(dados.tipo):
        v = res[f"test_{k}"]
        v = -v if k in ("mae", "rmse") else v        # o scikit-learn devolve erros negativos
        out[k] = (float(np.mean(v)), float(np.std(v)))
    return out, None


def _metrica_principal(tipo: str) -> str:
    return "r2" if tipo == "regressao" else "f1"


def avaliar_pipeline(pipe, dados: DadosPreparados) -> tuple:
    """Métricas no teste, métrica principal no treino e predições (para os gráficos)."""
    y_pred = np.asarray(pipe.predict(dados.X_test)).ravel()
    pred_tr = np.asarray(pipe.predict(dados.X_train)).ravel()
    if dados.tipo == "regressao":
        m = ev.metricas_regressao(dados.y_test, y_pred)
        m["residuos"] = ev.resumo_residuos(dados.y_test, y_pred)
        mt = {"r2": ev.metricas_regressao(dados.y_train, pred_tr)["r2"]}
        return m, mt, {"y_pred": y_pred}
    score = ev.pontuacao_classes(pipe, dados.X_test)
    k = len(dados.classes)
    m = ev.metricas_classificacao(dados.y_test, y_pred, score, k)
    m["matriz_confusao"] = ev.matriz_confusao(dados.y_test, y_pred, k)
    mt = {"f1": ev.metricas_classificacao(dados.y_train, pred_tr, None, k)["f1"]}
    return m, mt, {"y_pred": y_pred, "score": score, "curvas": ev.curvas_binarias(dados.y_test, score) if k == 2 else {}}


def _curto(e: Exception) -> str:
    texto = " ".join(str(e).split()) or e.__class__.__name__
    return f"{e.__class__.__name__}: {texto}"[:240]


def treinar_modelo(modelo_id: str, dados: DadosPreparados, params: dict = None, random_state: int = 42,
                   cv_folds: int = 0) -> ResultadoModelo:
    """Treina e avalia um algoritmo nos dados preparados. Nunca levanta erro por causa do algoritmo."""
    try:
        info = reg.obter(modelo_id)
    except KeyError as e:
        return ResultadoModelo(modelo_id, modelo_id, dados.tipo, FALHOU, str(e).strip("'\""))
    r = ResultadoModelo(info.id, info.nome, dados.tipo, complexidade=info.complexidade)
    if dados.tipo not in info.problemas or info.tipo_modelo != "tabular":
        r.estado, r.motivo = INCOMPATIVEL, f"{info.nome} não serve para {reg.NOMES_PROBLEMA[dados.tipo].lower()}."
        return r
    if not info.disponivel():
        r.estado, r.motivo = INDISPONIVEL, info.motivo_indisponivel()
        return r
    if info.verifica:
        try:
            ok, motivo = info.verifica(dados)
        except Exception as e:                       # uma verificação que rebenta conta como incompatível
            ok, motivo = False, _curto(e)
        if not ok:
            r.estado, r.motivo = INCOMPATIVEL, motivo
            return r
    r.params = reg.completar_parametros(info, params)
    try:
        pre = construir_preprocessador(dados.num_cols, dados.cat_cols, escalar=info.escalar, nao_negativo=info.nao_negativo)
        pipe = Pipeline([("prep", pre), ("modelo", info.construir(r.params, random_state))])
        with _vaga(), warnings.catch_warnings(record=True) as avisos_py:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            pipe.fit(dados.X_train, dados.y_train)
            r.tempo_treino = time.perf_counter() - t0
            r.metricas, r.metricas_treino, r.predicoes = avaliar_pipeline(pipe, dados)
            if cv_folds and cv_folds >= 2:
                try:
                    r.cv, motivo_cv = _validacao_cruzada(pipe, dados, cv_folds)
                    r.cv = r.cv or {}
                    if motivo_cv:
                        r.avisos.append(motivo_cv)
                except Exception as e:
                    r.avisos.append("A validação cruzada falhou: " + _curto(e))
        r.pipeline = pipe
        if any("onverge" in str(w.message) for w in avisos_py):
            r.avisos.append("O algoritmo não convergiu por completo; os resultados podem melhorar com mais iterações ou dados em escala.")
    except TimeoutError as e:
        r.estado, r.motivo = IGNORADO, str(e)
    except Exception as e:
        log.exception("Falha ao treinar %s", info.id)
        r.estado, r.motivo, r.pipeline = FALHOU, _curto(e), None
    return r


def treinar_referencia(dados: DadosPreparados, random_state: int = 42) -> ResultadoModelo:
    """Modelo de referência: média (regressão) ou classe mais frequente (classificação). Qualquer modelo tem de o superar."""
    if dados.tipo == "regressao":
        est, nome = DummyRegressor(strategy="mean"), "Referência (prever sempre a média)"
    else:
        est, nome = DummyClassifier(strategy="most_frequent"), "Referência (prever sempre a classe mais frequente)"
    r = ResultadoModelo("referencia", nome, dados.tipo, REFERENCIA)
    try:
        pipe = Pipeline([("prep", construir_preprocessador(dados.num_cols, dados.cat_cols, escalar=False)), ("modelo", est)])
        t0 = time.perf_counter()
        pipe.fit(dados.X_train, dados.y_train)
        r.tempo_treino = time.perf_counter() - t0
        r.metricas, r.metricas_treino, r.predicoes = avaliar_pipeline(pipe, dados)
        r.pipeline = pipe
    except Exception as e:
        r.estado, r.motivo = FALHOU, _curto(e)
    return r
