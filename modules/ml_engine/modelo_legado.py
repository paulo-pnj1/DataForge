"""Guardar modelos treinados nas páginas Classificação, Regressão e Comparador, com qualquer algoritmo.

Essas páginas pré-processam à mão (codificadores, scaler) e deixam o modelo ajustado em memória. Para o guardar sem o
retreinar, `ModeloLegado` junta tudo num só objecto com `predict(DataFrame)` e `predict_proba(DataFrame)`, igual ao Pipeline
que o resto da plataforma guarda: assim `model_persistence` e `prediction_service` tratam-no como qualquer outro modelo.

O que o objecto faz por ti em dados novos: aplica os mesmos codificadores e o mesmo scaler do treino, preenche os valores em
falta com a mediana do treino e trata categorias nunca vistas como a categoria mais comum (o `prediction_service` avisa).
"""
import io

import numpy as np
import pandas as pd

from modules.ml_engine import model_persistence as mp
from modules.ml_engine.ciclo_ml import _arestas, _proporcoes


class ErroLegado(Exception):
    """Mensagem pronta a mostrar ao estudante."""


class ModeloLegado:
    """Pré-processamento (codificadores + scaler) + algoritmo já treinado, num só objecto."""

    def __init__(self, features, cat_cols, encoders, scaler, modelo, tipo, valores_alvo=None, preenchimento=None, modas=None, com_nomes=False):
        self.com_nomes = com_nomes          # o Comparador ajusta os modelos com DataFrames (com nomes de colunas)
        self.features = list(features)
        self.cat_cols = list(cat_cols)
        self.encoders = dict(encoders or {})
        self.scaler = scaler
        self.modelo = modelo
        self.tipo = tipo
        self.valores_alvo = None if valores_alvo is None else np.asarray(valores_alvo)
        self.preenchimento = dict(preenchimento or {})
        self.modas = dict(modas or {})

    # ── dados novos → matriz que o algoritmo espera ──
    def _matriz(self, X) -> np.ndarray:
        X = pd.DataFrame(X)
        colunas = []
        for c in self.features:
            if c not in X.columns:
                raise ValueError(f"Falta a coluna «{c}».")
            s = X[c]
            if c in self.encoders:
                le = self.encoders[c]
                conhecidas = set(map(str, le.classes_))
                v = s.astype(str).str.strip()
                moda = self.modas.get(c, str(le.classes_[0]))
                v = v.where(v.isin(conhecidas), moda)
                colunas.append(le.transform(v).astype(float))
            else:
                n = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
                colunas.append(n.fillna(self.preenchimento.get(c, 0.0)).to_numpy())
        M = np.column_stack(colunas)
        M = self.scaler.transform(M) if self.scaler is not None else M
        return pd.DataFrame(M, columns=self.features) if getattr(self, "com_nomes", False) else M

    def predict(self, X):
        p = np.asarray(self.modelo.predict(self._matriz(X)))
        if self.tipo == "classificacao":
            return np.searchsorted(self.valores_alvo, p).astype(int)      # índice da classe, como nos Pipelines do Laboratório
        return p.astype(float)

    def predict_proba(self, X):
        if self.tipo != "classificacao" or not hasattr(self.modelo, "predict_proba"):
            raise AttributeError("Este modelo não calcula probabilidades.")
        bruto = self.modelo.predict_proba(self._matriz(X))
        classes = getattr(self.modelo, "classes_", None)
        if classes is None or len(classes) == len(self.valores_alvo):
            return bruto
        completo = np.zeros((bruto.shape[0], len(self.valores_alvo)))           # classe que faltou no treino: probabilidade 0
        completo[:, np.searchsorted(self.valores_alvo, classes)] = bruto
        return completo


# ── construção a partir do que cada página guarda ──────────────────────────────
def _preenchimento(df_features: pd.DataFrame, cat_cols) -> dict:
    return {c: float(pd.to_numeric(df_features[c], errors="coerce").median()) for c in df_features.columns if c not in cat_cols
            and pd.to_numeric(df_features[c], errors="coerce").notna().any()}


def _modas(df_features: pd.DataFrame, encoders: dict) -> dict:
    out = {}
    for c, le in encoders.items():
        if c in df_features.columns and len(df_features[c].dropna()):
            out[c] = str(le.classes_[int(df_features[c].mode().iloc[0])])
    return out


def _meta(*, tipo, nome, algoritmo, alvo, features, encoders, df_features, classes, metricas, cv, n_treino, n_teste, parametros, origem) -> dict:
    cat = list(encoders)
    num = [c for c in features if c not in cat]
    intervalos = {}
    for c in num:
        s = pd.to_numeric(df_features[c], errors="coerce")
        if s.notna().any():
            intervalos[c] = (float(s.min()), float(s.max()), float(s.median()))
    perfil = {"n_treino": int(n_treino), "numericas": {}, "categoricas": {}, "previsoes": None}
    for c in num:
        s = pd.to_numeric(df_features[c], errors="coerce")
        item = {"em_falta": float(s.isna().mean()), "media": float(s.mean()) if s.notna().any() else None,
                "desvio": float(s.std()) if s.notna().sum() > 1 else None}
        e = _arestas(s)
        if e is not None:
            item.update(arestas=[float(x) for x in e], proporcoes=_proporcoes(s, e))
        perfil["numericas"][c] = item
    for c, le in encoders.items():
        nomes = pd.Series(le.inverse_transform(df_features[c].astype(int))).value_counts(normalize=True)
        perfil["categoricas"][c] = {"em_falta": 0.0, "proporcoes": {str(k): float(v) for k, v in nomes.head(20).items()}}
    return mp._simples({
        "tipo": tipo, "nome": nome, "algoritmo_id": "legado", "algoritmo": algoritmo, "alvo": alvo, "features": list(features),
        "colunas_data": [], "colunas_modelo": list(features), "num_cols": num, "cat_cols": cat, "classes": classes,
        "categorias_conhecidas": {c: [str(x) for x in le.classes_][:50] for c, le in encoders.items()},
        "intervalos_numericos": intervalos, "parametros": parametros or {}, "metricas_teste": metricas, "cv": cv,
        "n_treino": int(n_treino), "n_teste": int(n_teste), "random_state": 42, "origem": origem, "avisos": [],
        "pipeline_completo": True, "referencia_monitorizacao": perfil,
    })


def _f(v):
    return None if v is None else float(v)


def de_classificacao(r: dict, algoritmo: str, nome: str) -> tuple:
    """Da página Classificação: (objecto a guardar, metadados)."""
    try:
        modelo, feats = r["model"], list(r["feature_names"])
        enc, df_f = r.get("encoders") or {}, r["df_features"]
    except KeyError:
        raise ErroLegado("Este resultado é antigo e não tem tudo o que é preciso. Treina o modelo outra vez.")
    classes = [str(c) for c in r["classes"]]
    obj = ModeloLegado(feats, list(enc), enc, r.get("scaler"), modelo, "classificacao", np.unique(r["y"]),
                       _preenchimento(df_f, enc), _modas(df_f, enc))
    cv = {"accuracy": [float(np.mean(r["cv_scores"])), float(np.std(r["cv_scores"]))]}
    metricas = {"accuracy": _f(r["acc"]), "precision": _f(r["prec"]), "recall": _f(r["rec"]), "f1": _f(r["f1"]), "roc_auc": _f(r.get("auc"))}
    meta = _meta(tipo="classificacao", nome=nome, algoritmo=algoritmo, alvo=(r.get("setup") or {}).get("target", "alvo"), features=feats,
                 encoders=enc, df_features=df_f, classes=classes, metricas=metricas, cv=cv, n_treino=len(r["X_train"]), n_teste=len(r["X_test"]),
                 parametros={}, origem="página Classificação")
    return obj, meta


def de_regressao(res: dict, nome: str) -> tuple:
    """Da página Regressão: (objecto a guardar, metadados)."""
    import joblib
    try:
        pacote = joblib.load(io.BytesIO(res["modelo_pkl"]))
        df_f = res["df_features"]
    except KeyError:
        raise ErroLegado("Este resultado é antigo e não tem tudo o que é preciso. Treina o modelo outra vez.")
    enc, feats = pacote.get("encoders") or {}, list(pacote["features"])
    obj = ModeloLegado(feats, list(enc), enc, pacote.get("scaler"), pacote["model"], "regressao", None, _preenchimento(df_f, enc), _modas(df_f, enc))
    n_teste = len(res["y_test"])
    meta = _meta(tipo="regressao", nome=nome, algoritmo=res["algo"], alvo=res["target"], features=feats, encoders=enc, df_features=df_f, classes=None,
                 metricas={"r2": _f(res["r2"]), "mae": _f(res["mae"]), "rmse": _f(res["rmse"]), "mse": _f(res["mse"])},
                 cv={"r2": [float(np.mean(res["cv_scores"])), float(np.std(res["cv_scores"]))]},
                 n_treino=int(res["n_linhas"]) - n_teste, n_teste=n_teste, parametros={}, origem="página Regressão")
    return obj, meta


def do_comparador(res: dict, algoritmo: str, nome: str) -> tuple:
    """Da página Comparador: só quando todas as colunas são numéricas (o comparador não guarda os codificadores de texto)."""
    if res.get("cat_cols"):
        raise ErroLegado("Este modelo foi treinado com colunas de texto, que o Comparador codifica sem guardar a codificação. "
                         "Para o guardar, treina-o na página Classificação, Regressão ou no Laboratório de ML.")
    if algoritmo not in res.get("modelos", {}):
        raise ErroLegado("Escolhe um dos modelos treinados.")
    clf = res["tipo"] == "Classificação"
    X, feats = res["X"], list(res["X"].columns)
    obj = ModeloLegado(feats, [], {}, None, res["modelos"][algoritmo], "classificacao" if clf else "regressao",
                       np.arange(len(res["classes"])) if clf else None, _preenchimento(X, []), {}, com_nomes=True)
    linha = next(x for x in res["resultados"] if x["Algoritmo"] == algoritmo)
    if clf:
        metricas, cv = {"accuracy": _f(linha["Accuracy"]), "f1": _f(linha["F1 Score"])}, {"accuracy": [_f(linha["CV Accuracy"]), 0.0]}
    else:
        metricas, cv = {"r2": _f(linha["R²"]), "rmse": _f(linha["RMSE"])}, {"r2": [_f(linha["CV R²"]), 0.0]}
    meta = _meta(tipo="classificacao" if clf else "regressao", nome=nome, algoritmo=algoritmo, alvo=res["alvo"], features=feats, encoders={}, df_features=X,
                 classes=[str(c) for c in res["classes"]] if clf else None, metricas=metricas, cv=cv, n_treino=len(res["X_tr"]), n_teste=len(res["X_te"]),
                 parametros={}, origem="página Comparador")
    return obj, meta
