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
from modules.ml_engine.dataforge_runtime import ModeloDummies, ModeloLegado  # noqa: F401  (as classes vivem no runtime autónomo)


class ErroLegado(Exception):
    """Mensagem pronta a mostrar ao estudante."""


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


def _meta(*, tipo, nome, algoritmo, alvo, features, cat_info, df_features, classes, metricas, cv, n_treino, n_teste, parametros, origem) -> dict:
    """`cat_info`: {coluna de texto: série com a categoria de cada linha de treino}."""
    cat = list(cat_info)
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
    for c, serie in cat_info.items():
        nomes = pd.Series(serie).astype(str).value_counts(normalize=True)
        perfil["categoricas"][c] = {"em_falta": 0.0, "proporcoes": {str(k): float(v) for k, v in nomes.head(20).items()}}
    return mp._simples({
        "tipo": tipo, "nome": nome, "algoritmo_id": "legado", "algoritmo": algoritmo, "alvo": alvo, "features": list(features),
        "colunas_data": [], "colunas_modelo": list(features), "num_cols": num, "cat_cols": cat, "classes": classes,
        "categorias_conhecidas": {c: sorted(pd.Series(serie).astype(str).unique().tolist())[:50] for c, serie in cat_info.items()},
        "intervalos_numericos": intervalos, "parametros": parametros or {}, "metricas_teste": metricas, "cv": cv,
        "n_treino": int(n_treino), "n_teste": int(n_teste), "random_state": 42, "origem": origem, "avisos": [],
        "pipeline_completo": True, "referencia_monitorizacao": perfil,
    })


def _cat_info(df_features: pd.DataFrame, encoders: dict) -> dict:
    """Categorias de cada linha de treino a partir dos códigos dos LabelEncoders."""
    return {c: pd.Series(le.inverse_transform(df_features[c].astype(int))) for c, le in encoders.items() if c in df_features.columns}


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
                       _preenchimento(df_f, enc), _modas(df_f, enc), nomes_classes=classes)
    cv = {"accuracy": [float(np.mean(r["cv_scores"])), float(np.std(r["cv_scores"]))]}
    metricas = {"accuracy": _f(r["acc"]), "precision": _f(r["prec"]), "recall": _f(r["rec"]), "f1": _f(r["f1"]), "roc_auc": _f(r.get("auc"))}
    meta = _meta(tipo="classificacao", nome=nome, algoritmo=algoritmo, alvo=(r.get("setup") or {}).get("target", "alvo"), features=feats,
                 cat_info=_cat_info(df_f, enc), df_features=df_f, classes=classes, metricas=metricas, cv=cv, n_treino=len(r["X_train"]), n_teste=len(r["X_test"]),
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
    meta = _meta(tipo="regressao", nome=nome, algoritmo=res["algo"], alvo=res["target"], features=feats, cat_info=_cat_info(df_f, enc), df_features=df_f, classes=None,
                 metricas={"r2": _f(res["r2"]), "mae": _f(res["mae"]), "rmse": _f(res["rmse"]), "mse": _f(res["mse"])},
                 cv={"r2": [float(np.mean(res["cv_scores"])), float(np.std(res["cv_scores"]))]},
                 n_treino=int(res["n_linhas"]) - n_teste, n_teste=n_teste, parametros={}, origem="página Regressão")
    return obj, meta


def do_comparador(res: dict, algoritmo: str, nome: str) -> tuple:
    """Da página Comparador: só quando todas as colunas são numéricas (o comparador não guarda os codificadores de texto)."""
    enc = res.get("encoders")
    if enc is None and res.get("cat_cols"):          # resultado de uma versão antiga: os codificadores de texto não ficaram guardados
        raise ErroLegado("Este resultado é antigo e não guardou a codificação das colunas de texto. Compara os modelos outra vez e guarda.")
    enc = enc or {}
    if algoritmo not in res.get("modelos", {}):
        raise ErroLegado("Escolhe um dos modelos treinados.")
    clf = res["tipo"] == "Classificação"
    X, feats = res["X"], list(res["X"].columns)
    obj = ModeloLegado(feats, list(enc), enc, None, res["modelos"][algoritmo], "classificacao" if clf else "regressao",
                       np.arange(len(res["classes"])) if clf else None, _preenchimento(X, enc), _modas(X, enc), com_nomes=True,
                       nomes_classes=res["classes"] if clf else None)
    linha = next(x for x in res["resultados"] if x["Algoritmo"] == algoritmo)
    if clf:
        metricas, cv = {"accuracy": _f(linha["Accuracy"]), "f1": _f(linha["F1 Score"])}, {"accuracy": [_f(linha["CV Accuracy"]), 0.0]}
    else:
        metricas, cv = {"r2": _f(linha["R²"]), "rmse": _f(linha["RMSE"])}, {"r2": [_f(linha["CV R²"]), 0.0]}
    meta = _meta(tipo="classificacao" if clf else "regressao", nome=nome, algoritmo=algoritmo, alvo=res["alvo"], features=feats, cat_info=_cat_info(X, enc), df_features=X,
                 classes=[str(c) for c in res["classes"]] if clf else None, metricas=metricas, cv=cv, n_treino=len(res["X_tr"]), n_teste=len(res["X_te"]),
                 parametros={}, origem="página Comparador")
    return obj, meta


def de_mlp(res: dict, tipo: str, nome: str) -> tuple:
    """Da página Redes Neurais (MLP): `tipo` é 'classificacao' ou 'regressao'."""
    try:
        setup, modelo, enc, df_f = res["setup"], res["model"], res.get("encoders") or {}, res["df_features"]
        feats = list(setup["features"])
    except KeyError:
        raise ErroLegado("Este resultado é antigo e não tem tudo o que é preciso. Treina a rede outra vez.")
    clf = tipo == "classificacao"
    classes = [str(c) for c in res["classes"]] if clf else None
    obj = ModeloLegado(feats, list(enc), enc, res.get("scaler"), modelo, tipo, np.asarray(res["valores_alvo"]) if clf else None,
                       _preenchimento(df_f, enc), _modas(df_f, enc), nomes_classes=classes)
    n_teste = int(setup.get("n_test", 0))
    metricas = ({"accuracy": _f(res["acc"]), "f1": _f(res["f1"])} if clf else {"r2": _f(res["r2"]), "rmse": _f(res["rmse"]), "mae": _f(res["mae"])})
    meta = _meta(tipo=tipo, nome=nome, algoritmo="MLP (rede neuronal)", alvo=setup["target"], features=feats, cat_info=_cat_info(df_f, enc), df_features=df_f,
                 classes=classes, metricas=metricas, cv={}, n_treino=len(df_f) - n_teste, n_teste=n_teste,
                 parametros={k: setup.get(k) for k in ("hidden", "activation", "solver", "lr", "max_iter")}, origem="página Redes Neurais")
    return obj, meta


def de_primeiro_modelo(res: dict, nome: str) -> tuple:
    """De «O Meu Primeiro Modelo»: o Random Forest original, com a mesma transformação (get_dummies) das colunas de texto."""
    try:
        bruto, modelo, cfg = res["X_bruto"], res["modelo"], res["cfg"]
    except KeyError:
        raise ErroLegado("Este resultado é antigo e não tem tudo o que é preciso. Treina o modelo outra vez.")
    clf = cfg["tipo"] == "clf"
    cat = [c for c in bruto.columns if not pd.api.types.is_numeric_dtype(bruto[c]) and not pd.api.types.is_bool_dtype(bruto[c])]   # texto: get_dummies
    num = [c for c in bruto.columns if c not in cat]
    classes = [str(c) for c in modelo.classes_] if clf else None
    obj = ModeloDummies(list(bruto.columns), cat, res["colunas"], modelo, "classificacao" if clf else "regressao",
                        np.asarray(modelo.classes_) if clf else None, _preenchimento(bruto[num], []) if num else {}, nomes_classes=classes)
    metricas = ({"accuracy": _f(res["acc"]), "precision": _f(res["prec"]), "recall": _f(res["rec"]), "f1": _f(res["f1"])} if clf
                else {"r2": _f(res["r2"]), "rmse": _f(res["rmse"]), "mae": _f(res["mae"])})
    cv = {("accuracy" if clf else "r2"): [float(np.mean(res["cv_scores"])), float(np.std(res["cv_scores"]))]}
    meta = _meta(tipo="classificacao" if clf else "regressao", nome=nome, algoritmo="Random Forest (Primeiro Modelo)", alvo=cfg["alvo"], features=list(bruto.columns),
                 cat_info={c: bruto[c] for c in cat}, df_features=bruto, classes=classes, metricas=metricas, cv=cv, n_treino=int(res["n_treino"]),
                 n_teste=int(res["n_teste"]), parametros={"n_estimators": 120}, origem="O Meu Primeiro Modelo")
    return obj, meta
