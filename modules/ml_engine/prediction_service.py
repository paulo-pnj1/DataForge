"""Previsões com modelos guardados: verifica os dados novos antes de prever e explica o que encontrou.

Não inventa nada: colunas em falta bloqueiam; categorias novas, valores fora do intervalo de treino e
números ilegíveis geram avisos (o pipeline trata-os, mas o resultado merece menos confiança).
"""
import numpy as np
import pandas as pd

from modules.ml_engine import forecasting as fc
from modules.ml_engine.preprocessing import expandir_datas, limpar_tipos


class ErroPrevisao(Exception):
    """Dados novos incompatíveis com o modelo (mensagem pronta a mostrar)."""


def verificar_entrada(df: pd.DataFrame, meta: dict) -> dict:
    """Compara a tabela nova com o que o modelo viu no treino. Devolve {erros, avisos, ignoradas}."""
    erros, avisos = [], []
    feats = meta.get("features") or []
    em_falta = [c for c in feats if c not in df.columns]
    if em_falta:
        erros.append("Faltam colunas que o modelo precisa: " + ", ".join(map(str, em_falta)) + ".")
    extra = [c for c in df.columns if c not in feats]
    if extra:
        avisos.append("Colunas que o modelo não usa (ignoradas): " + ", ".join(map(str, extra[:8])) + ("…" if len(extra) > 8 else "") + ".")
    if df.shape[0] == 0:
        erros.append("A tabela não tem linhas.")
    if erros:
        return {"erros": erros, "avisos": avisos, "ignoradas": extra}
    datas = set(meta.get("colunas_data") or [])
    for c in meta.get("num_cols") or []:
        if c in datas or c not in df.columns:
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        ilegiveis = int((num.isna() & df[c].notna()).sum())
        if ilegiveis:
            avisos.append(f"«{c}»: {ilegiveis} valores não são números e foram tratados como em falta.")
        falta = int(num.isna().sum())
        if falta and falta == len(df):
            erros.append(f"A coluna «{c}» não tem nenhum valor numérico válido.")
        iv = (meta.get("intervalos_numericos") or {}).get(c)
        if iv and len(iv) >= 2 and iv[0] is not None and iv[1] is not None:
            fora = int(((num < iv[0]) | (num > iv[1])).sum())
            if fora:
                avisos.append(f"«{c}»: {fora} valores estão fora do intervalo visto no treino ({iv[0]:.4g} a {iv[1]:.4g}); a previsão é uma extrapolação.")
    for c in meta.get("cat_cols") or []:
        if c not in df.columns:
            continue
        conhecidas = set(map(str, (meta.get("categorias_conhecidas") or {}).get(c, [])))
        if not conhecidas:
            continue
        novas = sorted({str(v).strip() for v in df[c].dropna()} - conhecidas)
        if novas:
            avisos.append(f"«{c}»: categorias nunca vistas no treino ({', '.join(novas[:5])}); o modelo trata-as como desconhecidas.")
    return {"erros": erros, "avisos": avisos, "ignoradas": extra}


def _preparar_X(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    feats = meta["features"]
    X = expandir_datas(df[feats].copy(), meta.get("colunas_data") or [])
    num, cat = meta.get("num_cols") or [], meta.get("cat_cols") or []
    for c in num + cat:
        if c not in X.columns:
            raise ErroPrevisao(f"Falta a coluna «{c}».")
    return limpar_tipos(X, num, cat)


def prever_tabela(modelo, meta: dict, df: pd.DataFrame) -> tuple:
    """Previsões para cada linha. Devolve (DataFrame com as colunas originais + previsão [+ probabilidades], relatório)."""
    if meta.get("tipo") not in ("regressao", "classificacao"):
        raise ErroPrevisao("Este modelo é uma série temporal: usa a previsão do futuro.")
    rel = verificar_entrada(df, meta)
    if rel["erros"]:
        raise ErroPrevisao(" ".join(rel["erros"]))
    X = _preparar_X(df, meta)
    saida = df.copy()
    alvo = meta.get("alvo", "alvo")
    nome = f"previsao_{alvo}"
    pred = modelo.predict(X)
    if meta["tipo"] == "classificacao":
        classes = meta.get("classes") or []
        pred = np.asarray(pred).astype(int)
        saida[nome] = [classes[i] if 0 <= i < len(classes) else i for i in pred]
        if hasattr(modelo, "predict_proba"):
            try:
                p = modelo.predict_proba(X)
                for j, cl in enumerate(classes[: p.shape[1]]):
                    saida[f"prob_{cl}"] = np.round(p[:, j], 4)
                saida["confianca"] = np.round(p.max(axis=1), 4)
            except Exception:
                rel["avisos"].append("Este modelo não calcula probabilidades.")
    else:
        saida[nome] = np.asarray(pred, dtype=float)
    return saida, rel


def prever_um(modelo, meta: dict, valores: dict) -> tuple:
    """Previsão para um único caso (dicionário coluna → valor). Devolve (previsao, detalhe, relatório)."""
    linha = pd.DataFrame([{c: valores.get(c, np.nan) for c in meta.get("features", [])}])
    saida, rel = prever_tabela(modelo, meta, linha)
    alvo = f"previsao_{meta.get('alvo', 'alvo')}"
    previsao = saida[alvo].iloc[0]
    detalhe = {c: float(saida[c].iloc[0]) for c in saida.columns if c.startswith("prob_")}
    return previsao, detalhe, rel


def prever_futuro(modelo, meta: dict, horizonte: int) -> pd.DataFrame:
    """Previsão de uma série guardada para os próximos `horizonte` períodos (continua a partir do fim dos dados de treino)."""
    if meta.get("tipo") != "serie":
        raise ErroPrevisao("Este modelo não é de séries temporais.")
    horizonte = int(horizonte)
    if not 1 <= horizonte <= 500:
        raise ErroPrevisao("O horizonte tem de estar entre 1 e 500 períodos.")
    prev = modelo.prever(horizonte)
    datas = fc.datas_futuras_de(pd.Timestamp(meta["ultima_data"]), meta.get("freq"), pd.Timedelta(days=1), horizonte)
    df = pd.DataFrame({"data": datas, "previsao": np.asarray(prev.media, dtype=float)})
    if prev.inferior is not None and prev.superior is not None:
        df["inferior"], df["superior"] = prev.inferior, prev.superior
    return df
