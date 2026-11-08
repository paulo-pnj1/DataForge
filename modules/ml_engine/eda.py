"""Exploração gráfica automática dos dados (EDA): resumo por coluna, distribuições, correlações e relação com o alvo.

Tudo aqui funciona sem Streamlit (testável). Só descreve os dados: não altera a tabela nem treina nada.

Regras:
- Nada é inventado: cada número vem da tabela que o estudante carregou.
- Em tabelas grandes usa-se uma amostra aleatória (fixa) só para os gráficos e para as medidas de associação.
- As medidas de associação são indícios, não provas: uma força alta pode ser fuga de informação e uma força baixa
  não exclui relações que não sejam lineares ou que só apareçam em conjunto com outras variáveis.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

MAX_LINHAS_EDA = 20_000
MAX_CATEGORIAS_GRAFICO = 15
MAX_CATEGORIAS_ASSOCIACAO = 50
MAX_COLUNAS_HISTOGRAMAS = 12
MAX_COLUNAS_CORRELACAO = 25
LIMITE_FUGA = 0.95
LIMITE_FRACA = 0.05


def _e_numerica(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def amostra_para_graficos(df: pd.DataFrame, max_linhas: int = MAX_LINHAS_EDA, random_state: int = 42) -> tuple:
    """(tabela, aviso ou None). Em tabelas grandes devolve uma amostra aleatória e fixa."""
    if len(df) > max_linhas:
        return (df.sample(n=max_linhas, random_state=random_state).reset_index(drop=True),
                f"A tabela tem {len(df):,} linhas; os gráficos usam uma amostra aleatória de {max_linhas:,}.".replace(",", " "))
    return df, None


# ── Resumo por coluna ─────────────────────────────────────────────────────────
def resumo_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por coluna: tipo, valores em falta, valores diferentes e um retrato rápido (intervalo ou mais frequente)."""
    linhas = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        falta = int(s.isna().sum())
        item = {"Coluna": str(c), "Tipo": "", "Em falta": f"{falta} ({falta / n:.0%})" if n else "0",
                "Valores diferentes": int(s.nunique(dropna=True)), "Retrato": ""}
        if _e_numerica(s):
            v = s.dropna()
            item["Tipo"] = "número"
            if len(v):
                item["Retrato"] = f"min {v.min():.4g} · mediana {v.median():.4g} · média {v.mean():.4g} · max {v.max():.4g}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            v = s.dropna()
            item["Tipo"] = "data"
            if len(v):
                item["Retrato"] = f"de {v.min():%Y-%m-%d} a {v.max():%Y-%m-%d}"
        else:
            item["Tipo"] = "categoria/texto"
            vc = s.dropna().astype(str).value_counts()
            if len(vc):
                item["Retrato"] = f"mais frequente: «{vc.index[0]}» ({vc.iloc[0] / max(len(s.dropna()), 1):.0%})"
        linhas.append(item)
    return pd.DataFrame(linhas)


# ── Distribuições ─────────────────────────────────────────────────────────────
def figura_distribuicao(s: pd.Series, nome: str = None) -> go.Figure:
    """Histograma (números) ou barras das categorias mais frequentes (texto)."""
    nome = nome or str(s.name)
    if _e_numerica(s):
        v = s.dropna()
        fig = go.Figure(go.Histogram(x=v, nbinsx=30))
        fig.update_layout(title=f"Distribuição de «{nome}»", xaxis_title=nome, yaxis_title="Quantos")
        return fig
    vc = s.dropna().astype(str).value_counts()
    top = vc.head(MAX_CATEGORIAS_GRAFICO)
    fig = go.Figure(go.Bar(x=top.index.tolist(), y=top.values.tolist()))
    extra = f" (as {MAX_CATEGORIAS_GRAFICO} mais frequentes de {len(vc)})" if len(vc) > MAX_CATEGORIAS_GRAFICO else ""
    fig.update_layout(title=f"Categorias de «{nome}»{extra}", xaxis_title=nome, yaxis_title="Quantos")
    return fig


def figuras_distribuicao(df: pd.DataFrame, colunas: list = None, max_colunas: int = MAX_COLUNAS_HISTOGRAMAS) -> list:
    """[(coluna, figura)] para as primeiras `max_colunas` colunas úteis (sem identificadores nem colunas vazias)."""
    cols = list(colunas) if colunas else list(df.columns)
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            continue
        if not _e_numerica(s) and not pd.api.types.is_datetime64_any_dtype(s) and s.nunique(dropna=True) == len(s.dropna()) and len(s) > 20:
            continue                                                  # parece um identificador: não há nada a ver
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        out.append((str(c), figura_distribuicao(s, str(c))))
        if len(out) >= max_colunas:
            break
    return out


# ── Correlações ───────────────────────────────────────────────────────────────
def matriz_correlacao(df: pd.DataFrame, max_colunas: int = MAX_COLUNAS_CORRELACAO) -> pd.DataFrame:
    """Correlação de Pearson entre as colunas numéricas (as de maior variância relativa ficam, se forem muitas)."""
    num = df.select_dtypes(include="number").copy()
    num = num.loc[:, num.nunique(dropna=True) > 1]
    if num.shape[1] < 2:
        return pd.DataFrame()
    if num.shape[1] > max_colunas:
        ordem = (num.std() / num.mean().abs().replace(0, np.nan)).abs().fillna(0).sort_values(ascending=False)
        num = num[ordem.index[:max_colunas]]
    return num.corr(method="pearson")


def figura_correlacao(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=np.round(corr.values, 2), texttemplate="%{text}", hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlação entre as colunas numéricas (−1 a 1)", yaxis_autorange="reversed")
    return fig


def pares_muito_correlacionados(corr: pd.DataFrame, limite: float = 0.9) -> list:
    """[(coluna_a, coluna_b, correlação)] com |correlação| ≥ limite (cada par uma vez, do mais forte para o mais fraco)."""
    if corr is None or corr.empty:
        return []
    out = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= limite:
                out.append((a, b, float(v)))
    return sorted(out, key=lambda t: -abs(t[2]))


# ── Relação de cada coluna com o alvo ─────────────────────────────────────────
def _razao_correlacao(categorias: pd.Series, valores: pd.Series) -> float:
    """η (eta): quanto da variação de `valores` é explicada pelas categorias. 0 = nada, 1 = tudo."""
    d = pd.DataFrame({"c": categorias.astype(str), "v": pd.to_numeric(valores, errors="coerce")}).dropna()
    if len(d) < 3 or d["c"].nunique() < 2:
        return float("nan")
    media = d["v"].mean()
    total = float(((d["v"] - media) ** 2).sum())
    if total <= 0:
        return float("nan")
    g = d.groupby("c")["v"]
    entre = float((g.count() * (g.mean() - media) ** 2).sum())
    return float(np.sqrt(min(max(entre / total, 0.0), 1.0)))


def _cramer_v(a: pd.Series, b: pd.Series) -> float:
    """V de Cramér entre duas colunas de categorias. 0 = independentes, 1 = uma determina a outra."""
    d = pd.DataFrame({"a": a.astype(str), "b": b.astype(str)})
    d = d[a.notna().to_numpy() & b.notna().to_numpy()]
    if len(d) < 3:
        return float("nan")
    tab = pd.crosstab(d["a"], d["b"]).to_numpy(dtype=float)
    if min(tab.shape) < 2:
        return float("nan")
    n = tab.sum()
    esperado = tab.sum(axis=1, keepdims=True) @ tab.sum(axis=0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((tab - esperado) ** 2 / esperado))
    return float(np.sqrt(min(max(chi2 / (n * (min(tab.shape) - 1)), 0.0), 1.0)))


def forca_associacao(df: pd.DataFrame, alvo: str, tipo: str = None) -> pd.DataFrame:
    """Quão ligada está cada coluna ao alvo (0 a 1), com a medida usada. Ordenada da mais forte para a mais fraca.

    Medidas: |Spearman| (número × número), η (número × categoria) e V de Cramér (categoria × categoria).
    `tipo`: 'regressao' ou 'classificacao' (se faltar, deduz-se pelo alvo).
    """
    if alvo not in df.columns:
        raise ValueError(f"A variável-alvo «{alvo}» não existe.")
    d, _ = amostra_para_graficos(df)
    y = d[alvo]
    tipo = tipo or ("regressao" if _e_numerica(y) and y.nunique(dropna=True) > 10 else "classificacao")
    y_num = tipo == "regressao"
    if y_num:
        y = pd.to_numeric(y, errors="coerce")
    linhas = []
    for c in d.columns:
        if c == alvo:
            continue
        s = d[c]
        ok = s.notna() & y.notna()
        if ok.sum() < 5 or s[ok].nunique() < 2:
            continue
        medida, v = None, float("nan")
        if _e_numerica(s):
            if y_num:
                medida, v = "|Spearman|", abs(float(s[ok].rank().corr(y[ok].rank())))
            else:
                medida, v = "η (número × classe)", _razao_correlacao(y[ok], s[ok])
        elif pd.api.types.is_datetime64_any_dtype(s):
            continue
        else:
            if s[ok].nunique() > MAX_CATEGORIAS_ASSOCIACAO:
                continue                                              # provável identificador ou texto livre
            if y_num:
                medida, v = "η (categoria × número)", _razao_correlacao(s[ok], y[ok])
            else:
                medida, v = "V de Cramér", _cramer_v(s[ok], y[ok])
        if v is None or not np.isfinite(v):
            continue
        obs = ""
        if v >= LIMITE_FUGA:
            obs = "Quase igual ao alvo: pode ser fuga de informação (a coluna só existe depois de saber a resposta)."
        elif v < LIMITE_FRACA:
            obs = "Quase sem relação com o alvo (sozinha)."
        linhas.append({"Coluna": str(c), "Força": float(v), "Medida": medida, "Observação": obs})
    tab = pd.DataFrame(linhas, columns=["Coluna", "Força", "Medida", "Observação"])
    return tab.sort_values("Força", ascending=False).reset_index(drop=True) if len(tab) else tab


def figura_forca(tab: pd.DataFrame, alvo: str) -> go.Figure:
    t = tab.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["Força"], y=t["Coluna"], orientation="h", hovertext=t["Medida"]))
    fig.update_layout(title=f"Ligação de cada coluna a «{alvo}» (0 = nenhuma, 1 = total)", xaxis_range=[0, 1])
    return fig


def figura_alvo_por_coluna(df: pd.DataFrame, coluna: str, alvo: str, tipo: str) -> go.Figure:
    """Como o alvo muda com uma coluna: dispersão (número × número), caixas (número × classe/ categoria × número) ou barras."""
    d, _ = amostra_para_graficos(df)
    d = d[[coluna, alvo]].dropna()
    s, y = d[coluna], d[alvo]
    if _e_numerica(s) and tipo == "regressao":
        fig = go.Figure(go.Scatter(x=s, y=pd.to_numeric(y, errors="coerce"), mode="markers", marker={"opacity": 0.5}))
        fig.update_layout(title=f"«{alvo}» em função de «{coluna}»", xaxis_title=coluna, yaxis_title=alvo)
    elif _e_numerica(s):
        fig = go.Figure()
        for classe, g in d.groupby(y.astype(str)):
            fig.add_trace(go.Box(y=g[coluna], name=str(classe)))
        fig.update_layout(title=f"«{coluna}» em cada classe de «{alvo}»", yaxis_title=coluna, showlegend=False)
    elif tipo == "regressao":
        top = s.astype(str).value_counts().head(MAX_CATEGORIAS_GRAFICO).index
        d2 = d[s.astype(str).isin(top)]
        fig = go.Figure()
        for cat, g in d2.groupby(d2[coluna].astype(str)):
            fig.add_trace(go.Box(y=pd.to_numeric(g[alvo], errors="coerce"), name=str(cat)))
        fig.update_layout(title=f"«{alvo}» em cada categoria de «{coluna}»", yaxis_title=alvo, showlegend=False)
    else:
        top = s.astype(str).value_counts().head(MAX_CATEGORIAS_GRAFICO).index
        d2 = d[s.astype(str).isin(top)]
        tab = pd.crosstab(d2[coluna].astype(str), d2[alvo].astype(str), normalize="index")
        fig = go.Figure([go.Bar(name=str(k), x=tab.index.tolist(), y=tab[k].tolist()) for k in tab.columns])
        fig.update_layout(barmode="stack", title=f"Proporção de cada classe de «{alvo}» por «{coluna}»", yaxis_tickformat=".0%")
    return fig
