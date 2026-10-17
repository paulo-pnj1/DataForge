"""
DataForge Business - Núcleo (sem interface).

Tudo o que o painel Empresa faz com os dados: ler CSV/Excel, diagnosticar, treinar modelos (prever vendas,
procura, clientes que vão sair), detectar fraude/anomalias, segmentar clientes, aplicar um modelo a dados
novos e gerar o relatório executivo em PDF.

Regras de desenho:
- Nada aqui desenha ecrãs: devolve dados simples (dicts, listas, DataFrames) que o ecrã mostra.
- O texto que depende do idioma não fica gravado: guardam-se números e modelos de frase (com `{}`);
  o ecrã (e o PDF) traduzem na hora com `tr(modelo).format(...)`, por isso mudar de idioma muda tudo.
"""
import base64
import datetime
import gzip
import io
import re
import time
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

try:                                    # a tradução só existe dentro da app (precisa do Streamlit)
    from modules.i18n import tr
except Exception:                       # pragma: no cover
    def tr(texto):
        return texto

MAX_LINHAS_TREINO = 50_000              # acima disto treina-se numa amostra (rápido e suficiente)
MAX_BYTES_DADOS = 3_000_000             # tamanho máximo (comprimido) de um conjunto de dados guardado
MAX_BYTES_MODELO = 4_000_000            # tamanho máximo (comprimido) de um modelo guardado
MAX_CATEGORIAS = 50                     # colunas com mais categorias do que isto não entram no modelo
SEMENTE = 42


def T(modelo: str, *args) -> str:
    """Traduz o modelo de frase e preenche os `{}`."""
    return tr(modelo).format(*args)


# ══════════════════════════════════════════════════════
# PROBLEMAS QUE A EMPRESA PODE RESOLVER
# ══════════════════════════════════════════════════════
PROBLEMAS = {
    "vendas": {
        "tipo": "regressao",
        "titulo": "Prever vendas ou receita",
        "descricao": "Estima quanto vais vender (em valor) nos próximos dias, semanas ou por loja.",
        "alvo": "Que coluna queres prever? (ex.: vendas, receita)",
        "palavras": ["vendas", "venda", "sales", "receita", "revenue", "faturacao", "faturação", "facturacao",
                     "facturação", "total", "montante", "valor"],
        "seguinte": ["Usa o modelo em «Previsões» para estimar as vendas dos próximos períodos.",
                     "Compara as previsões com o que acontece na realidade todos os meses."],
    },
    "stock": {
        "tipo": "regressao",
        "titulo": "Prever a procura e o stock",
        "descricao": "Estima quantas unidades vão ser pedidas, para comprar nem de mais nem de menos.",
        "alvo": "Que coluna queres prever? (ex.: unidades vendidas, procura)",
        "palavras": ["procura", "demanda", "demand", "unidades", "units", "quantidade", "qtd", "qty", "stock"],
        "seguinte": ["Usa o modelo em «Previsões» com as datas e preços previstos.",
                     "Define o stock mínimo por produto com base na previsão e na margem de segurança."],
    },
    "churn": {
        "tipo": "classificacao",
        "titulo": "Clientes que vão sair",
        "descricao": "Descobre que clientes estão em risco de cancelar, para os poderes contactar a tempo.",
        "alvo": "Que coluna diz se o cliente saiu? (ex.: saiu, cancelou)",
        "palavras": ["churn", "cancel", "saiu", "sair", "abandon", "left", "exited", "desist", "inativ", "inactiv"],
        "seguinte": ["Aplica o modelo à tua lista de clientes actuais em «Previsões».",
                     "Contacta primeiro quem tem o risco mais alto e mede se a acção funcionou."],
    },
    "fraude": {
        "tipo": "anomalia",
        "titulo": "Fraude e anomalias",
        "descricao": "Encontra operações fora do normal (transacções, faturas, despesas) para investigar.",
        "alvo": "",
        "palavras": ["fraud", "fraude", "suspeit"],
        "seguinte": ["Investiga primeiro as operações com a pontuação mais alta.",
                     "Confirma cada caso: o modelo aponta o que é invulgar, não prova que é fraude."],
    },
    "segmentos": {
        "tipo": "segmentacao",
        "titulo": "Segmentar clientes",
        "descricao": "Agrupa os clientes parecidos entre si, para tratar cada grupo de forma diferente.",
        "alvo": "",
        "palavras": [],
        "seguinte": ["Cria uma oferta ou mensagem diferente para cada segmento.",
                     "Aplica a segmentação a clientes novos em «Previsões»."],
    },
}

TIPOS_SUPERVISIONADOS = ("regressao", "classificacao")


# ══════════════════════════════════════════════════════
# LER E ARRUMAR TABELAS
# ══════════════════════════════════════════════════════
_RE_NUM_PONTO = re.compile(r"^-?\d+(\.\d+)?$")
_RE_NUM_VIRGULA = re.compile(r"^-?\d+(,\d+)?$")
_RE_DATA = re.compile(r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")


def _e_texto(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _detectar_sep(texto: str) -> str:
    linha = texto.split("\n", 1)[0]
    cont = {s: linha.count(s) for s in (";", ",", "\t", "|")}
    melhor = max(cont, key=cont.get)
    return melhor if cont[melhor] > 0 else ","


def _converter_numeros(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas escritas como texto mas que são números (inclui vírgula decimal e espaços de milhares)."""
    for c in df.columns:
        if not _e_texto(df[c]):
            continue
        s = df[c].dropna().astype(str).str.strip().str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
        if len(s) == 0:
            continue
        limpo = df[c].astype(str).str.strip().str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
        if s.map(lambda x: bool(_RE_NUM_PONTO.match(x))).mean() >= 0.9:
            df[c] = pd.to_numeric(limpo, errors="coerce").where(df[c].notna())
        elif s.map(lambda x: bool(_RE_NUM_VIRGULA.match(x))).mean() >= 0.9:
            df[c] = pd.to_numeric(limpo.str.replace(",", ".", regex=False), errors="coerce").where(df[c].notna())
    return df


def _converter_datas(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or not _e_texto(df[c]):
            continue
        amostra = df[c].dropna().astype(str).head(200)
        if len(amostra) == 0 or amostra.map(lambda x: bool(_RE_DATA.match(x))).mean() < 0.9:
            continue
        try:
            conv = pd.to_datetime(df[c], errors="coerce", dayfirst=not str(amostra.iloc[0]).strip()[:4].isdigit())
        except Exception:
            continue
        if conv.notna().mean() >= 0.9:
            df[c] = conv
    return df


def arrumar(df: pd.DataFrame) -> pd.DataFrame:
    """Nomes de colunas limpos e únicos, linhas/colunas vazias fora, números e datas reconhecidos."""
    df = df.copy()
    nomes, vistos = [], {}
    for i, c in enumerate(df.columns):
        n = " ".join(str(c).split()) or f"coluna_{i + 1}"
        vistos[n] = vistos.get(n, 0) + 1
        nomes.append(n if vistos[n] == 1 else f"{n}_{vistos[n]}")
    df.columns = nomes
    df = df.dropna(how="all").dropna(axis=1, how="all")
    for c in df.columns:
        if _e_texto(df[c]):
            df[c] = df[c].map(lambda x: " ".join(x.split()) if isinstance(x, str) else x)
            df[c] = df[c].replace("", np.nan)
    df = _converter_numeros(df)
    df = _converter_datas(df)
    return df.reset_index(drop=True)


def ler_tabela(nome: str, dados: bytes, folha=None) -> pd.DataFrame:
    """Lê CSV, TSV, TXT ou Excel (.xlsx/.xls). Levanta ValueError com uma mensagem clara."""
    ext = (nome or "").lower().rsplit(".", 1)[-1]
    try:
        if ext in ("xlsx", "xlsm", "xls"):
            df = pd.read_excel(io.BytesIO(dados), sheet_name=folha if folha is not None else 0)
        else:
            texto = None
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    texto = dados.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if texto is None or not texto.strip():
                raise ValueError("O ficheiro está vazio ou não é texto.")
            df = pd.read_csv(io.StringIO(texto), sep=_detectar_sep(texto))
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Não consegui ler o ficheiro: {e}")
    df = arrumar(df)
    if df.shape[1] < 2 or df.shape[0] < 2:
        raise ValueError("A tabela precisa de pelo menos 2 colunas e 2 linhas.")
    return df


def folhas_excel(dados: bytes) -> list:
    try:
        return list(pd.ExcelFile(io.BytesIO(dados)).sheet_names)
    except Exception:
        return []


# ══════════════════════════════════════════════════════
# GUARDAR / LER CONJUNTOS DE DADOS (CSV comprimido em texto)
# ══════════════════════════════════════════════════════
def tabela_para_texto(df: pd.DataFrame) -> str:
    bruto = df.to_csv(index=False).encode("utf-8")
    return base64.b64encode(gzip.compress(bruto, 6)).decode("ascii")


def texto_para_tabela(texto: str) -> pd.DataFrame:
    bruto = gzip.decompress(base64.b64decode(texto.encode("ascii")))
    return arrumar(pd.read_csv(io.BytesIO(bruto)))


def reduzir_para_guardar(df: pd.DataFrame):
    """Devolve (tabela, texto, foi_reduzida): se a tabela não cabe, guarda uma amostra aleatória que caiba."""
    texto = tabela_para_texto(df)
    if len(texto) <= MAX_BYTES_DADOS:
        return df, texto, False
    frac = max(0.05, (MAX_BYTES_DADOS / len(texto)) * 0.9)
    for _ in range(6):
        amostra = df.sample(frac=min(frac, 1.0), random_state=SEMENTE).sort_index().reset_index(drop=True)
        texto = tabela_para_texto(amostra)
        if len(texto) <= MAX_BYTES_DADOS:
            return amostra, texto, True
        frac *= 0.7
    raise ValueError("A tabela é grande demais para guardar. Reduz o número de colunas ou de linhas.")


def limpar_dados(df: pd.DataFrame):
    """Limpeza segura: linhas repetidas fora. Devolve (tabela, linhas_removidas)."""
    antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, antes - len(df)


# ══════════════════════════════════════════════════════
# DIAGNÓSTICO (linguagem de negócio)
# ══════════════════════════════════════════════════════
def _e_id(nome, s: pd.Series, n: int) -> bool:
    nome = str(nome).lower()
    nun = s.nunique(dropna=True)
    if n < 20 or nun == 0:
        return False
    por_nome = nome in ("id", "index", "codigo", "código", "code", "uuid") or nome.endswith(("_id", " id", "_codigo", "_code")) \
        or nome.startswith(("id_", "id ", "cod_"))
    return (por_nome and nun >= 0.9 * n) or (_e_texto(s) and nun >= 0.98 * n)


def colunas_id(df: pd.DataFrame) -> list:
    return [c for c in df.columns if _e_id(c, df[c], len(df))]


def colunas_data(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def diagnostico(df: pd.DataFrame) -> list:
    """Lista de (nivel, modelo_de_frase, argumentos): nivel em ok | aviso | erro | info."""
    out = []
    n, c = df.shape
    out.append(("info", "A tabela tem **{}** linhas e **{}** colunas.", (f"{n:,}".replace(",", " "), c)))
    if n < 100:
        out.append(("aviso", "Poucas linhas (menos de 100): os resultados serão pouco fiáveis. Quantas mais linhas, melhor.", ()))
    elif n < 500:
        out.append(("info", "Tens poucas linhas (menos de 500): os modelos funcionam, mas os resultados mexem com novos dados.", ()))

    dup = int(df.duplicated().sum())
    if dup:
        out.append(("aviso", "Há **{}** linhas repetidas ({}). Podes removê-las ao guardar os dados.", (dup, f"{dup / n:.0%}")))

    falta = df.isna().sum()
    falta = falta[falta > 0].sort_values(ascending=False)
    if len(falta):
        top = ", ".join(f"**{k}** ({v / n:.0%})" for k, v in falta.head(4).items())
        out.append(("aviso", "Há valores em falta em {} coluna(s): {}. O modelo preenche-os automaticamente, mas confirma se são erros de registo.", (len(falta), top)))
        for k, v in falta.items():
            if v / n > 0.5:
                out.append(("erro", "A coluna **{}** tem mais de metade dos valores em falta. Provavelmente é melhor ignorá-la.", (k,)))
    else:
        out.append(("ok", "Não há valores em falta.", ()))

    datas = colunas_data(df)
    if datas:
        d = df[datas[0]].dropna()
        if len(d):
            out.append(("info", "A coluna de datas **{}** vai de {} a {}. O modelo usa o mês, o dia da semana e o ano.",
                        (datas[0], d.min().strftime("%Y-%m-%d"), d.max().strftime("%Y-%m-%d"))))
    ids = colunas_id(df)
    if ids:
        out.append(("info", "As colunas {} parecem identificadores (valores todos diferentes). São ignoradas na previsão.",
                    (", ".join(f"**{i}**" for i in ids[:5]),)))

    for col in df.columns:
        if col in ids:
            continue
        s = df[col]
        nun = s.nunique(dropna=True)
        if nun <= 1:
            out.append(("aviso", "A coluna **{}** tem sempre o mesmo valor: não ajuda em nada.", (col,)))
        elif _e_texto(s) and nun > MAX_CATEGORIAS:
            out.append(("aviso", "A coluna **{}** tem {} categorias diferentes: é demasiado para o modelo e será ignorada.", (col, nun)))

    num = df.select_dtypes(include=[np.number])
    extremos = []
    for col in num.columns:
        s = num[col].dropna()
        if len(s) > 20 and s.std() > 0:
            k = int((((s - s.mean()) / s.std()).abs() > 4).sum())
            if k:
                extremos.append(f"**{col}** ({k})")
    if extremos:
        out.append(("aviso", "Valores muito fora do normal em: {}. Podem ser erros de escrita ou casos raros: confirma.", (", ".join(extremos[:4]),)))

    if num.shape[1] >= 2:
        corr = num.corr().abs().where(~np.eye(num.shape[1], dtype=bool))
        mx = corr.max().max()
        if pd.notna(mx) and mx > 0.97:
            a, b = corr.stack().idxmax()
            out.append(("aviso", "As colunas **{}** e **{}** são quase iguais (correlação {}). Se uma delas só existe depois do que queres prever, não a uses.", (a, b, f"{mx:.2f}")))
    return out


def resumo_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela com uma linha por coluna: tipo, em falta, valores diferentes e exemplo."""
    linhas = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            tipo = "Data"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            tipo = "Número"
        else:
            tipo = "Texto"
        ex = s.dropna()
        linhas.append({"Coluna": c, "Tipo": tipo, "Em falta (%)": round(float(s.isna().mean() * 100), 1),
                       "Valores diferentes": int(s.nunique(dropna=True)),
                       "Exemplo": str(ex.iloc[0])[:30] if len(ex) else ""})
    return pd.DataFrame(linhas)


def sugerir_alvo(df: pd.DataFrame, problema: str):
    """Coluna mais provável para o problema escolhido (ou None)."""
    cfg = PROBLEMAS.get(problema, {})
    tipo = cfg.get("tipo")
    if tipo not in TIPOS_SUPERVISIONADOS:
        return None
    ids = set(colunas_id(df))
    cands = []
    for c in df.columns:
        if c in ids or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        s = df[c]
        if tipo == "regressao" and not (pd.api.types.is_numeric_dtype(s) and s.nunique() > 10):
            continue
        if tipo == "classificacao" and not (2 <= s.nunique(dropna=True) <= 10):
            continue
        nome = str(c).lower()
        pontos = 0
        for i, p in enumerate(cfg.get("palavras", [])):
            if p in nome:
                pontos = max(pontos, 100 - i)
        cands.append((pontos, c))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    return cands[0][1] if cands[0][0] > 0 else None


def colunas_alvo_possiveis(df: pd.DataFrame, tipo: str) -> list:
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        nun = s.nunique(dropna=True)
        if tipo == "regressao" and pd.api.types.is_numeric_dtype(s) and nun > 2:
            out.append(c)
        elif tipo == "classificacao" and 2 <= nun <= 20:
            out.append(c)
    return out


# ══════════════════════════════════════════════════════
# PREPARAÇÃO DAS COLUNAS (igual no treino e nas previsões)
# ══════════════════════════════════════════════════════
def _expandir_datas(X: pd.DataFrame, datas: list) -> pd.DataFrame:
    X = X.copy()
    for c in datas:
        if c not in X.columns:
            continue
        d = pd.to_datetime(X[c], errors="coerce")
        X[f"{c}__ano"] = d.dt.year
        X[f"{c}__mes"] = d.dt.month
        X[f"{c}__dia"] = d.dt.day
        X[f"{c}__dia_da_semana"] = d.dt.dayofweek
        X = X.drop(columns=[c])
    return X


def _texto_limpo(v):
    return str(v).strip() if isinstance(v, (str, bool, int, float, np.integer, np.floating, np.bool_)) and v == v else np.nan


def _preparar_X(X: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Garante as colunas, a ordem e os tipos que o modelo espera. Colunas em falta ficam vazias."""
    X = X.copy()
    for c in meta["colunas_entrada"]:
        if c not in X.columns:
            X[c] = np.nan
    X = _expandir_datas(X[meta["colunas_entrada"]], meta["datas"])
    for c in meta["numericas"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in meta["categoricas"]:
        X[c] = X[c].astype(object).map(_texto_limpo).astype(object)
    return X[meta["numericas"] + meta["categoricas"]]


def _meta_entrada(df: pd.DataFrame, excluir: list, alvo=None) -> dict:
    """Decide que colunas entram no modelo e como (número, categoria ou data)."""
    ids = set(colunas_id(df))
    ignoradas, usadas = [], []
    for c in df.columns:
        if c == alvo or c in excluir:
            continue
        s = df[c]
        if c in ids:
            ignoradas.append((c, "identificador"))
        elif s.nunique(dropna=True) <= 1:
            ignoradas.append((c, "constante"))
        elif s.isna().mean() > 0.95:
            ignoradas.append((c, "vazia"))
        elif _e_texto(s) and s.nunique(dropna=True) > MAX_CATEGORIAS:
            ignoradas.append((c, "demasiadas categorias"))
        else:
            usadas.append(c)
    datas = [c for c in usadas if pd.api.types.is_datetime64_any_dtype(df[c])]
    X0 = _expandir_datas(df[usadas], datas)
    numericas = [c for c in X0.columns if pd.api.types.is_numeric_dtype(X0[c]) and not pd.api.types.is_bool_dtype(X0[c])]
    categoricas = [c for c in X0.columns if c not in numericas]
    meta = {"colunas_entrada": usadas, "datas": datas, "numericas": numericas, "categoricas": categoricas,
            "ignoradas": [[a, b] for a, b in ignoradas], "alvo": alvo, "tipos": {}, "referencia": {}}
    for c in usadas:
        s = df[c]
        if c in datas:
            d = s.dropna()
            meta["tipos"][c] = "data"
            meta["referencia"][c] = {"min": d.min().strftime("%Y-%m-%d") if len(d) else "",
                                     "max": d.max().strftime("%Y-%m-%d") if len(d) else ""}
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            d = s.dropna()
            meta["tipos"][c] = "numero"
            meta["referencia"][c] = {"mediana": float(d.median()) if len(d) else 0.0,
                                     "min": float(d.min()) if len(d) else 0.0,
                                     "max": float(d.max()) if len(d) else 0.0,
                                     "inteiro": bool(len(d) and bool((d % 1 == 0).all()))}
        else:
            vc = s.dropna().astype(str).value_counts()
            meta["tipos"][c] = "categoria"
            meta["referencia"][c] = {"valores": [str(v) for v in vc.index[:MAX_CATEGORIAS]]}
    return meta


def _preprocessador(meta: dict, escalar: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    passos_num = [("imp", SimpleImputer(strategy="median"))]
    if escalar:
        passos_num.append(("esc", StandardScaler()))
    partes = []
    if meta["numericas"]:
        partes.append(("num", Pipeline(passos_num), meta["numericas"]))
    if meta["categoricas"]:
        partes.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="(vazio)")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            meta["categoricas"]))
    return ColumnTransformer(partes, remainder="drop")


# ══════════════════════════════════════════════════════
# TREINO SUPERVISIONADO (regressão e classificação)
# ══════════════════════════════════════════════════════
def _candidatos(tipo: str, meta: dict):
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    if tipo == "classificacao":
        return [
            ("Regressão logística", lambda: Pipeline([("pre", _preprocessador(meta, True)),
                                                      ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))])),
            ("Floresta aleatória", lambda: Pipeline([("pre", _preprocessador(meta, False)),
                                                     ("m", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2,
                                                                                  class_weight="balanced_subsample", n_jobs=-1,
                                                                                  random_state=SEMENTE))])),
            ("Gradient boosting", lambda: Pipeline([("pre", _preprocessador(meta, False)),
                                                    ("m", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1,
                                                                                         class_weight="balanced", random_state=SEMENTE))])),
        ]
    return [
        ("Regressão linear (Ridge)", lambda: Pipeline([("pre", _preprocessador(meta, True)), ("m", Ridge(alpha=1.0))])),
        ("Floresta aleatória", lambda: Pipeline([("pre", _preprocessador(meta, False)),
                                                 ("m", RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=2,
                                                                             n_jobs=-1, random_state=SEMENTE))])),
        ("Gradient boosting", lambda: Pipeline([("pre", _preprocessador(meta, False)),
                                                ("m", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=SEMENTE))])),
    ]


def _pontuar(tipo, modelo, X, y, binario, positiva):
    from sklearn.metrics import f1_score, r2_score, roc_auc_score
    if tipo == "regressao":
        return float(r2_score(y, modelo.predict(X)))
    if binario:
        try:
            return float(roc_auc_score(y == positiva, modelo.predict_proba(X)[:, list(modelo.classes_).index(positiva)]))
        except Exception:
            pass
    return float(f1_score(y, modelo.predict(X), average="macro"))


def _dividir(Xn, y, tipo, por_data, col_data, teste):
    from sklearn.model_selection import train_test_split
    if por_data and col_data is not None:
        ordem = np.argsort(pd.to_datetime(col_data, errors="coerce").fillna(pd.Timestamp("1970-01-01")).values, kind="stable")
        corte = int(len(ordem) * (1 - teste))
        ti, ei = ordem[:corte], ordem[corte:]
        return Xn.iloc[ti], Xn.iloc[ei], y.iloc[ti], y.iloc[ei]
    estrat = None
    if tipo == "classificacao":
        vc = y.value_counts()
        if vc.min() >= 2 and len(vc) <= 20:
            estrat = y
    return train_test_split(Xn, y, test_size=teste, random_state=SEMENTE, stratify=estrat)


def _importancias(modelo, X, y, tipo, binario, positiva, n=1500):
    """Quanto piora a previsão quando uma coluna é baralhada (peso relativo, soma 1)."""
    from sklearn.inspection import permutation_importance
    try:
        if len(X) > n:
            idx = np.random.RandomState(SEMENTE).choice(len(X), n, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]
        if tipo == "regressao":
            scoring = "r2"
        elif binario and y.nunique() > 1:
            from sklearn.metrics import roc_auc_score
            ci = list(modelo.classes_).index(positiva)
            y = (y == positiva).astype(int)
            scoring = lambda est, Xs, ys: roc_auc_score(ys, est.predict_proba(Xs)[:, ci])
        else:
            scoring = "f1_macro"
        r = permutation_importance(modelo, X, y, n_repeats=3, random_state=SEMENTE, scoring=scoring)
        imp = {c: max(float(v), 0.0) for c, v in zip(X.columns, r.importances_mean)}
    except Exception:
        return []
    total = sum(imp.values())
    if total <= 0:
        return []
    ordenado = sorted(imp.items(), key=lambda kv: -kv[1])
    return [[c, round(v / total, 4)] for c, v in ordenado[:12] if v > 0]


def treinar(df: pd.DataFrame, alvo: str, tipo: str, excluir=(), por_data: bool = False, teste: float = 0.25) -> dict:
    """Treina 3 algoritmos, escolhe o melhor numa validação e mede-o em dados que nunca viu.
    Devolve {'resultado': dict simples (para gravar), 'modelo': Pipeline, 'meta': dict}."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                                 precision_score, r2_score, recall_score, roc_auc_score)
    t0 = time.time()
    if alvo not in df.columns:
        raise ValueError("A coluna a prever não existe nesta tabela.")
    excluir = [c for c in excluir if c in df.columns and c != alvo]
    d = df.dropna(subset=[alvo]).reset_index(drop=True)
    avisos = []
    if len(d) < len(df):
        avisos.append(("Há {} linhas sem valor na coluna a prever; foram deixadas de fora.", (len(df) - len(d),)))
    if len(d) < 30:
        raise ValueError("Precisas de pelo menos 30 linhas com valor na coluna a prever.")

    classes, positiva, binario = [], None, False
    y = d[alvo]
    if tipo == "regressao":
        y = pd.to_numeric(y, errors="coerce")
        if y.isna().mean() > 0.2:
            raise ValueError("A coluna a prever não é numérica. Escolhe outra ou muda o tipo de problema.")
        d = d.loc[y.notna()].reset_index(drop=True)
        y = y.loc[y.notna()].reset_index(drop=True)
    else:
        y = y.astype(str).str.strip()
        vc = y.value_counts()
        if len(vc) < 2:
            raise ValueError("A coluna a prever só tem um valor. Precisa de pelo menos 2 (ex.: sim e não).")
        if len(vc) > 20:
            raise ValueError("A coluna a prever tem demasiados valores diferentes para ser uma categoria.")
        classes = sorted(vc.index.tolist())
        binario = len(classes) == 2
        positiva = str(vc.index[-1]) if binario else None            # a menos frequente (ex.: «sim, saiu»)
        if binario:
            for pref in ("sim", "yes", "1", "true", "saiu", "fraude"):   # se houver um «sim» claro, esse é o alvo
                achados = [c for c in classes if c.lower() == pref]
                if achados:
                    positiva = achados[0]
                    break
        if vc.min() < 5:
            avisos.append(("Há valores da coluna a prever com menos de 5 exemplos: o modelo terá dificuldade com eles.", ()))
    y = y.reset_index(drop=True)

    if len(d) > MAX_LINHAS_TREINO:
        idx = d.sample(n=MAX_LINHAS_TREINO, random_state=SEMENTE).sort_index().index
        d, y = d.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
        avisos.append(("Os dados são grandes: o modelo foi treinado numa amostra de {} linhas.", (f"{MAX_LINHAS_TREINO:,}".replace(",", " "),)))

    meta = _meta_entrada(d, excluir, alvo)
    if not (meta["numericas"] or meta["categoricas"]):
        raise ValueError("Não sobrou nenhuma coluna útil para prever. Confirma as colunas ignoradas.")
    meta.update({"tipo": tipo, "classes": classes, "positiva": positiva})
    X = _preparar_X(d, meta)
    col_data = d[meta["datas"][0]] if (por_data and meta["datas"]) else None
    Xtr, Xte, ytr, yte = _dividir(X, y, tipo, por_data, col_data, teste)

    # escolhe o melhor numa validação interna (o conjunto de teste fica intocado)
    Xa, Xv, ya, yv = _dividir(Xtr, ytr, tipo, False, None, 0.2)
    candidatos = []
    for nome, fabrica in _candidatos(tipo, meta):
        t1 = time.time()
        try:
            m = fabrica().fit(Xa, ya)
            candidatos.append({"nome": nome, "pontuacao": round(_pontuar(tipo, m, Xv, yv, binario, positiva), 4),
                               "segundos": round(time.time() - t1, 1)})
        except Exception as e:                          # um algoritmo que falha não pára os outros
            candidatos.append({"nome": nome, "pontuacao": None, "segundos": 0, "erro": str(e)[:120]})
    validos = [c for c in candidatos if c["pontuacao"] is not None]
    if not validos:
        raise ValueError("Não foi possível treinar com estes dados. Confirma as colunas escolhidas.")
    melhor = max(validos, key=lambda c: c["pontuacao"])
    fabrica = dict(_candidatos(tipo, meta))[melhor["nome"]]
    modelo = fabrica().fit(Xtr, ytr)

    # medição honesta em dados que o modelo nunca viu
    pred = modelo.predict(Xte)
    res = {"tipo": tipo, "alvo": alvo, "melhor": melhor["nome"], "candidatos": candidatos,
           "n_linhas": int(len(d)), "n_treino": int(len(Xtr)), "n_teste": int(len(Xte)),
           "por_data": bool(col_data is not None), "classes": classes, "positiva": positiva}
    if tipo == "regressao":
        dummy = DummyRegressor(strategy="mean").fit(Xtr, ytr)
        mae, mae0 = mean_absolute_error(yte, pred), mean_absolute_error(yte, dummy.predict(Xte))
        media = float(np.mean(y))
        res["metricas"] = {"r2": round(float(r2_score(yte, pred)), 4), "mae": float(mae),
                           "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                           "mae_base": float(mae0), "media_alvo": media,
                           "erro_relativo": float(mae / abs(media)) if media else None}
        pts = np.random.RandomState(SEMENTE).choice(len(yte), min(300, len(yte)), replace=False)
        res["grafico"] = {"real": [round(float(v), 4) for v in np.asarray(yte)[pts]],
                          "previsto": [round(float(v), 4) for v in np.asarray(pred)[pts]]}
        if res["metricas"]["r2"] < 0.1:
            avisos.append(("O modelo quase não aprendeu nada com estes dados. Faltam colunas com informação relevante ou há muito ruído.", ()))
    else:
        dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
        m = {"accuracy": float(accuracy_score(yte, pred)), "accuracy_base": float(accuracy_score(yte, dummy.predict(Xte)))}
        if binario:
            ci = list(modelo.classes_).index(positiva)
            prob = modelo.predict_proba(Xte)[:, ci]
            m.update({"precisao": float(precision_score(yte, pred, pos_label=positiva, zero_division=0)),
                      "recall": float(recall_score(yte, pred, pos_label=positiva, zero_division=0)),
                      "f1": float(f1_score(yte, pred, pos_label=positiva, zero_division=0)),
                      "taxa_positiva": float((y == positiva).mean())})
            try:
                m["auc"] = float(roc_auc_score(yte == positiva, prob))
            except Exception:
                m["auc"] = None
        else:
            m["f1"] = float(f1_score(yte, pred, average="macro", zero_division=0))
        res["metricas"] = m
        labs = list(modelo.classes_)
        res["matriz"] = {"classes": labs, "valores": confusion_matrix(yte, pred, labels=labs).tolist()}
        if binario and m["taxa_positiva"] < 0.15:
            avisos.append(("Os casos «{}» são raros ({}). Por isso olha para o recall e a precisão, não só para a taxa de acerto.",
                           (positiva, f"{m['taxa_positiva']:.0%}")))
    res["importancias"] = _importancias(modelo, Xte, yte, tipo, binario, positiva)
    if res["importancias"] and res["importancias"][0][1] > 0.85:
        avisos.append(("A coluna **{}** pesa quase tudo na previsão. Se ela só é conhecida depois do que queres prever, o resultado é bom demais para ser verdade.",
                       (res["importancias"][0][0],)))
    nota = res["metricas"].get("r2") if tipo == "regressao" else (res["metricas"].get("auc") or res["metricas"].get("f1"))
    if nota is not None and nota > 0.99:
        avisos.append(("Resultado quase perfeito: desconfia. Confirma que nenhuma coluna contém já a resposta.", ()))
    if len(d) < 200:
        avisos.append(("Com menos de 200 linhas os resultados são pouco estáveis. Junta mais dados quando puderes.", ()))
    if meta["ignoradas"]:
        avisos.append(("Colunas ignoradas automaticamente: {}.", (", ".join(f"{a} ({b})" for a, b in meta["ignoradas"][:6]),)))
    res["avisos"] = [[a, list(b)] for a, b in avisos]

    # modelo final: volta a treinar com todos os dados (quando é barato) para usar em produção
    if len(X) <= 20000:
        try:
            modelo = fabrica().fit(X, y)
        except Exception:
            pass
    res["segundos"] = round(time.time() - t0, 1)
    return {"resultado": res, "modelo": modelo, "meta": meta}


# ══════════════════════════════════════════════════════
# FRAUDE E ANOMALIAS
# ══════════════════════════════════════════════════════
def _estatisticas_referencia(X: pd.DataFrame, meta: dict) -> dict:
    est = {"num": {}, "cat": {}}
    for c in meta["numericas"]:
        s = X[c].dropna()
        med = float(s.median()) if len(s) else 0.0
        mad = float((s - med).abs().median()) * 1.4826 if len(s) else 0.0
        est["num"][c] = (med, mad if mad > 0 else float(s.std() or 0.0))
    for c in meta["categoricas"]:
        est["cat"][c] = X[c].value_counts(normalize=True).to_dict()
    return est


def _motivos(X: pd.DataFrame, pos: int, meta: dict, est: dict) -> list:
    """Porque é que uma linha é invulgar: [[coluna, 'acima'|'abaixo'|'raro', valor]] (máx. 3)."""
    motivos = []
    for c in meta["numericas"]:
        v = X.iloc[pos][c]
        med, esc = est["num"][c]
        if pd.isna(v) or esc == 0:
            continue
        z = (v - med) / esc
        if abs(z) >= 3:
            motivos.append((abs(z), [c, "acima" if z > 0 else "abaixo", f"{v:.4g}"]))
    for c in meta["categoricas"]:
        v = X.iloc[pos][c]
        if isinstance(v, str) and est["cat"][c].get(v, 1.0) < 0.01:
            motivos.append((3.0, [c, "raro", v]))
    motivos.sort(key=lambda x: -x[0])
    return [m for _, m in motivos[:3]]


def _resultado_anomalias(pipe, X: pd.DataFrame, meta: dict, t0: float, taxa: float):
    score = -pipe.decision_function(X)                       # maior = mais invulgar
    suspeito = pipe.predict(X) == -1
    est = _estatisticas_referencia(X, meta)
    tabela = pd.DataFrame({"linha": np.arange(len(X)) + 2, "pontuacao": np.round(score, 4), "suspeito": suspeito})
    top_idx = np.argsort(-score)[:50]
    top = [{"linha": int(i) + 2, "pontuacao": round(float(score[i]), 4), "motivos": _motivos(X, int(i), meta, est)} for i in top_idx]
    amostra = np.random.RandomState(SEMENTE).choice(score, min(2000, len(score)), replace=False)
    res = {"tipo": "anomalia", "n_linhas": int(len(X)), "n_suspeitos": int(suspeito.sum()), "taxa": taxa,
           "top": top, "colunas": meta["numericas"] + meta["categoricas"], "avisos": [],
           "segundos": round(time.time() - t0, 1),
           "grafico": {"pontuacoes": [round(float(v), 4) for v in amostra],
                       "limiar": round(float(np.min(score[suspeito])) if suspeito.any() else float(np.max(score)), 4)}}
    if meta.get("ignoradas"):
        res["avisos"].append(["Colunas ignoradas automaticamente: {}.",
                              [", ".join(f"{a} ({b})" for a, b in meta["ignoradas"][:6])]])
    return res, tabela


def detectar_anomalias(df: pd.DataFrame, excluir=(), taxa: float = 0.02) -> dict:
    """Isolation Forest: aponta as linhas mais invulgares. `taxa` = fracção que se espera ser anómala."""
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    t0 = time.time()
    meta = _meta_entrada(df, list(excluir))
    if not (meta["numericas"] or meta["categoricas"]):
        raise ValueError("Não sobrou nenhuma coluna útil. Confirma as colunas ignoradas.")
    meta.update({"tipo": "anomalia", "classes": [], "positiva": None, "taxa": taxa})
    X = _preparar_X(df, meta)
    pipe = Pipeline([("pre", _preprocessador(meta, True)),
                     ("m", IsolationForest(n_estimators=200, contamination=taxa, random_state=SEMENTE))])
    pipe.fit(X.sample(n=min(len(X), MAX_LINHAS_TREINO), random_state=SEMENTE))
    res, tabela = _resultado_anomalias(pipe, X, meta, t0, taxa)
    return {"modelo": pipe, "meta": meta, "resultado": res, "tabela": tabela}


def texto_motivo(m: list) -> str:
    col, tipo, valor = m
    if tipo == "acima":
        return T("{}: valor muito acima do normal ({})", col, valor)
    if tipo == "abaixo":
        return T("{}: valor muito abaixo do normal ({})", col, valor)
    return T("{}: valor raro ({})", col, valor)


# ══════════════════════════════════════════════════════
# SEGMENTAÇÃO DE CLIENTES
# ══════════════════════════════════════════════════════
def perfil_segmentos(X: pd.DataFrame, rotulos, meta: dict) -> list:
    """Por segmento: tamanho, médias e as 2 características que mais o distinguem (alto/baixo)."""
    rotulos = np.asarray(rotulos)
    num = meta["numericas"]
    geral_med = X[num].mean()
    geral_std = X[num].std().replace(0, np.nan)
    perfis = []
    for g in sorted(set(rotulos.tolist())):
        sel = X[rotulos == g]
        medias = sel[num].mean()
        z = ((medias - geral_med) / geral_std).fillna(0)
        top = z.abs().sort_values(ascending=False).head(2).index.tolist()
        carac = [[c, "alto" if z[c] > 0 else "baixo", f"{medias[c]:.4g}", f"{geral_med[c]:.4g}"] for c in top if abs(z[c]) >= 0.3]
        perfis.append({"segmento": int(g) + 1, "tamanho": int(len(sel)), "percentagem": round(len(sel) / len(X), 4),
                       "medias": {c: round(float(medias[c]), 4) for c in num[:12]}, "caracteristicas": carac})
    return perfis


def segmentar(df: pd.DataFrame, excluir=(), k=None) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import Pipeline
    t0 = time.time()
    meta = _meta_entrada(df, list(excluir))
    if not meta["numericas"]:
        raise ValueError("A segmentação precisa de pelo menos uma coluna com números (ex.: gasto, idade, visitas).")
    meta.update({"tipo": "segmentacao", "classes": [], "positiva": None})
    X = _preparar_X(df, meta)
    amostra = X.sample(n=min(len(X), MAX_LINHAS_TREINO), random_state=SEMENTE)
    pre = _preprocessador(meta, True).fit(amostra)
    Z = pre.transform(amostra)
    if k is None:
        melhor, melhor_s = 3, -1.0
        zs = Z[np.random.RandomState(SEMENTE).choice(len(Z), min(len(Z), 4000), replace=False)]
        for kk in range(2, 8):
            if kk >= len(zs):
                break
            try:
                s = silhouette_score(zs, KMeans(n_clusters=kk, n_init=5, random_state=SEMENTE).fit_predict(zs))
            except Exception:
                continue
            if s > melhor_s:
                melhor, melhor_s = kk, s
        k = melhor
    km = KMeans(n_clusters=int(k), n_init=10, random_state=SEMENTE).fit(Z)
    pipe = Pipeline([("pre", pre), ("m", km)])
    rotulos = pipe.predict(X)
    perfis = perfil_segmentos(X, rotulos, meta)
    Zt = pre.transform(X)
    idx = np.random.RandomState(SEMENTE).choice(len(Zt), min(1500, len(Zt)), replace=False)
    if Zt.shape[1] >= 2:
        xy = PCA(n_components=2, random_state=SEMENTE).fit_transform(Zt[idx])
    else:
        xy = np.c_[Zt[idx, 0], np.zeros(len(idx))]
    avisos = []
    if meta.get("ignoradas"):
        avisos.append(["Colunas ignoradas automaticamente: {}.", [", ".join(f"{a} ({b})" for a, b in meta["ignoradas"][:6])]])
    res = {"tipo": "segmentacao", "k": int(k), "n_linhas": int(len(X)), "perfis": perfis, "avisos": avisos,
           "grafico": {"x": [round(float(v), 4) for v in xy[:, 0]], "y": [round(float(v), 4) for v in xy[:, 1]],
                       "segmento": [int(v) + 1 for v in rotulos[idx]]},
           "segundos": round(time.time() - t0, 1)}
    return {"modelo": pipe, "meta": meta, "resultado": res, "rotulos": rotulos + 1}


def nome_segmento(p: dict) -> str:
    if not p["caracteristicas"]:
        return T("Segmento {}: perfil médio", p["segmento"])
    partes = [f"{c} {tr('alto') if n == 'alto' else tr('baixo')}" for c, n, *_ in p["caracteristicas"]]
    return T("Segmento {}: {}", p["segmento"], ", ".join(partes))


# ══════════════════════════════════════════════════════
# APLICAR UM MODELO A DADOS NOVOS
# ══════════════════════════════════════════════════════
def nivel_risco(prob: float) -> str:
    return "alto" if prob >= 0.66 else ("médio" if prob >= 0.33 else "baixo")


def colunas_em_falta(meta: dict, df: pd.DataFrame) -> list:
    return [c for c in meta["colunas_entrada"] if c not in df.columns]


def aplicar_modelo(meta: dict, modelo, df: pd.DataFrame) -> pd.DataFrame:
    """Devolve a tabela original com as colunas de previsão à direita."""
    X = _preparar_X(df, meta)
    saida = df.reset_index(drop=True).copy()
    tipo = meta["tipo"]
    if tipo == "regressao":
        saida["previsao"] = np.round(modelo.predict(X), 2)
    elif tipo == "classificacao":
        saida["previsao"] = modelo.predict(X)
        if meta.get("positiva") is not None:
            ci = list(modelo.classes_).index(meta["positiva"])
            p = modelo.predict_proba(X)[:, ci]
            saida[f"probabilidade_{meta['positiva']}"] = np.round(p, 4)
            saida["risco"] = [nivel_risco(v) for v in p]
        else:
            saida["confianca"] = np.round(modelo.predict_proba(X).max(axis=1), 4)
    elif tipo == "anomalia":
        saida["pontuacao_anomalia"] = np.round(-modelo.decision_function(X), 4)
        saida["suspeito"] = np.where(modelo.predict(X) == -1, "sim", "não")
    elif tipo == "segmentacao":
        saida["segmento"] = modelo.predict(X) + 1
    return saida


def prever_um_caso(meta: dict, modelo, valores: dict) -> dict:
    """Previsão para uma só linha escrita à mão. Devolve a linha com as colunas de previsão."""
    return aplicar_modelo(meta, modelo, pd.DataFrame([valores])).iloc[0].to_dict()


# ══════════════════════════════════════════════════════
# GUARDAR / CARREGAR MODELOS
# ══════════════════════════════════════════════════════
def modelo_para_texto(modelo):
    """Texto (comprimido e em base64) do modelo, ou None se for grande demais para guardar."""
    import joblib
    buf = io.BytesIO()
    joblib.dump(modelo, buf, compress=3)
    texto = base64.b64encode(buf.getvalue()).decode("ascii")
    return texto if len(texto) <= MAX_BYTES_MODELO else None


def texto_para_modelo(texto: str):
    import joblib
    return joblib.load(io.BytesIO(base64.b64decode(texto.encode("ascii"))))


# ══════════════════════════════════════════════════════
# EXPLICAÇÕES EM LINGUAGEM SIMPLES
# ══════════════════════════════════════════════════════
def _num(v) -> str:
    v = float(v)
    if abs(v) >= 100:
        return f"{v:,.0f}".replace(",", " ")
    if abs(v) >= 1:
        return f"{v:,.2f}".replace(",", " ")
    return f"{v:.3g}"


def qualidade(res: dict):
    """(rótulo, cor) da qualidade geral do modelo supervisionado."""
    m = res.get("metricas", {})
    if res["tipo"] == "regressao":
        nota = m.get("r2")
    else:
        nota = m.get("auc") if m.get("auc") is not None else m.get("f1")
    if nota is None:
        return "Razoável", "amber"
    if nota >= 0.9:
        return "Excelente", "green"
    if nota >= 0.75:
        return "Bom", "green"
    if nota >= 0.6:
        return "Razoável", "amber"
    return "Fraco", "red"


def manchete(res: dict) -> str:
    """A frase principal do resultado, em linguagem de gestor."""
    m = res.get("metricas", {})
    if res["tipo"] == "regressao":
        frase = T("Em média, a previsão falha por **{}** (um palpite sem modelo falharia por **{}**).",
                  _num(m["mae"]), _num(m["mae_base"]))
        if m.get("erro_relativo") is not None:
            frase += " " + T("Isso é cerca de **{}** do valor médio.", f"{m['erro_relativo']:.0%}")
        return frase + " " + T("Explica **{}** da variação de **{}**.", f"{max(m['r2'], 0):.0%}", res["alvo"])
    base = T("O modelo acerta em **{}** dos casos (sem modelo, acertava-se **{}**).", f"{m['accuracy']:.0%}", f"{m['accuracy_base']:.0%}")
    if res.get("positiva") is not None and "recall" in m:
        if m["accuracy"] < m["accuracy_base"]:      # casos raros: a taxa de acerto geral engana
            base = ""
        base = (base + " " if base else "") + T("Dos casos «{}» reais, apanha **{}**; quando avisa, acerta em **{}** das vezes.",
                        res["positiva"], f"{m['recall']:.0%}", f"{m['precisao']:.0%}")
    return base


def texto_aviso(a) -> str:
    modelo, args = a
    return T(modelo, *args)


# ══════════════════════════════════════════════════════
# DADOS DE EXEMPLO (para experimentar sem ter ficheiros)
# ══════════════════════════════════════════════════════
EXEMPLOS = {
    "vendas": ("Vendas diárias por loja", "vendas"),
    "stock": ("Procura de produtos (stock)", "stock"),
    "clientes": ("Clientes e cancelamentos", "churn"),
    "transaccoes": ("Transacções (para detectar fraude)", "fraude"),
}


def exemplo(nome: str) -> pd.DataFrame:
    rs = np.random.RandomState(SEMENTE)
    if nome == "vendas":
        datas = pd.date_range("2024-01-01", periods=540, freq="D")
        dow = datas.dayofweek.values
        linhas = []
        for loja, base in (("Luanda", 1800), ("Benguela", 1100), ("Huambo", 800)):
            promo = (rs.rand(len(datas)) < 0.18).astype(int)
            visitantes = (base / 12 * (1 + 0.35 * (dow >= 4)) * (1 + 0.25 * promo) * rs.uniform(0.85, 1.15, len(datas))).round()
            preco = rs.uniform(900, 1300, len(datas)).round(0)
            vendas = (visitantes * preco * 0.11 * (1 + 0.2 * promo) * (1 + 0.1 * np.sin(np.arange(len(datas)) / 58))
                      * rs.uniform(0.9, 1.1, len(datas))).round(0)
            linhas.append(pd.DataFrame({"data": datas, "loja": loja, "promocao": np.where(promo == 1, "sim", "não"),
                                        "preco_medio": preco, "visitantes": visitantes, "vendas": vendas}))
        return pd.concat(linhas, ignore_index=True)
    if nome == "stock":
        datas = pd.date_range("2024-03-01", periods=300, freq="D")
        prods = {"Arroz 5kg": 4200, "Óleo 1L": 1800, "Açúcar 2kg": 2100, "Detergente": 1500, "Sumo 1L": 900}
        linhas = []
        for p, preco0 in prods.items():
            promo = (rs.rand(len(datas)) < 0.15).astype(int)
            preco = (preco0 * np.where(promo == 1, 0.88, 1) * rs.uniform(0.97, 1.05, len(datas))).round(0)
            fim = (datas.dayofweek.values >= 5).astype(int)
            proc = (40 + 14 * promo + 9 * fim + 6 * np.sin(np.arange(len(datas)) / 30) - (preco - preco0) / preco0 * 40
                    + rs.normal(0, 4, len(datas))).clip(2).round(0)
            linhas.append(pd.DataFrame({"data": datas, "produto": p, "preco": preco,
                                        "promocao": np.where(promo == 1, "sim", "não"), "unidades_vendidas": proc}))
        return pd.concat(linhas, ignore_index=True)
    if nome == "clientes":
        n = 1800
        meses = rs.randint(1, 72, n)
        plano = rs.choice(["Básico", "Standard", "Premium"], n, p=[0.5, 0.35, 0.15])
        gasto = (np.select([plano == "Básico", plano == "Standard"], [8000, 15000], 28000) * rs.uniform(0.7, 1.3, n)).round(0)
        susp = rs.poisson(1.4, n)
        atrasos = rs.poisson(0.6, n)
        app = rs.choice(["sim", "não"], n, p=[0.6, 0.4])
        z = -1.6 + 0.45 * susp + 0.5 * atrasos - 0.035 * meses + 0.7 * (app == "não") + 0.5 * (plano == "Básico")
        saiu = np.where(rs.rand(n) < 1 / (1 + np.exp(-z)), "sim", "não")
        return pd.DataFrame({"id_cliente": [f"C{i:05d}" for i in range(1, n + 1)], "antiguidade_meses": meses, "plano": plano,
                             "gasto_mensal_kz": gasto, "chamadas_suporte": susp, "atrasos_pagamento": atrasos,
                             "usa_app": app, "saiu": saiu})
    if nome == "transaccoes":
        n = 3000
        valor = np.exp(rs.normal(8.6, 0.8, n)).round(0)
        hora = np.clip(rs.normal(14, 4, n), 0, 23).round(0)
        dist = np.abs(rs.normal(6, 5, n)).round(1)
        tipo = rs.choice(["Supermercado", "Restaurante", "Combustível", "Loja online", "Farmácia"], n, p=[0.35, 0.2, 0.2, 0.15, 0.1])
        cidade = rs.choice(["Luanda", "Benguela", "Huambo", "Lobito"], n, p=[0.6, 0.15, 0.15, 0.1])
        fraude = np.zeros(n, dtype=int)
        idx = rs.choice(n, 60, replace=False)
        valor[idx] = valor[idx] * rs.uniform(8, 25, len(idx))
        hora[idx[:35]] = rs.choice([2, 3, 4], 35)
        dist[idx] = dist[idx] + rs.uniform(150, 600, len(idx))
        fraude[idx] = 1
        return pd.DataFrame({"id_transaccao": [f"T{i:06d}" for i in range(1, n + 1)], "valor_kz": valor, "hora": hora,
                             "tipo_comerciante": tipo, "cidade": cidade, "distancia_casa_km": dist,
                             "fraude": np.where(fraude == 1, "sim", "não")})
    raise ValueError("Exemplo desconhecido.")


# ══════════════════════════════════════════════════════
# RELATÓRIO EXECUTIVO EM PDF
# ══════════════════════════════════════════════════════
def _l(texto) -> str:
    """Texto seguro para as fontes base do PDF (Latin-1): troca o que não existe por equivalentes."""
    t = str(texto).replace("«", '"').replace("»", '"').replace("→", "->").replace("…", "...")
    t = t.replace("–", "-").replace("—", "-").replace("≥", ">=").replace("≤", "<=").replace("\u00a0", " ")
    return t.encode("latin-1", "replace").decode("latin-1")


def relatorio_pdf(empresa: str, analise: dict, autor: str = "") -> bytes:
    """Relatório executivo de uma análise (uma ou duas páginas). Usa o idioma activo da sessão."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    res = analise.get("resumo") or {}
    tipo = res.get("tipo", "")
    est = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=est["Title"], fontSize=20, textColor=colors.HexColor("#0B3D91"), spaceAfter=4, alignment=0)
    h2 = ParagraphStyle("h2", parent=est["Heading2"], fontSize=13, textColor=colors.HexColor("#0B3D91"), spaceBefore=12, spaceAfter=4)
    corpo = ParagraphStyle("c", parent=est["BodyText"], fontSize=10, leading=14)
    peq = ParagraphStyle("p", parent=corpo, fontSize=8.5, textColor=colors.HexColor("#555555"))
    cel = ParagraphStyle("cel", parent=corpo, fontSize=8.5, leading=11)
    cel_cab = ParagraphStyle("celc", parent=cel, fontName="Helvetica-Bold")

    def P(t, estilo=corpo):
        t = escape(_l(t))
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        return Paragraph(t, estilo)

    def tabela(linhas, larguras=None):
        dados = [[Paragraph(escape(_l(c)), cel_cab if i == 0 else cel) for c in l] for i, l in enumerate(linhas)]
        t = Table(dados, colWidths=larguras, hAlign="LEFT")
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF8"))]))
        return t

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=_l(analise.get("titulo", "Relatorio")))
    f = [P("DataForge Business", ParagraphStyle("b", parent=peq, fontSize=9, textColor=colors.HexColor("#0969DA"))),
         P(analise.get("titulo") or tr("Relatório"), h1)]
    data = str(analise.get("created_at", ""))[:10] or datetime.date.today().isoformat()
    f.append(P(T("{} · {} · gerado em {}", empresa, T("autor: {}", autor or analise.get("criado_por", "")), data), peq))
    f.append(P(T("Dados: {} · Problema: {}", analise.get("dataset_nome", ""),
                 tr(PROBLEMAS.get(analise.get("problema"), {}).get("titulo", ""))), peq))

    f.append(P(tr("Resumo para a gestão"), h2))
    if tipo in TIPOS_SUPERVISIONADOS:
        rot, _ = qualidade(res)
        f.append(P(manchete(res)))
        f.append(P(T("Qualidade geral do modelo: **{}**. Treinado com {} linhas e verificado em {} linhas que nunca viu (algoritmo escolhido: {}).",
                     tr(rot), res["n_treino"], res["n_teste"], tr(res["melhor"]))))
        m = res["metricas"]
        if tipo == "regressao":
            linhas = [[tr("Medida"), tr("Valor"), tr("O que significa")],
                      ["R2", f"{m['r2']:.2f}", tr("Quanto da variação o modelo explica (1 = perfeito).")],
                      [tr("Erro médio"), _num(m["mae"]), tr("Quanto a previsão falha, em média, na unidade do valor.")],
                      [tr("Erro sem modelo"), _num(m["mae_base"]), tr("Erro de um palpite pela média: o modelo deve ser bem melhor.")]]
        else:
            linhas = [[tr("Medida"), tr("Valor"), tr("O que significa")],
                      [tr("Taxa de acerto"), f"{m['accuracy']:.0%}", tr("Percentagem de casos em que o modelo acerta.")],
                      [tr("Sem modelo"), f"{m['accuracy_base']:.0%}", tr("Acerto de quem respondesse sempre o mais comum.")]]
            if "recall" in m:
                linhas += [[tr("Recall"), f"{m['recall']:.0%}", tr("Dos casos reais, quantos o modelo apanha.")],
                           [tr("Precisão"), f"{m['precisao']:.0%}", tr("Quando o modelo avisa, quantas vezes acerta.")]]
        f.append(Spacer(1, 6))
        f.append(tabela(linhas, [3.6 * cm, 2.4 * cm, 11 * cm]))
        if res.get("importancias"):
            f.append(P(tr("O que mais pesa na previsão"), h2))
            f.append(tabela([[tr("Coluna"), tr("Peso")]] + [[c, f"{v:.0%}"] for c, v in res["importancias"][:8]], [8 * cm, 3 * cm]))
    elif tipo == "anomalia":
        f.append(P(T("Foram analisadas **{}** operações e **{}** ({}) saem do normal e merecem ser investigadas.",
                     res["n_linhas"], res["n_suspeitos"], f"{res['n_suspeitos'] / max(res['n_linhas'], 1):.1%}")))
        linhas = [[tr("Linha"), tr("Pontuação"), tr("Porquê")]]
        for t in res["top"][:15]:
            linhas.append([str(t["linha"]), f"{t['pontuacao']:.2f}", "; ".join(texto_motivo(m) for m in t["motivos"]) or "-"])
        f.append(tabela(linhas, [1.8 * cm, 2.2 * cm, 13 * cm]))
    elif tipo == "segmentacao":
        f.append(P(T("Os clientes foram agrupados em **{}** segmentos.", res["k"])))
        linhas = [[tr("Segmento"), tr("Clientes"), tr("%")]]
        for p in res["perfis"]:
            linhas.append([nome_segmento(p), str(p["tamanho"]), f"{p['percentagem']:.0%}"])
        f.append(tabela(linhas, [11 * cm, 3 * cm, 3 * cm]))

    texto_ia = (res.get("texto_ia") or {}).get("texto")
    if texto_ia:
        f.append(P(tr("Comentário (escrito por IA)"), h2))
        for par in str(texto_ia).split("\n"):
            if par.strip():
                f.append(P(par.strip()))
        f.append(P(tr("Texto gerado por IA a partir dos números deste relatório. Confirma antes de decidir."), peq))

    avisos = res.get("avisos") or []
    if avisos:
        f.append(P(tr("Pontos de atenção"), h2))
        for a in avisos:
            f.append(P("- " + texto_aviso(a)))
    prox = PROBLEMAS.get(analise.get("problema"), {}).get("seguinte", [])
    if prox:
        f.append(P(tr("Próximos passos recomendados"), h2))
        for p in prox:
            f.append(P("- " + tr(p)))
    f.append(Spacer(1, 10))
    f.append(P(tr("Nota: um modelo aprende com o passado e pode errar. Usa-o para apoiar decisões, não para as substituir."), peq))
    doc.build(f)
    return buf.getvalue()
