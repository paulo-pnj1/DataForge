"""Preparação dos dados: divisão treino/teste e pipelines de pré-processamento.

Regras que este módulo garante:
- Tudo o que "aprende" com os dados (medianas, médias, categorias, escalas) está dentro de um Pipeline do scikit-learn
  e é ajustado SÓ com o treino (e, na validação cruzada, só com o treino de cada bloco). O teste nunca o influencia.
- As únicas transformações feitas antes da divisão são linha a linha (expandir uma data em ano/mês/dia, trocar infinitos
  por "em falta", converter tipos): não usam informação de outras linhas, por isso não há fuga de informação.
- Nenhuma linha é apagada sem ficar registada em `notas` (e mostrada ao utilizador).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

from modules.ml_engine.data_loader import ErroDados

MAX_CATEGORIAS = 50
MIN_LINHAS = 10


@dataclass
class DadosPreparados:
    tipo: str                         # 'regressao' | 'classificacao'
    alvo: str
    features: list                    # colunas escolhidas (nomes originais, por ordem)
    colunas_data: list                # colunas de data que foram expandidas
    colunas_modelo: list              # colunas que entram no pipeline (depois de expandir datas)
    num_cols: list
    cat_cols: list
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    classes: list = None              # rótulos originais (classificação), pela ordem dos códigos 0..k-1
    codificador: object = None
    categorias_conhecidas: dict = field(default_factory=dict)
    intervalos_numericos: dict = field(default_factory=dict)
    avisos: list = field(default_factory=list)
    notas: list = field(default_factory=list)
    n_original: int = 0
    n_usadas: int = 0
    estratificado: bool = False
    random_state: int = 42
    test_size: float = 0.2

    @property
    def n_treino(self) -> int:
        return len(self.X_train)

    @property
    def n_teste(self) -> int:
        return len(self.X_test)


# ── Transformações linha a linha (sem fuga de informação) ──────────────────────
def expandir_datas(X: pd.DataFrame, colunas_data: list) -> pd.DataFrame:
    """Troca cada coluna de data por ano, mês, dia e dia da semana."""
    X = X.copy()
    for c in colunas_data:
        if c not in X.columns:
            continue
        d = pd.to_datetime(X[c], errors="coerce")
        X[f"{c}__ano"] = d.dt.year.astype(float)
        X[f"{c}__mes"] = d.dt.month.astype(float)
        X[f"{c}__dia"] = d.dt.day.astype(float)
        X[f"{c}__dia_da_semana"] = d.dt.dayofweek.astype(float)
        X = X.drop(columns=[c])
    return X


def _texto(v):
    return np.nan if pd.isna(v) else str(v).strip()


def limpar_tipos(X: pd.DataFrame, num_cols: list, cat_cols: list) -> pd.DataFrame:
    """Números como float (infinitos viram 'em falta'), categorias como texto (em falta = NaN)."""
    X = X.copy()
    for c in num_cols:
        s = pd.to_numeric(X[c], errors="coerce").astype(float)
        X[c] = s.replace([np.inf, -np.inf], np.nan)
    for c in cat_cols:
        X[c] = X[c].astype(object).map(_texto).astype(object)
    return X[num_cols + cat_cols]


def separar_tipos(X: pd.DataFrame) -> tuple:
    """(colunas numéricas, colunas categóricas). Booleanos contam como categorias."""
    num, cat = [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def sugerir_features(df: pd.DataFrame, alvo: str, excluir=()) -> list:
    """Colunas preditoras por omissão: sem o alvo, sem colunas vazias, constantes ou identificadores."""
    try:
        from modules.business_core import colunas_id
        ids = set(colunas_id(df))
    except Exception:
        ids = set()
    out = []
    for c in df.columns:
        if c == alvo or c in excluir or c in ids:
            continue
        s = df[c]
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            continue
        out.append(c)
    return out


# ── Pipelines ─────────────────────────────────────────────────────────────────
def construir_preprocessador(num_cols: list, cat_cols: list, escalar: bool = True, nao_negativo: bool = False) -> ColumnTransformer:
    """Imputação + (escala) nos números; imputação + one-hot nas categorias. Sem estado até ao `fit`.

    `escalar=False` para árvores e florestas (não precisam). `nao_negativo=True` para o Naive Bayes multinomial
    (números levados a [0, 1], com corte, para nunca darem valores negativos).
    """
    partes = []
    if num_cols:
        passos = [("imputar", SimpleImputer(strategy="median"))]
        if nao_negativo:
            passos.append(("escala", MinMaxScaler(clip=True)))
        elif escalar:
            passos.append(("escala", StandardScaler()))
        partes.append(("num", Pipeline(passos), list(num_cols)))
    if cat_cols:
        partes.append(("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(cat_cols)))
    if not partes:
        raise ErroDados("Não há colunas preditoras utilizáveis.")
    return ColumnTransformer(partes, remainder="drop", verbose_feature_names_out=False)


def estimar_n_colunas(dados: DadosPreparados) -> int:
    """Quantas colunas o modelo vê depois do one-hot (aproximação pelo treino)."""
    n = len(dados.num_cols)
    for c in dados.cat_cols:
        n += int(dados.X_train[c].nunique(dropna=True))
    return n


def duplicados_entre_partes(X_train, y_train, X_test, y_test) -> int:
    """Quantas linhas do teste são cópias exactas de linhas do treino (mesmas variáveis e mesmo alvo)."""
    try:
        a = X_train.copy()
        a["__y"] = np.asarray(y_train)
        b = X_test.copy()
        b["__y"] = np.asarray(y_test)
        ha = set(pd.util.hash_pandas_object(a, index=False).tolist())
        hb = pd.util.hash_pandas_object(b, index=False).tolist()
        return sum(1 for h in hb if h in ha)
    except Exception:
        return 0


def _possivel_fuga(X: pd.DataFrame, y: np.ndarray, num_cols: list) -> list:
    """Colunas numéricas quase idênticas ao alvo (|correlação| > 0,98): suspeitas de fuga de informação."""
    suspeitas = []
    yy = pd.Series(np.asarray(y, dtype=float))
    if yy.nunique() < 2:
        return suspeitas
    for c in num_cols:
        x = pd.to_numeric(X[c], errors="coerce").reset_index(drop=True)
        ok = x.notna() & yy.notna()
        if ok.sum() < 10 or x[ok].nunique() < 2:
            continue
        r = np.corrcoef(x[ok].astype(float), yy[ok])[0, 1]
        if np.isfinite(r) and abs(r) > 0.98:
            suspeitas.append((c, float(r)))
    return suspeitas


def _possivel_fuga_cat(X: pd.DataFrame, y: np.ndarray, cat_cols: list, n_classes: int) -> list:
    """Colunas categóricas que quase determinam a classe (pureza > 99%, com poucas categorias): suspeitas de fuga."""
    suspeitas = []
    yy = pd.Series(np.asarray(y)).reset_index(drop=True)
    for c in cat_cols:
        x = X[c].reset_index(drop=True)
        k = x.nunique(dropna=True)
        if k < max(2, n_classes) or k > 20 or x.notna().sum() < 20:
            continue
        ct = pd.crosstab(x, yy)
        pureza = float(ct.max(axis=1).sum() / ct.values.sum())
        if pureza > 0.99:
            suspeitas.append((c, pureza))
    return suspeitas


# ── Preparação principal ──────────────────────────────────────────────────────
def preparar_dados(df: pd.DataFrame, alvo: str, tipo: str, features: list = None, test_size: float = 0.2,
                   random_state: int = 42, max_categorias: int = MAX_CATEGORIAS) -> DadosPreparados:
    """Separa preditoras e alvo, trata os tipos e divide em treino e teste (estratificado em classificação)."""
    if tipo not in ("regressao", "classificacao"):
        raise ErroDados("O tipo de problema tem de ser 'regressao' ou 'classificacao'.")
    if alvo not in df.columns:
        raise ErroDados(f"A variável-alvo «{alvo}» não existe na tabela.")
    if not 0.05 <= test_size <= 0.5:
        raise ErroDados("A percentagem de teste tem de estar entre 5% e 50%.")
    notas, avisos = [], []
    feats = list(features) if features else sugerir_features(df, alvo)
    feats = [f for i, f in enumerate(feats) if f != alvo and f not in feats[:i]]
    em_falta = [f for f in feats if f not in df.columns]
    if em_falta:
        raise ErroDados("Colunas que não existem na tabela: " + ", ".join(map(str, em_falta)))
    if not feats:
        raise ErroDados("Escolhe pelo menos uma coluna preditora.")

    d = df[feats + [alvo]].copy()
    n_original = len(d)

    # colunas sem nenhum valor: não há nada a aprender
    vazias = [c for c in feats if d[c].notna().sum() == 0]
    if vazias:
        feats = [c for c in feats if c not in vazias]
        notas.append("Colunas sem nenhum valor foram ignoradas: " + ", ".join(map(str, vazias)) + ".")
    # categorias a mais (provável identificador)
    muitas = [c for c in feats if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_datetime64_any_dtype(d[c])
              and d[c].nunique(dropna=True) > max_categorias]
    if muitas:
        feats = [c for c in feats if c not in muitas]
        notas.append(f"Colunas com mais de {max_categorias} categorias foram ignoradas (parecem identificadores ou texto livre): "
                     + ", ".join(map(str, muitas)) + ".")
    if not feats:
        raise ErroDados("Depois de retirar colunas inutilizáveis não sobrou nenhuma preditora.")

    colunas_data = [c for c in feats if pd.api.types.is_datetime64_any_dtype(d[c])]
    X = expandir_datas(d[feats], colunas_data)
    num_cols, cat_cols = separar_tipos(X)

    # alvo
    y = d[alvo]
    if tipo == "regressao":
        y = pd.to_numeric(y, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        if y.notna().sum() == 0:
            raise ErroDados(f"Para regressão o alvo «{alvo}» tem de ser numérico.")
    ok = y.notna()
    if (~ok).sum():
        notas.append(f"{int((~ok).sum())} linhas sem valor em «{alvo}» foram retiradas (não se pode aprender sem a resposta).")
    X, y = X.loc[ok], y.loc[ok]
    n = len(y)
    if n < MIN_LINHAS:
        raise ErroDados(f"Só há {n} linhas utilizáveis; são precisas pelo menos {MIN_LINHAS}.")

    X = limpar_tipos(X.reset_index(drop=True), num_cols, cat_cols)
    y = y.reset_index(drop=True)

    classes, codificador, estratificar = None, None, False
    if tipo == "classificacao":
        rotulos = y.astype(str) if y.dtype == object or pd.api.types.is_bool_dtype(y) else y
        codificador = LabelEncoder().fit(rotulos)
        classes = codificador.classes_.tolist()
        if len(classes) < 2:
            raise ErroDados(f"O alvo «{alvo}» tem uma só classe.")
        if len(classes) > MAX_CATEGORIAS:
            raise ErroDados(f"O alvo «{alvo}» tem {len(classes)} classes; demasiadas para classificar.")
        y_enc = codificador.transform(rotulos)
        contagem = np.bincount(y_enc)
        n_teste = int(round(n * test_size))
        estratificar = bool(contagem.min() >= 2 and n_teste >= len(classes) and (n - n_teste) >= len(classes))
        if not estratificar:
            avisos.append("Há classes com 1 só exemplo (ou poucos dados): a divisão treino/teste não mantém as proporções das classes.")
        topo = contagem.max() / n
        if topo > 0.8:
            avisos.append(f"Classes desequilibradas: a maior tem {topo:.0%} dos exemplos. A accuracy engana; usa F1, recall e a matriz de confusão.")
        y_final = y_enc
    else:
        y_final = y.to_numpy(dtype=float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_final, test_size=test_size, random_state=random_state, stratify=y_final if estratificar else None)
    X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)

    if len(X_tr) < 20:
        avisos.append(f"Só há {len(X_tr)} exemplos de treino e {len(X_te)} de teste: os resultados vão variar muito.")
    elif len(X_te) < 30:
        avisos.append(f"O teste tem só {len(X_te)} exemplos: as métricas são pouco fiáveis.")
    dup = duplicados_entre_partes(X_tr, y_tr, X_te, y_te)
    if dup and dup / len(X_te) > 0.05:
        avisos.append(f"{dup} linhas do teste ({dup / len(X_te):.0%}) são cópias exactas de linhas do treino: a nota do teste pode estar inflacionada.")
    if tipo == "classificacao":
        for c, p in _possivel_fuga_cat(X_tr, y_tr, cat_cols, len(classes)):
            avisos.append(f"Atenção: as categorias de «{c}» quase determinam a classe ({p:.0%} de pureza). Pode ser fuga de informação.")
    for c, r in _possivel_fuga(X_tr, y_tr, num_cols):
        avisos.append(f"Atenção: «{c}» está quase igual ao alvo (correlação {r:.2f}). Pode ser fuga de informação (a coluna só existe depois de saber a resposta).")

    return DadosPreparados(
        tipo=tipo, alvo=alvo, features=feats, colunas_data=colunas_data, colunas_modelo=num_cols + cat_cols,
        num_cols=num_cols, cat_cols=cat_cols, X_train=X_tr[num_cols + cat_cols], X_test=X_te[num_cols + cat_cols],
        y_train=y_tr, y_test=y_te, classes=classes, codificador=codificador,
        categorias_conhecidas={c: sorted(X_tr[c].dropna().unique().tolist())[:max_categorias] for c in cat_cols},
        intervalos_numericos={c: (float(X_tr[c].min()), float(X_tr[c].max()), float(X_tr[c].median()))
                              for c in num_cols if X_tr[c].notna().any()},
        avisos=avisos, notas=notas, n_original=n_original, n_usadas=n, estratificado=estratificar,
        random_state=random_state, test_size=test_size)
