"""Propostas de limpeza dos dados: o DataForge sugere, o estudante aceita ou recusa cada uma.

Só se propõem correções que NÃO usam informação de outras linhas para decidir o valor de uma linha (por isso não há
fuga de informação entre treino e teste, ao contrário de, por exemplo, limitar valores extremos com percentis de toda a
tabela):

- `texto_limpo`: tira espaços a mais e junta categorias que só diferem em maiúsculas/minúsculas («Luanda », «luanda»).
- `numeros_em_texto`: converte colunas de números escritos como texto («1 250,50», «12 %», «3 500 Kz»).
- `datas_em_texto`: converte colunas de datas escritas como texto (as datas depois viram ano, mês, dia e dia da semana).
- `duplicados`: remove linhas totalmente repetidas (o modelo veria o mesmo exemplo no treino e no teste).

As colunas vazias, constantes e os identificadores já saem sozinhos das variáveis preditoras (`sugerir_features`),
por isso não aparecem aqui. Nada é aplicado sem o estudante escolher, e cada alteração fica descrita em notas.
"""
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from modules.ml_engine.data_validation import _parece_data

ORDEM = ["texto_limpo", "numeros_em_texto", "datas_em_texto", "duplicados"]
LIMITE_NUMEROS = 0.90                 # fração de valores que têm de se ler como número para propor a conversão
LIMITE_DATAS = 0.95
_MOEDA = re.compile(r"(?i)\s*(us\$|r\$|kz|aoa|usd|eur|\$|€|%)\s*")
_NUMERO = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")
_ZEROS_A_ESQUERDA = re.compile(r"^0\d+$")
_FORMA_DATA = re.compile(r"^\s*\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")
_DIA_MES_ANO = re.compile(r"^\s*(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})\b")


@dataclass
class Proposta:
    id: str
    titulo: str
    explicacao: str
    afeta: int                                    # células (ou linhas, nos duplicados) que a correção muda
    colunas: list = field(default_factory=list)
    exemplos: list = field(default_factory=list)  # [(antes, depois)] para o estudante confirmar com os próprios olhos
    _aplicar: Callable = None

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._aplicar(df)


def _exemplos(antes: pd.Series, depois: pd.Series, n: int = 3) -> list:
    """Até `n` pares (antes, depois) de linhas em que o valor muda."""
    out = []
    for i in antes.index:
        a, b = antes.loc[i], depois.loc[i]
        if pd.isna(a) or (not pd.isna(b) and str(a) == str(b)):
            continue
        out.append((str(a), "(em falta)" if pd.isna(b) else str(b)))
        if len(out) >= n:
            break
    return out


# ── texto ─────────────────────────────────────────────────────────────────────
def _so_texto(s: pd.Series) -> bool:
    """Coluna de objectos em que todos os valores não nulos são texto."""
    if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
        return False
    v = s.dropna()
    return len(v) > 0 and bool(v.map(lambda x: isinstance(x, str)).all())


def _mapa_texto(s: pd.Series) -> dict:
    """{valor original: valor limpo}. Entre grafias que só diferem em maiúsculas, vence a mais frequente."""
    orig = s.dropna().astype(str)
    limpo = orig.str.strip().str.replace(r"\s+", " ", regex=True)
    chave = limpo.str.casefold()
    def _melhor(x):
        vc = x.value_counts()
        return max(vc.index, key=lambda g: (vc[g], not g.islower(), not g.isupper()))   # frequência; depois a grafia «normal»

    canon = limpo.groupby(chave).agg(_melhor)
    final = chave.map(canon)
    return dict(zip(orig, final))


def _propor_texto(df: pd.DataFrame):
    mudancas, exemplos = {}, []
    for c in df.columns:
        s = df[c]
        if not _so_texto(s) or s.nunique(dropna=True) > 1000:
            continue
        mapa = _mapa_texto(s)
        n = int(sum((str(v) != mapa.get(str(v), str(v))) for v in s.dropna()))
        if n:
            mudancas[c] = n
            if len(exemplos) < 3:
                exemplos += [(f"{c}: «{a}»", f"«{b}»") for a, b in
                             _exemplos(s, s.map(lambda v: v if pd.isna(v) else mapa.get(str(v), v)), 3 - len(exemplos))]
    if not mudancas:
        return None

    def aplicar(d):
        d = d.copy()
        for c in mudancas:
            if c in d.columns and _so_texto(d[c]):
                mapa = _mapa_texto(d[c])
                d[c] = d[c].map(lambda v: v if pd.isna(v) else mapa.get(str(v), v))
        return d

    return Proposta("texto_limpo", "Arrumar o texto das categorias",
                    "Tira espaços a mais e junta grafias que só diferem em maiúsculas/minúsculas "
                    "(por exemplo «Luanda », «luanda» e «Luanda» passam a ser a mesma categoria).",
                    sum(mudancas.values()), list(mudancas), exemplos, aplicar)


# ── números escritos como texto ───────────────────────────────────────────────
def _para_numero(valor) -> float:
    """Lê «1 250,50», «12 %», «3.500 Kz» ou «1,234.5». Devolve NaN se não for um número."""
    t = _MOEDA.sub("", str(valor)).replace("\u00a0", "").replace(" ", "").strip()
    if not t:
        return float("nan")
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".") if t.count(",") == 1 else t.replace(",", "")        # uma vírgula = decimal
    elif t.count(".") > 1:
        t = t.replace(".", "")                                                         # 1.234.567 = milhares
    return float(t) if _NUMERO.match(t) else float("nan")


def _converter_numeros(s: pd.Series) -> pd.Series:
    mapa = {v: _para_numero(v) for v in s.dropna().unique()}
    return s.map(lambda v: np.nan if pd.isna(v) else mapa.get(v, np.nan)).astype(float)


def _numerica_em_texto(s: pd.Series) -> float:
    """Fração de valores que se lêem como número (0 se a coluna não for texto ou parecer um código com zeros à esquerda)."""
    if not _so_texto(s):
        return 0.0
    v = s.dropna().astype(str).str.strip()
    if v.map(lambda x: bool(_ZEROS_A_ESQUERDA.match(x))).any():
        return 0.0
    amostra = v.head(2000)
    return float(amostra.map(lambda x: np.isfinite(_para_numero(x))).mean())


def _propor_numeros(df: pd.DataFrame, alvo: str = None):
    cols, perdidos, exemplos = [], 0, []
    for c in df.columns:
        if c == alvo:
            continue
        f = _numerica_em_texto(df[c])
        if f >= LIMITE_NUMEROS:
            cols.append(c)
            conv = _converter_numeros(df[c])
            perdidos += int((conv.isna() & df[c].notna()).sum())
            if len(exemplos) < 3:
                exemplos += [(f"{c}: «{a}»", b) for a, b in _exemplos(df[c], conv, 3 - len(exemplos))]
    if not cols:
        return None
    afeta = int(sum(df[c].notna().sum() for c in cols))
    expl = ("Estas colunas têm números escritos como texto (com espaços, vírgula decimal, %, Kz…). Convertidas para números, "
            "o modelo consegue usá-las.")
    if perdidos:
        expl += f" Atenção: {perdidos} valores que não são números ficam em falta."

    def aplicar(d):
        d = d.copy()
        for c in cols:
            if c in d.columns:
                d[c] = _converter_numeros(d[c])
        return d

    return Proposta("numeros_em_texto", "Converter números escritos como texto", expl, afeta, cols, exemplos, aplicar)


# ── datas escritas como texto ─────────────────────────────────────────────────
def _dia_primeiro(s: pd.Series) -> bool:
    """Em datas como 03/04/2024: dia primeiro (convenção em Portugal e Angola) salvo prova em contrário."""
    for v in s.dropna().astype(str).head(500):
        m = _DIA_MES_ANO.match(v)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > 12:
                return True
            if b > 12:
                return False
    return True


def _converter_datas(s: pd.Series) -> pd.Series:
    amostra = s.dropna().astype(str).head(500)
    dayfirst = bool(amostra.map(lambda x: bool(_DIA_MES_ANO.match(x))).any()) and _dia_primeiro(s)
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def _propor_datas(df: pd.DataFrame, alvo: str = None):
    cols, exemplos = [], []
    for c in df.columns:
        s = df[c]
        if c == alvo or not _so_texto(s) or _numerica_em_texto(s) >= LIMITE_NUMEROS:
            continue
        amostra = s.dropna().astype(str).head(300)
        if amostra.map(lambda x: bool(_FORMA_DATA.match(x))).mean() < 0.9:
            continue
        f = _parece_data(s)
        if f is not None and f >= LIMITE_DATAS:
            cols.append(c)
            if len(exemplos) < 3:
                conv = _converter_datas(s)
                exemplos += [(f"{c}: «{a}»", b[:10]) for a, b in _exemplos(s, conv, 3 - len(exemplos))]
    if not cols:
        return None
    afeta = int(sum(df[c].notna().sum() for c in cols))

    def aplicar(d):
        d = d.copy()
        for c in cols:
            if c in d.columns:
                d[c] = _converter_datas(d[c])
        return d

    return Proposta("datas_em_texto", "Converter datas escritas como texto",
                    "Estas colunas têm datas escritas como texto. Convertidas, o DataForge extrai o ano, o mês, o dia e o dia da semana "
                    "(datas como 03/04/2024 são lidas como dia/mês/ano, a não ser que os dados provem o contrário).",
                    afeta, cols, exemplos, aplicar)


# ── duplicados ────────────────────────────────────────────────────────────────
def _propor_duplicados(df: pd.DataFrame):
    n = int(df.duplicated().sum())
    if not n:
        return None
    return Proposta("duplicados", "Remover linhas repetidas",
                    f"Há {n} linhas totalmente iguais a outras. Se ficarem, o modelo pode ver o mesmo exemplo no treino e no teste "
                    "e a nota do teste fica inflacionada. Fica a primeira de cada grupo.",
                    n, [], [], lambda d: d.drop_duplicates().reset_index(drop=True))


# ── API ───────────────────────────────────────────────────────────────────────
def propor_limpeza(df: pd.DataFrame, alvo: str = None) -> list:
    """Lista de `Proposta` aplicáveis a esta tabela (vazia se os dados já estão arrumados)."""
    propostas = [p for p in (_propor_texto(df), _propor_numeros(df, alvo), _propor_datas(df, alvo), _propor_duplicados(df)) if p]
    return sorted(propostas, key=lambda p: ORDEM.index(p.id))


def aplicar_propostas(df: pd.DataFrame, propostas: list, ids: list) -> tuple:
    """Aplica as propostas escolhidas, pela ordem certa (texto → números → datas → duplicados). Devolve (tabela nova, notas)."""
    escolhidas = [p for p in propostas if p.id in set(ids)]
    out, notas = df.copy(), []
    for p in sorted(escolhidas, key=lambda p: ORDEM.index(p.id)):
        antes = len(out)
        out = p.aplicar(out)
        if p.id == "duplicados":
            notas.append(f"{antes - len(out)} linhas repetidas removidas.")
        elif p.colunas:
            notas.append(f"{p.titulo}: " + ", ".join(map(str, p.colunas)) + ".")
    return out.reset_index(drop=True), notas
