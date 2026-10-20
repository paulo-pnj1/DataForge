"""Validação dos dados e da variável-alvo antes de treinar.

Não altera nada: só diz o que está bem, o que merece atenção (avisos) e o que impede o treino (erros).
"""
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_AMOSTRAS_ERRO = 10
MIN_AMOSTRAS_AVISO = 50
MIN_POR_CLASSE_AVISO = 5
MAX_CLASSES = 50
LIMITE_DESEQUILIBRIO = 0.8
_NOME_DATA = re.compile(r"(data|date|dia|day|mes|month|ano|year|time|periodo|period|timestamp)", re.I)


@dataclass
class RelatorioValidacao:
    erros: list = field(default_factory=list)
    avisos: list = field(default_factory=list)
    infos: list = field(default_factory=list)
    colunas_vazias: list = field(default_factory=list)
    colunas_constantes: list = field(default_factory=list)
    colunas_id: list = field(default_factory=list)
    colunas_data: list = field(default_factory=list)
    colunas_data_suspeitas: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.erros


def _e_numerica(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def sugerir_tipo_problema(df: pd.DataFrame, alvo: str):
    """Sugestão (não decisão): ('classificacao' | 'regressao', motivo). O utilizador confirma."""
    s = df[alvo].dropna()
    n = len(s)
    nun = s.nunique()
    if n == 0:
        return "regressao", "A coluna não tem valores."
    if not _e_numerica(s):
        return "classificacao", f"A coluna tem texto ou categorias ({nun} valores diferentes)."
    inteiros = bool(np.all(np.isclose(s.astype(float) % 1, 0)))
    if nun <= 2:
        return "classificacao", "A coluna tem só 2 valores diferentes (parece sim/não)."
    if inteiros and nun <= 10 and nun <= max(2, 0.05 * n):
        return "classificacao", f"São números inteiros com só {nun} valores diferentes (parecem categorias)."
    return "regressao", f"A coluna é numérica com {nun} valores diferentes (parece uma medida contínua)."


def _tipos_misturados(s: pd.Series) -> tuple:
    """(percentagem de números, percentagem de texto) numa coluna de objectos; (0, 0) se não for mista."""
    nn = s.dropna()
    if len(nn) == 0 or s.dtype != object:
        return 0.0, 0.0
    e_num = nn.map(lambda v: isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool))
    e_txt = nn.map(lambda v: isinstance(v, str))
    pn, pt = float(e_num.mean()), float(e_txt.mean())
    return (pn, pt) if pn > 0 and pt > 0 else (0.0, 0.0)


def _parece_data(s: pd.Series):
    """Fracção de valores que se lêem como data (amostra). None se não se aplica."""
    if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
        return None
    amostra = s.dropna().astype(str).head(300)
    if len(amostra) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            conv = pd.to_datetime(amostra, errors="coerce")
        except Exception:
            return None
    return float(conv.notna().mean())


def validar_dados(df: pd.DataFrame, alvo: str = None, tipo: str = None, min_amostras: int = MIN_AMOSTRAS_ERRO) -> RelatorioValidacao:
    """Valida a tabela e, se `alvo` for dado, a variável-alvo. `tipo`: 'classificacao' | 'regressao' | None."""
    r = RelatorioValidacao()
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        r.erros.append("A tabela está vazia (sem linhas ou sem colunas).")
        return r
    n, c = df.shape
    r.infos.append(f"A tabela tem {n} linhas e {c} colunas.")

    if df.columns.duplicated().any():
        dup = sorted({str(x) for x in df.columns[df.columns.duplicated()]})
        r.erros.append("Há colunas com o mesmo nome: " + ", ".join(dup) + ". Renomeia-as antes de continuar.")
    nomes = [str(x) for x in df.columns]
    mangled = [x for x in nomes if re.fullmatch(r".+[._]\d+", x) and re.sub(r"[._]\d+$", "", x) in nomes]
    if mangled:
        r.avisos.append("Possíveis colunas duplicadas (nomes quase iguais): " + ", ".join(mangled[:5]) + ".")

    dup_linhas = int(df.duplicated().sum())
    if dup_linhas:
        r.avisos.append(f"Há {dup_linhas} linhas repetidas. Se existirem em treino e teste, a nota do teste fica inflacionada.")

    for col in df.columns:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            continue
        nulos = int(s.isna().sum())
        if nulos == n:
            r.colunas_vazias.append(col)
            continue
        if nulos / n > 0.5:
            r.avisos.append(f"A coluna «{col}» tem {nulos / n:.0%} de valores em falta.")
        if _e_numerica(s):
            inf = int(np.isinf(s.astype(float)).sum())
            if inf:
                r.avisos.append(f"A coluna «{col}» tem {inf} valores infinitos; serão tratados como valores em falta.")
        if s.nunique(dropna=True) <= 1:
            r.colunas_constantes.append(col)
        pn, pt = _tipos_misturados(s)
        if pn:
            r.avisos.append(f"A coluna «{col}» mistura números ({pn:.0%}) e texto ({pt:.0%}); será tratada como categoria.")
        if pd.api.types.is_datetime64_any_dtype(s):
            r.colunas_data.append(col)
        elif _NOME_DATA.search(str(col)):
            f = _parece_data(s)
            if f is not None and 0.3 <= f < 0.95:
                r.colunas_data_suspeitas.append(col)
                r.avisos.append(f"A coluna «{col}» parece uma data, mas só {f:.0%} dos valores têm um formato de data válido.")
            elif f is not None and f >= 0.95:
                r.infos.append(f"A coluna «{col}» parece uma data (texto): usa-a como coluna temporal nas séries.")
    if r.colunas_vazias:
        r.avisos.append("Colunas sem nenhum valor útil (vão ser ignoradas): " + ", ".join(map(str, r.colunas_vazias)) + ".")
    if r.colunas_constantes:
        r.avisos.append("Colunas com sempre o mesmo valor (não ajudam o modelo): " + ", ".join(map(str, r.colunas_constantes)) + ".")
    try:
        from modules.business_core import colunas_id
        r.colunas_id = [x for x in colunas_id(df) if x != alvo]
    except Exception:
        r.colunas_id = []
    if r.colunas_id:
        r.avisos.append("Colunas que parecem identificadores (não devem prever nada): " + ", ".join(map(str, r.colunas_id)) + ".")

    if alvo is not None:
        _validar_alvo(df, alvo, tipo, r, min_amostras)
    elif n < MIN_AMOSTRAS_AVISO:
        r.avisos.append(f"Poucas linhas ({n}): os resultados vão ser instáveis.")
    return r


def _validar_alvo(df: pd.DataFrame, alvo: str, tipo, r: RelatorioValidacao, min_amostras: int):
    if alvo not in df.columns:
        r.erros.append(f"A variável-alvo «{alvo}» não existe na tabela.")
        return
    y = df[alvo]
    n_total = len(y)
    validos = y.dropna()
    if _e_numerica(validos):
        validos = validos[np.isfinite(validos.astype(float))]
    n = len(validos)
    if n == 0:
        r.erros.append(f"A variável-alvo «{alvo}» não tem nenhum valor útil.")
        return
    if n < n_total:
        r.avisos.append(f"{n_total - n} linhas não têm valor em «{alvo}» e vão ser retiradas do treino (não se pode aprender sem a resposta).")
    if n < min_amostras:
        r.erros.append(f"Só há {n} linhas com valor em «{alvo}». São precisas pelo menos {min_amostras}.")
    elif n < MIN_AMOSTRAS_AVISO:
        r.avisos.append(f"Só há {n} linhas utilizáveis: os resultados vão ser instáveis.")
    nun = validos.nunique()
    if nun < 2:
        r.erros.append(f"A variável-alvo «{alvo}» tem sempre o mesmo valor: não há nada para prever.")
        return
    if tipo == "regressao":
        if not _e_numerica(validos):
            r.erros.append(f"Para regressão o alvo «{alvo}» tem de ser numérico.")
    elif tipo == "classificacao":
        if nun > MAX_CLASSES:
            r.erros.append(f"O alvo «{alvo}» tem {nun} valores diferentes: demasiadas classes. Se é uma medida contínua, usa regressão.")
            return
        contagem = validos.value_counts()
        pouca = contagem[contagem < MIN_POR_CLASSE_AVISO]
        if len(pouca):
            r.avisos.append("Classes com poucos exemplos (< %d): %s. As métricas dessas classes são pouco fiáveis."
                            % (MIN_POR_CLASSE_AVISO, ", ".join(f"{k} ({v})" for k, v in pouca.head(5).items())))
        if (contagem == 1).any():
            r.avisos.append("Há classes com 1 só exemplo: não é possível manter as proporções na divisão treino/teste.")
        topo = contagem.iloc[0] / n
        if topo > LIMITE_DESEQUILIBRIO:
            r.avisos.append(f"Classes muito desequilibradas: «{contagem.index[0]}» tem {topo:.0%} dos exemplos. "
                            "A accuracy engana (um modelo que responde sempre a mesma classe já acerta isso); olha para F1, recall e a matriz de confusão.")
        if _e_numerica(validos) and nun > 10 and not bool(np.all(np.isclose(validos.astype(float) % 1, 0))):
            r.avisos.append(f"O alvo «{alvo}» é numérico com casas decimais e {nun} valores diferentes: confirma que é mesmo uma categoria.")
