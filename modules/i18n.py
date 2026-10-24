"""DataForge EDU - Idiomas (Português / English).

O código-fonte continua em português. Quando o idioma é inglês, tudo o que vai
ser mostrado (texto, rótulos, opções, tabelas, gráficos) passa por `tr()`, que
procura a frase em `modules/locales/en.json`. Frases sem tradução ficam como estão.

Uso:
    from modules.i18n import tr
    tr("Texto em português")        # devolve inglês se o idioma for 'en'
"""
import functools
import json
import os
import re

import streamlit as st

from modules.i18n_seg import split_pieces, split_core

IDIOMAS = {"pt": "Português", "en": "English"}
_AQUI = os.path.dirname(os.path.abspath(__file__))
_FICHEIRO_EN = os.path.join(_AQUI, "locales", "en.json")

_dados = None   # {"exact": {...}, "frases": {primeira_palavra: [(chave, en, desvio)]}}


# ══════════════════════════════════════════════════════
# IDIOMA ACTUAL
# ══════════════════════════════════════════════════════
def idioma_actual() -> str:
    try:
        l = st.session_state.get("lang", "pt")
    except Exception:
        return "pt"
    return l if l in IDIOMAS else "pt"


def definir_idioma_sessao(lang: str):
    lang = lang if lang in IDIOMAS else "pt"
    st.session_state["lang"] = lang
    st.session_state["_idioma_radio"] = IDIOMAS[lang]
    try:
        st.query_params["lang"] = lang
    except Exception:
        pass


def iniciar_idioma():
    """Chamar no início de cada execução: lê ?lang= na primeira visita."""
    if "lang" not in st.session_state:
        q = None
        try:
            q = st.query_params.get("lang")
        except Exception:
            pass
        st.session_state["lang"] = q if q in IDIOMAS else "pt"
    st.session_state.setdefault("_idioma_radio", IDIOMAS[idioma_actual()])


# ══════════════════════════════════════════════════════
# DADOS DE TRADUÇÃO
# ══════════════════════════════════════════════════════
def _carregar():
    global _dados
    if _dados is not None:
        return _dados
    exact, inline = {}, set()
    try:
        with open(_FICHEIRO_EN, encoding="utf8") as f:
            raw = json.load(f)
        exact = raw.get("t", {})
        inline = set(raw.get("inline", []))
    except Exception:
        pass
    frases = {}
    tok = re.compile(r"[\wÀ-ÿ]+")
    for k, v in exact.items():
        if k == v:
            continue
        multi = len(k.split()) >= 2
        if not (multi or k in inline):
            continue
        m = tok.search(k)
        if not m:
            continue
        frases.setdefault(m.group(0), []).append((k, v, m.start()))
    for lst in frases.values():
        lst.sort(key=lambda x: -len(x[0]))
    # Modelos com valores dinâmicos: "Passo {} de {}" -> expressão regular + modelo inglês.
    # Servem para frases montadas com f-strings, em que o texto final nunca é igual ao do ficheiro.
    padroes = []
    for k, v in exact.items():
        if "{}" not in k or k.count("{}") != v.count("{}"):
            continue
        partes = k.split("{}")
        rx = re.compile("^" + "(.+?)".join(re.escape(x) for x in partes) + "$", re.DOTALL)
        padroes.append((partes[0], rx, v.split("{}")))
    padroes.sort(key=lambda x: -sum(len(t) for t in x[2]))
    _dados = {"exact": exact, "frases": frases, "padroes": padroes}
    return _dados


_TOK = re.compile(r"[\wÀ-ÿ]+")


def _frases_dentro(core: str, frases: dict) -> str:
    out, i, n = [], 0, len(core)
    for m in _TOK.finditer(core):
        if m.start() < i:
            continue
        for chave, en, off in frases.get(m.group(0), ()):
            ini = m.start() - off
            if ini < i:
                continue
            fim = ini + len(chave)
            if core.startswith(chave, ini) \
                    and (fim >= n or not chave[-1].isalnum() or not core[fim].isalnum()) \
                    and (ini == 0 or not chave[0].isalnum() or not core[ini - 1].isalnum()):
                out.append(core[i:ini])
                out.append(en)
                i = fim
                break
    if not out:
        return core
    out.append(core[i:])
    return "".join(out)


def _letras_fixas(partes_en: list) -> int:
    return sum(len(re.sub(r"[^A-Za-z]", "", t)) for t in partes_en)


def _por_modelo(core: str, padroes: list, fortes: bool = True):
    """Traduz uma frase montada com valores (ex.: «Passo 2 de 5») usando os modelos com {} do ficheiro.

    `fortes=True` só usa modelos com texto fixo a sério; `fortes=False` só os fracos («{} de {}», «{} - {}»...).
    Os fracos apanham quase qualquer frase com «de», por isso só entram quando mais nada traduziu."""
    for prefixo, rx, en_partes in padroes:
        forte = len(re.sub(r"[^A-Za-zÀ-ÿ]", "", prefixo)) >= 3 or _letras_fixas(en_partes) >= 8
        if forte != fortes:
            continue
        if prefixo and not core.startswith(prefixo):
            continue
        m = rx.match(core)
        if not m:
            continue
        valores = [_traduzir_en(g) for g in m.groups()]       # os valores também podem ter texto a traduzir
        out = [en_partes[0]]
        for val, resto in zip(valores, en_partes[1:]):
            out.append(val)
            out.append(resto)
        return "".join(out)
    return None


@functools.lru_cache(maxsize=8192)
def _traduzir_en(texto: str) -> str:
    d = _carregar()
    exact, frases = d["exact"], d["frases"]
    res = []
    for e_texto, p in split_pieces(texto):
        if not e_texto:
            res.append(p)
            continue
        pre, core, suf = split_core(p)
        if not core:
            res.append(p)
            continue
        en = exact.get(core)
        if en is None:
            en = _por_modelo(core, d["padroes"], True)
        if en is None:
            f = _frases_dentro(core, frases)
            if f != core:
                en = f
        if en is None:
            en = _por_modelo(core, d["padroes"], False)
        if en is None:
            en = core
        res.append(pre + en + suf)
    return "".join(res)


def tr(texto):
    """Traduz `texto` para o idioma actual. Valores que não são texto passam inalterados."""
    if not isinstance(texto, str) or not texto or idioma_actual() == "pt":
        return texto
    return _traduzir_en(texto)


def tr_en(texto):
    """Como tr(), mas ignora o idioma da sessão (útil para ficheiros gerados)."""
    if not isinstance(texto, str) or not texto:
        return texto
    return _traduzir_en(texto)


# ══════════════════════════════════════════════════════
# GRÁFICOS E TABELAS
# ══════════════════════════════════════════════════════
def traduzir_figura_plotly(fig):
    """Traduz títulos, eixos, legendas e anotações de uma figura Plotly (altera a figura)."""
    if idioma_actual() == "pt" or fig is None:
        return fig
    try:
        lay = fig.layout
        if lay.title is not None and lay.title.text:
            lay.title.text = tr(lay.title.text)
        for nome in list(lay):
            if nome.startswith(("xaxis", "yaxis", "coloraxis", "scene")):
                ax = lay[nome]
                try:
                    if ax.title is not None and ax.title.text:
                        ax.title.text = tr(ax.title.text)
                except Exception:
                    pass
                for sub in ("xaxis", "yaxis", "zaxis"):
                    try:
                        a = ax[sub]
                        if a.title is not None and a.title.text:
                            a.title.text = tr(a.title.text)
                    except Exception:
                        pass
        for an in (lay.annotations or ()):
            if an.text:
                an.text = tr(an.text)
        for tr_ in fig.data:
            if getattr(tr_, "name", None):
                tr_.name = tr(tr_.name)
            if getattr(tr_, "text", None) is not None and isinstance(tr_.text, str):
                tr_.text = tr(tr_.text)
            for atr in ("x", "y"):
                v = getattr(tr_, atr, None)
                if v is not None and getattr(tr_, "type", "") in ("bar", "heatmap", "box", "violin") \
                        and hasattr(v, "__iter__") and not isinstance(v, str):
                    try:
                        lst = list(v)
                        if lst and all(isinstance(x, str) for x in lst[:50]):
                            setattr(tr_, atr, [tr(x) for x in lst])
                    except Exception:
                        pass
    except Exception:
        pass
    return fig


def traduzir_figura_matplotlib(fig):
    if idioma_actual() == "pt" or fig is None:
        return fig
    try:
        for ax in fig.axes:
            ax.set_title(tr(ax.get_title()))
            ax.set_xlabel(tr(ax.get_xlabel()))
            ax.set_ylabel(tr(ax.get_ylabel()))
            leg = ax.get_legend()
            if leg is not None:
                for t in leg.get_texts():
                    t.set_text(tr(t.get_text()))
            for t in ax.texts:
                t.set_text(tr(t.get_text()))
        if fig._suptitle is not None:
            fig._suptitle.set_text(tr(fig._suptitle.get_text()))
    except Exception:
        pass
    return fig


def traduzir_tabela(df):
    """Devolve uma cópia da tabela com cabeçalhos (e índice de texto) traduzidos."""
    if idioma_actual() == "pt":
        return df
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            novo = df.copy()
            novo.columns = [tr(c) if isinstance(c, str) else c for c in novo.columns]
            if novo.index.dtype == object:
                novo.index = [tr(i) if isinstance(i, str) else i for i in novo.index]
            return novo
    except Exception:
        pass
    return df
