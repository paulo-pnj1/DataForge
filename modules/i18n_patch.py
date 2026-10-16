"""DataForge EDU - Liga a tradução ao Streamlit.

`instalar()` envolve os métodos de desenho do Streamlit para que o texto passe por
`tr()` quando o idioma é inglês. O resto do código não precisa de mudar e os valores
devolvidos pelos widgets continuam a ser os originais (as opções são só traduzidas
na apresentação, via format_func).
"""
import functools
import inspect

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from modules.i18n import (idioma_actual, tr, traduzir_figura_plotly, traduzir_figura_matplotlib,
                          traduzir_tabela)

# métodos cujo 1.º argumento (e alguns nomeados) é texto visível
_TEXTO = ["markdown", "write", "caption", "title", "header", "subheader", "text", "success", "info",
          "warning", "error", "toast", "html", "code_caption", "status", "popover", "expander",
          "metric", "progress", "button", "download_button", "link_button", "checkbox", "toggle",
          "radio", "selectbox", "multiselect", "slider", "select_slider", "text_input", "text_area",
          "number_input", "date_input", "time_input", "file_uploader", "color_picker", "camera_input",
          "form_submit_button", "page_link", "segmented_control", "pills", "feedback", "chat_input",
          "exception_note"]
_KW_TEXTO = ("label", "help", "placeholder", "body", "text", "caption", "title", "description",
             "delta", "value_label", "message")
_COM_OPCOES = {"radio", "selectbox", "multiselect", "select_slider", "segmented_control", "pills"}
_TABELAS = ("dataframe", "table")


def _tr_args(args, kw, nome):
    args = list(args)
    if nome == "write":
        args = [tr(a) if isinstance(a, str) else a for a in args]
    elif args and isinstance(args[0], str):
        args[0] = tr(args[0])
    for k in _KW_TEXTO:
        v = kw.get(k)
        if isinstance(v, str) and not (nome == "metric" and k == "value"):
            kw[k] = tr(v)
    if nome == "metric":
        if len(args) > 1 and isinstance(args[1], str):
            args[1] = tr(args[1])
        if isinstance(kw.get("value"), str):
            kw["value"] = tr(kw["value"])
    if nome in _COM_OPCOES:
        ff = kw.get("format_func")
        kw["format_func"] = (lambda x, _f=ff: tr(_f(x) if _f else (x if isinstance(x, str) else str(x))))
    return tuple(args), kw


def _envolver_texto(nome, original):
    @functools.wraps(original)
    def metodo(self, *args, **kw):
        if idioma_actual() != "en":
            return original(self, *args, **kw)
        args, kw = _tr_args(args, dict(kw), nome)
        return original(self, *args, **kw)
    metodo._df_i18n = True
    return metodo


def _envolver_tabela(original):
    @functools.wraps(original)
    def metodo(self, data=None, *args, **kw):
        if idioma_actual() == "en":
            data = traduzir_tabela(data)
        return original(self, data, *args, **kw)
    metodo._df_i18n = True
    return metodo


def _envolver_tabs(original):
    @functools.wraps(original)
    def metodo(self, tabs, *args, **kw):
        if idioma_actual() == "en":
            tabs = [tr(t) if isinstance(t, str) else t for t in tabs]
        return original(self, tabs, *args, **kw)
    metodo._df_i18n = True
    return metodo


def _envolver_plotly(original):
    @functools.wraps(original)
    def f(fig, *args, **kw):
        if idioma_actual() == "en":
            traduzir_figura_plotly(fig)
        return original(fig, *args, **kw)
    f._df_i18n = True
    return f


def _envolver_pyplot(original):
    @functools.wraps(original)
    def f(fig=None, *args, **kw):
        if idioma_actual() == "en":
            if fig is None:
                try:
                    import matplotlib.pyplot as plt
                    fig = plt.gcf()
                except Exception:
                    pass
            traduzir_figura_matplotlib(fig)
        return original(fig, *args, **kw)
    f._df_i18n = True
    return f


def _envolver_funcao_texto(original):
    @functools.wraps(original)
    def f(*args, **kw):
        if idioma_actual() != "en":
            return original(*args, **kw)
        args = list(args)
        if args and isinstance(args[0], str):
            args[0] = tr(args[0])
        for k in ("text", "label", "body", "page_title"):
            if isinstance(kw.get(k), str):
                kw[k] = tr(kw[k])
        return original(*args, **kw)
    f._df_i18n = True
    return f


def instalar():
    """Idempotente: pode ser chamada em todas as execuções."""
    if getattr(DeltaGenerator, "_df_i18n_ok", False):
        return
    for nome in _TEXTO:
        orig = getattr(DeltaGenerator, nome, None)
        if orig is not None and not getattr(orig, "_df_i18n", False):
            setattr(DeltaGenerator, nome, _envolver_texto(nome, orig))
    for nome in _TABELAS:
        orig = getattr(DeltaGenerator, nome, None)
        if orig is not None and not getattr(orig, "_df_i18n", False):
            setattr(DeltaGenerator, nome, _envolver_tabela(orig))
    orig = getattr(DeltaGenerator, "tabs", None)
    if orig is not None and not getattr(orig, "_df_i18n", False):
        DeltaGenerator.tabs = _envolver_tabs(orig)
    for nome in ("spinner", "toast"):
        orig = getattr(st, nome, None)
        if orig is not None and not getattr(orig, "_df_i18n", False):
            setattr(st, nome, _envolver_funcao_texto(orig))
    if not getattr(st.plotly_chart, "_df_i18n", False):
        st.plotly_chart = _envolver_plotly(st.plotly_chart)
    if not getattr(st.pyplot, "_df_i18n", False):
        st.pyplot = _envolver_pyplot(st.pyplot)
    # st.markdown, st.button, ... são métodos "presos" ao DeltaGenerator principal no momento
    # do import: voltar a ligá-los aos métodos já envolvidos.
    for nome in list(_TEXTO) + list(_TABELAS) + ["tabs"]:
        cur = getattr(st, nome, None)
        if inspect.ismethod(cur) and isinstance(cur.__self__, DeltaGenerator):
            setattr(st, nome, getattr(cur.__self__, nome))
    DeltaGenerator._df_i18n_ok = True
