"""Leitor de ecrã para o Tutor IA.

O tutor é um modelo de texto: não vê pixels. Este módulo regista, durante cada execução de uma página, o que o aluno
tem à frente (títulos, avisos, controlos e os valores escolhidos) e entrega isso ao tutor como texto. Funciona em
TODAS as páginas sem as alterar: intercepta os componentes do Streamlit (st.selectbox, st.slider, ...).

Privacidade: campos de texto e de ficheiros só dizem se estão preenchidos, nunca o conteúdo.
Nunca parte a página: qualquer falha aqui é ignorada.
"""
import html
import re

import streamlit as st

_CHAVE = "_ecra_linhas"
_LIMITE = 1800          # caracteres enviados ao tutor
_ACTIVO = {"on": False}

# componente -> (tipo mostrado ao tutor, devolve valor?)
_ENTRADAS = {
    "selectbox": "lista", "multiselect": "lista (vários)", "radio": "opções", "select_slider": "deslizador",
    "slider": "deslizador", "number_input": "número", "checkbox": "caixa", "toggle": "interruptor",
    "date_input": "data", "pills": "opções", "segmented_control": "opções",
}
_TEXTOS = {"text_input": "campo de texto", "text_area": "campo de texto", "file_uploader": "carregar ficheiro",
           "chat_input": "campo de texto"}
_BOTOES = {"button": "botão", "download_button": "botão de download", "form_submit_button": "botão"}
_AVISOS = {"info": "Aviso", "warning": "Atenção", "success": "Sucesso", "error": "Erro"}
_TITULOS = {"title": "Título", "header": "Secção", "subheader": "Subsecção"}

_originais = {}


def _curto(x, n=90) -> str:
    s = " ".join(str(x).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _sem_html(t: str) -> str:
    """Tira CSS, scripts e etiquetas HTML e fica só com o texto que o aluno lê."""
    if "<" in t:
        t = re.sub(r"(?is)<(style|script)\b.*?</\1>", " ", t)
        t = re.sub(r"(?s)<[^>]*>", " ", t)
        t = html.unescape(t)
    t = " ".join(t.replace("**", "").replace("`", "").split())
    return "" if t.startswith(("{", "@media", ".")) else t


def _valor(v) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(_curto(i, 40) for i in v) if v else "(nada escolhido)"
    if v is None:
        return "(nada escolhido)"
    return _curto(v, 70)


def _linhas() -> list:
    return st.session_state.setdefault(_CHAVE, [])


def _juntar(linha: str):
    try:
        ls = _linhas()
        linha = linha[:220]
        if linha and (not ls or ls[-1] != linha):
            ls.append(linha)
    except Exception:
        pass


def _embrulhar(nome: str, fn, metodo: bool = False):
    def f(*args, **kw):
        res = fn(*args, **kw)
        a = args[1:] if metodo else args      # nos métodos da classe o 1.º argumento é o próprio objecto
        if not _ACTIVO["on"]:
            return res
        try:
            if nome == "markdown" or nome == "caption":
                # só textos curtos e sem HTML (o CSS/HTML decorativo não interessa ao tutor)
                t = _sem_html(str(a[0]) if a else str(kw.get("body", "")))
                if t and len(t) <= 220:
                    _juntar(f"Texto: {_curto(t, 200)}")
            elif nome in _TITULOS:
                _juntar(f"{_TITULOS[nome]}: {_curto(a[0] if a else kw.get('body', ''), 100)}")
            elif nome in _AVISOS:
                _juntar(f"{_AVISOS[nome]}: {_curto(a[0] if a else kw.get('body', ''), 220)}")
            elif nome in _ENTRADAS:
                rotulo = a[0] if a else kw.get("label", "")
                if rotulo:
                    _juntar(f"{_curto(rotulo, 60)} ({_ENTRADAS[nome]}): {_valor(res)}")
            elif nome in _TEXTOS:
                rotulo = a[0] if a else kw.get("label", "")
                if rotulo and kw.get("type") != "password":
                    estado = "preenchido" if res else "vazio"
                    _juntar(f"{_curto(rotulo, 60)} ({_TEXTOS[nome]}): {estado}")
            elif nome in _BOTOES:
                rotulo = a[0] if a else kw.get("label", "")
                if rotulo:
                    _juntar(f"Botão: {_curto(rotulo, 60)}")
            elif nome == "expander":
                rotulo = a[0] if a else kw.get("label", "")
                if rotulo:
                    _juntar(f"Secção que se abre/fecha: {_curto(rotulo, 70)}")
            elif nome == "metric":
                rotulo = a[0] if a else kw.get("label", "")
                valor = a[1] if len(a) > 1 else kw.get("value", "")
                if rotulo:
                    _juntar(f"Número em destaque: {_curto(rotulo, 50)} = {_curto(valor, 30)}")
            elif nome == "tabs":
                rotulos = a[0] if a else kw.get("tabs", [])
                _juntar("Separadores: " + ", ".join(_curto(r, 30) for r in rotulos))
            elif nome in ("plotly_chart", "pyplot", "line_chart", "bar_chart", "area_chart", "scatter_chart"):
                _juntar("Há um gráfico no ecrã (o tutor não o vê: só sabe o que os números e títulos dizem)")
            elif nome == "dataframe" or nome == "table" or nome == "data_editor":
                _juntar("Há uma tabela de dados no ecrã")
        except Exception:
            pass
        return res
    f.__name__ = getattr(fn, "__name__", nome)
    f.__doc__ = getattr(fn, "__doc__", None)
    f._ecra_ia = True
    return f


_TODOS = (list(_ENTRADAS) + list(_TEXTOS) + list(_BOTOES) + list(_AVISOS) + list(_TITULOS)
          + ["markdown", "caption", "expander", "metric", "tabs", "plotly_chart", "pyplot", "line_chart", "bar_chart",
             "area_chart", "scatter_chart", "dataframe", "table", "data_editor"])


def instalar():
    """Intercepta os componentes (uma única vez por processo). Cobre st.xxx e também coluna.xxx / sidebar.xxx."""
    if _originais:
        return
    from streamlit.delta_generator import DeltaGenerator
    for nome in _TODOS:
        # 1) método da classe (coluna.selectbox, st.sidebar.selectbox, container.selectbox)
        orig_cls = getattr(DeltaGenerator, nome, None)
        if orig_cls is not None and not getattr(orig_cls, "_ecra_ia", False):
            _originais[("cls", nome)] = orig_cls
            setattr(DeltaGenerator, nome, _embrulhar(nome, orig_cls, metodo=True))
        # 2) atalho do módulo (st.selectbox): já está ligado ao objecto principal, por isso embrulha-se à parte
        orig_mod = getattr(st, nome, None)
        if orig_mod is not None and not getattr(orig_mod, "_ecra_ia", False):
            _originais[("mod", nome)] = orig_mod
            try:
                setattr(st, nome, _embrulhar(nome, orig_mod))
            except Exception:
                pass


def iniciar():
    """Chamar no início do desenho de cada página: apaga o ecrã anterior e começa a registar."""
    try:
        instalar()
        st.session_state[_CHAVE] = []
        _ACTIVO["on"] = True
    except Exception:
        _ACTIVO["on"] = False


def parar() -> str:
    """Chamar quando a página acabou de ser desenhada: pára de registar e devolve a descrição em texto."""
    _ACTIVO["on"] = False
    try:
        ls = st.session_state.get(_CHAVE, [])
        out, total = [], 0
        for l in ls:
            if total + len(l) + 1 > _LIMITE:
                out.append("(…o ecrã tem mais elementos)")
                break
            out.append("- " + l)
            total += len(l) + 1
        return "\n".join(out)
    except Exception:
        return ""
