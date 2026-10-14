"""DataForge EDU - Pesquisa global (lições, dicionário e páginas)."""
import unicodedata

import streamlit as st

from modules.utils import ir_para


def _norm(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def pesquisar(q: str, paginas: list, limite: int = 6) -> list:
    """paginas: lista de (rótulo, chave). Devolve [(tipo, titulo, accao)]."""
    from modules.escola import LICOES
    from modules.ajuda import GLOSSARIO
    q = _norm(q.strip())
    if len(q) < 2:
        return []
    res = []
    for rotulo, chave in paginas:
        if q in _norm(rotulo):
            res.append(("Página", rotulo, ("pagina", chave, None)))
    for l in LICOES:
        if q in _norm(l["titulo"]) or q in _norm(l["ideia"]):
            res.append(("Lição", l["titulo"], ("licao", "escola", l["id"])))
    for termo, cat, definicao, _ in GLOSSARIO:
        if q in _norm(termo) or q in _norm(definicao):
            res.append(("Dicionário", termo, ("termo", "dicionario", termo)))
    return res[:limite]


def render_pesquisa_sidebar(paginas: list):
    q = st.text_input("Pesquisar na plataforma", placeholder="ex.: overfitting, KNN...", key="pesq_q", label_visibility="collapsed")
    if not q:
        return
    res = pesquisar(q, paginas)
    if not res:
        st.caption("Nada encontrado.")
        return
    for i, (tipo, titulo, accao) in enumerate(res):
        st.button(f"{tipo}: {titulo}", key=f"pesq_{i}", width="stretch", on_click=_abrir, args=accao)


def _abrir(kind: str, pagina: str, extra):
    """Callback: pode alterar o estado dos widgets antes da próxima execução."""
    if kind == "licao":
        st.session_state.escola_aberta = extra
    elif kind == "termo":
        st.session_state["dic_busca"] = extra
    st.session_state["pesq_q"] = ""
    st.session_state.pagina = pagina
