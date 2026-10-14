"""
DataForge EDU - Revisão espaçada
Volta a perguntar, ao fim de 1, 3, 7, 14 e 30 dias, as perguntas do quiz em que o aluno errou.
"""
import datetime
import random
import zlib

import streamlit as st

from modules.settings import PONTOS_REVISAO
from modules.utils import (
    page_header, get_user_progress, add_pontos, _gravar_progress, ir_para,
    C_SURFACE, C_BORDER, C_GREEN, C_TEXT, C_TEXT_SEC,
)

INTERVALOS = [1, 3, 7, 14, 30]


def pendentes(progress: dict) -> list:
    hoje = datetime.date.today().isoformat()
    return sorted(k for k, v in progress.get("revisao", {}).items() if v.get("proxima", hoje) <= hoje)


def _pergunta(chave: str):
    from modules.escola import LICAO_POR_ID
    lid, idx = chave.split(":")
    licao = LICAO_POR_ID.get(lid)
    if not licao or int(idx) >= len(licao["quiz"]):
        return None
    return licao, licao["quiz"][int(idx)]


def render_revisao(username: str):
    page_header("Revisão do dia", "Perguntas em que erraste, a voltar na altura certa para ficarem na memória")
    progress = get_user_progress(username)
    fila = [k for k in pendentes(progress) if _pergunta(k)]

    fb = st.session_state.get("rev_fb")
    if fb:
        (st.success if fb["ok"] else st.error)(fb["msg"])
        if st.button("Próxima pergunta", type="primary", key="rev_next"):
            st.session_state.pop("rev_fb")
            st.rerun()
        return

    total = len(progress.get("revisao", {}))
    if not fila:
        st.success("Não tens revisões para hoje. Boa!")
        st.caption(f"Perguntas em acompanhamento: {total}. Quando erras num quiz da Escola de ML, a pergunta volta aqui no dia seguinte.")
        if st.button("Ir para a Escola de ML", key="rev_escola"):
            ir_para("escola")
        return

    chave = fila[0]
    licao, (pergunta, certa, erradas) = _pergunta(chave)
    st.caption(f"Faltam {len(fila)} pergunta(s) hoje · da lição: {licao['titulo']}")
    opcoes = [certa] + list(erradas)
    random.Random(zlib.crc32(chave.encode())).shuffle(opcoes)
    resp = st.radio(pergunta, opcoes, index=None, key=f"rev_{chave}")
    if st.button("Responder", type="primary", key="rev_ok") and resp is not None:
        item = progress["revisao"][chave]
        if resp == certa:
            i = INTERVALOS.index(item.get("intervalo", 1)) if item.get("intervalo", 1) in INTERVALOS else 0
            if i + 1 >= len(INTERVALOS):
                del progress["revisao"][chave]  # dominada
                msg = "Certo! Esta pergunta está dominada e sai da revisão."
            else:
                novo = INTERVALOS[i + 1]
                item["intervalo"] = novo
                item["proxima"] = (datetime.date.today() + datetime.timedelta(days=novo)).isoformat()
                msg = f"Certo! Volto a perguntar daqui a {novo} dias. +{PONTOS_REVISAO} pontos."
            progress["revisoes_ok"] = progress.get("revisoes_ok", 0) + 1
            _gravar_progress(username, progress)
            add_pontos(username, PONTOS_REVISAO, "Revisão: resposta certa")
            st.session_state["rev_fb"] = {"ok": True, "msg": msg}
        else:
            item["intervalo"] = 1
            item["proxima"] = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
            _gravar_progress(username, progress)
            st.session_state["rev_fb"] = {"ok": False, "msg": f"Não foi desta. A resposta certa era: **{certa}**. Volto a perguntar amanhã."}
        st.rerun()
