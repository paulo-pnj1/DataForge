"""
DataForge EDU - Conquistas: medalhas, sequência de dias, desafio da semana e ranking da turma.
"""
import datetime

import pandas as pd
import streamlit as st

from modules import db
from modules.utils import (
    page_header, progresso_bar, get_user_progress, get_historico_modelos,
    C_SURFACE, C_BORDER, C_ACCENT, C_GREEN, C_AMBER, C_TEXT, C_TEXT_SEC, C_TEXT_MUTE,
)


def _datas_actividade(progress: dict) -> set:
    return {h["data"][:10] for h in progress.get("historico_pontos", []) if h.get("data")}


def sequencia(progress: dict) -> int:
    """Dias seguidos com actividade (conta se a última foi hoje ou ontem)."""
    datas = _datas_actividade(progress)
    dia = datetime.date.today()
    if dia.isoformat() not in datas:
        dia -= datetime.timedelta(days=1)
    n = 0
    while dia.isoformat() in datas:
        n += 1
        dia -= datetime.timedelta(days=1)
    return n


def _segunda() -> str:
    hoje = datetime.date.today()
    return (hoje - datetime.timedelta(days=hoje.weekday())).isoformat()


def _conta(progress, prefixo: str) -> int:
    ini = _segunda()
    return sum(1 for h in progress.get("historico_pontos", [])
               if h.get("data", "")[:10] >= ini and h.get("motivo", "").startswith(prefixo))


DESAFIOS = [
    ("Treina 3 modelos esta semana", 3, "Modelo"),
    ("Conclui 2 lições da Escola de ML esta semana", 2, "Lição concluída"),
    ("Responde a 5 perguntas da Revisão do dia", 5, "Revisão"),
    ("Faz 2 comparações de modelos", 2, "Comparação"),
]


def desafio_semana(progress: dict, historico: list) -> dict:
    semana = datetime.date.today().isocalendar()[1]
    titulo, meta, prefixo = DESAFIOS[semana % len(DESAFIOS)]
    if prefixo == "Modelo":
        ini = _segunda()
        feito = sum(1 for h in historico if str(h.get("data", ""))[:10] >= ini)
    else:
        feito = _conta(progress, prefixo)
    return {"titulo": titulo, "meta": meta, "feito": min(feito, meta), "ok": feito >= meta}


def medalhas(progress: dict, n_modelos: int) -> list:
    from modules.escola import LICOES
    licoes = progress.get("licoes", [])
    marcos = progress.get("marcos", {})
    seq = sequencia(progress)
    return [
        ("Primeiro modelo", "Treinaste o teu primeiro modelo.", "primeiro_modelo" in progress.get("modulos", {})),
        ("Aluno curioso", "Concluíste 5 lições.", len(licoes) >= 5),
        ("Diplomado da Escola", "Concluíste todas as lições.", len(licoes) >= len(LICOES)),
        ("Cinco modelos", "Treinaste 5 modelos.", n_modelos >= 5),
        ("Comparador", "Comparaste algoritmos.", "comparador" in marcos),
        ("Projecto concluído", "Terminaste o projecto final.", "projeto" in marcos),
        ("Memória de elefante", "Acertaste 10 perguntas de revisão.", progress.get("revisoes_ok", 0) >= 10),
        ("Sequência de 3 dias", "3 dias seguidos a aprender.", seq >= 3),
        ("Sequência de 7 dias", "7 dias seguidos a aprender.", seq >= 7),
    ]


def render_conquistas(username: str):
    page_header("Conquistas e Desafios", "Medalhas, sequência de dias, desafio da semana e ranking da turma")
    progress = get_user_progress(username)
    n_modelos = len(get_historico_modelos(username))
    seq = sequencia(progress)

    c1, c2, c3 = st.columns(3)
    c1.metric("Pontos", progress.get("pontos", 0))
    c2.metric("Sequência", f"{seq} dia(s)", help="Dias seguidos com actividade na plataforma.")
    c3.metric("Modelos treinados", n_modelos)

    d = desafio_semana(progress, get_historico_modelos(username))
    st.markdown("### Desafio da semana")
    progresso_bar(d["feito"] / d["meta"], f"{d['titulo']} ({d['feito']}/{d['meta']})")
    if d["ok"]:
        st.success("Desafio cumprido! Volta na segunda-feira para o próximo.")

    st.markdown("### Medalhas")
    ms = medalhas(progress, n_modelos)
    cols = st.columns(3)
    for i, (nome, desc, ok) in enumerate(ms):
        cor = C_GREEN if ok else C_BORDER
        with cols[i % 3]:
            st.markdown(
                f'<div style="background:{C_SURFACE};border:1px solid {cor};border-radius:12px;padding:.8rem 1rem;margin-bottom:.6rem;opacity:{1 if ok else .55};">'
                f'<div style="font-weight:800;color:{C_TEXT};">{nome}</div>'
                f'<div style="font-size:12.5px;color:{C_TEXT_SEC};">{desc}</div>'
                f'<div style="font-size:11px;font-weight:700;color:{C_GREEN if ok else C_TEXT_MUTE};margin-top:4px;">{"CONQUISTADA" if ok else "POR CONQUISTAR"}</div></div>',
                unsafe_allow_html=True)

    st.markdown("### Ranking da turma")
    user = db.get_user(username)
    if not user or not user.get("turma"):
        st.info("Entra numa turma (em **Perfil**) para veres o ranking. O ranking é opcional e mostra só quem escolheu aparecer.")
        return
    membros = [u for u in db.list_users(turma=user["turma"]) if u.get("ranking")]
    if not user.get("ranking"):
        st.caption("Tu não estás a aparecer no ranking. Podes ligar essa opção em Perfil.")
    if not membros:
        st.info("Ainda ninguém activou o ranking nesta turma.")
        return
    progs = db.list_progress([u["username"] for u in membros])
    linhas = sorted(({"Nome": u["name"], "Pontos": progs.get(u["username"], {}).get("pontos", 0),
                      "Lições": len(progs.get(u["username"], {}).get("licoes", []))} for u in membros),
                    key=lambda r: -r["Pontos"])
    df = pd.DataFrame(linhas[:10])
    df.insert(0, "#", range(1, len(df) + 1))
    st.dataframe(df, width="stretch", hide_index=True)
