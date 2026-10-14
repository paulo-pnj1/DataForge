"""
DataForge EDU - Painel do Professor
Turmas com código de convite, progresso de cada aluno, lições mais difíceis e reposição de palavras-passe.
"""
import datetime
import secrets

import pandas as pd
import streamlit as st

from modules import db
from modules.auth import _hash
from modules.utils import page_header, C_ACCENT, C_SURFACE, C_TEXT


def _dias_desde(iso: str):
    if not iso:
        return None
    try:
        return (datetime.datetime.now() - datetime.datetime.fromisoformat(iso)).days
    except Exception:
        return None


def _codigo_novo() -> str:
    while True:
        c = secrets.token_hex(3).upper()
        if not db.get_turma(c):
            return c


def render_painel_professor(username: str, role: str):
    from modules.escola import LICOES, LICAO_POR_ID
    page_header("Painel do Professor", "Turmas, progresso dos alunos e onde eles têm mais dificuldades")

    turmas = db.list_turmas() if role == "admin" else db.list_turmas(username)

    with st.expander("Criar uma turma nova", expanded=not turmas):
        nome = st.text_input("Nome da turma (ex.: Engenharia Informática 3.º ano)", key="prof_nome_turma")
        if st.button("Criar turma", type="primary", key="prof_criar") and nome.strip():
            codigo = _codigo_novo()
            db.create_turma(codigo, nome.strip(), username)
            st.success(f"Turma criada! Código para os alunos: **{codigo}** (usam-no ao criar conta ou em Perfil).")
            st.rerun()

    if not turmas:
        st.info("Ainda não tens turmas. Cria uma acima e partilha o código com os alunos.")
        return

    opcoes = {f"{t['nome']} ({t['codigo']})": t for t in turmas}
    turma = opcoes[st.selectbox("Turma", list(opcoes.keys()), key="prof_turma")]
    st.caption(f"Código de convite: **{turma['codigo']}** · professor: {turma['professor']}")

    alunos = [u for u in db.list_users(turma=turma["codigo"]) if u.get("role", "aluno") == "aluno"]
    if not alunos:
        st.info("Ainda não há alunos nesta turma. Partilha o código acima.")
        return

    progs = db.list_progress([a["username"] for a in alunos])
    total = len(LICOES)
    linhas = []
    for a in alunos:
        p = progs.get(a["username"], {})
        dias = _dias_desde(p.get("ultima_actividade"))
        estado = "Nunca entrou" if dias is None else ("Parado há %d dias" % dias if dias > 7 else "Activo")
        linhas.append({"Nome": a["name"], "Utilizador": a["username"], "Pontos": p.get("pontos", 0),
                       "Lições": f"{len(p.get('licoes', []))}/{total}", "_licoes": len(p.get("licoes", [])),
                       "Módulos": len(p.get("completados", [])), "Estado": estado})
    df = pd.DataFrame(linhas)

    c1, c2, c3 = st.columns(3)
    c1.metric("Alunos", len(df))
    c2.metric("Média de lições", f"{df['_licoes'].mean():.1f} / {total}")
    c3.metric("A precisar de atenção", int((df["Estado"] != "Activo").sum()))

    st.markdown("### Progresso dos alunos")
    mostrar = df.drop(columns=["_licoes"]).sort_values("Pontos", ascending=False)
    st.dataframe(mostrar, width="stretch", hide_index=True)
    st.download_button("Descarregar (CSV)", mostrar.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"turma_{turma['codigo']}.csv", mime="text/csv", key="prof_csv")

    st.markdown("### Onde os alunos têm mais dificuldade")
    erros = {}
    for p in progs.values():
        for lid, n in p.get("quiz_erros", {}).items():
            erros[lid] = erros.get(lid, 0) + n
    if erros:
        top = sorted(erros.items(), key=lambda x: -x[1])[:6]
        dfe = pd.DataFrame({"Lição": [LICAO_POR_ID[k]["titulo"] if k in LICAO_POR_ID else k for k, _ in top],
                            "Respostas erradas": [v for _, v in top]}).set_index("Lição")
        st.bar_chart(dfe, color=C_ACCENT)
        st.caption("Estas são as lições onde mais alunos erraram nos mini-testes. Vale a pena reforçá-las na aula.")
    else:
        st.info("Ainda não há erros registados nos mini-testes.")

    st.markdown("### Repor palavra-passe de um aluno")
    quem = st.selectbox("Aluno", [f"{a['name']} ({a['username']})" for a in alunos], key="prof_reset_quem")
    if st.button("Gerar palavra-passe temporária", key="prof_reset"):
        uname = quem.rsplit("(", 1)[1].rstrip(")")
        temp = secrets.token_urlsafe(6)
        db.update_user(uname, {"password_hash": _hash(temp)})
        st.success(f"Nova palavra-passe de **{uname}**: `{temp}`. Entrega-a ao aluno; ele pode mudá-la em Perfil.")


def render_admin_utilizadores(username: str):
    page_header("Utilizadores", "Gestão de papéis (só administradores)")
    users = db.list_users()
    if not users:
        st.info("Ainda não há utilizadores na base de dados.")
        return
    df = pd.DataFrame([{"Utilizador": u["username"], "Nome": u["name"], "Papel": u.get("role", "aluno"),
                        "Turma": u.get("turma") or ""} for u in users])
    st.dataframe(df, width="stretch", hide_index=True)
    st.markdown("### Mudar o papel de um utilizador")
    c1, c2 = st.columns(2)
    alvo = c1.selectbox("Utilizador", [u["username"] for u in users], key="adm_user")
    papel = c2.selectbox("Novo papel", ["aluno", "professor", "admin"], key="adm_papel")
    if st.button("Guardar papel", type="primary", key="adm_save"):
        if alvo == username and papel != "admin":
            st.error("Não podes retirar o teu próprio papel de administrador.")
        else:
            db.update_user(alvo, {"role": papel})
            st.success(f"{alvo} agora é {papel}. A pessoa deve sair e voltar a entrar.")
