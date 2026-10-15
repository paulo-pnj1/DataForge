"""
DataForge EDU - Painel do Professor
Turmas com código de convite, progresso de cada aluno, lições mais difíceis e reposição de palavras-passe.
"""
import datetime
import secrets

import pandas as pd
import streamlit as st

from modules import acesso, db, email_service
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
    return acesso.codigo_novo()


def render_painel_professor(username: str, role: str):
    from modules.escola import LICOES, LICAO_POR_ID
    if role == "admin":   # o admin do sistema não tem acesso ao painel do professor
        st.error("O administrador do sistema não tem acesso ao painel do professor.")
        return
    page_header("Painel do Professor", "Turmas, progresso dos alunos e onde eles têm mais dificuldades")

    turmas = db.list_turmas(username)

    with st.expander("Criar uma turma nova", expanded=not turmas):
        nome = st.text_input("Nome da turma (ex.: Engenharia Informática 3.º ano)", key="prof_nome_turma")
        if st.button("Criar turma", type="primary", key="prof_criar") and nome.strip():
            codigo = _codigo_novo()
            escolas = [m for m in db.membros_do_utilizador(username) if m["tipo"] == "escola"]
            db.create_turma(codigo, nome.strip(), username, escolas[0]["org_id"] if escolas else None)
            st.success(f"Turma criada! Código para os alunos: **{codigo}** (usam-no ao criar conta ou em Perfil).")
            st.rerun()

    if not turmas:
        st.info("Ainda não tens turmas. Cria uma acima e partilha o código com os alunos.")
        return

    opcoes = {f"{t['nome']} ({t['codigo']})": t for t in turmas}
    turma = opcoes[st.selectbox("Turma", list(opcoes.keys()), key="prof_turma")]
    st.caption(f"Código de convite: **{turma['codigo']}** · professor: {turma['professor']}")

    _render_gestao_turma(username, turma)

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
        db.update_user(uname, {"password_hash": _hash(temp), "must_change_pw": True})
        st.success(f"Nova palavra-passe de **{uname}**: `{temp}`. Entrega-a ao aluno; no primeiro acesso ele escolhe uma nova.")


def _render_gestao_turma(username: str, turma: dict):
    """Pedidos de entrada, aprovação ligada/desligada e criação de alunos em lote."""
    cod = turma["codigo"]

    pedidos = db.list_pedidos_turma(codigo=cod)
    if pedidos:
        st.warning(f"{len(pedidos)} aluno(s) a pedir para entrar nesta turma.")
        nomes = {u["username"]: u["name"] for u in db.list_users()}
        for p in pedidos:
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(f"**{nomes.get(p['username'], p['username'])}** ({p['username']})")
            if c2.button("Aceitar", key=f"ped_ok_{cod}_{p['username']}"):
                acesso.aceitar_pedido(cod, p["username"])
                st.rerun()
            if c3.button("Recusar", key=f"ped_no_{cod}_{p['username']}"):
                acesso.recusar_pedido(cod, p["username"])
                st.rerun()

    aprov = st.toggle("Quem usa o código só entra depois de eu aceitar", value=bool(turma.get("aprovacao", True)),
                      key=f"prof_aprov_{cod}")
    if aprov != bool(turma.get("aprovacao", True)):
        db.update_turma(cod, {"aprovacao": aprov})
        st.rerun()

    with st.expander("Criar contas de alunos"):
        st.caption("Um aluno por linha: `Nome completo` ou `Nome completo, email`. O sistema gera o utilizador e uma "
                   f"palavra-passe temporária (o aluno escolhe a sua no primeiro acesso). Máximo {acesso.MAX_ALUNOS_POR_LOTE} por vez.")
        texto = st.text_area("Lista de alunos", key=f"prof_lote_{cod}", height=140,
                             placeholder="Ana Silva\nJoão Pedro Costa, joao@exemplo.com")
        por_email = st.checkbox("Enviar os dados por email aos alunos que tenham email", value=False,
                                key=f"prof_lote_mail_{cod}", disabled=not email_service.configurado())
        if st.button("Criar contas", type="primary", key=f"prof_lote_btn_{cod}") and texto.strip():
            try:
                res = acesso.criar_alunos(username, cod, texto.splitlines(), por_email)
            except db.DBError as e:
                st.error(str(e))
            else:
                st.session_state[f"_lote_{cod}"] = res
        res = st.session_state.get(f"_lote_{cod}")
        if res:
            criados = [r for r in res if not r["erro"]]
            st.success(f"{len(criados)} conta(s) criada(s). Guarda esta lista agora: as palavras-passe não voltam a aparecer.")
            df = pd.DataFrame([{"Nome": r["nome"], "Utilizador": r["utilizador"], "Palavra-passe temporária": r["palavra_passe"],
                                "Erro": r["erro"]} for r in res])
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button("Descarregar lista (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"alunos_{cod}.csv", mime="text/csv", key=f"prof_lote_csv_{cod}")


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
