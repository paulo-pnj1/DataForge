"""
DataForge - Painel de Administração (só administradores globais).
Visão geral, organizações (escolas e empresas) e membros, conteúdos e registo de erros.
A gestão de papéis dos utilizadores continua em modules.professor.render_admin_utilizadores.
"""
import secrets

import pandas as pd
import streamlit as st

from modules import db
from modules.utils import page_header


def _novo_codigo() -> str:
    while True:
        c = secrets.token_hex(3).upper()
        if not db.get_org_por_codigo(c) and not db.get_turma(c):
            return c


def adicionar_membro_por_codigo(username: str, codigo: str, papel_preferido: str = "aluno"):
    """Junta uma pessoa a uma organização pelo código de convite. Devolve (ok, mensagem)."""
    org = db.get_org_por_codigo(codigo)
    if not org or not org.get("activa", True):
        return False, "Não encontrei nenhuma organização activa com esse código."
    if any(m["username"] == username for m in db.list_membros(org_id=org["id"])):
        return True, f"Já fazes parte de {org['nome']}."   # não altera o papel que já tinha
    papeis = db.PAPEIS_ORG[org["tipo"]]
    papel = papel_preferido if papel_preferido in papeis else papeis[-1]   # escola: aluno; empresa: membro
    db.add_membro(org["id"], username, papel)
    return True, f"Ligado a {org['nome']} como {papel}."


def render_admin_home(username: str):
    page_header("Visão geral", "O estado da plataforma num relance")
    users = db.list_users()
    orgs = db.list_orgs()
    turmas = db.list_turmas()
    erros = db.list_erros(500)

    por_papel = {}
    for u in users:
        por_papel[u.get("role", "aluno")] = por_papel.get(u.get("role", "aluno"), 0) + 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Utilizadores", len(users))
    c2.metric("Organizações", len(orgs))
    c3.metric("Turmas", len(turmas))
    c4.metric("Erros registados", len(erros))

    st.markdown("### Os dois produtos")
    escolas = [o for o in orgs if o["produto"] == "edu"]
    empresas = [o for o in orgs if o["produto"] == "business"]
    st.dataframe(pd.DataFrame([
        {"Produto": "DataForge EDU", "Organizações": len(escolas), "Activas": sum(1 for o in escolas if o.get("activa", True)),
         "Para quem": "Escolas, professores e alunos"},
        {"Produto": "DataForge Business", "Organizações": len(empresas), "Activas": sum(1 for o in empresas if o.get("activa", True)),
         "Para quem": "Empresas (ML nos dados do negócio)"},
    ]), width="stretch", hide_index=True)

    st.markdown("### Utilizadores por papel")
    if por_papel:
        st.bar_chart(pd.DataFrame({"Utilizadores": por_papel}))
    else:
        st.info("Ainda não há utilizadores na base de dados.")

    st.caption("Utilizadores sem organização continuam a funcionar como alunos independentes do EDU.")


def render_admin_orgs(username: str):
    page_header("Organizações", "Escolas e empresas, com os seus membros. Cada uma só vê os seus dados.")

    with st.expander("Criar uma organização", expanded=not db.list_orgs()):
        c1, c2 = st.columns([2, 1])
        nome = c1.text_input("Nome (ex.: Universidade X, Empresa Y)", key="org_nome")
        tipo = c2.selectbox("Tipo", list(db.TIPOS_ORG.keys()), key="org_tipo",
                            format_func=lambda t: {"escola": "Escola (EDU)", "empresa": "Empresa (Business)"}[t])
        if st.button("Criar organização", type="primary", key="org_criar"):
            if len(nome.strip()) < 3:
                st.error("Escreve o nome da organização.")
            else:
                codigo = _novo_codigo()
                db.create_org(secrets.token_hex(4), nome.strip(), tipo, codigo)
                st.success(f"Criada! Código de convite: **{codigo}** (usado ao criar conta ou em Perfil).")
                st.rerun()

    orgs = db.list_orgs()
    if not orgs:
        st.info("Ainda não há organizações. Cria a primeira acima.")
        return

    contagem = {}
    for m in db.list_membros():
        contagem[m["org_id"]] = contagem.get(m["org_id"], 0) + 1
    st.dataframe(pd.DataFrame([{
        "Nome": o["nome"], "Tipo": o["tipo"], "Produto": o["produto"], "Código": o["codigo_convite"],
        "Membros": contagem.get(o["id"], 0), "Activa": "Sim" if o.get("activa", True) else "Não"} for o in orgs]),
        width="stretch", hide_index=True)

    st.markdown("### Gerir uma organização")
    opcoes = {f"{o['nome']} ({o['codigo_convite']})": o for o in orgs}
    org = opcoes[st.selectbox("Organização", list(opcoes.keys()), key="org_sel")]

    activa = org.get("activa", True)
    if st.button("Desactivar organização" if activa else "Reactivar organização", key="org_toggle"):
        db.update_org(org["id"], {"activa": not activa})
        st.rerun()
    if not activa:
        st.warning("Organização desactivada: os membros deixam de ter acesso ao painel desta organização.")

    membros = db.list_membros(org_id=org["id"])
    nomes = {u["username"]: u["name"] for u in db.list_users()}
    st.markdown(f"**Membros ({len(membros)})**")
    if membros:
        st.dataframe(pd.DataFrame([{"Utilizador": m["username"], "Nome": nomes.get(m["username"], "?"), "Papel": m["papel"]}
                                   for m in membros]), width="stretch", hide_index=True)

    papeis = db.PAPEIS_ORG[org["tipo"]]
    c1, c2, c3 = st.columns([2, 1, 1])
    candidato = c1.selectbox("Adicionar utilizador", sorted(nomes.keys()) or [""], key="org_add_user")
    papel = c2.selectbox("Papel", papeis, key="org_add_papel")
    c3.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    if c3.button("Adicionar", key="org_add") and candidato:
        db.add_membro(org["id"], candidato, papel)
        st.session_state.pop("_membros_cache", None)
        st.success(f"{candidato} é agora {papel} em {org['nome']}. Deve sair e voltar a entrar.")
        st.rerun()

    if membros:
        c1, c2 = st.columns([3, 1])
        quem = c1.selectbox("Remover membro", [m["username"] for m in membros], key="org_rm_user")
        c2.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if c2.button("Remover", key="org_rm"):
            db.remove_membro(org["id"], quem)
            st.session_state.pop("_membros_cache", None)
            st.rerun()


def render_admin_conteudos(username: str):
    from modules.escola import LICOES
    page_header("Conteúdos", "Lições da Escola de ML (só leitura nesta fase)")
    st.dataframe(pd.DataFrame([{
        "ID": l["id"], "Lição": l["titulo"], "Tempo": l["tempo"], "Perguntas": len(l.get("quiz", [])),
        "Prática": l["pratica"][0] if l.get("pratica") else ""} for l in LICOES]),
        width="stretch", hide_index=True)
    st.caption(f"{len(LICOES)} lições. A edição de conteúdos e as trilhas por nível chegam com o EDU completo.")


def render_admin_erros(username: str):
    page_header("Erros", "O que correu mal nas páginas (registado automaticamente)")
    erros = db.list_erros(300)
    if not erros:
        st.success("Sem erros registados.")
        return
    df = pd.DataFrame([{"Quando": (e.get("created_at") or "")[:16].replace("T", " "), "Utilizador": e.get("username", ""),
                        "Página": e.get("pagina", ""), "Mensagem": (e.get("mensagem") or "").strip().splitlines()[-1][:160]
                        if (e.get("mensagem") or "").strip() else ""} for e in erros])
    st.markdown("### Por página")
    st.bar_chart(df["Página"].value_counts())
    st.markdown("### Mais recentes")
    st.dataframe(df, width="stretch", hide_index=True)
    with st.expander("Ver o erro completo mais recente"):
        st.code(erros[0].get("mensagem", ""))


def render_business_home(username: str, name: str):
    """Marcador do painel Business (MVP na Fase 3)."""
    page_header("DataForge Business", "Machine Learning nos dados da tua empresa")
    st.info("O painel Business está em construção. Vai permitir carregar dados (CSV ou Excel), fazer um diagnóstico "
            "automático, escolher o problema (prever vendas, clientes que vão sair, fraude, stock), treinar com "
            "explicação simples e exportar um relatório executivo em PDF.")
    membros = db.membros_do_utilizador(username)
    if membros:
        st.markdown("**As tuas organizações**")
        for m in membros:
            st.markdown(f"- {m['nome']} ({m['tipo']}, {m['papel']})")
