"""
DataForge EDU - Painel do Professor
Turmas com código de convite, progresso de cada aluno, lições mais difíceis e reposição de palavras-passe.
"""
import datetime
import secrets

import pandas as pd
import streamlit as st

from modules import acesso, db, email_service, ia
from modules.auth import _hash
from modules.i18n import idioma_actual
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

    erros_por_licao = sorted(erros.items(), key=lambda x: -x[1])
    _render_ia_professor(username, turma, df, erros_por_licao, total)

    st.markdown("### Repor palavra-passe de um aluno")
    quem = st.selectbox("Aluno", [f"{a['name']} ({a['username']})" for a in alunos], key="prof_reset_quem")
    if st.button("Gerar palavra-passe temporária", key="prof_reset"):
        uname = quem.rsplit("(", 1)[1].rstrip(")")
        temp = secrets.token_urlsafe(6)
        db.update_user(uname, {"password_hash": _hash(temp), "must_change_pw": True})
        st.success(f"Nova palavra-passe de **{uname}**: `{temp}`. Entrega-a ao aluno; no primeiro acesso ele escolhe uma nova.")


# ══════════════════════════════════════════════════════
# ASSISTENTE DE IA DO PROFESSOR (rascunhos para rever)
# ══════════════════════════════════════════════════════
def _dados_turma(df, erros_por_licao: list, total: int) -> dict:
    """O que a IA vê da turma: só totais e contagens (sem nomes nem utilizadores)."""
    from modules.escola import LICAO_POR_ID
    return {"alunos": int(len(df)), "activos": int((df["Estado"] == "Activo").sum()),
            "parados_ha_mais_de_7_dias": int(df["Estado"].str.startswith("Parado").sum()),
            "nunca_entraram": int((df["Estado"] == "Nunca entrou").sum()),
            "media_de_licoes_concluidas": round(float(df["_licoes"].mean()), 1), "total_de_licoes": total,
            "alunos_com_todas_as_licoes": int((df["_licoes"] >= total).sum()),
            "licoes_com_mais_respostas_erradas": [{"licao": LICAO_POR_ID[k]["titulo"] if k in LICAO_POR_ID else str(k), "respostas_erradas": n}
                                                  for k, n in erros_por_licao[:5]]}


def _caixa_rascunho(chave: str, rotulo: str, altura: int = 180):
    """Mostra o rascunho gerado (editável) para o professor rever e copiar."""
    if chave in st.session_state:
        st.text_area(rotulo, key=chave, height=altura)
        st.caption("Rascunho gerado por IA: lê, corrige e só depois usa. A IA pode enganar-se.")


def _exercicios_em_texto(lista: list) -> str:
    out = []
    for i, e in enumerate(lista, 1):
        out.append(f"{i}. {e['pergunta']}")
        out += [f"   {'ABCD'[j]}) {o}" for j, o in enumerate(e["opcoes"])]
        out.append(f"   Resposta certa: {'ABCD'[e['certa']]}" + (f" — {e['explicacao']}" if e["explicacao"] else ""))
        out.append("")
    return "\n".join(out)


def _render_ia_professor(username: str, turma: dict, df, erros_por_licao: list, total: int):
    from modules.escola import LICOES, LICAO_POR_ID
    if not ia.configurada():
        return
    cod = turma["codigo"]
    idp = ia.id_professor(username)
    st.markdown("### Assistente de IA")
    st.caption("Rascunhos para tu rever: resumo da turma, aula de reforço, exercícios, mensagens e feedback. Os nomes dos alunos "
               "nunca seguem para a IA. A IA pode enganar-se, por isso nada é enviado a ninguém sem passares por ti.")
    st.caption(f"Pedidos à IA que restam hoje: {ia.pedidos_restantes(idp, ia.limite_professor())}")
    t1, t2, t3, t4, t5 = st.tabs(["Resumo da turma", "Aula de reforço", "Exercícios", "Mensagem", "Feedback ao projecto"])
    dados = _dados_turma(df, erros_por_licao, total)

    with t1:
        st.caption("Resume em poucas frases onde a turma está bem e onde está a falhar. Seguem só totais e contagens.")
        chave = f"prof_ia_resumo_{cod}"
        if st.button("Gerar resumo", key=f"prof_ia_b1_{cod}", type="primary"):
            try:
                with st.spinner("A pensar..."):
                    st.session_state[chave] = ia.resumo_turma(dados, username, idioma_actual())
            except ia.IAErro as e:
                st.error(str(e))
        _caixa_rascunho(chave, "Resumo da turma")

    with t2:
        st.caption("Uma actividade curta para explicar um conceito que a turma está a errar. Segue só o texto da lição.")
        ids = [l["id"] for l in LICOES]
        por_omissao = next((k for k, _ in erros_por_licao if k in LICAO_POR_ID), ids[0])
        lid = st.selectbox("Lição", ids, index=ids.index(por_omissao), key=f"prof_ia_licao_{cod}",
                           format_func=lambda i: LICAO_POR_ID[i]["titulo"])
        chave = f"prof_ia_aula_{cod}"
        if st.button("Sugerir aula de reforço", key=f"prof_ia_b2_{cod}", type="primary"):
            try:
                with st.spinner("A pensar..."):
                    st.session_state[chave] = ia.sugerir_aula(LICAO_POR_ID[lid], dict(erros_por_licao).get(lid, 0), len(df),
                                                              username, idioma_actual())
            except ia.IAErro as e:
                st.error(str(e))
        _caixa_rascunho(chave, "Sugestão de aula", 260)

    with t3:
        st.caption("Perguntas de escolha múltipla sobre um tema, com a resposta certa e a explicação. Confirma-as antes de as usares.")
        c1, c2 = st.columns([3, 1])
        tema = c1.text_input("Tema", max_chars=200, key=f"prof_ia_tema_{cod}", placeholder="ex.: overfitting e validação cruzada")
        n = c2.number_input("Quantas", min_value=1, max_value=8, value=3, key=f"prof_ia_n_{cod}")
        chave = f"prof_ia_ex_{cod}"
        if st.button("Criar exercícios", key=f"prof_ia_b3_{cod}", type="primary"):
            try:
                with st.spinner("A pensar..."):
                    st.session_state[chave] = ia.gerar_exercicios(tema, int(n), username, idioma_actual())
            except ia.IAErro as e:
                st.error(str(e))
        lista = st.session_state.get(chave)
        if lista:
            for i, e in enumerate(lista, 1):
                st.markdown(f"**{i}. {e['pergunta']}**")
                for j, o in enumerate(e["opcoes"]):
                    st.markdown(f"- {'ABCD'[j]}) {o}" + (" (certa)" if j == e["certa"] else ""))
                if e["explicacao"]:
                    st.caption(e["explicacao"])
            st.download_button("Descarregar (texto)", _exercicios_em_texto(lista).encode("utf-8"),
                               file_name=f"exercicios_{cod}.txt", mime="text/plain", key=f"prof_ia_dl_{cod}")
            st.caption("Rascunho gerado por IA: confirma cada pergunta e resposta antes de usar.")

    with t4:
        st.caption("Escolhe para quem é e o tom. A IA escreve a mensagem sem saber o nome: se quiseres, o nome é posto aqui, no servidor.")
        destino = st.radio("Para quem", ["Um aluno", "Toda a turma"], horizontal=True, key=f"prof_ia_dest_{cod}")
        tom = st.selectbox("Tom", list(ia.TONS_MENSAGEM.keys()), key=f"prof_ia_tom_{cod}",
                           format_func=lambda k: {"encorajar": "Encorajar", "atrasado": "Avisar que está atrasado",
                                                  "parabens": "Dar os parabéns", "lembrar": "Lembrar o que falta"}[k])
        nome_aluno = uname_aluno = None
        usar_nome = False
        if destino == "Um aluno":
            quem = st.selectbox("Aluno", [f"{r['Nome']} ({r['Utilizador']})" for _, r in df.iterrows()], key=f"prof_ia_aluno_{cod}")
            uname_aluno = quem.rsplit("(", 1)[1].rstrip(")")
            linha = df[df["Utilizador"] == uname_aluno].iloc[0]
            nome_aluno = str(linha["Nome"]).split()[0]
            usar_nome = st.checkbox("Tratar o aluno pelo primeiro nome", value=True, key=f"prof_ia_usanome_{cod}")
            situacao = {"estado": str(linha["Estado"]), "licoes_concluidas": int(linha["_licoes"]), "total_de_licoes": total,
                        "pontos": int(linha["Pontos"])}
        else:
            situacao = {k: dados[k] for k in ("alunos", "activos", "parados_ha_mais_de_7_dias", "nunca_entraram",
                                              "media_de_licoes_concluidas", "total_de_licoes")}
        nota = st.text_input("Algo a incluir (opcional, sem dados pessoais)", max_chars=200, key=f"prof_ia_nota_{cod}",
                             placeholder="ex.: a prova é na próxima semana")
        chave = f"prof_ia_msg_{cod}"
        if st.button("Escrever mensagem", key=f"prof_ia_b4_{cod}", type="primary"):
            try:
                with st.spinner("A pensar..."):
                    texto = ia.redigir_mensagem("aluno" if destino == "Um aluno" else "turma", tom, situacao, nota, username, idioma_actual())
                    st.session_state[chave] = ia.aplicar_nome(texto, nome_aluno if usar_nome else None)
            except ia.IAErro as e:
                st.error(str(e))
        _caixa_rascunho(chave, "Mensagem", 150)

    with t5:
        st.caption("Cola o texto do relatório ou das respostas do aluno e recebe um rascunho de comentário por critérios.")
        st.warning("O texto colado segue para o serviço de IA. Tira o nome do aluno e quaisquer dados pessoais antes de colares "
                   "(emails e números de telefone são retirados automaticamente).")
        texto = st.text_area("Texto do aluno", max_chars=3000, height=180, key=f"prof_ia_txt_{cod}")
        criterios = st.multiselect("Critérios", ia.CRITERIOS_PROJECTO, default=ia.CRITERIOS_PROJECTO, key=f"prof_ia_crit_{cod}")
        ok = st.checkbox("Confirmo que tirei o nome do aluno e dados pessoais do texto", key=f"prof_ia_ok_{cod}")
        chave = f"prof_ia_fb_{cod}"
        if st.button("Escrever rascunho de feedback", key=f"prof_ia_b5_{cod}", type="primary", disabled=not ok):
            try:
                with st.spinner("A pensar..."):
                    st.session_state[chave] = ia.rascunho_feedback(texto, criterios, username, idioma_actual())
            except ia.IAErro as e:
                st.error(str(e))
        _caixa_rascunho(chave, "Feedback", 280)


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
    papel = c2.selectbox("Novo papel", ["aluno", "professor", "empresa", "admin"], key="adm_papel")
    if st.button("Guardar papel", type="primary", key="adm_save"):
        if alvo == username and papel != "admin":
            st.error("Não podes retirar o teu próprio papel de administrador.")
        else:
            db.update_user(alvo, {"role": papel})
            st.success(f"{alvo} agora é {papel}. A pessoa deve sair e voltar a entrar.")
