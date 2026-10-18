"""
DataForge - Painel de Administração (só administradores globais).
Visão geral, organizações (escolas e empresas) e membros, conteúdos e registo de erros.
A gestão de papéis dos utilizadores continua em modules.professor.render_admin_utilizadores.
"""
import secrets

import pandas as pd
import streamlit as st

from modules import acesso, db, email_service, ia
from modules.utils import page_header


def _novo_codigo() -> str:
    while True:
        c = secrets.token_hex(3).upper()
        if not db.get_org_por_codigo(c) and not db.get_turma(c):
            return c


def adicionar_membro_por_codigo(username: str, codigo: str, papel_preferido: str = "aluno"):
    """Junta uma pessoa a uma organização pelo código de convite. Devolve (ok, mensagem)."""
    u = db.get_user(username)
    if u and u.get("role") == "admin":
        return False, "O administrador do sistema não pode juntar-se a escolas nem a empresas."
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

    pend = acesso.orgs_pendentes()
    if pend:
        st.warning(f"Há {len(pend)} organização(ões) a aguardar a tua aprovação, em Organizações.")
    pend_conv = acesso.pedidos_convite_pendentes()
    if pend_conv:
        st.warning(f"Há {len(pend_conv)} professor(es) a pedir convite, em Convites.")

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

    if ia.configurada():
        g = ia.estado_global()
        st.markdown("### IA: uso da chave hoje")
        d1, d2, d3 = st.columns(3)
        d1.metric("Pedidos hoje", f"{g['dia']} / {g['tecto_dia'] or '∞'}")
        d2.metric("Neste minuto", f"{g['minuto']} / {g['tecto_minuto'] or '∞'}")
        d3.metric("Reservado a empresas e professores", f"{g['reserva_pct']}%")
        st.caption(f"Provedor {g['provedor']} · modelo geral {g['modelo']} · modelo do tutor {g['modelo_tutor']}. "
                   "Os tectos são os de ia_limite_global_dia e ia_limite_global_minuto: confirma na consola do teu provedor quais são os limites do teu plano.")

    st.caption("Utilizadores sem organização continuam a funcionar como alunos independentes do EDU.")


def render_admin_orgs(username: str):
    """Supervisão: o admin NÃO cria organizações, NÃO entra nelas e NÃO vê os seus membros nem dados.
    Só aprova/recusa empresas novas e suspende/reactiva as existentes."""
    page_header("Organizações", "Aprovação e supervisão. Cada escola ou empresa só vê os seus dados.")

    pend = acesso.orgs_pendentes()
    st.markdown(f"### A aguardar aprovação ({len(pend)})")
    if not pend:
        st.caption("Não há pedidos pendentes.")
    for o in pend:
        dono = db.get_user(o.get("dono") or "") or {}
        with st.container(border=True):
            st.markdown(f"**{o['nome']}** ({o['tipo']})  \nResponsável: {dono.get('name', '?')} · "
                        f"{o.get('dono', '?')} · {dono.get('email') or 'sem email'}")
            c1, c2, c3 = st.columns([1, 2, 1])
            if c1.button("Aprovar", type="primary", key=f"org_ok_{o['id']}"):
                acesso.aprovar_org(o["id"])
                st.rerun()
            motivo = c2.text_input("Motivo da recusa (opcional)", key=f"org_mot_{o['id']}", label_visibility="collapsed",
                                   placeholder="Motivo da recusa (opcional)")
            if c3.button("Recusar", key=f"org_no_{o['id']}"):
                acesso.recusar_org(o["id"], motivo)
                st.rerun()

    st.info("O administrador do sistema não cria nem entra em organizações. Aqui só vê os totais, "
            "aprova pedidos e suspende ou reactiva uma organização (por exemplo, por abuso ou falta de pagamento).")

    orgs = [o for o in db.list_orgs() if o.get("estado") != "pendente"]
    if not orgs:
        st.info("Ainda não há organizações aprovadas.")
        return

    contagem = {}
    for m in db.list_membros():
        contagem[m["org_id"]] = contagem.get(m["org_id"], 0) + 1
    st.markdown("### Organizações")
    st.dataframe(pd.DataFrame([{
        "Nome": o["nome"], "Tipo": o["tipo"], "Produto": o["produto"], "Estado": o.get("estado") or "aprovada",
        "Membros": contagem.get(o["id"], 0), "Activa": "Sim" if o.get("activa", True) else "Não"} for o in orgs]),
        width="stretch", hide_index=True)

    gerivel = [o for o in orgs if o.get("estado") in (None, "", "aprovada")]      # recusadas não se reactivam aqui
    if not gerivel:
        return
    st.markdown("### Suspender ou reactivar")
    opcoes = {o["nome"]: o for o in gerivel}
    org = opcoes[st.selectbox("Organização", list(opcoes.keys()), key="org_sel")]
    activa = org.get("activa", True)
    if st.button("Suspender organização" if activa else "Reactivar organização", key="org_toggle"):
        db.update_org(org["id"], {"activa": not activa})
        st.rerun()
    if not activa:
        st.warning("Organização suspensa: os membros deixam de ter acesso ao painel desta organização.")


def _render_pedidos_convite(username: str):
    """Pedidos feitos pelos próprios professores no formulário de registo."""
    pedidos = acesso.pedidos_convite_pendentes()
    st.markdown(f"### Pedidos de convite ({len(pedidos)})")
    resultado = st.session_state.pop("_pedido_resultado", None)
    if resultado:
        (st.success if resultado[0] == "ok" else st.warning)(resultado[1])
        if resultado[2]:
            st.code(resultado[2])
    if not pedidos:
        st.caption("Não há pedidos à espera. Os professores novos podem pedir convite no ecrã «Criar conta».")
        return
    for p in pedidos:
        with st.container(border=True):
            st.markdown(f"**{p['nome']}** · {p['email']}  \n{p.get('escola') or 'Escola não indicada'}")
            if p.get("mensagem"):
                st.caption(p["mensagem"])
            motivo = st.text_input("Motivo (só se recusares)", key=f"pc_motivo_{p['id']}")
            c1, c2, _ = st.columns([1.2, 1, 3])
            if c1.button("Aprovar e enviar convite", type="primary", key=f"pc_ok_{p['id']}"):
                try:
                    codigo, estado, msg = acesso.aprovar_pedido_convite(p["id"], username)
                except db.DBError as e:
                    st.session_state["_pedido_resultado"] = ("aviso", str(e), "")
                else:
                    if estado == "enviado":
                        st.session_state["_pedido_resultado"] = ("ok", f"Convite {codigo} enviado para {p['email']}.", "")
                    else:
                        st.session_state["_pedido_resultado"] = (
                            "aviso", f"Convite {codigo} criado, mas o email não saiu ({msg}). Envia-o tu, por WhatsApp ou outro meio:",
                            acesso.link_convite(codigo) or codigo)
                st.rerun()
            if c2.button("Recusar", key=f"pc_no_{p['id']}"):
                try:
                    acesso.recusar_pedido_convite(p["id"], username, motivo)
                    st.session_state["_pedido_resultado"] = ("ok", f"Pedido de {p['nome']} recusado.", "")
                except db.DBError as e:
                    st.session_state["_pedido_resultado"] = ("aviso", str(e), "")
                st.rerun()


def render_admin_convites(username: str):
    page_header("Convites de professor", "Cada convite é de uso único e expira em %d dias" % acesso.VALIDADE_CONVITE_DIAS)

    _render_pedidos_convite(username)

    st.markdown("### Convidar um professor directamente")
    with st.form("form_convite", clear_on_submit=True):
        email = st.text_input("Email do professor")
        criar = st.form_submit_button("Criar convite e enviar", type="primary")
    if criar:
        try:
            codigo, estado, msg = acesso.criar_convite_professor(email, username)
        except db.DBError as e:
            st.error(str(e))
        else:
            link = acesso.link_convite(codigo)
            if estado == "enviado":
                st.success(f"Convite {codigo} enviado para {email}.")
            else:
                st.warning(f"Convite {codigo} criado, mas o email não saiu ({msg}). Envia-o tu, por WhatsApp ou outro meio:")
            st.code(link or codigo)

    convites = db.list_convites()
    if not convites:
        st.info("Ainda não criaste convites.")
        return
    st.dataframe(pd.DataFrame([{
        "Código": c["codigo"], "Email": c["email"], "Estado": acesso.estado_convite(c),
        "Usado por": c.get("usado_por") or "", "Expira": (c.get("expira_em") or "")[:10]} for c in convites]),
        width="stretch", hide_index=True)

    pendentes = [c["codigo"] for c in convites if acesso.estado_convite(c) == "Pendente"]
    if pendentes:
        c1, c2 = st.columns([3, 1])
        alvo = c1.selectbox("Cancelar um convite pendente", pendentes, key="conv_cancel_sel")
        c2.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if c2.button("Cancelar", key="conv_cancel"):
            acesso.cancelar_convite(alvo)
            st.rerun()


def render_admin_email(username: str):
    page_header("Email", "Estado do envio, quota diária e fila")
    if email_service.configurado():
        st.success("O envio de email está configurado.")
    else:
        st.warning("O envio de email ainda não está configurado. Preenche `smtp_host`, `smtp_user`, `smtp_password` "
                   "e `email_from` nos segredos (ver `.streamlit/secrets.toml.example`). Até lá, os convites e códigos "
                   "aparecem no ecrã para os enviares à mão.")

    enviados, limite, fila = email_service.enviados_hoje(), email_service.limite_diario(), email_service.em_fila()
    c1, c2, c3 = st.columns(3)
    c1.metric("Enviados hoje", f"{enviados} / {limite}")
    c2.metric("Na fila", fila)
    c3.metric("Quota restante", max(limite - enviados, 0))
    if limite and enviados >= 0.8 * limite:
        st.warning("Estás perto do limite diário. O que passar do limite fica na fila e sai quando houver quota.")

    if fila and st.button("Enviar agora o que está na fila", key="mail_fila"):
        r = email_service.processar_fila()
        st.info(f"Enviados: {r['enviados']}. Ainda na fila: {r['restam']}.")

    st.markdown("### Email de teste")
    destino = st.text_input("Enviar para", key="mail_teste_dest")
    if st.button("Enviar email de teste", key="mail_teste") and destino.strip():
        estado, msg = email_service.enviar(destino.strip(), "Teste do DataForge", "Se estás a ler isto, o envio de email funciona.")
        (st.success if estado == "enviado" else st.error)(msg)

    recentes = db.list_emails(limite=30)
    if recentes:
        st.markdown("### Últimos emails")
        st.dataframe(pd.DataFrame([{"Quando": (e.get("created_at") or "")[:16].replace("T", " "), "Para": e["para"],
                                    "Assunto": e["assunto"], "Estado": e["estado"], "Erro": e.get("erro") or ""}
                                   for e in recentes]), width="stretch", hide_index=True)


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
    """Mantido por compatibilidade: o painel Business completo está em modules/business.py."""
    if (db.get_user(username) or {}).get("role") == "admin":
        st.error("O administrador do sistema não tem acesso ao painel Business.")
        return
    from modules.business import render_inicio
    render_inicio(username, name)


def render_empresa_pendente(username: str, org: dict):
    """Ecrã do dono enquanto a empresa aguarda aprovação (ou depois de recusada)."""
    page_header("DataForge Business", org["nome"])
    if org.get("estado") == "pendente":
        st.info("O teu pedido está a aguardar a aprovação do administrador. Vais receber um email quando for aprovado. "
                "Volta a entrar mais tarde para ver o estado.")
        return
    st.error(f"O pedido não foi aprovado. Motivo: {org.get('motivo') or 'não indicado'}.")
    nome = st.text_input("Nome da empresa para um novo pedido", value=org["nome"], key="emp_novo_nome")
    if st.button("Submeter novo pedido", type="primary", key="emp_novo_btn"):
        try:
            novo = acesso.novo_pedido_empresa(username, nome)
            email_service.enviar_a_admins(*email_service.t_empresa_pendente_admin(
                novo["nome"], (db.get_user(username) or {}).get("name", username), (db.get_user(username) or {}).get("email", "")))
            st.rerun()
        except db.DBError as e:
            st.error(str(e))
