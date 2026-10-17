"""
DataForge EDU - Módulo de Autenticação
Login, registo, perfil do utilizador
"""

import time
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import re
import bcrypt
from modules import acesso, db
from modules.settings import secret, contas_demo
from modules.utils import (
    inject_css, C_ACCENT, C_SURFACE, C_BORDER, C_TEXT, C_TEXT_SEC,
    C_TEXT_MUTE, C_BG, C_SURFACE2, C_GREEN, C_AMBER, get_user_progress, badge,
    render_seletor_tema, render_seletor_tema_perfil, render_seletor_idioma
)

CONFIG_PATH = "config.yaml"
UTILIZADOR_RE = re.compile(r"^[a-z0-9_]{3,20}$")


def _hash(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def _cookie_cfg() -> dict:
    return {"expiry_days": 7,
            "key": str(secret("cookie_key", "dataforge_edu_secret_key_2024")),
            "name": "dataforge_edu_cookie"}


def _contas_demo() -> dict:
    """Contas de demonstração (config.yaml). Só activas em modo local ou se contas_demo=true."""
    if not contas_demo():
        return {}
    if not os.path.exists(CONFIG_PATH):
        cfg = {"credentials": {"usernames": {
            "admin":     {"email": "admin@dataforge.ao", "name": "Administrador",  "password": _hash("admin123"), "role": "admin"},
            "professor": {"email": "prof@dataforge.ao",  "name": "Professor Demo", "password": _hash("prof123"),  "role": "professor"},
            "aluno":     {"email": "aluno@dataforge.ao", "name": "Aluno Demo",     "password": _hash("aluno123"), "role": "aluno"},
        }}}
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        except OSError:
            pass
        return cfg["credentials"]["usernames"]
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=SafeLoader) or {}
    return cfg.get("credentials", {}).get("usernames", {})


def load_config() -> dict:
    """Credenciais = contas demo (se activas) + utilizadores da base de dados."""
    usernames = dict(_contas_demo())
    for u in db.list_users():
        usernames[u["username"]] = {
            "email": u.get("email") or "", "name": u["name"], "password": u["password_hash"],
            "role": u.get("role", "aluno"), "turma": u.get("turma"),
        }
    return {"credentials": {"usernames": usernames}, "cookie": _cookie_cfg()}


def get_authenticator(force: bool = False):
    if force or "authenticator" not in st.session_state or "auth_config" not in st.session_state:
        config = load_config()
        st.session_state.auth_config = config
        st.session_state.authenticator = stauth.Authenticate(
            config["credentials"], config["cookie"]["name"], config["cookie"]["key"], config["cookie"]["expiry_days"])
    return st.session_state.authenticator, st.session_state.auth_config


def _tinha_cookie(config) -> bool:
    """True se o navegador já enviou o cookie de sessão (login automático, sem escrever cookie novo)."""
    try:
        return bool(st.context.cookies.get(config["cookie"]["name"]))
    except Exception:
        return False


# ══════════════════════════════════════════════════════
# TERMINAR SESSÃO (Sair)
# ══════════════════════════════════════════════════════
# Porque é que «Sair» não chegava: o cookie de sessão vive no navegador e é apagado por um componente
# que só executa se a página ainda existir. O st.rerun() logo a seguir destruía-o antes de correr, o
# cookie ficava lá e, ao actualizar a página, o login automático reabria a sessão.
# Agora há três camadas: (1) o cookie é apagado no navegador e damos-lhe tempo para isso,
# (2) as sessões abertas até ao momento do «Sair» ficam revogadas no servidor (o cookie velho, se
# ainda existir, deixa de servir) e (3) o cookie é verificado contra essa revogação sempre que é lido.
def _chave_sessao(username: str) -> str:
    # os nomes de utilizador não podem ter «:», por isso esta chave nunca colide com uma conta real
    return f"sessao:{username}"


def _marca_logout(username: str) -> float:
    try:
        return float((db.get_progress(_chave_sessao(username)) or {}).get("logout_em") or 0)
    except Exception:
        return 0.0


def _revogar_sessoes(username: str):
    """Marca o instante do «Sair»: cookies emitidos até agora deixam de abrir sessão."""
    try:
        db.save_progress(_chave_sessao(username), {"logout_em": time.time()})
    except Exception:
        pass            # sem base de dados continua a valer o apagar do cookie no navegador


def _cookie_revogado(token: dict, config) -> bool:
    """True se o cookie (JWT já descodificado) foi emitido antes do último «Sair» dessa conta."""
    try:
        marca = _marca_logout(token.get("username", ""))
        if not marca:
            return False
        emitido = float(token.get("exp_date", 0)) - float(config["cookie"]["expiry_days"]) * 86400
        return emitido <= marca
    except Exception:
        return False


def _descodificar_cookie(bruto, config):
    try:
        import jwt
        token = jwt.decode(bruto, config["cookie"]["key"], algorithms=["HS256"])
        return token if token.get("username") else None
    except Exception:
        return None


def _apagar_cookie_no_navegador(nome: str):
    """Apaga o cookie de sessão no navegador (JavaScript num iframe do mesmo domínio)."""
    try:
        import streamlit.components.v1 as components
        js = ("<script>(function(){var n=%r,p='Thu, 01 Jan 1970 00:00:00 GMT';"
              "function k(d){try{d.cookie=n+'=; expires='+p+'; path=/';d.cookie=n+'=; Max-Age=0; path=/';"
              "d.cookie=n+'=; expires='+p+'; path=/; SameSite=Lax';}catch(e){}}"
              "k(document);try{k(window.parent.document);}catch(e){}})();</script>" % nome)
        components.html(js, height=0)
    except Exception:
        pass


def _terminar_sessao(username: str, authenticator, config=None):
    """Sai da conta de forma definitiva (também depois de actualizar a página)."""
    nome_cookie = (config or st.session_state.get("auth_config") or {}).get("cookie", {}).get("name", "dataforge_edu_cookie")
    _revogar_sessoes(username)
    st.session_state["_df_saiu"] = True      # impede que o cookie ainda por apagar reabra a sessão
    try:
        authenticator.logout(location="unrendered")
    except Exception:
        pass
    try:
        if _google_activo() and st.user.is_logged_in:
            st.logout()
    except Exception:
        pass
    _apagar_cookie_no_navegador(nome_cookie)
    # Dá tempo ao navegador para executar o apagar do cookie antes do st.rerun() (que destrói o componente).
    time.sleep(1.2)
    # Esquece tudo o que era da pessoa que saiu: a próxima a entrar começa limpa.
    for k in list(st.session_state.keys()):
        if k in ("authentication_status", "username", "name", "email", "roles", "painel", "pagina", "painel_sel",
                 "_membros_cache", "_prog_cache", "_pw_ok", "guiado_step", "_df_cookie_t0") \
                or str(k).startswith(("_tema_lido_", "_idioma_lido_")):
            st.session_state.pop(k, None)
    st.session_state["authentication_status"] = None
    st.session_state["logout"] = True        # o authenticator deixa de ler o cookie nesta sessão
    st.rerun()


def _login_por_cookie_do_navegador(authenticator, config) -> bool:
    """Restaura a sessão a partir do cookie guardado no navegador.

    O streamlit-authenticator 0.4.x lê o cookie no servidor (st.context.cookies), mas no Streamlit Cloud o
    proxy não o reencaminha: o cookie existe no navegador e o servidor nunca o vê, por isso cada
    recarregamento caía no login. Aqui o cookie é lido no próprio navegador (componente) e validado
    com a mesma chave (JWT), e a sessão é aberta como se o authenticator o tivesse lido."""
    if st.session_state.get("_df_saiu"):
        return False
    try:
        import extra_streamlit_components as stx
        import jwt
        nome = config["cookie"]["name"]
        cookies = stx.CookieManager(key="df_leitor_cookie").get_all() or {}
        # Espera pelo navegador durante os primeiros 5 s da sessão (não só na primeira execução: a ligação
        # ao servidor pode cair e voltar, e a página volta a executar antes do cookie chegar).
        inicio = st.session_state.setdefault("_df_cookie_t0", time.time())
        restante = 5.0 - (time.time() - inicio)
        if not cookies and restante > 0:
            aviso = st.empty()
            # Cada actualização do aviso é um comando do Streamlit, que interrompe a espera logo que o
            # navegador responde (a página volta a executar já com o cookie).
            for i in range(int(restante / 0.2) + 1):
                time.sleep(0.2)
                aviso.markdown(f"<div style='position:fixed;inset:0;z-index:999999;display:flex;align-items:center;justify-content:center;background:var(--df-bg,#FFFFFF);color:{C_TEXT_SEC};font-size:15px;'>A carregar{'.' * (i % 4)}</div>",
                               unsafe_allow_html=True)
            aviso.empty()
            return False
        bruto = cookies.get(nome)
        if not bruto:
            return False
        token = jwt.decode(bruto, config["cookie"]["key"], algorithms=["HS256"])
        if not token.get("username") or float(token.get("exp_date", 0)) <= time.time():
            return False
        utilizador = config["credentials"]["usernames"].get(token["username"])
        if not utilizador:
            return False
        if _cookie_revogado(token, config):      # a pessoa clicou em «Sair» depois deste cookie ser emitido
            _apagar_cookie_no_navegador(nome)
            return False
        try:
            authenticator.authentication_controller.login(token=token)
        except Exception:
            pass
        if st.session_state.get("authentication_status") is not True:
            st.session_state["authentication_status"] = True
            st.session_state["username"] = token["username"]
            st.session_state["name"] = utilizador.get("name", token["username"])
            st.session_state["email"] = utilizador.get("email", "")
        return st.session_state.get("authentication_status") is True
    except Exception:
        return False


def _google_activo() -> bool:
    try:
        return "auth" in st.secrets and hasattr(st, "login")
    except Exception:
        return False


def _utilizador_google():
    """Se o utilizador entrou com Google (OIDC do Streamlit), devolve (username, nome); senão None."""
    try:
        if not (_google_activo() and st.user.is_logged_in):
            return None
        email = (st.user.get("email") or "").lower()
        nome = st.user.get("name") or email.split("@")[0]
        if not email:
            return None
        for u in db.list_users(email=email):
            return u["username"], u["name"]
        base = re.sub(r"[^a-z0-9_]", "", email.split("@")[0].lower())[:16] or "aluno"
        username, n = base, 1
        while db.get_user(username) or username in _contas_demo():
            n += 1
            username = f"{base}{n}"
        db.create_user({"username": username, "name": nome, "email": email, "password_hash": _hash(os.urandom(12).hex()), "role": "aluno"})
        return username, nome
    except Exception:
        return None


def _registar(papel, nome, utilizador, email, pw, pw2, **extra) -> tuple:
    """Valida e cria a conta (regras em modules.acesso). Devolve (ok, mensagem)."""
    ok, msg = acesso.registar(papel, nome, utilizador, email, pw, pw2, reservados=set(_contas_demo()), **extra)
    if ok:
        get_authenticator(force=True)
    return ok, msg


_PAPEIS_REGISTO = {"Sou aluno": "aluno", "Sou professor": "professor", "Sou uma empresa (Business)": "empresa"}


def _render_registo():
    convite_url = st.query_params.get("convite", "")
    if convite_url:
        st.session_state.setdefault("reg_papel", "Sou professor")
        st.info("Recebeste um convite de professor. Preenche o formulário abaixo com o email que recebeu o convite.")
    papel = _PAPEIS_REGISTO[st.radio("Quem és?", list(_PAPEIS_REGISTO), key="reg_papel", horizontal=True)]
    modo = "nova"
    if papel == "empresa":
        modo = "nova" if st.radio("A tua empresa", ["Criar uma empresa nova", "Entrar numa empresa que já existe"],
                                  key="reg_emp_modo").startswith("Criar") else "existente"
        if modo == "nova":
            st.caption("A empresa fica a aguardar a aprovação do administrador. Avisamos-te por email.")

    extra = {}
    with st.form("form_registo", clear_on_submit=False):
        r_nome = st.text_input("Nome completo")
        r_user = st.text_input("Utilizador (letras minúsculas, números ou _)")
        r_mail = st.text_input("Email (obrigatório)" if papel != "aluno" else "Email (opcional)")
        r_pw = st.text_input("Palavra-passe (mínimo 6 caracteres)", type="password")
        r_pw2 = st.text_input("Repete a palavra-passe", type="password")
        if papel == "aluno":
            extra["turma"] = st.text_input("Código da turma (opcional, dado pelo teu professor)")
            extra["cod_org"] = st.text_input("Código da escola (opcional)")
        elif papel == "professor":
            extra["cod_convite"] = st.text_input("Código de convite de professor", value=convite_url)
            extra["cod_org"] = st.text_input("Código da escola (opcional, se a tua escola já existe)")
        elif modo == "nova":
            extra["empresa_nome"] = st.text_input("Nome da empresa")
        else:
            extra["empresa_cod"] = st.text_input("Código da empresa (dado pelo responsável)")
        enviar = st.form_submit_button("Criar conta", type="primary")
    if enviar:
        try:
            ok, msg = _registar(papel, r_nome, r_user, r_mail, r_pw, r_pw2, empresa_modo=modo, **extra)
        except db.DBError as e:
            ok, msg = False, f"Erro na base de dados: {e}"
        (st.success if ok else st.error)(msg)
    if papel == "professor":
        _render_pedir_convite(aberto=not convite_url)


def _render_pedir_convite(aberto: bool = True):
    """Professor novo sem convite: pede acesso aqui e o admin aprova (o convite chega ao email)."""
    with st.expander("Não tens convite? Pede acesso", expanded=aberto):
        st.caption("Diz-nos quem és. O administrador analisa o pedido e envia-te o convite por email: "
                   "depois voltas a este ecrã, escolhes «Sou professor» e usas o código.")
        with st.form("form_pedir_convite", clear_on_submit=True):
            p_nome = st.text_input("Nome completo", key="pc_nome")
            p_mail = st.text_input("Email onde queres receber o convite", key="pc_mail")
            p_escola = st.text_input("Escola ou instituição onde ensinas", key="pc_escola")
            p_msg = st.text_area("Mensagem (opcional)", key="pc_msg", height=80, max_chars=300,
                                 placeholder="Ex.: Dou Machine Learning no 3.º ano de Engenharia Informática.")
            pedir = st.form_submit_button("Pedir convite", type="primary")
        if pedir:
            try:
                ok, msg = acesso.pedir_convite(p_nome, p_mail, p_escola, p_msg)
            except db.DBError as e:
                ok, msg = False, f"Erro na base de dados: {e}"
            (st.success if ok else st.error)(msg)


def _render_esqueci():
    st.markdown("Escreve o teu **utilizador** ou **email**. Se a conta tiver email, enviamos-te um link para escolheres "
                "uma palavra-passe nova (válido 1 hora).")
    ident = st.text_input("Utilizador ou email", key="esq_ident")
    if st.button("Enviar link", key="esq_btn") and ident.strip():
        try:
            acesso.pedir_reposicao(ident)
        except Exception:
            pass
        st.success("Se a conta existir e tiver email, o link foi enviado. Confirma também o spam.")
    st.caption("Não tens email na conta (por exemplo, alunos criados pelo professor)? Pede ao teu **professor** "
               "(Painel do Professor > Repor palavra-passe) ou ao **administrador**.")


def _render_reset(token: str):
    st.markdown("### Escolhe uma palavra-passe nova")
    if not acesso.validar_reposicao(token):
        st.error("Este link já foi usado ou expirou. Pede um novo em «Esqueci a palavra-passe».")
    else:
        with st.form("form_reset"):
            n1 = st.text_input("Nova palavra-passe (mínimo 6)", type="password")
            n2 = st.text_input("Repete a nova", type="password")
            ok = st.form_submit_button("Guardar", type="primary")
        if ok:
            if len(n1) < 6 or n1 != n2:
                st.error("A palavra-passe deve ter 6+ caracteres e ser igual nas duas caixas.")
            elif acesso.aplicar_reposicao(token, n1):
                get_authenticator(force=True)
                st.success("Palavra-passe alterada. Já podes entrar.")
            else:
                st.error("Não foi possível alterar. Pede um link novo.")
    if st.button("Ir para o login", key="reset_voltar"):
        st.query_params.clear()
        st.rerun()


def render_mudar_pw_obrigatoria(username: str):
    """Primeiro acesso de um aluno criado pelo professor: tem de escolher a sua própria palavra-passe."""
    from modules.utils import page_header
    page_header("Escolhe a tua palavra-passe", "A palavra-passe que recebeste é temporária")
    with st.form("form_pw_obrig"):
        n1 = st.text_input("Nova palavra-passe (mínimo 6)", type="password")
        n2 = st.text_input("Repete a nova", type="password")
        ok = st.form_submit_button("Guardar e continuar", type="primary")
    if ok:
        if len(n1) < 6 or n1 != n2:
            st.error("A palavra-passe deve ter 6+ caracteres e ser igual nas duas caixas.")
        else:
            db.update_user(username, {"password_hash": _hash(n1), "must_change_pw": False})
            st.session_state["_pw_ok"] = username
            get_authenticator(force=True)
            st.rerun()


def render_login_page():
    # (o CSS já é injectado em app.py em cada execução; repeti-lo aqui duplicava ~900 linhas de CSS por clique)

    authenticator, config = get_authenticator()

    # Estado actual antes de qualquer tentativa de (re)autenticação.
    auth_status = st.session_state.get("authentication_status")

    # Primeira execução da sessão: se o navegador ainda traz um cookie de uma sessão já terminada com «Sair»,
    # o authenticator não o pode usar (senão actualizar a página reabria a conta).
    if auth_status is not True and "_df_guarda_cookie" not in st.session_state:
        st.session_state["_df_guarda_cookie"] = True
        try:
            bruto = st.context.cookies.get(config["cookie"]["name"])
            tk = _descodificar_cookie(bruto, config) if bruto else None
            if tk and _cookie_revogado(tk, config):
                st.session_state["_df_saiu"] = True
                st.session_state["logout"] = True
        except Exception:
            pass

    if auth_status is not True:
        g = _utilizador_google()
        if g:
            st.session_state["authentication_status"] = True
            st.session_state["username"], st.session_state["name"] = g
            auth_status = True

    if auth_status is not True and _login_por_cookie_do_navegador(authenticator, config):
        auth_status = True

    if auth_status is not True:
        # Ainda não autenticado: desenha o ecrã de login completo (cabeçalho,
        # formulário - que também tenta o cookie guardado - e mensagens).
        col1, col2, col3 = st.columns([1, 1.6, 1])
        with col2:
            render_seletor_idioma(None)
            st.markdown(f"""<div style=" display:flex; flex-direction:column; align-items:center; padding: 3rem 1rem 2rem; "><div style=" width:72px; height:72px; background: linear-gradient(135deg, {C_ACCENT}, #BC8CFF); border-radius:20px; display:flex; align-items:center; justify-content:center; font-size:22px; font-weight:900; color:var(--df-on-accent,#0D1117); letter-spacing:-1px; margin-bottom:1.2rem; box-shadow: 0 8px 32px rgba(88,166,255,0.3); font-family: Inter, sans-serif; ">DF</div><div style=" font-size:32px; font-weight:800; color:{C_TEXT}; letter-spacing:-.5px; margin-bottom:.4rem; ">DataForge EDU</div><div style=" font-size:15px; color:{C_TEXT_SEC}; font-weight:500; margin-bottom:2.4rem; ">Plataforma de Aprendizagem de Machine Learning</div><div style=" font-size:15px; font-weight:600; color:{C_TEXT_SEC}; margin-bottom:.8rem; text-transform:uppercase; letter-spacing:.06em; font-size:12px; ">Entra na tua conta</div></div>""", unsafe_allow_html=True)
            if st.query_params.get("reset"):
                _render_reset(st.query_params.get("reset"))
                return authenticator, st.session_state.get("authentication_status"), st.session_state.get("username"), config
            if st.query_params.get("convite"):
                # Link de convite: abre logo o formulário (o Streamlit não muda de separador sozinho)
                _render_registo()
                if st.button("Já tenho conta: ir para o login", key="convite_login"):
                    st.query_params.clear()
                    st.session_state.pop("reg_papel", None)
                    st.rerun()
                return authenticator, st.session_state.get("authentication_status"), st.session_state.get("username"), config
            tab_in, tab_novo, tab_esq = st.tabs(["Entrar", "Criar conta", "Esqueci a palavra-passe"])
            with tab_in:
                try:
                    authenticator.login(location="main", key="login_form")
                except Exception:
                    authenticator.login(location="main")
                auth_status = st.session_state.get("authentication_status")
                if auth_status is True:
                    st.session_state.pop("_df_saiu", None)
                if auth_status is True and not _tinha_cookie(config):
                    # Login novo pelo formulário: o cookie é escrito no navegador por um componente que só
                    # executa se a página ainda existir. Qualquer st.rerun() logo a seguir (tema, etc.)
                    # destruía-o antes de gravar e, ao recarregar, voltava o login. Damos-lhe tempo.
                    time.sleep(1.5)
                if auth_status is False:
                    st.markdown(f"""<div style=" background:rgba(248,81,73,0.1); border:1px solid rgba(248,81,73,0.3); border-left:3px solid #F85149; border-radius:8px; padding:.75rem 1rem; color:#F85149; font-size:14px; font-weight:600; margin-top:.8rem; ">Utilizador ou palavra-passe incorrectos.</div>""", unsafe_allow_html=True)
                if _google_activo():
                    st.markdown("---")
                    if st.button("Entrar com Google", key="btn_google", width="stretch"):
                        st.login("google")
                if auth_status is None and contas_demo():
                    st.markdown(f"""<div style=" background:rgba(88,166,255,0.08); border:1px solid rgba(88,166,255,0.2); border-radius:8px; padding:.65rem 1rem; font-size:13px; color:{C_TEXT_SEC}; text-align:center; margin-top:1rem; ">Conta demo: <code style=" background:rgba(88,166,255,0.15); color:{C_ACCENT}; padding:2px 6px; border-radius:4px; font-family:monospace; ">aluno</code>&nbsp;/&nbsp;<code style=" background:rgba(88,166,255,0.15); color:{C_ACCENT}; padding:2px 6px; border-radius:4px; font-family:monospace; ">aluno123</code></div>""", unsafe_allow_html=True)
            with tab_novo:
                _render_registo()
            with tab_esq:
                _render_esqueci()

        # Feature cards no rodapé (fora da coluna, largura própria)
        st.markdown(f"""<div style=" display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; max-width:420px; margin:2rem auto 0; "><div style=" background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:12px; padding:.9rem .8rem; text-align:center; "><div style="margin-bottom:8px;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#58A6FF" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg></div><div style="font-size:11px; font-weight:700; color:{C_TEXT_SEC}; line-height:1.4;">Teoria<br>+ Prática</div></div><div style=" background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:12px; padding:.9rem .8rem; text-align:center; "><div style="margin-bottom:8px;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D29922" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg></div><div style="font-size:11px; font-weight:700; color:{C_TEXT_SEC}; line-height:1.4;">Certificados<br>Verificáveis</div></div><div style=" background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:12px; padding:.9rem .8rem; text-align:center; "><div style="margin-bottom:8px;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#39D0C0" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg></div><div style="font-size:11px; font-weight:700; color:{C_TEXT_SEC}; line-height:1.4;">50+<br>Algoritmos</div></div></div><div style=" text-align:center; font-size:11px; color:{C_TEXT_MUTE}; margin-top:2rem; padding-bottom:1rem; ">DataForge EDU v1.0 · Feito em Angola 🇦🇴</div>""", unsafe_allow_html=True)
    else:
        # Já autenticado: apenas confirma/renova o estado via cookie, sem
        # desenhar absolutamente nada - assim não fica HTML de login
        # "escondido" no DOM por trás da sidebar (visível se ela colapsar).
        try:
            authenticator.login(location="unrendered")
        except Exception:
            pass

    auth_status = st.session_state.get("authentication_status")
    username    = st.session_state.get("username")
    return authenticator, auth_status, username, config


def render_sidebar_user(name: str, username: str, role: str, authenticator, produto: str = "DataForge EDU",
                        papel_rotulo: str = None, mostrar_pontos: bool = True, subtitulo: str = "Machine Learning"):
    """Cartão da pessoa no menu lateral. No painel Business não há pontos nem papel de aluno:
    `papel_rotulo` mostra o papel na empresa (Gestor, Analista, Membro) e `mostrar_pontos=False` esconde os pontos."""
    initial = name[0].upper() if name else "U"
    role_labels = {"admin": ("Admin", "red"), "professor": ("Professor", "amber"), "aluno": ("Aluno", "blue"),
                   "empresa": ("Empresa", "teal")}
    role_label, role_color = role_labels.get(role, ("Utilizador", "blue"))
    if papel_rotulo:
        role_label, role_color = papel_rotulo, "teal"
    pontos_html = ""
    if mostrar_pontos:
        pontos = get_user_progress(username).get("pontos", 0)
        pontos_html = f" &nbsp;⭐ {pontos} pts"
    st.markdown(f"""<div class="sidebar-brand"><div class="sidebar-brand-icon" style="font-size:15px;font-weight:900;color:var(--df-on-accent,#0D1117);letter-spacing:-1px;font-family:Inter,sans-serif;">DF</div><div><div class="sidebar-brand-name">{produto}</div><div class="sidebar-brand-sub">{subtitulo}</div></div></div><div class="user-pill"><div class="user-avatar">{initial}</div><div style="flex:1;min-width:0;"><div class="user-name">{name}</div><div class="user-role">{badge(role_label, role_color)}{pontos_html}</div></div></div>""", unsafe_allow_html=True)
    render_seletor_tema(username)
    render_seletor_idioma(username)
    if st.button("Sair", key="logout_btn", use_container_width=True):
        _terminar_sessao(username, authenticator)


def get_user_role(username: str, config: dict) -> str:
    try:
        return config["credentials"]["usernames"][username].get("role", "aluno")
    except Exception:
        try:
            u = db.get_user(username)
            return u.get("role", "aluno") if u else "aluno"
        except Exception:
            return "aluno"


# ══════════════════════════════════════════════════════
# PERFIL
# ══════════════════════════════════════════════════════
def render_perfil(username: str, name: str, role: str):
    from modules.utils import page_header
    page_header("O Meu Perfil", "Palavra-passe, turma e preferências")
    registo = db.get_user(username)
    demo = registo is None  # conta demo (só existe no config.yaml)

    st.markdown(f"**Nome:** {name}  \n**Utilizador:** {username}  \n**Papel:** {role}")

    st.markdown("### Preferências")
    tam = st.select_slider("Tamanho do texto", options=["Pequeno", "Normal", "Grande", "Muito grande"],
                           value=st.session_state.get("tamanho_texto", "Normal"), key="perfil_tam")
    st.session_state["tamanho_texto"] = tam
    render_seletor_tema_perfil(username)

    if demo:
        st.info("As contas demo não podem ser alteradas. Cria uma conta real para gravar a turma, o ranking e a palavra-passe.")
        return

    if role == "admin":
        _form_palavra_passe(registo)
        return

    if role == "aluno":
        st.markdown("### Turma")
        atual = registo.get("turma") or ""
        if atual:
            t = db.get_turma(atual)
            st.success(f"Estás na turma **{t['nome'] if t else atual}** (código {atual}).")
        for p in db.list_pedidos_turma(username=username):
            t = db.get_turma(p["codigo"])
            st.info(f"Pedido pendente para a turma **{t['nome'] if t else p['codigo']}**: aguarda a aprovação do professor.")
        cod = st.text_input("Entrar noutra turma (código dado pelo professor)", key="perfil_cod").strip().upper()
        c1, c2 = st.columns(2)
        if c1.button("Entrar na turma", key="perfil_turma_btn") and cod:
            estado, msg = acesso.entrar_na_turma(username, cod)
            if estado == "erro":
                st.error(msg)
            else:
                st.success(msg)
                if estado == "ok":
                    st.rerun()
        if atual and c2.button("Sair da turma", key="perfil_sair_turma"):
            db.update_user(username, {"turma": None})
            st.rerun()

    if role == "professor":
        st.markdown("### A minha escola")
        for o in acesso.escolas_do_dono(username):
            st.success(f"**{o['nome']}** · código de convite: **{o['codigo_convite']}** "
                       "(dá-o aos outros professores e aos alunos).")
        nome_escola = st.text_input("Criar uma escola nova", key="perfil_nova_escola", placeholder="Ex.: Instituto Politécnico X")
        if st.button("Criar escola", key="perfil_criar_escola") and nome_escola.strip():
            try:
                org = acesso.criar_escola(username, nome_escola)
                st.session_state.pop("_membros_cache", None)
                st.success(f"Escola criada! Código de convite: **{org['codigo_convite']}**.")
            except db.DBError as e:
                st.error(str(e))

    st.markdown("### Escola ou empresa")
    for m in db.membros_do_utilizador(username):
        st.markdown(f"- **{m['nome']}** ({m['tipo']}, {m['papel']})")
    cod_o = st.text_input("Juntar-me a uma escola ou empresa (código de convite)", key="perfil_cod_org").strip().upper()
    if st.button("Juntar-me", key="perfil_org_btn") and cod_o:
        from modules.admin import adicionar_membro_por_codigo
        ok_o, msg_o = adicionar_membro_por_codigo(username, cod_o, "professor" if role == "professor" else "aluno")
        if ok_o:
            st.session_state.pop("_membros_cache", None)
            st.success(msg_o + " Sai e volta a entrar para ver o painel novo.")
        else:
            st.error(msg_o)

    if role == "aluno":
        st.markdown("### Ranking da turma")
        rank = st.toggle("Aparecer no ranking da minha turma", value=bool(registo.get("ranking")), key="perfil_rank")
        if rank != bool(registo.get("ranking")):
            db.update_user(username, {"ranking": rank})
            st.success("Preferência guardada.")

    _form_palavra_passe(registo)


def _form_palavra_passe(registo):
    st.markdown("### Mudar palavra-passe")
    with st.form("form_pw"):
        atual_pw = st.text_input("Palavra-passe actual", type="password")
        nova = st.text_input("Nova palavra-passe (mínimo 6)", type="password")
        nova2 = st.text_input("Repete a nova", type="password")
        ok = st.form_submit_button("Mudar palavra-passe")
    if ok:
        if not bcrypt.checkpw(atual_pw.encode(), registo["password_hash"].encode()):
            st.error("A palavra-passe actual está errada.")
        elif len(nova) < 6 or nova != nova2:
            st.error("A nova palavra-passe deve ter 6+ caracteres e ser igual nas duas caixas.")
        else:
            db.update_user(username, {"password_hash": _hash(nova)})
            get_authenticator(force=True)
            st.success("Palavra-passe alterada.")
