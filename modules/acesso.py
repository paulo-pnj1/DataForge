"""
DataForge - Regras de acesso (sem interface).
Convites de professor, escolas, empresas (aprovadas pelo admin), turmas, alunos e repor palavra-passe.
O admin do sistema NUNCA cria turmas/organizações nem entra nelas: aqui só ele aprova/recusa empresas.
"""
import datetime
import hashlib
import re
import secrets
import unicodedata

import bcrypt

from modules import db, email_service
from modules.settings import secret, app_url

ALFABETO = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"       # sem 0/O/1/I para não haver confusões
VALIDADE_CONVITE_DIAS = 7
VALIDADE_RESET_HORAS = 1
MAX_ALUNOS_POR_LOTE = 200
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
UTILIZADOR_RE = re.compile(r"^[a-z0-9_]{3,20}$")
RESERVADOS = {"admin", "professor", "aluno"}


def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def _agora() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _dt(iso):
    try:
        d = datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return None


def _codigo_curto(n: int = 6) -> str:
    return "".join(secrets.choice(ALFABETO) for _ in range(n))


def codigo_novo() -> str:
    """Código único para turmas e organizações (nunca colide com os existentes)."""
    while True:
        c = _codigo_curto(6)
        if not db.get_org_por_codigo(c) and not db.get_turma(c):
            return c


# ══════════════════════════════════════════════════════
# CONVITES DE PROFESSOR (criados pelo admin, uso único)
# ══════════════════════════════════════════════════════
def normalizar_convite(codigo: str) -> str:
    c = re.sub(r"[^A-Za-z0-9]", "", codigo or "").upper()
    return f"{c[:4]}-{c[4:]}" if len(c) == 8 else (codigo or "").strip().upper()


def link_convite(codigo: str) -> str:
    return f"{app_url()}/?convite={codigo}" if app_url() else ""


def criar_convite_professor(email: str, criado_por: str):
    """Cria o convite e tenta enviá-lo. Devolve (codigo, estado_email, mensagem_email)."""
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise db.DBError("Escreve um email válido.")
    raw = _codigo_curto(8)
    codigo = f"{raw[:4]}-{raw[4:]}"
    expira = (_agora() + datetime.timedelta(days=VALIDADE_CONVITE_DIAS)).isoformat()
    db.create_convite({"codigo": codigo, "email": email, "criado_por": criado_por, "expira_em": expira, "cancelado": False})
    assunto, corpo = email_service.t_convite_professor(codigo, link_convite(codigo), VALIDADE_CONVITE_DIAS)
    estado, msg = email_service.enviar(email, assunto, corpo)
    return codigo, estado, msg


def estado_convite(c: dict) -> str:
    if c.get("cancelado"):
        return "Cancelado"
    if c.get("usado_por"):
        return "Usado"
    exp = _dt(c.get("expira_em"))
    if exp and exp < _agora():
        return "Expirado"
    return "Pendente"


def validar_convite(codigo: str, email: str):
    """Devolve (convite, mensagem_de_erro)."""
    c = db.get_convite(normalizar_convite(codigo))
    if not c:
        return None, "Convite de professor inválido."
    est = estado_convite(c)
    if est != "Pendente":
        return None, f"Este convite está {est.lower()}. Pede um novo ao administrador."
    if (email or "").strip().lower() != (c.get("email") or "").lower():
        return None, "Este convite foi enviado para outro email. Usa o email que recebeu o convite."
    return c, ""


def cancelar_convite(codigo: str):
    db.update_convite(codigo, {"cancelado": True})


# ══════════════════════════════════════════════════════
# PEDIDOS DE CONVITE (professor novo, sem falar com o admin)
# ══════════════════════════════════════════════════════
MAX_PEDIDOS_PENDENTES = 100       # trava contra spam: acima disto não se aceitam mais pedidos novos


def pedir_convite(nome: str, email: str, escola: str = "", mensagem: str = ""):
    """Regista o pedido e avisa o admin. Devolve (ok, mensagem). Um pedido pendente por email."""
    nome = " ".join((nome or "").split())
    email = (email or "").strip().lower()
    escola = " ".join((escola or "").split())[:120]
    mensagem = (mensagem or "").strip()[:300]
    if len(nome) < 3:
        return False, "Escreve o teu nome completo."
    if not EMAIL_RE.match(email):
        return False, "Escreve um email válido: é para lá que enviamos o convite."
    if len(escola) < 3:
        return False, "Escreve o nome da escola ou instituição onde ensinas."
    pendentes = db.list_pedidos_convite(estado="pendente")
    if any((p.get("email") or "").lower() == email for p in pendentes):
        return False, "Já tens um pedido a aguardar aprovação. Assim que for aprovado, o convite chega ao teu email."
    if len(pendentes) >= MAX_PEDIDOS_PENDENTES:
        return False, "Há muitos pedidos em espera neste momento. Tenta novamente daqui a uns dias."
    db.create_pedido_convite({"id": secrets.token_hex(6), "nome": nome, "email": email, "escola": escola,
                              "mensagem": mensagem, "estado": "pendente"})
    email_service.enviar_a_admins(*email_service.t_pedido_convite_admin(nome, email, escola, mensagem))
    return True, ("Pedido enviado! O administrador vai analisá-lo e o convite chega ao email que indicaste "
                  "(confirma também o spam).")


def pedidos_convite_pendentes() -> list:
    return sorted(db.list_pedidos_convite(estado="pendente"), key=lambda p: str(p.get("created_at", "")))


def aprovar_pedido_convite(pedido_id: str, admin: str):
    """Cria o convite (uso único, ligado ao email do pedido) e envia-o. Devolve (codigo, estado_email, mensagem_email)."""
    p = db.get_pedido_convite(pedido_id)
    if not p or p.get("estado") != "pendente":
        raise db.DBError("Este pedido já foi tratado.")
    codigo, estado, msg = criar_convite_professor(p["email"], admin)
    db.update_pedido_convite(pedido_id, {"estado": "aprovado", "codigo_convite": codigo,
                                         "decidido_por": admin, "decidido_em": db._agora()})
    return codigo, estado, msg


def recusar_pedido_convite(pedido_id: str, admin: str, motivo: str = ""):
    p = db.get_pedido_convite(pedido_id)
    if not p or p.get("estado") != "pendente":
        raise db.DBError("Este pedido já foi tratado.")
    motivo = (motivo or "").strip()[:300]
    db.update_pedido_convite(pedido_id, {"estado": "recusado", "motivo": motivo,
                                         "decidido_por": admin, "decidido_em": db._agora()})
    email_service.enviar(p["email"], *email_service.t_pedido_convite_recusado(p.get("nome", ""), motivo))


# ══════════════════════════════════════════════════════
# ESCOLAS (criadas pelo professor) E EMPRESAS (aprovadas pelo admin)
# ══════════════════════════════════════════════════════
def criar_escola(username: str, nome: str):
    nome = (nome or "").strip()
    if len(nome) < 3:
        raise db.DBError("Escreve o nome da escola.")
    u = db.get_user(username) or {}
    if u.get("role") != "professor":
        raise db.DBError("Só professores podem criar uma escola.")
    org_id = secrets.token_hex(4)
    db.create_org(org_id, nome, "escola", codigo_novo(), dono=username, estado="aprovada")
    db.add_membro(org_id, username, "gestor")
    return db.get_org(org_id)


def escolas_do_dono(username: str) -> list:
    return [o for o in db.list_orgs(dono=username) if o["tipo"] == "escola"]


def criar_empresa_pendente(username: str, nome: str):
    """A empresa nasce PENDENTE e inactiva; só fica activa quando o admin aprova."""
    nome = (nome or "").strip()
    if len(nome) < 3:
        raise db.DBError("Escreve o nome da empresa.")
    org_id = secrets.token_hex(4)
    db.create_org(org_id, nome, "empresa", codigo_novo(), dono=username, estado="pendente", activa=False)
    db.add_membro(org_id, username, "gestor")
    return db.get_org(org_id)


def empresa_do_dono(username: str):
    """Pedido de empresa mais recente do utilizador que ainda não foi aprovado (pendente ou recusada)."""
    for o in db.list_orgs(dono=username):                # mais recentes primeiro
        if o["tipo"] == "empresa" and o.get("estado") in ("pendente", "recusada"):
            return o
    return None


def orgs_pendentes() -> list:
    return [o for o in db.list_orgs() if o.get("estado") == "pendente"]


def _email_do_dono(org: dict) -> str:
    return (db.get_user(org.get("dono") or "") or {}).get("email") or ""


def aprovar_org(org_id: str):
    org = db.get_org(org_id)
    if not org:
        raise db.DBError("Organização inexistente.")
    db.update_org(org_id, {"estado": "aprovada", "activa": True, "motivo": None})
    if _email_do_dono(org):
        email_service.enviar(_email_do_dono(org), *email_service.t_empresa_aprovada(org["nome"], org["codigo_convite"]))


def recusar_org(org_id: str, motivo: str = ""):
    org = db.get_org(org_id)
    if not org:
        raise db.DBError("Organização inexistente.")
    db.update_org(org_id, {"estado": "recusada", "activa": False, "motivo": (motivo or "").strip()[:300]})
    if _email_do_dono(org):
        email_service.enviar(_email_do_dono(org), *email_service.t_empresa_recusada(org["nome"], motivo))


def novo_pedido_empresa(username: str, nome: str):
    """Depois de recusada, o dono pode submeter outro pedido (o antigo é apagado)."""
    antigo = empresa_do_dono(username)
    if antigo and antigo.get("estado") == "recusada":
        db.remove_membro(antigo["id"], username)
        db._delete("organizacoes", {"id": antigo["id"]})
    return criar_empresa_pendente(username, nome)


# ══════════════════════════════════════════════════════
# TURMAS: entrada com aprovação do professor
# ══════════════════════════════════════════════════════
def entrar_na_turma(username: str, codigo: str):
    """Devolve (estado, mensagem): 'ok' | 'pendente' | 'erro'."""
    codigo = (codigo or "").strip().upper()
    t = db.get_turma(codigo)
    if not t:
        return "erro", "Código de turma não encontrado."
    if t.get("aprovacao", True):
        db.add_pedido_turma(codigo, username)
        prof = db.get_user(t["professor"]) or {}
        if prof.get("email"):
            u = db.get_user(username) or {}
            email_service.enviar(prof["email"], *email_service.t_pedido_turma(u.get("name", username), t["nome"]))
        return "pendente", f"Pedido enviado: aguarda a aprovação do professor para entrares na turma {t['nome']}."
    db.update_user(username, {"turma": codigo})
    return "ok", f"Já estás na turma {t['nome']}!"


def aceitar_pedido(codigo: str, username: str):
    db.update_user(username, {"turma": codigo})
    db.delete_pedido_turma(codigo, username)


def recusar_pedido(codigo: str, username: str):
    db.delete_pedido_turma(codigo, username)


# ══════════════════════════════════════════════════════
# ALUNOS CRIADOS PELO PROFESSOR (em lote)
# ══════════════════════════════════════════════════════
def _slug(nome: str) -> str:
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    partes = [p for p in s.split("_") if p]
    base = "_".join([partes[0], partes[-1]]) if len(partes) > 1 else (partes[0] if partes else "aluno")
    return (base[:15] or "aluno").ljust(3, "0")


def criar_alunos(professor: str, turma_codigo: str, linhas: list, enviar_email: bool = False) -> list:
    """Cada linha: 'Nome' ou 'Nome, email'. Devolve [{nome, utilizador, palavra_passe, email, erro}]."""
    t = db.get_turma(turma_codigo)
    if not t or t["professor"] != professor:
        raise db.DBError("Só podes criar alunos nas tuas turmas.")
    linhas = [l.strip() for l in linhas if l.strip()][:MAX_ALUNOS_POR_LOTE]
    out, usados = [], set()
    for linha in linhas:
        nome, _, mail = linha.partition(",")
        nome, mail = nome.strip(), mail.strip().lower()
        if len(nome) < 3:
            out.append({"nome": nome, "utilizador": "", "palavra_passe": "", "email": mail, "erro": "Nome demasiado curto."})
            continue
        if mail and not EMAIL_RE.match(mail):
            out.append({"nome": nome, "utilizador": "", "palavra_passe": "", "email": mail, "erro": "Email inválido."})
            continue
        base, user, n = _slug(nome), None, 1
        user = base
        while user in usados or user in RESERVADOS or db.get_user(user):
            n += 1
            user = f"{base[:17]}{n}"
        usados.add(user)
        pw = secrets.token_urlsafe(6)
        db.create_user({"username": user, "name": nome, "email": mail, "password_hash": hash_pw(pw),
                        "role": "aluno", "turma": turma_codigo, "must_change_pw": True})
        if enviar_email and mail:
            email_service.enviar(mail, *email_service.t_aluno_criado(nome, user, pw, t["nome"]))
        out.append({"nome": nome, "utilizador": user, "palavra_passe": pw, "email": mail, "erro": ""})
    return out


# ══════════════════════════════════════════════════════
# REPOR PALAVRA-PASSE POR EMAIL
# ══════════════════════════════════════════════════════
def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def pedir_reposicao(identificador: str) -> None:
    """Resposta neutra para quem pede (não revela se a conta existe). Só envia se houver email e app_url."""
    ident = (identificador or "").strip().lower()
    if not ident:
        return
    user = db.get_user(ident) if UTILIZADOR_RE.match(ident) else None
    if not user and EMAIL_RE.match(ident):
        encontrados = db.list_users(email=ident)
        user = encontrados[0] if len(encontrados) == 1 else None
    if not user or not user.get("email") or not app_url():
        return
    token = secrets.token_urlsafe(24)
    db.save_reset(_h(token), user["username"], (_agora() + datetime.timedelta(hours=VALIDADE_RESET_HORAS)).isoformat())
    email_service.enviar(user["email"], *email_service.t_repor_pw(f"{app_url()}/?reset={token}"))


def validar_reposicao(token: str):
    r = db.get_reset(_h(token or ""))
    if not r or r.get("usado"):
        return None
    exp = _dt(r.get("expira_em"))
    return r["username"] if exp and exp > _agora() else None


def aplicar_reposicao(token: str, nova: str) -> bool:
    user = validar_reposicao(token)
    if not user or len(nova or "") < 6:
        return False
    db.update_user(user, {"password_hash": hash_pw(nova), "must_change_pw": False})
    db.update_reset(_h(token), {"usado": True})
    return True


# ══════════════════════════════════════════════════════
# REGISTO (formulário «Criar conta»)
# ══════════════════════════════════════════════════════
def registar(papel, nome, utilizador, email, pw, pw2, turma="", cod_org="", cod_convite="",
             empresa_modo="nova", empresa_nome="", empresa_cod="", reservados=()):
    """papel: 'aluno' | 'professor' | 'empresa'. Devolve (ok, mensagem)."""
    utilizador = (utilizador or "").strip().lower()
    nome = (nome or "").strip()
    email = (email or "").strip().lower()
    if papel not in ("aluno", "professor", "empresa"):
        return False, "Escolhe quem és."
    if len(nome) < 3:
        return False, "Escreve o teu nome completo."
    if not UTILIZADOR_RE.match(utilizador):
        return False, "O utilizador deve ter 3 a 20 caracteres: letras minúsculas, números ou _ (sem espaços nem acentos)."
    if db.get_user(utilizador) or utilizador in set(reservados) or utilizador in RESERVADOS:
        return False, "Esse utilizador já existe. Escolhe outro."
    if email and not EMAIL_RE.match(email):
        return False, "O email não parece válido."
    if papel in ("professor", "empresa") and not email:
        return False, "O email é obrigatório para professores e empresas."
    if len(pw or "") < 6:
        return False, "A palavra-passe deve ter pelo menos 6 caracteres."
    if pw != pw2:
        return False, "As palavras-passe não coincidem."

    role, turma, org, convite, nova_empresa = "aluno", (turma or "").strip().upper(), None, None, None
    cod_org = (cod_org or "").strip().upper()

    if papel == "aluno":
        if turma and not db.get_turma(turma):
            return False, "Não encontrei nenhuma turma com esse código. Confirma com o teu professor ou deixa vazio."
        if cod_org:
            org = db.get_org_por_codigo(cod_org)
            if not org or not org.get("activa", True):
                return False, "Não encontrei nenhuma escola ou empresa com esse código. Confirma-o ou deixa vazio."
    elif papel == "professor":
        turma = ""
        codigo = (cod_convite or "").strip()
        if secret("admin_code") and codigo == str(secret("admin_code")):
            if any(u.get("role") == "admin" for u in db.list_users()):
                return False, "O código de administrador só serve para criar o primeiro administrador. Pede um convite de professor."
            role = "admin"
        else:
            convite, erro = validar_convite(codigo, email)
            if erro:
                return False, erro
            role = "professor"
        if cod_org and role != "admin":
            org = db.get_org_por_codigo(cod_org)
            if not org or not org.get("activa", True) or org["tipo"] != "escola":
                return False, "Não encontrei nenhuma escola activa com esse código. Confirma-o ou deixa vazio."
    else:  # empresa
        turma = ""
        if empresa_modo == "nova":
            if len((empresa_nome or "").strip()) < 3:
                return False, "Escreve o nome da empresa."
        else:
            org = db.get_org_por_codigo(empresa_cod)
            if not org or not org.get("activa", True) or org["tipo"] != "empresa":
                return False, "Não encontrei nenhuma empresa activa com esse código."

    db.create_user({"username": utilizador, "name": nome, "email": email, "password_hash": hash_pw(pw),
                    "role": role, "turma": None})
    if convite:
        db.update_convite(convite["codigo"], {"usado_por": utilizador, "usado_em": db._agora()})

    aviso = ""
    if papel == "empresa" and empresa_modo == "nova":
        nova_empresa = criar_empresa_pendente(utilizador, empresa_nome)
        email_service.enviar_a_admins(*email_service.t_empresa_pendente_admin(nova_empresa["nome"], nome, email))
        return True, ("Conta criada. A tua empresa aguarda a aprovação do administrador: "
                      "vais receber um email quando for aprovada. Já podes entrar no separador 'Entrar'.")
    if org:
        from modules.admin import adicionar_membro_por_codigo
        adicionar_membro_por_codigo(utilizador, org["codigo_convite"],
                                    "professor" if role == "professor" else ("membro" if org["tipo"] == "empresa" else "aluno"))
    if papel == "aluno" and turma:
        estado, msg = entrar_na_turma(utilizador, turma)
        aviso = " " + msg if estado == "pendente" else ""
    return True, f"Conta criada ({role}).{aviso} Já podes entrar no separador 'Entrar'."
