"""
DataForge - Envio de emails por SMTP (Brevo, Gmail, Zoho...).
- Configurado em secrets: smtp_host, smtp_port, smtp_user, smtp_password, email_from.
- Respeita o limite diário (email_limite_diario); o que passa do limite fica em fila.
- Nunca bloqueia o fluxo: devolve o estado e quem chamou decide o que mostrar.
"""
import datetime
import secrets
import smtplib
import ssl
from email.message import EmailMessage

from modules import db
from modules.settings import secret


def configurado() -> bool:
    return all(secret(k) for k in ("smtp_host", "smtp_user", "smtp_password", "email_from"))


def limite_diario() -> int:
    try:
        return int(secret("email_limite_diario", 300))
    except (TypeError, ValueError):
        return 300


def _hoje() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def enviados_hoje() -> int:
    try:
        return sum(1 for e in db.list_emails(estado="enviado") if (e.get("enviado_em") or "")[:10] == _hoje())
    except Exception:
        return 0


def em_fila() -> int:
    try:
        return len(db.list_emails(estado="fila"))
    except Exception:
        return 0


def _limpar(txt: str) -> str:
    return " ".join(str(txt or "").split())


def _smtp(para: str, assunto: str, texto: str):
    """Envia mesmo. Devolve (ok, erro)."""
    try:
        msg = EmailMessage()
        nome = _limpar(secret("email_from_nome", "DataForge"))
        msg["From"] = f"{nome} <{secret('email_from')}>"
        msg["To"] = _limpar(para)
        msg["Subject"] = _limpar(assunto)
        msg.set_content(texto)
        host, porta = str(secret("smtp_host")), int(secret("smtp_port", 587))
        ctx = ssl.create_default_context()
        if porta == 465:
            with smtplib.SMTP_SSL(host, porta, context=ctx, timeout=15) as s:
                s.login(str(secret("smtp_user")), str(secret("smtp_password")))
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, porta, timeout=15) as s:
                s.starttls(context=ctx)
                s.login(str(secret("smtp_user")), str(secret("smtp_password")))
                s.send_message(msg)
        return True, ""
    except Exception as e:   # rede, autenticação, remetente não verificado...
        return False, str(e)[:300]


def _registar(para, assunto, corpo, estado, erro=""):
    try:
        db.log_email({"id": secrets.token_hex(8), "para": para, "assunto": assunto, "corpo": corpo,
                      "estado": estado, "erro": erro,
                      "enviado_em": db._agora() if estado == "enviado" else None})
    except Exception:
        pass


def enviar(para: str, assunto: str, texto: str):
    """Devolve (estado, mensagem). Estados: enviado | fila | falhou | nao_configurado."""
    para = (para or "").strip()
    if not para:
        return "falhou", "Sem destinatário."
    if not configurado():
        return "nao_configurado", "O envio de email ainda não está configurado."
    if enviados_hoje() >= limite_diario():
        _registar(para, assunto, texto, "fila")
        return "fila", "Limite diário atingido: o email fica na fila e sai quando houver quota."
    ok, erro = _smtp(para, assunto, texto)
    if ok:
        _registar(para, assunto, texto, "enviado")
        return "enviado", "Email enviado."
    _registar(para, assunto, texto, "falhou", erro)
    return "falhou", f"Não consegui enviar: {erro}"


def processar_fila() -> dict:
    """Envia o que ficou em fila enquanto houver quota. Devolve {'enviados': n, 'restam': n}."""
    enviados = 0
    for e in reversed(db.list_emails(estado="fila")):          # mais antigos primeiro
        if enviados_hoje() >= limite_diario():
            break
        ok, erro = _smtp(e["para"], e["assunto"], e.get("corpo") or "")
        db.update_email(e["id"], {"estado": "enviado", "enviado_em": db._agora(), "erro": ""} if ok
                        else {"estado": "falhou", "erro": erro})
        enviados += ok
    return {"enviados": enviados, "restam": em_fila()}


def emails_admin() -> list:
    lista = []
    if secret("admin_email"):
        lista.append(str(secret("admin_email")).strip())
    try:
        lista += [u["email"] for u in db.list_users(role="admin") if u.get("email")]
    except Exception:
        pass
    return list(dict.fromkeys(e for e in lista if e))


def enviar_a_admins(assunto: str, texto: str):
    for e in emails_admin():
        enviar(e, assunto, texto)


# ══════════════════════════════════════════════════════
# MODELOS DE EMAIL (assunto, texto)
# ══════════════════════════════════════════════════════
_ASS = "\n\n— Equipa DataForge"


def t_convite_professor(codigo: str, link: str, dias: int):
    corpo = (f"Olá!\n\nFoste convidado(a) para ser professor(a) no DataForge.\n\n"
             f"Código de convite: {codigo}\n")
    if link:
        corpo += f"Link directo: {link}\n"
    corpo += (f"\nAbre o DataForge, vai a «Criar conta», escolhe «Sou professor» e usa este mesmo email. "
              f"O convite é de uso único e válido por {dias} dias.")
    return "Convite para professor no DataForge", corpo + _ASS


def t_empresa_pendente_admin(nome_empresa: str, dono: str, email: str):
    return ("Empresa a aguardar aprovação no DataForge",
            f"Há uma empresa nova à espera de aprovação.\n\nEmpresa: {nome_empresa}\nResponsável: {dono}\n"
            f"Email: {email}\n\nEntra em Administração > Organizações para aprovar ou recusar." + _ASS)


def t_empresa_aprovada(nome_empresa: str, codigo: str):
    return ("A tua empresa foi aprovada no DataForge",
            f"Boas notícias! A empresa «{nome_empresa}» foi aprovada.\n\nCódigo de convite para os teus colegas: {codigo}\n\n"
            "Volta a entrar no DataForge para aceder ao painel Business." + _ASS)


def t_empresa_recusada(nome_empresa: str, motivo: str):
    return ("Sobre o teu pedido de empresa no DataForge",
            f"O pedido para a empresa «{nome_empresa}» não foi aprovado.\n\nMotivo: {motivo or 'não indicado'}\n\n"
            "Podes entrar no DataForge e submeter um novo pedido." + _ASS)


def t_aluno_criado(nome: str, utilizador: str, pw_temp: str, turma_nome: str):
    return ("A tua conta no DataForge",
            f"Olá, {nome}!\n\nO teu professor criou-te uma conta na turma {turma_nome}.\n\n"
            f"Utilizador: {utilizador}\nPalavra-passe temporária: {pw_temp}\n\n"
            "No primeiro acesso vais ter de escolher uma palavra-passe nova." + _ASS)


def t_pedido_turma(aluno: str, turma_nome: str):
    return ("Pedido de entrada na tua turma",
            f"{aluno} pediu para entrar na turma «{turma_nome}».\n\nEntra no Painel do Professor para aceitar ou recusar." + _ASS)


def t_repor_pw(link: str):
    return ("Repor a palavra-passe do DataForge",
            f"Pediste para repor a palavra-passe.\n\nAbre este link (válido 1 hora, uso único):\n{link}\n\n"
            "Se não foste tu, ignora este email." + _ASS)
