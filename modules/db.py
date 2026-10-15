"""
DataForge EDU - Camada de dados
Usa o Supabase (API REST/PostgREST) quando há credenciais em st.secrets;
caso contrário usa TinyDB local (data/db.json) para desenvolvimento.

Segurança: usa a chave service_role, que fica SÓ no servidor (st.secrets).
Activa RLS nas tabelas sem políticas (ver schema.sql) para a chave pública ficar sem acesso.
"""
import json
import datetime
import os

import requests
from tinydb import TinyDB

from modules.settings import secret, usando_supabase


class DBError(Exception):
    pass


def _agora() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ══════════════════════════════════════════════════════
# BACKENDS
# ══════════════════════════════════════════════════════
def _sb(method: str, tabela: str, params=None, body=None, prefer=None):
    base = str(secret("supabase_url")).rstrip("/")
    chave = str(secret("supabase_key"))
    headers = {"apikey": chave, "Authorization": f"Bearer {chave}", "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    try:
        r = requests.request(method, f"{base}/rest/v1/{tabela}", params=params,
                             data=json.dumps(body) if body is not None else None, headers=headers, timeout=15)
    except requests.RequestException as e:
        raise DBError(f"Sem ligação à base de dados: {e}")
    if r.status_code >= 400:
        raise DBError(f"Supabase {r.status_code}: {r.text[:300]}")
    return r.json() if r.text.strip() else None


def _local(tabela: str):
    os.makedirs("data", exist_ok=True)
    return TinyDB("data/db.json").table(tabela)


def _select(tabela: str, order=None, desc=False, limit=None, **eq) -> list:
    if usando_supabase():
        params = {"select": "*"}
        for k, v in eq.items():
            params[k] = f"eq.{str(v).lower() if isinstance(v, bool) else v}"
        if order:
            params["order"] = f"{order}.{'desc' if desc else 'asc'}"
        if limit:
            params["limit"] = str(limit)
        return _sb("GET", tabela, params=params) or []
    rows = [dict(r) for r in _local(tabela).all() if all(r.get(k) == v for k, v in eq.items())]
    if order:
        rows.sort(key=lambda r: str(r.get(order, "")), reverse=desc)
    return rows[:limit] if limit else rows


def _insert(tabela: str, row: dict):
    if usando_supabase():
        _sb("POST", tabela, body=row, prefer="return=minimal")
    else:
        _local(tabela).insert(dict(row))


def _upsert(tabela: str, row: dict, chave: str):
    if usando_supabase():
        _sb("POST", tabela, params={"on_conflict": chave}, body=row, prefer="resolution=merge-duplicates,return=minimal")
    else:
        t = _local(tabela)
        existentes = t.search(lambda r: r.get(chave) == row[chave])
        if existentes:
            t.update(dict(row), lambda r: r.get(chave) == row[chave])
        else:
            t.insert(dict(row))


def _update(tabela: str, eq: dict, campos: dict):
    if usando_supabase():
        _sb("PATCH", tabela, params={k: f"eq.{v}" for k, v in eq.items()}, body=campos, prefer="return=minimal")
    else:
        _local(tabela).update(dict(campos), lambda r: all(r.get(k) == v for k, v in eq.items()))


def _delete(tabela: str, eq: dict):
    if usando_supabase():
        _sb("DELETE", tabela, params={k: f"eq.{v}" for k, v in eq.items()}, prefer="return=minimal")
    else:
        _local(tabela).remove(lambda r: all(r.get(k) == v for k, v in eq.items()))


# ══════════════════════════════════════════════════════
# UTILIZADORES
# ══════════════════════════════════════════════════════
def get_user(username: str):
    r = _select("users", username=username)
    return r[0] if r else None


def list_users(**eq) -> list:
    return _select("users", **eq)


def create_user(row: dict):
    row = dict(row)
    row.setdefault("created_at", _agora())
    _insert("users", row)


def update_user(username: str, campos: dict):
    _update("users", {"username": username}, campos)


# ══════════════════════════════════════════════════════
# PROGRESSO (um documento JSON por utilizador)
# ══════════════════════════════════════════════════════
def get_progress(username: str) -> dict:
    rows = _select("progress", username=username)
    if not rows:
        return {}
    r = rows[0]
    if "data" in r and isinstance(r["data"], dict):
        return r["data"]
    return {k: v for k, v in r.items() if k not in ("username", "updated_at")}  # formato antigo (TinyDB)


def save_progress(username: str, data: dict):
    _upsert("progress", {"username": username, "data": data, "updated_at": _agora()}, "username")


def list_progress(usernames: list) -> dict:
    """Progresso de vários utilizadores: {username: dict}."""
    if not usernames:
        return {}
    out = {}
    if usando_supabase():
        lista = ",".join(usernames)
        rows = _sb("GET", "progress", params={"select": "*", "username": f"in.({lista})"}) or []
    else:
        rows = [dict(r) for r in _local("progress").all() if r.get("username") in usernames]
    for r in rows:
        out[r["username"]] = r["data"] if isinstance(r.get("data"), dict) else {k: v for k, v in r.items() if k != "username"}
    return out


# ══════════════════════════════════════════════════════
# HISTORIAL DE MODELOS
# ══════════════════════════════════════════════════════
def add_historico(username: str, entrada: dict):
    _insert("historico", {"username": username, "data": entrada, "created_at": _agora()})


def get_historico(username: str, limite: int = 200) -> list:
    rows = _select("historico", order="created_at", desc=True, limit=limite, username=username)
    rows = list(reversed(rows))
    return [r["data"] if isinstance(r.get("data"), dict) else r for r in rows]


# ══════════════════════════════════════════════════════
# TURMAS, CERTIFICADOS, ERROS
# ══════════════════════════════════════════════════════
def create_turma(codigo: str, nome: str, professor: str, org_id=None):
    row = {"codigo": codigo, "nome": nome, "professor": professor, "created_at": _agora()}
    if org_id:
        row["org_id"] = org_id
    _insert("turmas", row)


def get_turma(codigo: str):
    r = _select("turmas", codigo=codigo)
    return r[0] if r else None


def list_turmas(professor=None) -> list:
    return _select("turmas", professor=professor) if professor else _select("turmas")


def save_cert(row: dict):
    row = dict(row)
    row.setdefault("emitido_em", _agora())
    _upsert("certificados", row, "codigo")


def get_cert(codigo: str):
    r = _select("certificados", codigo=codigo)
    return r[0] if r else None


def list_certs(username: str) -> list:
    return _select("certificados", username=username)


def log_erro(username: str, pagina: str, mensagem: str):
    try:
        _insert("erros", {"username": username or "", "pagina": pagina or "", "mensagem": str(mensagem)[:1500], "created_at": _agora()})
    except Exception:
        pass


def list_erros(limite: int = 200) -> list:
    return _select("erros", order="created_at", desc=True, limit=limite)


# ══════════════════════════════════════════════════════
# ORGANIZAÇÕES (escolas e empresas) E MEMBROS
# ══════════════════════════════════════════════════════
TIPOS_ORG = {"escola": "edu", "empresa": "business"}                 # tipo -> produto
PAPEIS_ORG = {"escola": ["professor", "aluno", "gestor"], "empresa": ["gestor", "analista", "membro"]}


def create_org(org_id: str, nome: str, tipo: str, codigo_convite: str, dono=None, estado=None, activa: bool = True):
    """estado: 'pendente' | 'aprovada' | 'recusada' (vazio = organização antiga, tratada como aprovada)."""
    if tipo not in TIPOS_ORG:
        raise DBError(f"Tipo de organização inválido: {tipo}")
    row = {"id": org_id, "nome": nome, "tipo": tipo, "produto": TIPOS_ORG[tipo],
           "codigo_convite": codigo_convite, "activa": activa, "created_at": _agora()}
    if dono:
        row["dono"] = dono
    if estado:
        row["estado"] = estado
    _insert("organizacoes", row)


def get_org(org_id: str):
    r = _select("organizacoes", id=org_id)
    return r[0] if r else None


def get_org_por_codigo(codigo: str):
    r = _select("organizacoes", codigo_convite=(codigo or "").strip().upper())
    return r[0] if r else None


def list_orgs(**eq) -> list:
    return _select("organizacoes", order="created_at", desc=True, **eq)


def update_org(org_id: str, campos: dict):
    _update("organizacoes", {"id": org_id}, campos)


def add_membro(org_id: str, username: str, papel: str):
    org = get_org(org_id)
    if not org:
        raise DBError("Organização inexistente.")
    if papel not in PAPEIS_ORG[org["tipo"]]:
        raise DBError(f"Papel '{papel}' inválido para uma {org['tipo']}.")
    _upsert_membro({"org_id": org_id, "username": username, "papel": papel, "created_at": _agora()})


def _upsert_membro(row: dict):
    """Chave composta (org_id, username)."""
    if usando_supabase():
        _sb("POST", "membros", params={"on_conflict": "org_id,username"}, body=row,
            prefer="resolution=merge-duplicates,return=minimal")
    else:
        _local("membros").upsert(dict(row), lambda r: r.get("org_id") == row["org_id"] and r.get("username") == row["username"])


def remove_membro(org_id: str, username: str):
    _delete("membros", {"org_id": org_id, "username": username})


def list_membros(org_id=None, username=None) -> list:
    eq = {}
    if org_id:
        eq["org_id"] = org_id
    if username:
        eq["username"] = username
    return _select("membros", **eq)


def membros_do_utilizador(username: str) -> list:
    """Pertenças activas de uma pessoa: [{org_id, papel, tipo, produto, nome}]."""
    out = []
    for m in list_membros(username=username):
        org = get_org(m["org_id"])
        if org and org.get("activa", True):
            out.append({"org_id": org["id"], "papel": m["papel"], "tipo": org["tipo"],
                        "produto": org["produto"], "nome": org["nome"],
                        "codigo_convite": org["codigo_convite"]})
    return out


# ══════════════════════════════════════════════════════
# TURMAS (aprovação de entradas) E PEDIDOS
# ══════════════════════════════════════════════════════
def update_turma(codigo: str, campos: dict):
    _update("turmas", {"codigo": codigo}, campos)


def add_pedido_turma(codigo: str, username: str):
    _upsert("pedidos_turma", {"id": f"{codigo}:{username}", "codigo": codigo, "username": username,
                              "created_at": _agora()}, "id")


def list_pedidos_turma(codigo=None, username=None) -> list:
    eq = {}
    if codigo:
        eq["codigo"] = codigo
    if username:
        eq["username"] = username
    return _select("pedidos_turma", order="created_at", **eq)


def delete_pedido_turma(codigo: str, username: str):
    _delete("pedidos_turma", {"id": f"{codigo}:{username}"})


# ══════════════════════════════════════════════════════
# CONVITES DE PROFESSOR (uso único)
# ══════════════════════════════════════════════════════
def create_convite(row: dict):
    row = dict(row)
    row.setdefault("created_at", _agora())
    _insert("convites_professor", row)


def get_convite(codigo: str):
    r = _select("convites_professor", codigo=codigo)
    return r[0] if r else None


def list_convites(limite: int = 200) -> list:
    return _select("convites_professor", order="created_at", desc=True, limit=limite)


def update_convite(codigo: str, campos: dict):
    _update("convites_professor", {"codigo": codigo}, campos)


# ══════════════════════════════════════════════════════
# REPOSIÇÃO DE PALAVRA-PASSE (só se guarda o hash do token)
# ══════════════════════════════════════════════════════
def save_reset(token_hash: str, username: str, expira_em: str):
    _insert("reset_tokens", {"token_hash": token_hash, "username": username, "expira_em": expira_em,
                             "usado": False, "created_at": _agora()})


def get_reset(token_hash: str):
    r = _select("reset_tokens", token_hash=token_hash)
    return r[0] if r else None


def update_reset(token_hash: str, campos: dict):
    _update("reset_tokens", {"token_hash": token_hash}, campos)


# ══════════════════════════════════════════════════════
# EMAILS (registo, contador diário e fila)
# ══════════════════════════════════════════════════════
def log_email(row: dict):
    row = dict(row)
    row.setdefault("created_at", _agora())
    _insert("emails", row)


def list_emails(limite: int = 2000, **eq) -> list:
    return _select("emails", order="created_at", desc=True, limit=limite, **eq)


def update_email(email_id: str, campos: dict):
    _update("emails", {"id": email_id}, campos)
