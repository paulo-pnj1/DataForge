"""Carregamento de dados: ficheiro (CSV, TSV, TXT, Excel .xlsx), link público (incluindo Google Sheets e GitHub) e,
quando o administrador o ativa, uma consulta SQL só de leitura.

Reutiliza o leitor do painel Business (`business_core.ler_tabela`): separador automático, vírgula decimal,
datas reconhecidas e nomes de colunas limpos e únicos. Aqui acrescentam-se os limites de tamanho e mensagens claras.

Segurança (o servidor é que vai buscar os dados, por isso é ele que tem de se proteger):
- Links: só http/https, nas portas 80 e 443, e só para endereços públicos (nunca localhost, redes internas ou o
  endereço de metadados da cloud). Cada redirecionamento é validado de novo. O tamanho é limitado enquanto se descarrega.
- SQL: desligado por omissão (segredo `permitir_sql`). Só uma consulta SELECT/WITH, sem comentários nem palavras de escrita,
  só PostgreSQL/MySQL/SQL Server e só para endereços públicos. Não substitui uma conta de base de dados só de leitura.
"""
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

EXTENSOES = ("csv", "tsv", "txt", "xlsx", "xlsm")
MAX_BYTES = 50 * 1024 * 1024
MAX_LINHAS_PADRAO = 200_000


class ErroDados(ValueError):
    """Problema com os dados (mensagem já escrita para o utilizador)."""


def extensao(nome: str) -> str:
    return (nome or "").lower().rsplit(".", 1)[-1] if "." in (nome or "") else ""


def carregar_ficheiro(nome: str, dados: bytes, folha=None, max_bytes: int = MAX_BYTES) -> pd.DataFrame:
    """Lê um ficheiro carregado. Levanta `ErroDados` com uma mensagem clara se não puder ser usado."""
    if not dados:
        raise ErroDados("O ficheiro está vazio.")
    if len(dados) > max_bytes:
        raise ErroDados(f"O ficheiro tem {len(dados) / 1e6:.0f} MB. O máximo é {max_bytes / 1e6:.0f} MB.")
    ext = extensao(nome)
    if ext == "xls":
        raise ErroDados("O formato antigo .xls não é suportado. Abre o ficheiro no Excel e guarda como .xlsx ou .csv.")
    if ext not in EXTENSOES:
        raise ErroDados(f"Formato «.{ext or '?'}» não suportado. Usa CSV, TSV, TXT ou Excel (.xlsx).")
    from modules.business_core import ler_tabela
    try:
        return ler_tabela(nome, dados, folha)
    except ErroDados:
        raise
    except ValueError as e:
        raise ErroDados(str(e))


def folhas_excel(nome: str, dados: bytes) -> list:
    """Nomes das folhas de um Excel (lista vazia para os outros formatos)."""
    if extensao(nome) not in ("xlsx", "xlsm"):
        return []
    from modules.business_core import folhas_excel as _folhas
    return _folhas(dados)


def amostrar(df: pd.DataFrame, max_linhas: int = MAX_LINHAS_PADRAO, random_state: int = 42):
    """Se a tabela for muito grande, treina-se numa amostra aleatória. Devolve (tabela, aviso ou None)."""
    if max_linhas and len(df) > max_linhas:
        amostra = df.sample(n=max_linhas, random_state=random_state).sort_index().reset_index(drop=True)
        return amostra, (f"A tabela tem {len(df):,} linhas; para ser rápido e caber na memória usa-se uma amostra aleatória "
                         f"de {max_linhas:,}.".replace(",", " "))
    return df, None


# ══════════════════════════════════════════════════════════════════════════════
# Links (CSV/Excel públicos, Google Sheets, GitHub)
# ══════════════════════════════════════════════════════════════════════════════
MAX_REDIRECIONAMENTOS = 3
TIMEOUT_LIGACAO = (5, 20)                       # (ligar, ler) em segundos
PORTAS_PERMITIDAS = (None, 80, 443)
_GOOGLE_SHEETS = re.compile(r"docs\.google\.com/spreadsheets/d/([\w-]+)")
_GID = re.compile(r"[?&#]gid=(\d+)")
_GITHUB_BLOB = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/blob/(.+)$")


def converter_link(url: str) -> tuple:
    """(link para descarregar, nome sugerido ou None). Converte Google Sheets e GitHub (blob) em ligações diretas."""
    url = (url or "").strip()
    m = _GOOGLE_SHEETS.search(url)
    if m:
        g = _GID.search(url)
        return (f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv" + (f"&gid={g.group(1)}" if g else ""),
                "google_sheets.csv")
    m = _GITHUB_BLOB.match(url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}", None
    return url, None


def _ip_publico(ip_texto: str) -> bool:
    ip = ipaddress.ip_address(ip_texto)
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global


def validar_destino(url: str) -> None:
    """Levanta `ErroDados` se o link não for http(s) para um endereço público."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ErroDados("O link tem de começar por http:// ou https://.")
    if p.username or p.password:
        raise ErroDados("Links com utilizador e palavra-passe não são aceites.")
    try:
        porta = p.port
    except ValueError:
        raise ErroDados("O link tem uma porta inválida.")
    if porta not in PORTAS_PERMITIDAS:
        raise ErroDados("Só se aceitam links nas portas normais da web (80 e 443).")
    try:
        infos = socket.getaddrinfo(p.hostname, porta or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ErroDados("Não consegui encontrar o endereço deste link. Confirma que está certo.")
    if not infos or not all(_ip_publico(i[4][0]) for i in infos):
        raise ErroDados("Este endereço não é público (rede interna ou local), por isso não o posso abrir.")


def descarregar_url(url: str, max_bytes: int = MAX_BYTES) -> tuple:
    """Descarrega um link público. Devolve (nome do ficheiro, bytes). Levanta `ErroDados` com mensagem clara."""
    import requests
    atual, nome_sugerido = converter_link(url)
    if not atual:
        raise ErroDados("Cola um link.")
    for _ in range(MAX_REDIRECIONAMENTOS + 1):
        validar_destino(atual)
        try:
            r = requests.get(atual, stream=True, timeout=TIMEOUT_LIGACAO, allow_redirects=False,
                             headers={"User-Agent": "DataForge-EDU/1.0"})
        except requests.RequestException:
            raise ErroDados("Não consegui abrir o link (sem resposta ou demorou demasiado).")
        try:
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                atual = urljoin(atual, r.headers["Location"])
                continue
            if r.status_code in (401, 403):
                raise ErroDados("O link é privado. Partilha-o como «Qualquer pessoa com o link» e tenta de novo.")
            if r.status_code == 404:
                raise ErroDados("O link não existe (erro 404).")
            if r.status_code != 200:
                raise ErroDados(f"O servidor respondeu com o erro {r.status_code}.")
            partes, total = [], 0
            for bloco in r.iter_content(65536):
                total += len(bloco)
                if total > max_bytes:
                    raise ErroDados(f"O ficheiro do link tem mais de {max_bytes / 1e6:.0f} MB. O máximo é {max_bytes / 1e6:.0f} MB.")
                partes.append(bloco)
        finally:
            r.close()
        dados = b"".join(partes)
        if not dados:
            raise ErroDados("O link devolveu um ficheiro vazio.")
        if dados[:512].lstrip().lower().startswith((b"<!doctype html", b"<html")):
            raise ErroDados("O link abre uma página web e não um ficheiro de dados. Se é uma folha do Google, "
                            "partilha-a como «Qualquer pessoa com o link».")
        nome = nome_sugerido or (urlparse(atual).path.rsplit("/", 1)[-1] or "")
        if extensao(nome) not in EXTENSOES:
            nome = "dados.xlsx" if dados[:2] == b"PK" else "dados.csv"
        return nome, dados
    raise ErroDados("O link redireciona demasiadas vezes.")


def carregar_url(url: str, folha=None) -> pd.DataFrame:
    """Lê uma tabela de um link público (CSV, Excel, Google Sheets, GitHub)."""
    nome, dados = descarregar_url(url)
    return carregar_ficheiro(nome, dados, folha)


# ══════════════════════════════════════════════════════════════════════════════
# SQL (só leitura; desligado por omissão)
# ══════════════════════════════════════════════════════════════════════════════
DRIVERS_SQL = ("postgresql", "mysql", "mariadb", "mssql")
_SQL_PROIBIDO = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|attach|detach|pragma|merge|into)\b", re.I)


def validar_consulta(consulta: str) -> str:
    """Devolve a consulta limpa se for um único SELECT/WITH sem comentários nem palavras de escrita."""
    q = (consulta or "").strip().rstrip(";").strip()
    if not q:
        raise ErroDados("Escreve a consulta SQL.")
    if ";" in q:
        raise ErroDados("Só é permitida uma consulta de cada vez (sem «;» no meio).")
    if "--" in q or "/*" in q:
        raise ErroDados("Comentários SQL não são permitidos.")
    if not re.match(r"(?is)^(select|with)\b", q):
        raise ErroDados("Só se aceitam consultas de leitura (SELECT ou WITH).")
    if _SQL_PROIBIDO.search(q):
        raise ErroDados("A consulta tem uma palavra de escrita (INSERT, UPDATE, DELETE, DROP, INTO…). Só se pode ler.")
    return q


def carregar_sql(ligacao: str, consulta: str, max_linhas: int = MAX_LINHAS_PADRAO) -> pd.DataFrame:
    """Executa uma consulta SELECT numa base de dados (PostgreSQL, MySQL/MariaDB ou SQL Server) e devolve a tabela."""
    q = validar_consulta(consulta)
    try:
        import sqlalchemy
        from sqlalchemy.engine import make_url
    except ImportError:
        raise ErroDados("Falta o pacote «sqlalchemy» no servidor para ler de bases de dados SQL.")
    try:
        url = make_url(ligacao.strip())
    except Exception:
        raise ErroDados("A ligação à base de dados não está bem escrita (exemplo: postgresql://utilizador:senha@servidor/base).")
    if url.get_backend_name() not in DRIVERS_SQL:
        raise ErroDados("Só se aceitam bases PostgreSQL, MySQL/MariaDB e SQL Server.")
    if not url.host:
        raise ErroDados("A ligação tem de indicar o servidor da base de dados.")
    try:
        publico = all(_ip_publico(i[4][0]) for i in socket.getaddrinfo(url.host, url.port or 0, proto=socket.IPPROTO_TCP))
    except socket.gaierror:
        raise ErroDados("Não consegui encontrar o servidor da base de dados.")
    if not publico:
        raise ErroDados("O servidor da base de dados não é público, por isso não me posso ligar.")
    motor = None
    try:
        motor = sqlalchemy.create_engine(url, connect_args={"connect_timeout": 10} if url.get_backend_name() != "mssql" else {})
        partes, total = [], 0
        with motor.connect() as con:
            for bloco in pd.read_sql_query(sqlalchemy.text(q), con, chunksize=5000):
                partes.append(bloco)
                total += len(bloco)
                if total >= max_linhas:
                    break
    except Exception as e:
        raise ErroDados("A base de dados recusou a consulta ou a ligação: " + " ".join(str(e).split())[:200])
    finally:
        if motor is not None:
            motor.dispose()
    if not partes or total == 0:
        raise ErroDados("A consulta não devolveu nenhuma linha.")
    df = pd.concat(partes, ignore_index=True).head(max_linhas)
    df.columns = [str(c) for c in df.columns]
    return df
