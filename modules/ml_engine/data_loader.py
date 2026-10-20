"""Carregamento de dados (CSV, TSV, TXT, Excel .xlsx).

Reutiliza o leitor do painel Business (`business_core.ler_tabela`): separador automático, vírgula decimal,
datas reconhecidas e nomes de colunas limpos e únicos. Aqui acrescentam-se os limites de tamanho e mensagens claras.
"""
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
