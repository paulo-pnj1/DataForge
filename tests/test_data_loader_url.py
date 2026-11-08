"""Carregar por link (Google Sheets, GitHub, CSV público) e SQL de leitura: proteção do servidor e erros claros."""
import io
import ipaddress
import os
import socket
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_engine import data_loader as dl  # noqa: E402


class _Resp:
    def __init__(self, status=200, corpo=b"a,b\n1,2\n3,4\n", headers=None):
        self.status_code, self._corpo, self.headers = status, corpo, headers or {}

    def iter_content(self, n):
        for i in range(0, len(self._corpo), n):
            yield self._corpo[i:i + n]

    def close(self):
        pass


@pytest.fixture
def rede(monkeypatch):
    """DNS falso: cada nome aponta para o IP que o teste indicar; requests.get devolve o que o teste preparar."""
    estado = {"ips": {}, "respostas": [], "pedidos": []}

    def getaddrinfo(host, porta, *a, **k):
        try:                                                              # um IP escrito no link resolve para si próprio
            ipaddress.ip_address(host)
            ip = host
        except ValueError:
            ip = estado["ips"].get(host, "93.184.216.34")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, porta))]

    def get(url, **k):
        estado["pedidos"].append(url)
        return estado["respostas"].pop(0)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr("requests.get", get)
    return estado


@pytest.mark.parametrize("url", ["http://127.0.0.1/a.csv", "http://10.0.0.5/a.csv", "http://169.254.169.254/latest/meta-data",
                                 "http://[::1]/a.csv", "ftp://exemplo.com/a.csv", "http://localhost:8501/x", "https://u:p@exemplo.com/a.csv",
                                 "https://exemplo.com:6379/a.csv"])
def test_destinos_perigosos_sao_recusados(url, rede):
    rede["ips"]["localhost"] = "127.0.0.1"
    with pytest.raises(dl.ErroDados):
        dl.validar_destino(url)


def test_nome_que_aponta_para_rede_interna_e_recusado(rede):
    rede["ips"]["interno.exemplo.com"] = "192.168.1.10"
    with pytest.raises(dl.ErroDados, match="não é público"):
        dl.descarregar_url("https://interno.exemplo.com/dados.csv")
    assert rede["pedidos"] == []                                          # nem sequer pediu


def test_csv_publico_carrega(rede):
    rede["respostas"] = [_Resp()]
    df = dl.carregar_url("https://exemplo.com/dados.csv")
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_google_sheets_vira_exportacao_csv_com_gid(rede):
    rede["respostas"] = [_Resp()]
    dl.carregar_url("https://docs.google.com/spreadsheets/d/AbC-123/edit#gid=77")
    assert rede["pedidos"] == ["https://docs.google.com/spreadsheets/d/AbC-123/export?format=csv&gid=77"]


def test_github_blob_vira_raw():
    assert dl.converter_link("https://github.com/u/r/blob/main/x/y.csv")[0] == "https://raw.githubusercontent.com/u/r/main/x/y.csv"


def test_redirecionamento_para_rede_interna_e_travado(rede):
    rede["ips"]["interno.exemplo.com"] = "10.1.1.1"
    rede["respostas"] = [_Resp(302, b"", {"Location": "https://interno.exemplo.com/x.csv"})]
    with pytest.raises(dl.ErroDados, match="não é público"):
        dl.carregar_url("https://exemplo.com/dados.csv")
    assert rede["pedidos"] == ["https://exemplo.com/dados.csv"]


def test_demasiados_redirecionamentos(rede):
    rede["respostas"] = [_Resp(302, b"", {"Location": "https://exemplo.com/outro"}) for _ in range(dl.MAX_REDIRECIONAMENTOS + 1)]
    with pytest.raises(dl.ErroDados, match="redireciona"):
        dl.carregar_url("https://exemplo.com/dados.csv")


def test_ficheiro_demasiado_grande_e_cortado_a_meio(rede):
    rede["respostas"] = [_Resp(corpo=b"x" * 5000)]
    with pytest.raises(dl.ErroDados, match="mais de"):
        dl.descarregar_url("https://exemplo.com/dados.csv", max_bytes=1000)


@pytest.mark.parametrize("status,texto", [(403, "privado"), (401, "privado"), (404, "404"), (500, "500")])
def test_erros_http_tem_mensagem_clara(rede, status, texto):
    rede["respostas"] = [_Resp(status, b"")]
    with pytest.raises(dl.ErroDados, match=texto):
        dl.carregar_url("https://exemplo.com/dados.csv")


def test_pagina_html_nao_e_dados(rede):
    rede["respostas"] = [_Resp(corpo=b"<!DOCTYPE html><html>Entrar na conta</html>")]
    with pytest.raises(dl.ErroDados, match="página web"):
        dl.carregar_url("https://docs.google.com/spreadsheets/d/AbC/edit")


def test_excel_por_link_e_reconhecido_sem_extensao(rede):
    import pandas as pd
    buf = io.BytesIO()
    pd.DataFrame({"x": [1, 2], "z": [3, 4]}).to_excel(buf, index=False)
    rede["respostas"] = [_Resp(corpo=buf.getvalue())]
    df = dl.carregar_url("https://exemplo.com/baixar?id=7")
    assert list(df.columns) == ["x", "z"]


# ── SQL ──
def test_consulta_valida_passa():
    assert dl.validar_consulta("SELECT a, b FROM t WHERE c > 3;") == "SELECT a, b FROM t WHERE c > 3"
    assert dl.validar_consulta("with x as (select 1 as a) select * from x").startswith("with")


@pytest.mark.parametrize("q", ["", "DROP TABLE t", "select 1; drop table t", "select * from t -- x", "select /* x */ 1",
                               "update t set a = 1", "with a as (select 1) insert into t select * from a", "select * into novo from t"])
def test_consultas_de_escrita_ou_suspeitas_sao_recusadas(q):
    with pytest.raises(dl.ErroDados):
        dl.validar_consulta(q)


def test_sql_recusa_sqlite_e_servidor_interno(rede):
    pytest.importorskip("sqlalchemy")
    with pytest.raises(dl.ErroDados, match="PostgreSQL"):
        dl.carregar_sql("sqlite:///dados.db", "select 1")
    rede["ips"]["db.interno"] = "10.0.0.9"
    with pytest.raises(dl.ErroDados, match="não é público"):
        dl.carregar_sql("postgresql://u:p@db.interno/base", "select 1")
