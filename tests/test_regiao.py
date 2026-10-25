"""Região (país): Angola, Moçambique, Brasil e Portugal. Angola é o ponto de partida e não pode mudar nada."""
import os
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


def _app(corpo: str, regiao: str, lang: str = "pt"):
    script = (f"import sys\nsys.path.insert(0, {RAIZ!r})\nimport streamlit as st\n"
              "from modules.i18n_patch import instalar\ninstalar()\n"
              f"st.session_state['regiao'] = {regiao!r}\nst.session_state['lang'] = {lang!r}\n" + corpo)
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _loc(texto, cod, idioma="pt"):
    from modules import regiao
    regiao.st.session_state["regiao"] = cod
    return regiao.localizar(texto, idioma)


# ── texto ──────────────────────────────────────────────────────────────────
def test_angola_nao_muda_nada():
    t = "Escolhe o ficheiro. O projecto custa 4.500 Kz em Luanda, ecrã e acção. Alunos em Angola."
    assert _loc(t, "AO") == t


def test_moeda_cidade_e_pais():
    t = "Casas em Luanda (fictício): preço em milhões de Kz. Alunos em Angola. Taxista angolano: 4.500 Kz."
    assert _loc(t, "BR") == "Casas em São Paulo (fictício): preço em mil R$. Alunos no Brasil. Taxista brasileiro: R$ 4.500."
    assert "Maputo" in _loc(t, "MZ") and "mil MT" in _loc(t, "MZ") and "moçambicano" in _loc(t, "MZ")
    pt = _loc(t, "PT")
    assert "Lisboa" in pt and "mil €" in pt and "4.500 €" in pt and "em Portugal" in pt


def test_ortografia_do_acordo_menos_em_angola():
    t = "O projecto tem uma acção óptima."
    assert _loc(t, "AO") == t
    for cod in ("MZ", "PT", "BR"):
        assert _loc(t, cod) == "O projeto tem uma ação ótima."
    assert _loc("A detecção", "PT") == "A deteção" and _loc("A detecção", "MZ") == "A deteção"
    assert _loc("A detecção", "BR") == "A detecção"        # no Brasil o «c» pronuncia-se


def test_portugal_e_mocambique_mantem_o_tu():
    t = "Escolhe um algoritmo e clica em Treinar. Tens o teu ficheiro?"
    assert _loc(t, "PT") == t and _loc(t, "MZ") == t


def test_brasil_usa_voce_e_vocabulario_brasileiro():
    r = _loc("Escolhe um algoritmo e clica em Treinar. Tens de carregar um ficheiro no teu telemóvel.", "BR")
    assert r == "Escolha um algoritmo e clique em Treinar. Você precisa enviar um arquivo no seu celular."
    assert _loc("Vê o gráfico no ecrã e usa o separador.", "BR") == "Veja o gráfico na tela e use a aba."
    assert _loc("Acertaste! Já tens o resultado, queres continuar?", "BR") == "Você acertou! Você já tem o resultado, quer continuar?"


def test_brasil_nao_converte_descricoes_na_terceira_pessoa():
    """«Procura a fronteira...» e «Dá intervalos...» descrevem um algoritmo: não são ordens ao aluno."""
    for d in ("Dá intervalos de previsão.", "Procura uma «faixa» que contenha os dados; boa com poucos exemplos.",
              "Rede especializada em imagens: procura padrões como cantos."):
        assert _loc(d, "BR") == d
    assert _loc("Treina o teu primeiro modelo", "BR") == "Treine o seu primeiro modelo"      # com «teu» é ordem ao aluno
    assert _loc("Treina o modelo", "BR") == "Treine o modelo"                                 # rótulo curto de botão


def test_brasil_acrescenta_o_sujeito_que_o_portugues_europeu_dispensa():
    assert _loc("Tens poucos dados. Terminaste o projecto final.", "BR") == "Você tem poucos dados. Você terminou o projeto final."
    assert _loc("Os modelos que já treinaste.", "BR") == "Os modelos que você já treinou."
    assert _loc("Tu não estás a aparecer no ranking.", "BR") == "Você não está aparecendo no ranking."


def test_brasil_imperativo_depois_de_adverbio_de_abertura():
    assert _loc("Já viste a teoria. Agora experimenta na prática.", "BR") == "Você já viu a teoria. Agora experimente na prática."


def test_brasil_so_converte_imperativos_no_inicio_da_frase():
    """«treina e testa» no meio de uma frase é a 3.ª pessoa: não pode virar «treina e teste»."""
    assert _loc("O modelo treina e testa. O tutor vê o teu ecrã.", "BR") == "O modelo treina e testa. O tutor vê a sua tela."


def test_brasil_nao_parte_palavras_ao_meio():
    """Regressão: «a ti» apanhava o meio de «tinha» e «tirar» e dava «vocênha» e «vocêrar»."""
    assert _loc("nunca tinha visto. Para tirar dúvidas, fala para ti.", "BR") == "nunca tinha visto. Para tirar dúvidas, fala para você."
    assert _loc("Não tens de esperar. Já tens o ficheiro?", "BR") == "Você não precisa esperar. Você já tem o arquivo?"


def test_nome_dos_paises_no_seletor_nao_se_adapta():
    from modules.regiao import NOMES_PARA_SELETOR as N
    assert _loc(N["AO"], "BR") == N["AO"] and _loc(N["BR"], "PT") == N["BR"] and _loc(N["AO"], "BR", "en") == N["AO"]
    assert not any(ord(ch) >= 0x1F1E6 and ord(ch) <= 0x1F1FF for n in N.values() for ch in n)      # sem bandeiras emoji


def test_etiquetas_html_nao_se_tocam():
    r = _loc("<div class='ecrã' style='x'><b>Usa</b> o ficheiro e escolhe o separador.</div>", "BR")
    assert r == "<div class='ecrã' style='x'><b>Use</b> o arquivo e escolha a aba.</div>"
    assert _loc("<a href='https://x.pt/ficheiro?ecrã=1'>Escolhe</a>", "BR") == "<a href='https://x.pt/ficheiro?ecrã=1'>Escolha</a>"


def test_ingles_tambem_adapta_moeda_e_cidade():
    assert _loc("Houses in Luanda: price in million Kz (4,500 Kz)", "BR", "en") == "Houses in São Paulo: price in thousand R$ (R$ 4,500)"
    assert _loc("Angolan students in Angola", "PT", "en") == "Portuguese students in Portugal"


# ── datasets de exemplo ───────────────────────────────────────────────────
@pytest.mark.parametrize("cod", ["AO", "MZ", "BR", "PT"])
def test_datasets_de_exemplo_por_regiao(cod):
    from modules import regiao
    from modules.utils import _gerar_casas_luanda, _gerar_banca, _gerar_alunos
    c = regiao.REGIOES[cod]
    casas = _gerar_casas_luanda(regiao=cod)
    assert set(casas["bairro"]) <= set(c["bairros"]) and "preco" in casas.columns and casas["preco"].min() >= 8 * c["fator_casas"] - 1
    assert "Kz" not in " ".join(casas.columns) or cod == "AO"
    banca = _gerar_banca(regiao=cod)
    assert {"preco_do_kg", "vendas_dia"} <= set(banca.columns) and banca["vendas_dia"].min() > 0
    alunos = _gerar_alunos(regiao=cod)
    assert alunos["nota_anterior"].max() <= c["nota_max"]


def test_angola_continua_com_os_mesmos_numeros():
    """Impressões digitais dos datasets antes de existirem regiões: só os nomes das colunas mudaram."""
    from modules.utils import _gerar_casas_luanda, _gerar_banca, _gerar_alunos
    a, b, c = _gerar_casas_luanda(), _gerar_banca(), _gerar_alunos()
    assert a["preco"].head(3).tolist() == [114.7, 124.0, 81.0] and round(float(a["preco"].sum()), 1) == 31536.5
    assert b["vendas_dia"].head(3).tolist() == [34300.0, 31800.0, 30500.0] and float(b["vendas_dia"].sum()) == 11341600.0
    assert b["preco_do_kg"].head(3).tolist() == [295, 287, 339]
    assert c["nota_anterior"].head(3).tolist() == [9.3, 10.0, 16.6] and round(float(c["nota_anterior"].sum()), 1) == 5452.2
    assert int((c["passou_de_ano"] == "sim").sum()) == 260


def test_brasil_usa_notas_de_0_a_10_e_mais_nada_muda_no_resultado():
    from modules.utils import _gerar_alunos
    ao, br = _gerar_alunos(regiao="AO"), _gerar_alunos(regiao="BR")
    assert br["nota_anterior"].max() <= 10 < ao["nota_anterior"].max()
    assert (ao["passou_de_ano"] == br["passou_de_ano"]).all()       # o mesmo aluno passa ou chumba: só a escala muda


def test_cache_do_dataset_depende_da_regiao():
    at = _app("from modules.utils import carregar_dataset_embutido\n"
              "df, desc, _ = carregar_dataset_embutido('Casas em Luanda (fictício) - Preços')\n"
              "st.text(','.join(sorted(df['bairro'].unique())[:3]))", "BR")
    assert "Capão Redondo" in at.text[0].value or "Butantã" in at.text[0].value
    at2 = _app("from modules.utils import carregar_dataset_embutido\n"
               "df, desc, _ = carregar_dataset_embutido('Casas em Luanda (fictício) - Preços')\n"
               "st.text(','.join(sorted(df['bairro'].unique())[:3]))", "AO")
    assert "Cacuaco" in at2.text[0].value or "Benfica" in at2.text[0].value


# ── ligação ao resto do sistema ───────────────────────────────────────────
def test_streamlit_mostra_o_texto_adaptado():
    at = _app("st.markdown('Escolhe um ficheiro no teu telemóvel')\nst.button('Treina o modelo')", "BR")
    assert at.markdown[0].value == "Escolha um arquivo no seu celular"
    assert at.button[0].label == "Treine o modelo"
    at2 = _app("st.markdown('Escolhe um ficheiro no teu telemóvel')", "AO")
    assert at2.markdown[0].value == "Escolhe um ficheiro no teu telemóvel"


def test_ingles_com_regiao():
    at = _app("st.markdown('Casas em Luanda (fictício) - Preços')", "BR", "en")
    assert at.markdown[0].value == "Houses in São Paulo (fictional) - Prices"


def test_ia_responde_na_variante_do_pais_e_a_cache_nao_mistura_paises():
    at = _app("from modules import ia\nst.text(ia._idioma_pedido('pt') + '|' + ia._chave_cache('x'))", "BR")
    var_br, chave_br = at.text[0].value.split("|")
    at2 = _app("from modules import ia\nst.text(ia._idioma_pedido('pt') + '|' + ia._chave_cache('x'))", "AO")
    var_ao, chave_ao = at2.text[0].value.split("|")
    assert "Brasil" in var_br and "Angola" in var_ao and chave_br != chave_ao


@pytest.mark.parametrize("cod", ["AO", "MZ", "BR", "PT"])
def test_licoes_com_dados_da_regiao(cod):
    from modules import regiao
    at = _app("from modules import escola\nescola._licao_2()\nescola._licao_4()\nescola._licao_13()", cod)
    todo = " ".join(m.value for m in at.markdown) + " ".join(str(d.value) for d in at.dataframe)
    assert regiao.REGIOES[cod]["bairros_licao"][0] in todo or regiao.REGIOES[cod]["bairros_licao"][0] in " ".join(w.value for w in at.warning + at.success + at.info)


def test_regiao_fica_guardada_na_sessao_e_na_conta():
    at = _app("from modules.utils import definir_regiao\ndefinir_regiao('teste', 'MZ')\nst.text(st.session_state['regiao'])", "AO")
    assert at.text[0].value == "MZ"
