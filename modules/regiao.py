"""Região (país) do aluno: Angola, Moçambique, Brasil ou Portugal.

O que muda com a região:
  - moeda, cidade e bairros dos datasets de exemplo (casas, banca de rua, alunos) e das lições;
  - gentílico e «em/de + país» nos textos;
  - a variante do português: Brasil usa «você», «tela», «arquivo»...; Moçambique, Portugal e Brasil usam a ortografia
    do Acordo de 1990 (projeto, ação), enquanto Angola mantém a que o sistema já tinha (projecto, acção);
  - a variante e os exemplos do dia-a-dia que a IA do tutor usa nas respostas.

A região NÃO muda a lógica do sistema: os nomes internos dos datasets (por exemplo «Casas em Luanda (fictício) - Preços»)
continuam os mesmos; só o texto mostrado ao aluno é adaptado (por `localizar`, chamada em `i18n.tr`).
Angola é o ponto de partida: para Angola `localizar` não altera nada.
"""
import re

import streamlit as st

REGIAO_PADRAO = "AO"

# ══════════════════════════════════════════════════════
# CONFIGURAÇÃO POR PAÍS
# ══════════════════════════════════════════════════════
REGIOES = {
    "AO": {
        "nome": "Angola", "nome_en": "Angola", "em": "em Angola", "de": "de Angola", "bandeira": "🇦🇴",
        "gentilicio": ("angolano", "angolanos", "angolana", "angolanas"), "gentilicio_en": "Angolan",
        "moeda": "Kz", "cidade": "Luanda",
        "unidade_casas": "milhões de Kz", "unidade_casas_en": "million Kz", "fator_casas": 1.0,
        "bairros": {
            "Ingombota": (25, 2), "Miramar": (30, 4), "Maianga": (10, 5), "Talatona": (20, 15),
            "Kilamba": (8, 25), "Benfica": (3, 18), "Camama": (0, 14), "Cazenga": (-5, 10),
            "Viana": (-6, 20), "Cacuaco": (-8, 22),
        },
        "bairros_licao": ["Talatona", "Maianga", "Viana", "Cazenga", "Kilamba", "Benfica", "Ingombota", "Camama"],
        "taxi": "para o Kilamba são uns 4.500 Kz",
        "fator_preco_kg": 1.0, "fator_vendas": 1.0, "casas_vendas": 0,
        "salario": {"fator": 1.0, "unidade": "milhares de Kz"},
        "nota_max": 20,
        "banca": "Banca de rua",
        "lingua": "português de Angola (europeu, sem brasileirismos)",
        "exemplos": "de Angola: mercado, táxi, loja, escola",
        "variante": "ao", "temp_media": 29,
    },
    "MZ": {
        "nome": "Moçambique", "nome_en": "Mozambique", "em": "em Moçambique", "de": "de Moçambique", "bandeira": "🇲🇿",
        "gentilicio": ("moçambicano", "moçambicanos", "moçambicana", "moçambicanas"), "gentilicio_en": "Mozambican",
        "moeda": "MT", "cidade": "Maputo",
        "unidade_casas": "mil MT", "unidade_casas_en": "thousand MT", "fator_casas": 70.0,
        "bairros": {
            "Polana": (30, 3), "Sommerschield": (28, 4), "Costa do Sol": (20, 9), "Coop": (14, 4),
            "Alto-Maé": (10, 3), "Malhangalene": (8, 4), "Matola": (2, 15), "Machava": (-4, 12),
            "Zimpeto": (-6, 14), "Magoanine": (-8, 16),
        },
        "bairros_licao": ["Costa do Sol", "Coop", "Zimpeto", "Machava", "Matola", "Alto-Maé", "Polana", "Magoanine"],
        "taxi": "para a Matola são uns 150 MT",
        "fator_preco_kg": 0.2, "fator_vendas": 0.2,
        "salario": {"fator": 100.0, "unidade": "MT"},
        "nota_max": 20,
        "banca": "Banca de rua",
        "lingua": "português de Moçambique (norma europeia, sem brasileirismos)",
        "exemplos": "de Moçambique: mercado, chapa (o minibus), loja, escola",
        "variante": "mz", "temp_media": 27,
    },
    "BR": {
        "nome": "Brasil", "nome_en": "Brazil", "em": "no Brasil", "de": "do Brasil", "bandeira": "🇧🇷",
        "gentilicio": ("brasileiro", "brasileiros", "brasileira", "brasileiras"), "gentilicio_en": "Brazilian",
        "moeda": "R$", "cidade": "São Paulo",
        "unidade_casas": "mil R$", "unidade_casas_en": "thousand R$", "fator_casas": 10.0,
        "bairros": {
            "Pinheiros": (30, 6), "Moema": (28, 9), "Vila Mariana": (22, 6), "Tatuapé": (12, 8),
            "Mooca": (10, 5), "Santana": (8, 7), "Butantã": (6, 10), "Itaquera": (-4, 20),
            "Capão Redondo": (-8, 22), "Guaianases": (-9, 26),
        },
        "bairros_licao": ["Moema", "Vila Mariana", "Itaquera", "Capão Redondo", "Butantã", "Tatuapé", "Pinheiros", "Guaianases"],
        "taxi": "para a Vila Madalena são uns R$ 35",
        "fator_preco_kg": 0.02, "fator_vendas": 0.008,
        "salario": {"fator": 10.0, "unidade": "R$"},
        "nota_max": 10,
        "banca": "Barraca de feira",
        "lingua": "português do Brasil (use «você», «tela», «arquivo» e a ortografia brasileira; evite expressões de Portugal)",
        "exemplos": "do Brasil: feira, ônibus, padaria, escola",
        "variante": "br", "temp_media": 25,
    },
    "PT": {
        "nome": "Portugal", "nome_en": "Portugal", "em": "em Portugal", "de": "de Portugal", "bandeira": "🇵🇹",
        "gentilicio": ("português", "portugueses", "portuguesa", "portuguesas"), "gentilicio_en": "Portuguese",
        "moeda": "€", "cidade": "Lisboa",
        "unidade_casas": "mil €", "unidade_casas_en": "thousand €", "fator_casas": 5.0,
        "bairros": {
            "Chiado": (30, 1), "Estrela": (26, 2), "Parque das Nações": (22, 7), "Alvalade": (14, 4),
            "Benfica": (8, 6), "Odivelas": (0, 10), "Amadora": (-3, 12), "Loures": (-5, 14),
            "Sacavém": (-6, 11), "Cacém": (-8, 18),
        },
        "bairros_licao": ["Parque das Nações", "Alvalade", "Amadora", "Loures", "Odivelas", "Benfica", "Chiado", "Sacavém"],
        "taxi": "para o Parque das Nações são uns 12 €",
        "fator_preco_kg": 0.01, "fator_vendas": 0.004,
        "salario": {"fator": 6.0, "unidade": "€"},
        "nota_max": 20,
        "banca": "Banca de mercado",
        "lingua": "português de Portugal (europeu)",
        "exemplos": "de Portugal: mercado, metro, loja, escola",
        "variante": "pt", "temp_media": 19,
    },
}

NOMES_PARA_SELETOR = {c: f"{r['bandeira']} {r['nome']}" for c, r in REGIOES.items()}


# ══════════════════════════════════════════════════════
# ESTADO (sessão)
# ══════════════════════════════════════════════════════
def regiao_actual() -> str:
    try:
        r = st.session_state.get("regiao", REGIAO_PADRAO)
    except Exception:
        return REGIAO_PADRAO
    return r if r in REGIOES else REGIAO_PADRAO


def cfg(codigo: str = None) -> dict:
    return REGIOES[codigo or regiao_actual()]


def adapta() -> bool:
    """True se a região obriga a mexer nos textos (tudo menos Angola, que é o ponto de partida)."""
    return regiao_actual() != REGIAO_PADRAO


def definir_regiao_sessao(codigo: str):
    st.session_state["regiao"] = codigo if codigo in REGIOES else REGIAO_PADRAO
    st.session_state["_regiao_sel"] = NOMES_PARA_SELETOR[st.session_state["regiao"]]
    try:
        st.query_params["pais"] = st.session_state["regiao"].lower()
    except Exception:
        pass


def iniciar_regiao():
    """No início de cada execução: lê ?pais=br na primeira visita."""
    if "regiao" not in st.session_state:
        q = None
        try:
            q = st.query_params.get("pais")
        except Exception:
            pass
        q = str(q).upper() if q else ""
        st.session_state["regiao"] = q if q in REGIOES else REGIAO_PADRAO
    st.session_state.setdefault("_regiao_sel", NOMES_PARA_SELETOR[regiao_actual()])


# ══════════════════════════════════════════════════════
# O QUE A IA DO TUTOR DEVE SABER
# ══════════════════════════════════════════════════════
def lingua_da_ia(lang: str = "pt") -> str:
    if lang == "en":
        return "English"
    return cfg()["lingua"]


def exemplos_da_ia() -> str:
    return cfg()["exemplos"]


# ══════════════════════════════════════════════════════
# ADAPTAÇÃO DE TEXTO
# ══════════════════════════════════════════════════════
_TAGS = re.compile(r"(<[^>]*>)")


def _manter_maiuscula(orig: str, novo: str) -> str:
    if orig.isupper() and len(orig) > 1:
        return novo.upper()
    if orig[:1].isupper():
        return novo[:1].upper() + novo[1:]
    return novo


def _regex_palavras(mapa: dict):
    """Um regex que apanha qualquer chave do mapa como palavra inteira (sem distinguir maiúsculas)."""
    chaves = sorted(mapa, key=len, reverse=True)
    return re.compile(r"(?<![\wÀ-ÿ-])(" + "|".join(re.escape(k) for k in chaves) + r")(?![\wÀ-ÿ])", re.I)


def _aplicar_palavras(texto: str, rx, mapa: dict) -> str:
    return rx.sub(lambda m: _manter_maiuscula(m.group(1), mapa[m.group(1).lower()]), texto)


# Ortografia do Acordo de 1990. Angola mantém a ortografia que o sistema já tinha (projecto, acção).
# Letras que deixam de se escrever nas três variantes (o c/p era mudo em todas):
_AO90_COMUM = {
    "projecto": "projeto", "projectos": "projetos", "acção": "ação", "acções": "ações", "selecção": "seleção",
    "correcção": "correção", "correcções": "correções", "direcção": "direção", "óptimo": "ótimo", "óptima": "ótima",
    "ópticas": "óticas", "óptico": "ótico", "optimizar": "otimizar", "optimizado": "otimizado", "optimizada": "otimizada",
    "optimização": "otimização", "activo": "ativo", "activa": "ativa", "activar": "ativar", "activado": "ativado",
    "actual": "atual", "actuais": "atuais", "actualizar": "atualizar", "actualizado": "atualizado", "actualização": "atualização",
    "objectivo": "objetivo", "objectivos": "objetivos", "directo": "direto", "directa": "direta", "directamente": "diretamente",
    "exactamente": "exatamente", "exacto": "exato", "exacta": "exata", "excepção": "exceção", "inspecção": "inspeção",
    "protecção": "proteção", "reacção": "reação", "redacção": "redação", "tracção": "tração", "factor": "fator",
    "factores": "fatores", "interactivo": "interativo", "interactiva": "interativa", "interactivos": "interativos",
    "colectivo": "coletivo", "colecção": "coleção", "adopção": "adoção", "afectar": "afetar", "efectivo": "efetivo",
}
# Em Portugal e Moçambique o c/p mudo cai também nestas (no Brasil mantêm-se porque se pronunciam):
_AO90_PT = {
    "detecção": "deteção", "infecção": "infeção", "recepção": "receção", "concepção": "conceção",
    "aspecto": "aspeto", "aspectos": "aspetos", "perspectiva": "perspetiva", "respectivo": "respetivo",
}

# Brasil: vocabulário e ortografia própria
_BR_VOCAB = {
    "ecrã": "tela", "ecrãs": "telas", "ficheiro": "arquivo", "ficheiros": "arquivos",
    "telemóvel": "celular", "telemóveis": "celulares", "utilizador": "usuário", "utilizadores": "usuários",
    "palavra-passe": "senha", "palavras-passe": "senhas", "equipa": "equipe", "equipas": "equipes",
    "grelha": "grade", "grelhas": "grades", "ligação": "conexão", "ligações": "conexões",
    "descarregar": "baixar", "descarregado": "baixado", "descarregada": "baixada", "descarrega": "baixe",
    "separador": "aba", "separadores": "abas", "registo": "registro", "registos": "registros",
    "contacto": "contato", "contactos": "contatos", "guardar": "salvar", "guardado": "salvo", "guardada": "salva",
    "guardados": "salvos", "guardadas": "salvas", "autocarro": "ônibus", "comboio": "trem",
    "casa de banho": "banheiro", "frigorífico": "geladeira", "desporto": "esporte", "desportivo": "esportivo",
    "desportiva": "esportiva", "cancro": "câncer", "rapariga": "garota", "rapaz": "garoto", "miúdos": "crianças",
    "miúdo": "criança", "monitorizar": "monitorar", "monitorização": "monitoramento", "gerir": "gerenciar",
    "riscas": "linhas", "risca": "linha", "facto": "fato", "factos": "fatos", "de facto": "de fato",
    "neurónio": "neurônio", "neurónios": "neurônios", "económico": "econômico", "económica": "econômica",
    "fenómeno": "fenômeno", "género": "gênero", "ténis": "tênis", "bebé": "bebê", "pára": "para",
    "bué": "muito", "kamba": "amigo", "kambas": "amigos",
}
_BR_VOCAB = {k: v for k, v in _BR_VOCAB.items() if k != v}

_RX_AO90 = _regex_palavras(_AO90_COMUM)
_RX_AO90_PT = _regex_palavras(_AO90_PT)
_RX_BR_VOCAB = _regex_palavras(_BR_VOCAB)

# Brasil: «tu» -> «você». A conversão tem quatro partes, por esta ordem:
#  1) imperativos («Escolhe» -> «Escolha»); 2) verbos na 2.ª pessoa («tens» -> «você tem»); 3) pronomes e possessivos
#  («teu» -> «seu»); 4) «estar a + infinitivo» -> gerúndio. Só se mexe onde a regra é segura: a mesma forma («Procura»,
#  «Dá») é 3.ª pessoa numa descrição («Procura a fronteira...») e aí NÃO se converte.
_BR_V2P = {
    "tens": "tem", "és": "é", "estás": "está", "podes": "pode", "queres": "quer", "sabes": "sabe", "vais": "vai",
    "precisas": "precisa", "consegues": "consegue", "fazes": "faz", "vês": "vê", "ficas": "fica", "deves": "deve",
    "gostas": "gosta", "percebes": "percebe", "aprendes": "aprende", "dizes": "diz", "pedes": "pede", "escolhes": "escolhe",
    "achas": "acha", "dás": "dá", "dizes": "diz", "usas": "usa", "tentas": "tenta", "pensas": "pensa", "olhas": "olha", "colocas": "coloca",
    "escreves": "escreve", "clicas": "clica", "treinas": "treina", "testas": "testa", "mudas": "muda", "alteras": "altera",
    "aumentas": "aumenta", "comparas": "compara", "selecionas": "seleciona", "executas": "executa", "carregas": "carrega",
    "defines": "define", "ajustas": "ajusta", "observas": "observa", "preferes": "prefere", "entendes": "entende",
    "lembras": "lembra", "imaginas": "imagina", "experimentas": "experimenta", "descobres": "descobre", "ouves": "ouve",
    "arrastas": "arrasta", "pareces": "parece", "reparas": "repara", "tiveste": "teve", "acertaste": "acertou",
    "erraste": "errou", "fizeste": "fez", "completaste": "completou", "terminaste": "terminou", "ganhaste": "ganhou",
    "conseguiste": "conseguiu", "escolheste": "escolheu", "treinaste": "treinou", "chegaste": "chegou",
    "atingiste": "atingiu", "estiveste": "esteve", "foste": "foi", "disseste": "disse", "viste": "viu",
    "começaste": "começou", "usaste": "usou", "testaste": "testou", "guardaste": "salvou", "tentaste": "tentou",
    "aprendeste": "aprendeu", "mudaste": "mudou", "abriste": "abriu", "fechaste": "fechou", "criaste": "criou",
    "carregaste": "enviou", "compraste": "comprou", "resolveste": "resolveu", "descobriste": "descobriu",
}
_BR_PRON = {"tu": "você", "ti": "você", "teu": "seu", "tua": "sua", "teus": "seus", "tuas": "suas"}

# Imperativos. «SEGUROS» quase só aparecem como instrução ao aluno; «AMBIGUOS» também abrem descrições na 3.ª pessoa
# («Usa vizinhos...», «Calcula a probabilidade...»): só se convertem quando há um «tu» na frase ou quando é um rótulo curto.
_BR_IMP_SEGUROS = {
    "escolhe": "escolha", "clica": "clique", "tenta": "tente", "experimenta": "experimente", "começa": "comece",
    "continua": "continue", "volta": "volte", "espera": "espere", "abre": "abra", "seleciona": "selecione",
    "selecciona": "selecione", "confirma": "confirme", "escreve": "escreva", "arrasta": "arraste", "responde": "responda",
    "lê": "leia", "vê": "veja", "entra": "entre", "reinicia": "reinicie", "sai": "saia", "imagina": "imagine",
    "pensa": "pense", "repara": "repare", "recorda": "lembre", "descobre": "descubra", "completa": "complete",
    "digita": "digite", "relê": "releia", "olha": "olhe", "toca": "toque", "avança": "avance", "cola": "cole",
    "repete": "repita", "regista": "registre", "verifica": "verifique", "decide": "decida", "marca": "marque",
}
_BR_IMP_AMBIGUOS = {
    "usa": "use", "faz": "faça", "treina": "treine", "compara": "compare", "testa": "teste", "ajusta": "ajuste",
    "mostra": "mostre", "carrega": "envie", "liga": "ative", "desliga": "desative", "move": "mova", "prepara": "prepare",
    "cria": "crie", "define": "defina", "indica": "indique", "pesquisa": "pesquise", "procura": "procure",
    "identifica": "identifique", "calcula": "calcule", "junta": "junte", "observa": "observe", "passa": "passe",
    "segue": "siga", "dá": "dê", "põe": "ponha", "pede": "peça", "tira": "tire", "guarda": "salve", "grava": "grave",
    "aprende": "aprenda", "diz": "diga", "conclui": "conclua", "revê": "revise", "muda": "mude", "aumenta": "aumente",
    "diminui": "diminua", "ordena": "ordene", "filtra": "filtre", "explora": "explore", "analisa": "analise",
}
_BR_IMP = {**_BR_IMP_SEGUROS, **_BR_IMP_AMBIGUOS}
_BR_VOCE_SO_INICIO = set(_BR_IMP)       # nome antigo, usado nos testes

_FRONTEIRA = re.compile(r"([.!?]\s+|\n+)")        # «:» e «;» NÃO acabam a frase: depois deles costuma vir uma descrição
_RX_PRIMEIRA_PALAVRA = re.compile(r"((?:\s|\x00|[-*•»\"“(\[]|\d+[.)])*)([A-Za-zÀ-ÿ-]+)")      # usado com .match(texto, pos)
_ADVERBIOS_DE_ABERTURA = {"agora", "depois", "primeiro", "então", "também", "antes", "sempre", "só", "por favor"}
_IMP_ALT = "|".join(re.escape(k) for k in sorted(_BR_IMP, key=len, reverse=True))
_RX_IMP_DEPOIS_DE_E = re.compile(r"(\b(?:e|ou|depois|então)\s+|,\s+)(" + _IMP_ALT + r")(?![\wÀ-ÿ])", re.I)
_RX_CUE_TU = re.compile(r"(?<![\wÀ-ÿ-])(tu|ti|teu|tua|teus|tuas|contigo|" + "|".join(_BR_V2P) + r")(?![\wÀ-ÿ])", re.I)

_RX_V2P = re.compile(r"(?<![\wÀ-ÿ-])((?:(?:já|não|nunca|só|também|ainda|sempre|agora|então)\s+)*)(" +
                     "|".join(sorted(_BR_V2P, key=len, reverse=True)) + r")(?![\wÀ-ÿ])", re.I)
_RX_PRON = re.compile(r"(?<![\wÀ-ÿ-])(" + "|".join(_BR_PRON) + r")(?![\wÀ-ÿ])", re.I)
_RX_ESTAR_A = re.compile(r"(?<![\wÀ-ÿ-])(estou|está|estás|estamos|estão|estava|estavam|estarei|estará|anda|andam|ando)\s+a\s+([a-zà-ÿ]*?)(ar|er|ir)(?![\wÀ-ÿ])", re.I)
_GER = {"ar": "ando", "er": "endo", "ir": "indo"}


def _e_inicio_de_frase(antes: str) -> bool:
    return bool(re.search(r"(^|[.!?]\s*|\n\s*)(?:\x00|[-*•»\"“(\[]|\s|\d+[.)])*$", antes))


def _imperativos_para_voce(parte: str) -> str:
    """«Escolhe um algoritmo e clica...» -> «Escolha um algoritmo e clique...»."""
    pedacos = _FRONTEIRA.split(parte)
    for k in range(0, len(pedacos), 2):
        pedaco = pedacos[k]
        m = _RX_PRIMEIRA_PALAVRA.match(pedaco, 0)
        if m and m.group(2).lower() in _ADVERBIOS_DE_ABERTURA:      # «Agora experimenta...»: o imperativo vem a seguir
            m = _RX_PRIMEIRA_PALAVRA.match(pedaco, m.end())
        if not m:
            continue
        palavra = m.group(2).lower()
        if palavra not in _BR_IMP:
            continue
        em_destaque = pedaco[m.end():m.end() + 1] == "\x00" and "\x00" in (m.group(1) or pedaco[:m.start(2)])     # «<b>Usa</b> ...»: a negrito é instrução
        if palavra not in _BR_IMP_SEGUROS and not em_destaque:
            e_rotulo = len(pedaco.split()) <= 6 and not re.search(r"[.:;]\s*$", pedaco.strip()) and k + 2 >= len(pedacos)
            if not (_RX_CUE_TU.search(pedaco) or e_rotulo):
                continue
        nova = _manter_maiuscula(m.group(2), _BR_IMP[palavra])
        resto = pedaco[m.end():]
        resto = _RX_IMP_DEPOIS_DE_E.sub(lambda x: x.group(1) + _manter_maiuscula(x.group(2), _BR_IMP[x.group(2).lower()]), resto)
        pedacos[k] = pedaco[:m.start(2)] + nova + resto
    return "".join(pedacos)


def _verbos_2p_para_voce(parte: str) -> str:
    """«Tens poucos dados» -> «Você tem poucos dados»; «Já tens» -> «Você já tem».
    Se a frase já tem «tu»/«você», só se troca o verbo («Você já tem o resultado, quer continuar?»)."""
    saida, pos = [], 0
    for m in _RX_V2P.finditer(parte):
        saida.append(parte[pos:m.start()])
        antes = "".join(saida)                      # o que já está escrito (com os «você» que acabámos de pôr)
        adv, verbo = m.group(1), m.group(2)
        novo = _BR_V2P[verbo.lower()]
        desde_a_frase = re.split(r"[.!?]\s+|\n+", antes)[-1]
        inicio = _e_inicio_de_frase(antes)
        ja_tem_sujeito = re.search(r"(?<![\wÀ-ÿ])(tu|você)(?![\wÀ-ÿ])", desde_a_frase, re.I)
        if ja_tem_sujeito:
            saida.append(adv + novo)
        else:
            saida.append(("Você " if inicio else "você ") + (adv.lower() if inicio else adv) + novo)
        pos = m.end()
    saida.append(parte[pos:])
    return "".join(saida)


def _gerundio(parte: str) -> str:
    """«está a aparecer» -> «está aparecendo» (no Brasil usa-se o gerúndio, não «estar a + infinitivo»)."""
    return _RX_ESTAR_A.sub(lambda m: f"{m.group(1)} {m.group(2)}{_GER[m.group(3).lower()]}", parte)


# Brasil: palavras com género diferente. «ecrã» (m.) vira «tela» (f.): os artigos e determinantes acompanham.
_GENERO = {"o": "a", "um": "uma", "no": "na", "num": "numa", "do": "da", "dum": "duma", "ao": "à", "este": "esta", "esse": "essa",
           "aquele": "aquela", "pelo": "pela", "seu": "sua", "meu": "minha", "nosso": "nossa", "os": "as", "uns": "umas",
           "nos": "nas", "dos": "das", "aos": "às", "estes": "estas", "esses": "essas", "seus": "suas", "meus": "minhas"}
_ART = "o|um|no|num|do|dum|ao|este|esse|aquele|pelo|os|uns|nos|dos|aos|estes|esses"
_POS = "seu|meu|nosso|seus|meus"
_RX_GENERO = re.compile(r"\b(?:(" + _ART + r")\s+)?(?:(" + _POS + r")\s+)?(tela|telas|aba|abas)\b", re.I)

# Locuções feitas de várias palavras (aplicadas primeiro, sempre com limites de palavra)
_BR_FRASES = [
    ("Carrega um ficheiro", "Envie um arquivo"), ("Carrega o ficheiro", "Envie o arquivo"), ("carrega um ficheiro", "envie um arquivo"),
    ("Carrega ficheiro", "Enviar arquivo"), ("Carregar ficheiro", "Enviar arquivo"), ("carregar um ficheiro", "enviar um arquivo"),
    ("Upload CSV", "Enviar CSV"), ("Carregar CSV", "Enviar CSV"), ("carrega o teu CSV", "envie o seu CSV"),
    ("Descarregar", "Baixar"), ("Não tens de", "Você não precisa"), ("não tens de", "você não precisa"),
    ("Não precisas de", "Você não precisa de"), ("não precisas de", "você não precisa de"),
    ("tens de", "você precisa"), ("Tens de", "Você precisa"), ("tu mesmo", "você mesmo"), ("para ti", "para você"),
    ("a ti", "a você"), ("contigo", "com você"), ("Imagina que és", "Imagine que você é"), ("imagina que és", "imagine que você é"),
]
_RX_FRASES = [(re.compile(r"(?<![\wÀ-ÿ])" + re.escape(a) + r"(?![\wÀ-ÿ])"), b)
              for a, b in sorted(_BR_FRASES, key=lambda x: -len(x[0]))]     # as mais compridas primeiro


def _concordar_genero(parte: str) -> str:
    def f(m):
        art, pos, nome = m.group(1), m.group(2), m.group(3)
        if not art and not pos:
            return m.group(0)
        out = ""
        if art:
            out += _manter_maiuscula(art, _GENERO[art.lower()]) + " "
        if pos:
            out += _manter_maiuscula(pos, _GENERO[pos.lower()]) + " "
        return out + nome
    return _RX_GENERO.sub(f, parte)


def _para_brasil(parte: str) -> str:
    for rx, b in _RX_FRASES:
        parte = rx.sub(b, parte)
    parte = _imperativos_para_voce(parte)          # primeiro: precisa de ver as formas «tu» originais
    parte = _verbos_2p_para_voce(parte)
    parte = _aplicar_palavras(parte, _RX_PRON, _BR_PRON)
    parte = _gerundio(parte)
    parte = _aplicar_palavras(parte, _RX_BR_VOCAB, _BR_VOCAB)
    return _concordar_genero(parte)


def _geo(parte: str, c: dict) -> str:
    """Moeda, cidade, bairros, gentílico e país (só trocam quando a região não é Angola)."""
    ao = REGIOES["AO"]
    # valores com moeda primeiro: «4.500 Kz» -> «R$ 4.500», «12 €»
    def moeda(m):
        n = m.group(1)
        return {"R$": f"R$ {n}", "€": f"{n} €", "MT": f"{n} MT"}.get(c["moeda"], f"{n} {c['moeda']}")
    parte = re.sub(r"(\d[\d.,]*)\s*Kz\b", moeda, parte)
    parte = parte.replace("milhões de Kz", c["unidade_casas"]).replace("milhões Kz", c["unidade_casas"])
    parte = parte.replace("milhares de Kz", "milhares de " + c["moeda"] if c["moeda"] != "R$" else "milhares de R$")
    parte = re.sub(r"\bKz\b", c["moeda"], parte)
    parte = parte.replace("Luanda", c["cidade"])
    parte = parte.replace("em Angola", c["em"]).replace("de Angola", c["de"])
    parte = re.sub(r"\bAngola\b", c["nome"], parte)
    for forma_ao, forma_c in zip(ao["gentilicio"], c["gentilicio"]):
        parte = re.sub(rf"\b{forma_ao}\b", forma_c, parte, flags=re.I)
    parte = parte.replace("Banca de rua (fictício)", f"{c['banca']} (fictício)")
    return parte


def _geo_en(parte: str, c: dict) -> str:
    ao = REGIOES["AO"]
    parte = re.sub(r"(\d[\d.,]*)\s*Kz\b", lambda m: {"R$": f"R$ {m.group(1)}", "€": f"€{m.group(1)}"}.get(c["moeda"], f"{m.group(1)} {c['moeda']}"), parte)
    parte = parte.replace("million Kz", c["unidade_casas_en"]).replace("thousands of Kz", "thousands of " + c["moeda"])
    parte = re.sub(r"\bKz\b", c["moeda"], parte)
    parte = parte.replace("Luanda", c["cidade"])
    parte = re.sub(r"\bAngolan\b", c["gentilicio_en"], parte)
    parte = re.sub(r"\bAngola\b", c["nome_en"], parte)
    return parte


def localizar(texto, idioma: str = "pt"):
    """Adapta um texto mostrado ao aluno à região escolhida. Para Angola devolve o texto como está."""
    if not isinstance(texto, str) or not texto:
        return texto
    cod = regiao_actual()
    if cod == REGIAO_PADRAO:
        return texto
    if "\U0001F1E6" <= texto[0] <= "\U0001F1FF":     # bandeira: é o nome de um país no seletor, não se adapta
        return texto
    c = REGIOES[cod]
    if idioma == "en":
        return _geo_en(texto, c)
    etiquetas = []

    def guardar(m):
        etiquetas.append(m.group(0))
        return "\x00"
    parte = _TAGS.sub(guardar, texto)          # as etiquetas HTML não se tocam (classes, estilos, links)
    parte = _geo(parte, c)
    parte = _aplicar_palavras(parte, _RX_AO90, _AO90_COMUM)
    if cod != "BR":
        parte = _aplicar_palavras(parte, _RX_AO90_PT, _AO90_PT)
    else:
        parte = _para_brasil(parte)
    it = iter(etiquetas)
    return re.sub("\x00", lambda m: next(it, ""), parte)


# ══════════════════════════════════════════════════════
# DADOS DE EXEMPLO DA REGIÃO (lições e datasets)
# ══════════════════════════════════════════════════════
def bairros_licao() -> list:
    return cfg()["bairros_licao"]


def preco_casa(valor_ao: float) -> float:
    """Converte um preço dado em «milhões de Kz» para a unidade da região (mil R$, mil €...)."""
    return round(valor_ao * cfg()["fator_casas"])


def salario(valores_ao: list) -> tuple:
    """(valores, unidade) para o exemplo dos salários, a partir de valores dados em milhares de Kz."""
    s = cfg()["salario"]
    f = s["fator"]
    return [round(v * f) if f >= 1 else round(v * f, 1) for v in valores_ao], s["unidade"]


def taxi_exemplo() -> str:
    return cfg()["taxi"]


def nota_maxima() -> int:
    return cfg()["nota_max"]


# ══════════════════════════════════════════════════════
# SELETOR (menu lateral e ecrã de entrada)
# ══════════════════════════════════════════════════════
def codigo_do_rotulo(rotulo: str) -> str:
    for c, r in NOMES_PARA_SELETOR.items():
        if r == rotulo:
            return c
    return REGIAO_PADRAO
