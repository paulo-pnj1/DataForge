"""
DataForge - Painéis e menus por papel.
Cada pessoa cai no seu painel ao entrar e só vê o menu desse painel.
O router (app.py) usa `paginas_permitidas` para recusar páginas de outros painéis.
"""

PAINEIS = {"admin": "Administração", "professor": "Professor", "aluno": "Aluno", "business": "Business"}
PRODUTO_DO_PAINEL = {"admin": "DataForge", "professor": "DataForge EDU", "aluno": "DataForge EDU", "business": "DataForge Business"}
INICIO = {"admin": "admin_home", "professor": "professor", "aluno": "dashboard", "business": "business_home"}


def paineis_disponiveis(role: str, membros: list, tem_turma: bool = False) -> list:
    """Painéis a que a pessoa tem direito (o primeiro é o de entrada).

    - admin global: SÓ o painel de Administração (gere a plataforma, não usa os painéis dos utilizadores).
    - conta de empresa (papel «empresa», ou aluno sem turma nem escola que só pertence a empresas):
      SÓ o painel Business. Não vê nada de aluno (lições, pontos, turma).
    - professor: Professor (+ Business se também pertencer a uma empresa).
    - aluno: Aluno (+ Business se também pertencer a uma empresa).
    """
    if role == "admin":
        return ["admin"]
    membros = membros or []
    em_empresa = any(m.get("tipo") == "empresa" for m in membros)
    em_escola = any(m.get("tipo") == "escola" for m in membros)
    if role == "empresa":
        return ["business"] if em_empresa else []      # sem empresa activa (pendente ou removida): não há painel
    if role == "professor":
        return ["professor", "business"] if em_empresa else ["professor"]
    if em_empresa and not tem_turma and not em_escola:
        return ["business"]
    return ["aluno", "business"] if em_empresa else ["aluno"]


def menu_do_painel(painel: str, rotulo_revisao: str = "Revisão do dia") -> list:
    """Lista de (rótulo, chave, secção, ícone, cor) do painel."""
    AZ, VD, AM, RS = "#6BA3FF", "#4ADE80", "#FFC107", "#F778BA"
    if painel == "admin":
        return [
            ("Visão geral",   "admin_home",      "ADMINISTRAÇÃO", "home",   RS),
            ("Organizações",  "admin_orgs",      None,            "school", RS),
            ("Convites",      "admin_convites",  None,            "rocket", RS),
            ("Utilizadores",  "admin_users",     None,            "menu",   RS),
            ("Email",         "admin_email",     None,            "text",   RS),
            ("Conteúdos",     "admin_conteudos", None,            "book",   RS),
            ("Erros",         "admin_erros",     None,            "warning", RS),
            ("Perfil",        "perfil",          None,            "menu",   RS),
        ]
    if painel == "professor":
        return [
            ("Turmas e alunos", "professor", "ENSINAR",        "school", RS),
            ("Escola de ML",    "escola",    "CONTEÚDO",       "book",   VD),
            ("Dicionário de ML", "dicionario", None,           "book",   VD),
            ("Perfil",          "perfil",    None,             "menu",   AM),
        ]
    if painel == "business":
        return [
            ("Início",     "business_home",      "DATAFORGE BUSINESS", "home",      AZ),
            ("Dados",      "business_dados",     None,                 "chart-bar", AZ),
            ("Análises",   "business_analises",  "RESOLVER PROBLEMAS", "brain",     VD),
            ("Previsões",  "business_previsoes", None,                 "trend-up",  VD),
            ("Relatórios", "business_relatorios", None,                "book",      VD),
            ("Equipa",     "business_equipa",    "EMPRESA",            "network",   AM),
            ("Perfil",     "perfil",             None,                 "menu",      AM),
        ]
    # aluno: o menu segue o caminho de aprendizagem (começar → módulos 1 a 4 → ferramentas → a minha jornada)
    return [
        ("Início",                   "dashboard",       "COMEÇAR AQUI",                     "home",     AZ),
        ("Escola de ML",             "escola",          None,                               "school",   VD),
        ("O Meu Primeiro Modelo",    "primeiro_modelo", None,                               "rocket",   VD),
        ("Classificação",            "classificacao",   "MÓDULO 1 · PREVER",                "chart-bar", AZ),
        ("Regressão",                "regressao",       None,                               "trend-up", AZ),
        ("Clustering",               "clustering",      "MÓDULO 2 · DESCOBRIR PADRÕES",     "circles",  AZ),
        ("Redução Dimensionalidade", "reducao",         None,                               "network",  AZ),
        ("Detecção de Anomalias",    "anomalias",       None,                               "warning",  AZ),
        ("MLP / Redes Neurais",      "mlp",             "MÓDULO 3 · REDES E DEEP LEARNING", "brain",    AZ),
        ("Visão Computacional",      "visao",           None,                               "eye",      AZ),
        ("NLP",                      "nlp",             None,                               "text",     AZ),
        ("Séries Temporais",         "series",          None,                               "clock",    AZ),
        ("Modelos Generativos",      "generativo",      None,                               "sparkle",  AZ),
        ("Aprendizagem por Reforço", "reforco",         "MÓDULO 4 · REFORÇO",               "game",     AZ),
        ("Que algoritmo uso?",       "assistente",      "FERRAMENTAS",                      "compass2", VD),
        ("Comparador de Modelos",    "comparador",      None,                               "compare",  VD),
        ("Dicionário de ML",         "dicionario",      None,                               "book",     VD),
        (rotulo_revisao,             "revisao",         "A MINHA JORNADA",                  "history",  AM),
        ("Projecto Final",           "projeto",         None,                               "rocket",   AM),
        ("Conquistas e Desafios",    "conquistas",      None,                               "trophy",   AM),
        ("Certificados",             "certificados",    None,                               "trophy",   AM),
        ("Meu Historial",            "historico",       None,                               "history",  AM),
        ("Perfil",                   "perfil",          "CONTA",                            "menu",     AM),
    ]


def paginas_permitidas(painel: str) -> set:
    return {item[1] for item in menu_do_painel(painel)}


def resolver_painel(pedido, disponiveis: list) -> str:
    """Painel a mostrar: o pedido, se a pessoa tiver direito a ele; senão o primeiro."""
    return pedido if pedido in disponiveis else disponiveis[0]


def agrupar_menu(menu: list) -> list:
    """Agrupa os itens do menu por secção, na ordem em que aparecem: [(secção, [itens])].
    Os itens sem secção juntam-se à secção anterior."""
    grupos = []
    for item in menu:
        if item[2] or not grupos:
            grupos.append((item[2] or "", []))
        grupos[-1][1].append(item)
    return grupos


# Apoio didáctico do menu do aluno: o que cada módulo ensina e o que cada página faz, em linguagem simples.
MODULOS_INFO = {
    "COMEÇAR AQUI": ("🚀", "Primeiro percebe a ideia e treina o teu primeiro modelo."),
    "MÓDULO 1 · PREVER": ("🎯", "Tens exemplos com a resposta certa? O computador aprende com eles e prevê casos novos."),
    "MÓDULO 2 · DESCOBRIR PADRÕES": ("🔍", "Sem respostas prontas: o computador procura grupos e coisas estranhas nos dados."),
    "MÓDULO 3 · REDES E DEEP LEARNING": ("🧠", "Modelos inspirados no cérebro: imagens, texto, séries no tempo e criação de dados novos."),
    "MÓDULO 4 · REFORÇO": ("🎮", "Um agente aprende por tentativa e erro, ganhando prémios e perdendo pontos."),
    "FERRAMENTAS": ("🧰", "Para tirar dúvidas, escolher o algoritmo e comparar resultados."),
    "A MINHA JORNADA": ("🏅", "Revê o que falhaste, faz o projecto final e ganha medalhas e certificados."),
    "CONTA": ("👤", "Os teus dados e a tua palavra-passe."),
}

AJUDA_PAGINA = {
    "dashboard": "O teu próximo passo e o resumo do que já fizeste.",
    "escola": "14 lições curtas com analogias e experiências para mexer.",
    "primeiro_modelo": "Treina um modelo com 1 clique e vê o resultado em linguagem simples.",
    "classificacao": "Prever uma categoria: sim/não, spam/normal, espécie A/B/C.",
    "regressao": "Prever um número: preço, vendas, temperatura.",
    "clustering": "Juntar casos parecidos em grupos, sem ninguém dizer quais são os grupos.",
    "reducao": "Resumir muitas colunas em 2 ou 3 para conseguires ver os dados.",
    "anomalias": "Encontrar casos estranhos: fraudes, avarias, erros.",
    "mlp": "A rede neural básica: camadas de neurónios que aprendem padrões.",
    "visao": "Ensinar o computador a reconhecer imagens.",
    "nlp": "Ensinar o computador a perceber texto.",
    "series": "Prever o que vem a seguir a partir do passado, como as vendas de cada mês.",
    "generativo": "Modelos que criam dados novos, como imagens.",
    "reforco": "Um agente aprende por tentativa e erro, com prémios e castigos.",
    "assistente": "Responde a perguntas simples e sugere o algoritmo certo para o teu problema.",
    "comparador": "Põe vários algoritmos à prova nos mesmos dados e vê qual ganha.",
    "dicionario": "Explica os termos de Machine Learning em palavras simples.",
    "revisao": "Repete as perguntas que falhaste, no momento certo para fixar.",
    "projeto": "Projecto guiado em 5 passos, com relatório em PDF.",
    "conquistas": "Medalhas, sequência de dias e desafio da semana.",
    "certificados": "O teu certificado em PDF, com código de verificação.",
    "historico": "Os modelos que já treinaste.",
    "perfil": "Os teus dados e a palavra-passe.",
}
