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
    # aluno
    return [
        ("Início",                   "dashboard",       "COMEÇAR AQUI",       "home",     AZ),
        ("Escola de ML",             "escola",          None,                 "school",   VD),
        ("O Meu Primeiro Modelo",    "primeiro_modelo", None,                 "rocket",   VD),
        ("Que algoritmo uso?",       "assistente",      None,                 "compass2", VD),
        ("Dicionário de ML",         "dicionario",      None,                 "book",     VD),
        (rotulo_revisao,             "revisao",         "A MINHA JORNADA",    "history",  AM),
        ("Projecto Final",           "projeto",         None,                 "rocket",   AM),
        ("Conquistas e Desafios",    "conquistas",      None,                 "trophy",   AM),
        ("Certificados",             "certificados",    None,                 "trophy",   AM),
        ("Perfil",                   "perfil",          None,                 "menu",     AM),
        ("Classificação",            "classificacao",   "SUPERVISIONADO",     "chart-bar", AZ),
        ("Regressão",                "regressao",       None,                 "trend-up", AZ),
        ("Clustering",               "clustering",      "NÃO SUPERVISIONADO", "circles",  AZ),
        ("Redução Dimensionalidade", "reducao",         None,                 "network",  AZ),
        ("Detecção de Anomalias",    "anomalias",       None,                 "warning",  AZ),
        ("MLP / Redes Neurais",      "mlp",             "REDES NEURAIS",      "brain",    AZ),
        ("Visão Computacional",      "visao",           "DEEP LEARNING",      "eye",      AZ),
        ("NLP",                      "nlp",             None,                 "text",     AZ),
        ("Séries Temporais",         "series",          None,                 "clock",    AZ),
        ("Modelos Generativos",      "generativo",      None,                 "sparkle",  AZ),
        ("Aprendizagem por Reforço", "reforco",         "REFORÇO",            "game",     AZ),
        ("Comparador de Modelos",    "comparador",      "FERRAMENTAS",        "compare",  AZ),
        ("Meu Historial",            "historico",       None,                 "history",  AZ),
    ]


def paginas_permitidas(painel: str) -> set:
    return {item[1] for item in menu_do_painel(painel)}


def resolver_painel(pedido, disponiveis: list) -> str:
    """Painel a mostrar: o pedido, se a pessoa tiver direito a ele; senão o primeiro."""
    return pedido if pedido in disponiveis else disponiveis[0]
