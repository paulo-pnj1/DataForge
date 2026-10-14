"""
DataForge EDU - Ajuda
  - Dicionário de ML (glossário pesquisável em linguagem simples)
  - Assistente "Que algoritmo devo usar?" (3 perguntas -> recomendação)
"""
import streamlit as st

from modules.utils import (
    page_header, ir_para,
    C_SURFACE, C_BORDER, C_ACCENT, C_GREEN, C_AMBER, C_TEAL,
    C_TEXT, C_TEXT_SEC, C_TEXT_MUTE,
)

# ══════════════════════════════════════════════════════
# DICIONÁRIO
# (termo, categoria, definição simples, exemplo/analogia)
# ══════════════════════════════════════════════════════
GLOSSARIO = [
    # ── Básico ──
    ("Inteligência Artificial (IA)", "Básico", "Área grande que junta tudo o que faz máquinas parecerem 'inteligentes'. O Machine Learning é uma parte dela.", "Um assistente que responde a perguntas."),
    ("Machine Learning (ML)", "Básico", "Ensinar o computador a aprender com exemplos, em vez de lhe escrever todas as regras.", "A zungueira que aprende quanto peixe levar com a experiência dos dias anteriores."),
    ("Deep Learning", "Básico", "Machine Learning que usa redes neuronais com muitas camadas. Muito bom para imagens, voz e texto.", "Reconhecer um rosto numa fotografia."),
    ("Modelo", "Básico", "O 'cérebro' que resulta do treino: guarda o que o algoritmo aprendeu e faz previsões.", "O aluno depois de estudar."),
    ("Algoritmo", "Básico", "A receita que o computador segue para aprender (KNN, Random Forest, etc.).", "A forma de estudar: resumos, exercícios, decorar."),
    ("Treinar", "Básico", "Mostrar exemplos ao algoritmo para ele ajustar o modelo.", "Estudar para a prova."),
    ("Previsão (inferência)", "Básico", "O que o modelo responde quando lhe damos um caso novo.", "A resposta na prova."),
    # ── Dados ──
    ("Dataset", "Dados", "O conjunto de dados, normalmente uma tabela.", "Um caderno de vendas de uma loja."),
    ("Exemplo (amostra)", "Dados", "Uma linha da tabela: um caso concreto.", "Uma venda, um cliente, uma casa."),
    ("Feature (característica)", "Dados", "Uma coluna usada para o modelo decidir.", "Área, nº de quartos e bairro de uma casa."),
    ("Target (alvo)", "Dados", "A coluna que queremos prever.", "O preço da casa."),
    ("Valor nulo (em falta)", "Dados", "Uma célula vazia na tabela. Deve ser tratada antes de treinar.", "Um cliente sem idade registada."),
    ("Outlier", "Dados", "Um valor muito fora do normal. Pode ser um erro ou um caso raro.", "Uma casa a custar 100 vezes mais que as outras."),
    ("Normalização / Escala", "Dados", "Pôr todas as colunas em escalas parecidas para uma não 'gritar' mais alto que as outras.", "Comparar metros com quilómetros sem converter."),
    ("Variável categórica", "Dados", "Coluna com categorias em vez de números.", "Bairro: Talatona, Viana, Cazenga."),
    ("Encoding", "Dados", "Transformar categorias em números para o modelo conseguir ler.", "Talatona = 0, Viana = 1."),
    ("Fuga de informação (data leakage)", "Dados", "Quando uma coluna já contém a resposta ou espreita o futuro. A nota fica falsamente óptima.", "Ver o gabarito antes da prova."),
    ("Dados desbalanceados", "Dados", "Uma classe é muito mais rara do que a outra. A accuracy engana.", "1 doente em cada 100 pessoas."),
    # ── Avaliação ──
    ("Treino e teste", "Avaliação", "Separar os dados: uma parte para aprender e outra para a prova final.", "Exercícios de estudo vs prova com perguntas novas."),
    ("Validação cruzada", "Avaliação", "Repetir treino e teste várias vezes com partes diferentes para ter uma nota mais fiável.", "Fazer 5 provas diferentes em vez de uma só."),
    ("Accuracy", "Avaliação", "Percentagem total de acertos.", "Acertou 9 em cada 10."),
    ("Precision", "Avaliação", "Quando o modelo diz 'sim', quantas vezes tem razão.", "Dos que o teste deu como doentes, quantos estavam mesmo."),
    ("Recall", "Avaliação", "Dos casos que realmente eram 'sim', quantos o modelo encontrou.", "Dos doentes verdadeiros, quantos o teste apanhou."),
    ("F1-Score", "Avaliação", "Nota única que junta Precision e Recall.", "Só é alta se as duas forem boas."),
    ("Matriz de confusão", "Avaliação", "Tabela que mostra acertos e quais classes o modelo troca.", "Diagonal = acertos; fora da diagonal = erros."),
    ("ROC-AUC", "Avaliação", "Mede se o modelo separa bem as classes. 1.0 = perfeito, 0.5 = moeda ao ar.", "Distinguir bem doentes de saudáveis."),
    ("R²", "Avaliação", "Quanto da variação dos dados o modelo explica (regressão). 1.0 = perfeito, 0 = tão mau como prever a média.", "R² 0.80 = explica 80% do que faz o preço mudar."),
    ("MAE", "Avaliação", "Erro médio em valor absoluto, na mesma unidade do alvo.", "Em média falha por 3 milhões de Kz."),
    ("RMSE", "Avaliação", "Parecido com o MAE mas castiga mais os erros grandes.", "Um erro de 10 pesa mais do que dois erros de 5."),
    ("Baseline", "Avaliação", "Resultado de um palpite muito simples. O teu modelo tem de o bater para valer a pena.", "Dizer sempre a resposta mais comum."),
    # ── Modelos e conceitos ──
    ("Classificação", "Modelos", "Prever uma categoria.", "Spam ou não spam."),
    ("Regressão", "Modelos", "Prever um número.", "O preço de uma casa."),
    ("Clustering (agrupamento)", "Modelos", "Encontrar grupos nos dados sem respostas prévias.", "Separar clientes por comportamento."),
    ("Aprendizagem supervisionada", "Modelos", "Aprender com exemplos que já têm a resposta certa.", "Estudar com gabarito."),
    ("Aprendizagem não supervisionada", "Modelos", "Procurar padrões em dados sem respostas.", "Arrumar roupa em montes sem instruções."),
    ("Aprendizagem por reforço", "Modelos", "Um agente aprende por tentativa e erro com prémios e castigos.", "Treinar um cão com petiscos."),
    ("Hiperparâmetro", "Modelos", "Uma definição que tu escolhes ANTES de treinar (ex.: K no KNN).", "O número de horas que decides estudar."),
    ("Overfitting", "Modelos", "O modelo decorou o treino e falha com casos novos.", "O aluno que decora as respostas e não percebe a matéria."),
    ("Underfitting", "Modelos", "O modelo é simples demais e não aprendeu nem o treino.", "Estudar só o título dos capítulos."),
    ("Regularização", "Modelos", "Travão que impede o modelo de ficar complicado demais.", "Limitar o tamanho do resumo."),
    ("Ensemble", "Modelos", "Juntar muitos modelos para votarem em conjunto.", "Perguntar a 100 especialistas em vez de a um."),
    ("Importância das features", "Modelos", "Mostra que colunas mais pesaram nas decisões do modelo.", "Perceber que a área pesa mais que o bairro."),
    ("Redução de dimensionalidade", "Modelos", "Resumir muitas colunas em poucas, perdendo o mínimo de informação.", "Resumir um livro num parágrafo."),
    ("Anomalia", "Modelos", "Um caso muito diferente do normal.", "Uma transferência bancária suspeita."),
    # ── Redes neuronais ──
    ("Neurónio artificial", "Redes neuronais", "Pequena unidade que soma entradas vezes pesos e decide um valor de saída.", "Decidir se vais à praia pesando tempo, amigos e dinheiro."),
    ("Peso", "Redes neuronais", "A importância dada a cada entrada. É o que a rede aprende.", "O quanto o tempo bom pesa na tua decisão."),
    ("Camada", "Redes neuronais", "Um grupo de neurónios lado a lado. Uma rede tem várias camadas em sequência.", "Uma fila de pessoas a passar um recado: cada camada trabalha e passa à seguinte."),
    ("Função de activação", "Redes neuronais", "Regra que decide quanto o neurónio 'dispara' depois da soma.", "O limiar a partir do qual decides ir."),
    ("Época", "Redes neuronais", "Uma passagem completa por todos os dados de treino.", "Ler o livro inteiro uma vez."),
    ("Taxa de aprendizagem (learning rate)", "Redes neuronais", "O tamanho de cada ajuste dos pesos. Alta = passos grandes e instáveis; baixa = lento.", "O tamanho do passo ao descer uma ladeira."),
    ("Função de perda (loss)", "Redes neuronais", "A nota de erro que a rede tenta reduzir.", "A distância entre a resposta e a certa."),
    ("CNN (rede convolucional)", "Redes neuronais", "Rede especializada em imagens: procura padrões como cantos, formas e texturas.", "Reconhecer uma cara."),
    ("LSTM / GRU", "Redes neuronais", "Redes com 'memória', boas para sequências como texto e séries temporais.", "Lembrar o início da frase para entender o fim."),
    ("GAN", "Redes neuronais", "Duas redes em competição: uma inventa dados falsos e outra tenta descobrir.", "O falsificador e o detective."),
]
CATEGORIAS = ["Todas"] + sorted({g[1] for g in GLOSSARIO}, key=lambda c: ["Básico", "Dados", "Avaliação", "Modelos", "Redes neuronais"].index(c))
COR_CAT = {"Básico": C_TEAL, "Dados": C_ACCENT, "Avaliação": C_GREEN, "Modelos": C_AMBER, "Redes neuronais": "#BC8CFF"}


def render_dicionario(username: str):
    page_header("Dicionário de ML", "Todos os termos explicados em linguagem simples, com um exemplo")

    c1, c2 = st.columns([3, 2])
    with c1:
        busca = st.text_input("Pesquisar um termo", placeholder="ex.: overfitting, recall, feature...", key="dic_busca")
    with c2:
        cat = st.selectbox("Categoria", CATEGORIAS, key="dic_cat")

    q = busca.strip().lower()
    itens = [g for g in GLOSSARIO
             if (cat == "Todas" or g[1] == cat)
             and (not q or q in g[0].lower() or q in g[2].lower() or q in g[3].lower())]

    st.caption(f"{len(itens)} termo(s)")
    if not itens:
        st.info("Não encontrei esse termo. Tenta uma palavra mais curta ou muda a categoria.")
        return

    for termo, categoria, definicao, exemplo in itens:
        cor = COR_CAT.get(categoria, C_ACCENT)
        st.markdown(
            f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-left:4px solid {cor};'
            f'border-radius:10px;padding:.8rem 1.1rem;margin-bottom:.6rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="font-size:16px;font-weight:800;color:{C_TEXT};">{termo}</div>'
            f'<div style="font-size:10px;font-weight:700;color:{cor};text-transform:uppercase;letter-spacing:.06em;">{categoria}</div></div>'
            f'<div style="font-size:14px;color:{C_TEXT};line-height:1.6;margin-top:4px;">{definicao}</div>'
            f'<div style="font-size:13px;color:{C_TEXT_SEC};margin-top:4px;"><b>Exemplo:</b> {exemplo}</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════
# ASSISTENTE
# ══════════════════════════════════════════════════════
OBJECTIVOS = {
    "Prever um número (preço, temperatura, vendas...)": "regressao",
    "Prever uma categoria (sim/não, tipo A/B/C...)": "classificacao",
    "Encontrar grupos nos dados (sem categorias prévias)": "clustering",
    "Descobrir casos estranhos (fraudes, avarias...)": "anomalias",
    "Simplificar ou visualizar dados com muitas colunas": "reducao",
    "Trabalhar com texto (opiniões, mensagens...)": "nlp",
    "Trabalhar com imagens (fotos)": "visao",
    "Prever o futuro de uma série no tempo (vendas por dia...)": "series",
    "Ensinar um agente por tentativa e erro (jogos, robôs)": "reforco",
    "Gerar ou comprimir dados (autoencoders, GAN)": "generativo",
}

# destino -> (nome do módulo, algoritmo recomendado, porquê, alternativa)
FIXAS = {
    "clustering": ("Clustering", "K-Means (começa aqui) ou DBSCAN",
                   "O K-Means é simples e rápido. Se os grupos tiverem formas estranhas (anéis, luas) ou houver ruído, usa DBSCAN.",
                   "Gaussian Mixture se os grupos se sobrepõem."),
    "anomalias": ("Detecção de Anomalias", "Isolation Forest",
                  "Funciona bem sem exemplos de fraude: isola os pontos que são fáceis de separar do resto.",
                  "Local Outlier Factor se as anomalias são locais num grupo."),
    "reducao": ("Redução de Dimensionalidade", "PCA para resumir, t-SNE/UMAP para visualizar",
                "O PCA é rápido e dá-te um resumo fiel. Para ver grupos num gráfico 2D, t-SNE ou UMAP separam melhor.",
                "Nunca uses as coordenadas do t-SNE como features de outro modelo."),
    "nlp": ("NLP", "TF-IDF + Logistic Regression (começa aqui)",
            "Transforma o texto em números e classifica. É rápido, funciona bem e é fácil de explicar. Só passa para LSTM se for preciso.",
            "LSTM quando a ordem das palavras importa muito."),
    "visao": ("Visão Computacional", "CNN simples (ou transferência de aprendizagem)",
              "Redes convolucionais são feitas para imagens. Com poucas fotos, usa um modelo pré-treinado (MobileNet/ResNet).",
              "Transfer learning poupa tempo e dados."),
    "series": ("Séries Temporais", "ARIMA para séries simples, LSTM/GRU para padrões complexos",
               "Começa pelo ARIMA: é leve e dá uma boa referência. Se não chegar, experimenta LSTM/GRU.",
               "Nunca misturar dados do futuro no treino."),
    "reforco": ("Aprendizagem por Reforço", "Q-Learning (começa aqui), DQN para ambientes grandes",
                "O Q-Learning cabe numa tabela e ajuda a perceber a ideia. O DQN usa uma rede neuronal quando há demasiados estados.",
                "Começa pelo GridWorld para ver o agente a aprender."),
    "generativo": ("Modelos Generativos", "Autoencoder (começa aqui), VAE ou GAN",
                   "O Autoencoder comprime e reconstrói dados. O VAE e a GAN geram dados novos.",
                   "GANs são mais difíceis de treinar."),
}

# tabela: (tipo, prioridade, tamanho) -> (algoritmo, porquê, alternativa)
def _recomendar_supervisionado(tipo: str, prioridade: str, tamanho: str):
    clf = tipo == "classificacao"
    if prioridade.startswith("Explicar"):
        if clf:
            return ("Decision Tree (ou Logistic Regression)",
                    "A árvore mostra as perguntas que faz, uma a uma, e qualquer pessoa consegue seguir. A Logistic Regression mostra o peso de cada coluna.",
                    "Random Forest se a árvore sozinha for pouco precisa (mas fica mais difícil de explicar).")
        return ("Linear Regression (ou Ridge)",
                "A recta mostra directamente quanto cada coluna influencia o resultado.",
                "Decision Tree Regressor se a relação não for uma linha recta.")
    if prioridade.startswith("Rapidez"):
        if clf:
            return ("Logistic Regression (ou Naive Bayes)",
                    "São rápidos, treinam em segundos, dão um bom ponto de partida e raramente dão problemas.",
                    "KNN se o conjunto for pequeno.")
        return ("Linear Regression (ou Ridge)",
                "Rápida, simples e uma boa referência para comparar com modelos mais complicados.",
                "SGD Regressor para tabelas muito grandes.")
    # melhor nota
    if tamanho.startswith("Poucas"):
        if clf:
            return ("Random Forest (ou SVM)",
                    "Com poucos dados, a Random Forest é robusta e quase não precisa de ajustes. O SVM também brilha em conjuntos pequenos.",
                    "Gradient Boosting para tentar espremer mais uns pontos.")
        return ("Random Forest Regressor (ou SVR)",
                "Robusta com poucos dados e quase não precisa de ajustes.",
                "Gradient Boosting Regressor.")
    if clf:
        return ("Gradient Boosting (XGBoost / LightGBM)",
                "Costuma ganhar em tabelas. Cada árvore corrige os erros da anterior. O LightGBM é o mais rápido em tabelas grandes.",
                "Random Forest, se quiseres menos ajustes.")
    return ("Gradient Boosting Regressor (XGBoost / LightGBM)",
            "Costuma ter o menor erro em tabelas, aprendendo com os erros de cada árvore anterior.",
            "Random Forest Regressor, se quiseres menos ajustes.")


def _cartao_resultado(modulo: str, algoritmo: str, porque: str, alternativa: str):
    st.markdown(
        f'<div style="background:{C_SURFACE};border:2px solid {C_GREEN};border-radius:14px;padding:1.2rem 1.4rem;margin:1rem 0;">'
        f'<div style="font-size:11px;font-weight:800;color:{C_GREEN};text-transform:uppercase;letter-spacing:.07em;">A minha recomendação</div>'
        f'<div style="font-size:22px;font-weight:800;color:{C_TEXT};margin:4px 0 8px;">{algoritmo}</div>'
        f'<div style="font-size:15px;color:{C_TEXT};line-height:1.7;"><b>Porquê:</b> {porque}</div>'
        f'<div style="font-size:14px;color:{C_TEXT_SEC};line-height:1.6;margin-top:6px;"><b>Alternativa:</b> {alternativa}</div>'
        f'<div style="font-size:13px;color:{C_TEXT_MUTE};margin-top:8px;">Módulo da plataforma: {modulo}</div></div>',
        unsafe_allow_html=True,
    )


def render_assistente(username: str):
    page_header("Que algoritmo devo usar?",
                "Responde a 1 a 3 perguntas e eu digo-te por onde começar")

    obj = st.radio("1. O que queres fazer?", list(OBJECTIVOS.keys()), index=None, key="ass_obj")
    if obj is None:
        st.info("Escolhe uma opção para continuar.")
        return
    destino = OBJECTIVOS[obj]

    if destino in FIXAS:
        modulo, algoritmo, porque, alternativa = FIXAS[destino]
        _cartao_resultado(modulo, algoritmo, porque, alternativa)
        if st.button(f"Abrir {modulo}", type="primary", key="ass_abrir"):
            ir_para(destino)
        return

    # classificação / regressão: 2 perguntas extra
    tamanho = st.radio("2. Quantas linhas (exemplos) tem a tua tabela?",
                       ["Poucas (menos de 1.000)", "Muitas (mais de 1.000)"], index=None, key="ass_tam", horizontal=True)
    prioridade = st.radio("3. O que é mais importante para ti?",
                          ["Explicar a decisão a outras pessoas", "Obter a melhor nota possível", "Rapidez e simplicidade"],
                          index=None, key="ass_prio")
    if tamanho is None or prioridade is None:
        st.info("Responde às perguntas 2 e 3 para receber a recomendação.")
        return

    algoritmo, porque, alternativa = _recomendar_supervisionado(destino, prioridade, tamanho)
    modulo = "Classificação" if destino == "classificacao" else "Regressão"
    _cartao_resultado(modulo, algoritmo, porque, alternativa)

    st.markdown("**Receita para começar (3 passos):**")
    st.markdown(f"1. Abre **{modulo}** e carrega os teus dados (ou escolhe um dataset de exemplo).\n"
                f"2. Escolhe **{algoritmo.split(' (')[0]}** e deixa as definições por omissão.\n"
                "3. Treina, lê a explicação dos resultados e compara com outro algoritmo no **Comparador de Modelos**.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"Abrir {modulo}", type="primary", key="ass_abrir2", width="stretch"):
            ir_para(destino)
    with c2:
        if st.button("Comparar vários algoritmos", key="ass_comp", width="stretch"):
            ir_para("comparador")
