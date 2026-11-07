"""Texto pedagógico das 8 etapas do Laboratório de Avaliação e Optimização (só conteúdo, sem lógica).

Cada etapa responde às mesmas 6 perguntas: o que é, porquê, como funciona, como interpretar, exemplo prático e um desafio.
"""

ETAPAS = [
    ("avaliar", "Avaliar"),
    ("generalizar", "Generalização"),
    ("comparar", "Comparar"),
    ("optimizar", "Optimizar"),
    ("guardar", "Guardar"),
    ("prever", "Prever"),
    ("implementar", "Implementar"),
    ("monitorizar", "Monitorizar"),
]

LICOES = {
    "avaliar": {
        "titulo": "Etapa 1 · Avaliar o modelo",
        "o_que": "Avaliar é medir, com números, quão bem o modelo responde a casos que nunca viu (o conjunto de teste).",
        "porque": "Um modelo que «parece bom» pode estar errado. Só as métricas dizem quanto confiar nele.",
        "como": "O modelo faz previsões para o teste. Compara-se cada previsão com a resposta real. Em classificação contam-se acertos e erros; em regressão mede-se o tamanho dos erros.",
        "interpretar": "Nenhuma métrica chega sozinha. Com classes muito desiguais, a accuracy engana: olha para recall, F1 e para a matriz de confusão. Em regressão, compara o erro com a escala do alvo.",
        "exemplo": "Num teste com 95 alunos que passam e 5 que chumbam, um modelo que responde sempre «passa» tem 95% de accuracy e não encontra nenhum aluno em risco (recall 0% nessa classe).",
        "desafio": {
            "pergunta": "Um modelo tem 96% de accuracy, mas 96% dos casos pertencem à mesma classe. O que concluis?",
            "opcoes": ["O modelo é excelente.", "A accuracy não chega: um palpite fixo já daria 96%. É preciso ver recall e F1.", "O modelo está com sobreajuste."],
            "certa": 1,
            "explicacao": "Quando uma classe domina, acertar muito é fácil. As métricas por classe mostram se o modelo encontra também a classe rara.",
        },
    },
    "generalizar": {
        "titulo": "Etapa 2 · Generalização",
        "o_que": "Generalizar é funcionar bem em dados novos, não só nos que serviram para treinar.",
        "porque": "Um modelo que decora o treino tem notas altas lá e fracas na vida real (sobreajuste). Um modelo simples demais falha em tudo (subajuste).",
        "como": "Comparam-se as notas do treino e do teste (e da validação cruzada). A diferença entre elas e a distância à referência simples dão indícios do que se passa.",
        "interpretar": "Treino muito acima do teste: sobreajuste. Treino e teste ambos fracos: subajuste. Notas próximas e acima da referência: bom sinal. Com poucos dados nada disto é conclusivo.",
        "exemplo": "Uma árvore de decisão sem limite de profundidade pode ter F1 de 100% no treino e 70% no teste: aprendeu o ruído do treino.",
        "desafio": {
            "pergunta": "O R² é 0,98 no treino e 0,55 no teste. Qual é o diagnóstico mais provável?",
            "opcoes": ["Subajuste.", "Sobreajuste.", "Bom ajustamento."],
            "certa": 1,
            "explicacao": "A grande queda do treino para o teste indica que o modelo decorou o treino e não generaliza.",
        },
    },
    "comparar": {
        "titulo": "Etapa 3 · Comparar algoritmos",
        "o_que": "Treinar vários algoritmos nos mesmos dados, com a mesma divisão e a mesma validação, para os comparar de forma justa.",
        "porque": "Nenhum algoritmo é o melhor em tudo. Só comparando se descobre o que serve para os teus dados.",
        "como": "Todos usam o mesmo treino e o mesmo teste. A escolha do melhor usa a validação cruzada só com o treino; o teste fica reservado para a nota final.",
        "interpretar": "Critérios diferentes podem eleger modelos diferentes (por exemplo, F1 e recall). Se a diferença for menor do que o desvio da validação cruzada, os modelos são equivalentes: prefere o mais simples e rápido.",
        "exemplo": "Uma floresta aleatória pode ganhar por 0,5 pontos de F1 a uma regressão logística, mas treinar 50 vezes mais devagar. Nesse caso a diferença raramente compensa.",
        "desafio": {
            "pergunta": "O modelo A tem F1 de 0,84 ± 0,05 e o modelo B 0,82 ± 0,04 na validação cruzada. O que fazes?",
            "opcoes": ["Escolho o A, porque tem a nota mais alta.", "Vejo-os como equivalentes e escolho o mais simples ou mais rápido.", "Descarto os dois."],
            "certa": 1,
            "explicacao": "A diferença (0,02) é menor do que o desvio (0,04 a 0,05): está dentro do ruído.",
        },
    },
    "optimizar": {
        "titulo": "Etapa 4 · Optimizar hiperparâmetros",
        "o_que": "Os hiperparâmetros são as definições do algoritmo que tu escolhes antes de treinar (profundidade, número de árvores, C...). Optimizar é procurar os melhores valores.",
        "porque": "Os valores por omissão servem, mas raramente são os melhores para os teus dados.",
        "como": "A procura em grelha (GridSearchCV) testa todas as combinações. A procura aleatória (RandomizedSearchCV) testa algumas ao acaso, e é mais rápida. As duas usam validação cruzada só com o treino.",
        "interpretar": "O teste só é usado no fim, uma vez, para comparar antes e depois. Se o modelo melhorar na validação cruzada mas não no teste, a procura ajustou-se demasiado aos blocos de validação.",
        "exemplo": "Numa floresta aleatória, testar 3 valores de n_estimators e 3 de max_depth são 9 combinações. Com 3 blocos de validação, são 27 treinos.",
        "desafio": {
            "pergunta": "Porque é que NÃO se escolhem hiperparâmetros olhando para o conjunto de teste?",
            "opcoes": ["Porque é mais lento.", "Porque o teste deixa de ser «dados novos» e a nota final fica optimista.", "Porque o teste é demasiado pequeno."],
            "certa": 1,
            "explicacao": "Se o teste decide, o modelo acaba por se ajustar a ele e já não temos uma nota honesta.",
        },
    },
    "guardar": {
        "titulo": "Etapa 5 · Guardar e exportar",
        "o_que": "Guardar o modelo treinado, com tudo o que é preciso para o voltar a usar, num ficheiro.",
        "porque": "Sem guardar, perde-se o trabalho ao fechar a aplicação. E sem o pré-processamento, o modelo dá resultados errados em dados novos.",
        "como": "Guarda-se o Pipeline completo (imputação, escala, one-hot e algoritmo), mais os metadados: algoritmo, hiperparâmetros, métricas, características e classes.",
        "interpretar": "O Pipeline garante que os dados novos recebem exactamente as mesmas transformações do treino. Guarda também as versões das bibliotecas: com versões diferentes, as previsões podem mudar.",
        "exemplo": "Se guardasses só o algoritmo, terias de repetir à mão a escala e a codificação das categorias, e qualquer diferença estragava as previsões.",
        "desafio": {
            "pergunta": "Porque se guarda o Pipeline inteiro e não só o algoritmo?",
            "opcoes": ["Para o ficheiro ficar maior.", "Para os dados novos serem transformados exactamente como no treino.", "Porque o algoritmo não se pode guardar sozinho."],
            "certa": 1,
            "explicacao": "O pré-processamento aprende com o treino (médias, categorias). Tem de viajar com o modelo.",
        },
    },
    "prever": {
        "titulo": "Etapa 6 · Testar com novos dados",
        "o_que": "Usar o modelo em casos que não participaram no treino, escrevendo-os à mão ou carregando um ficheiro CSV.",
        "porque": "É a prova real: o modelo serve para o que foi feito, prever casos novos.",
        "como": "Antes de prever, a plataforma valida as colunas obrigatórias, os tipos e os valores. Faltar uma coluna bloqueia; categorias novas ou valores fora do intervalo do treino geram avisos.",
        "interpretar": "Um aviso de «fora do intervalo» quer dizer que o modelo está a extrapolar: a previsão merece menos confiança. A confiança de um classificador é a probabilidade da classe prevista, não uma garantia.",
        "exemplo": "Um modelo treinado com casas de 40 a 200 m² dá uma previsão pouco fiável para uma casa de 900 m².",
        "desafio": {
            "pergunta": "Um valor novo está muito fora do intervalo visto no treino. O que fazes com a previsão?",
            "opcoes": ["Confio, o modelo é bom.", "Desconfio: o modelo está a extrapolar.", "Apago a coluna."],
            "certa": 1,
            "explicacao": "Fora do que viu no treino, o modelo não tem base para acertar.",
        },
    },
    "implementar": {
        "titulo": "Etapa 7 · Preparar a implementação",
        "o_que": "Deployment é pôr o modelo a funcionar fora do caderno: numa API, numa aplicação web ou dentro de outra aplicação.",
        "porque": "Um modelo que só existe no teu computador não ajuda ninguém. Mas implementar com cuidado é uma tarefa de engenharia.",
        "como": "Carrega-se o Pipeline guardado, recebe-se um pedido com os dados, valida-se, prevê-se e devolve-se a resposta. Os exemplos abaixo fazem exactamente isso.",
        "interpretar": "Os exemplos servem para aprender e para protótipos. Para produção real precisas de autenticação, limites de uso, registos, testes e infra-estrutura adequada. A plataforma não faz esse alojamento.",
        "exemplo": "Com FastAPI, outra aplicação envia um pedido POST com os dados de um aluno e recebe a previsão em JSON.",
        "desafio": {
            "pergunta": "O que falta num exemplo de FastAPI antes de o usar com dados de clientes reais?",
            "opcoes": ["Nada, já está pronto.", "Autenticação, validação mais rigorosa, registos, testes e alojamento seguro.", "Só um nome mais bonito."],
            "certa": 1,
            "explicacao": "O exemplo mostra a ideia. Produção exige segurança e operação.",
        },
    },
    "monitorizar": {
        "titulo": "Etapa 8 · Monitorizar o modelo",
        "o_que": "Acompanhar o modelo depois de implementado, para saber se continua a funcionar bem.",
        "porque": "O mundo muda. Um modelo treinado com dados de ontem pode falhar com os de amanhã (deriva de dados).",
        "como": "Comparam-se os dados novos com o retrato do treino (PSI), a distribuição das previsões e, quando já se sabe a resposta real, as métricas em dados novos.",
        "interpretar": "PSI abaixo de 0,10 é estável; entre 0,10 e 0,25 é deriva moderada; acima de 0,25 é forte. São regras práticas: a deriva é um sinal para investigar, não uma prova de que o modelo falha. Só as métricas com respostas reais o confirmam.",
        "exemplo": "Se uma aplicação de crédito começa a receber clientes de outra faixa de rendimentos, o PSI desse rendimento sobe, e convém reavaliar o modelo.",
        "desafio": {
            "pergunta": "O PSI de uma variável é 0,40. Qual é a conclusão mais correcta?",
            "opcoes": ["O modelo está com certeza errado.", "Há deriva forte nesta variável: convém investigar e talvez reavaliar o modelo.", "Não tem importância nenhuma."],
            "certa": 1,
            "explicacao": "A deriva mostra que os dados mudaram. Só com valores reais se sabe se o desempenho piorou.",
        },
    },
}
