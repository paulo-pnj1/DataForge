# DataForge EDU

**Aprende Machine Learning do zero, em português, com experiências que podes mexer.**
Feita em Angola para o contexto universitário lusófono.

---

## Por onde começar (3 minutos)

1. Abre a app e entra com a conta demo (`aluno` / `aluno123`).
2. No menu, secção **COMEÇAR AQUI**:
   - **Escola de ML**: 11 lições de ~5 minutos, cada uma com analogia do dia-a-dia, uma demonstração interactiva e um mini-teste.
   - **O Meu Primeiro Modelo**: escolhes um problema, carregas num botão e o computador aprende. O resultado vem explicado em palavras simples.
   - **Que algoritmo uso?**: respondes a 1 a 3 perguntas e recebes uma recomendação.
   - **Dicionário de ML**: 54 termos explicados com exemplos.
3. O **Início** mostra sempre o teu **próximo passo** recomendado.

---

## O que há de novo

| Novidade | Para que serve |
|---|---|
| Escola de ML (11 lições) | Ensina desde "o que é ML" até redes neuronais, com sliders e jogos |
| O Meu Primeiro Modelo | Treino em 1 clique, sem configurar nada, com 6 problemas (incl. preços de casas em Luanda) |
| "Explica-me estes números" | Painel em Classificação e Regressão que traduz Accuracy, Recall, R², etc. para linguagem simples |
| Tooltips nas métricas | Passa o rato sobre cada métrica e lê o que significa |
| Que algoritmo uso? | Assistente de decisão com receita de 3 passos |
| Dicionário de ML | Pesquisável, com categorias e exemplos |
| Início com "próximo passo" | O sistema diz-te o que fazer a seguir |
| Comparador educativo (8 abas) | Palpite antes do resultado, porquê ganhou, onde cada modelo erra, fronteiras de decisão, "mexer e ver", curva de aprendizagem, custo real, relatório PDF |
| Dataset local de casas em Luanda | Funciona sem internet (o da Califórnia precisa de descarregar) |

### Correcções
- O progresso do percurso nunca avançava (nenhum módulo ficava "concluído"). Agora treinar num módulo marca-o como concluído.
- O módulo **Aprendizagem por Reforço** (Q-Learning e DQN) já existia mas o menu mostrava "Em breve". Está ligado.
- O DQN dava erro sem PyTorch; agora mostra um aviso amigável.
- No Comparador, a Regressão com Gradient Boosting usava um classificador por engano; e trocar de dataset esvaziava a lista de colunas.
- A "Dica do Dia" mudava a cada reinício; agora muda por dia.
- Módulos já não ficam "bloqueados": mostram "recomendado fazer antes" mas podes abrir qualquer um.
- Removido um `<script>` que o Streamlit nunca executava.

---

## Instalação

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt           # núcleo (Classificação, Regressão, Clustering, Escola...)
pip install -r requirements-dl.txt        # opcional: Deep Learning e Reforço (PyTorch, gymnasium)

streamlit run app.py
```

Abre em **http://localhost:8501**. O ficheiro `config.yaml` de utilizadores já vem incluído (e é recriado automaticamente se faltar).

## Contas de demonstração

| Utilizador | Password | Papel |
|-----------|----------|-------|
| `aluno` | `aluno123` | Aluno |
| `professor` | `prof123` | Professor |
| `admin` | `admin123` | Administrador |

> Antes de pôr em produção, muda estas palavras-passe e a `cookie.key` do `config.yaml`.

---

## Estrutura

```
app.py                       # Entrada, menu e roteador
config.yaml                  # Utilizadores
requirements.txt / requirements-dl.txt

modules/
├── escola.py                # NOVO  Escola de ML (11 lições interactivas + quizzes)
├── primeiro_modelo.py       # NOVO  Treino em 1 clique com explicação simples
├── ajuda.py                 # NOVO  Dicionário de ML + assistente "Que algoritmo uso?"
├── explicacoes.py           # NOVO  Métricas traduzidas para linguagem simples
├── utils.py                 # Paleta, CSS, datasets, teoria, progresso
├── auth.py   dashboard.py   comparador.py
├── supervisionado/          # classificacao.py, regressao.py
├── nao_supervisionado/      # clustering.py, reducao.py, anomalias.py
├── redes_neurais/           # mlp.py
├── deep_learning/           # visao.py, nlp.py, series_temporais.py, generativo.py
└── reforco/                 # q_learning.py, dqn.py
```

## Como acrescentar uma lição

Em `modules/escola.py`, cria uma função `_licao_N()` com a analogia e a demonstração, e acrescenta
um dicionário à lista `LICOES` (título, ideia numa frase, resumo, botão de prática e perguntas do quiz).
A primeira resposta de cada pergunta é a certa; o sistema mistura as opções automaticamente.

## Por fazer
- Certificados em PDF com QR code (a página ainda mostra "Em desenvolvimento").
- Modo professor: acompanhar o progresso dos alunos.
