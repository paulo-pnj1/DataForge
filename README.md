# DataForge EDU

**Aprende Machine Learning do zero, em português, com experiências que podes mexer.**
Feita em Angola para o contexto universitário lusófono.

## Por onde começar
1. Cria uma conta (separador **Criar conta**) ou usa a conta demo (só em modo local: `aluno` / `aluno123`).
2. Menu **COMEÇAR AQUI**: Escola de ML (14 lições), O Meu Primeiro Modelo, Que algoritmo uso?, Dicionário de ML.
3. O **Início** mostra sempre o teu próximo passo.

## O que tem
| Área | O que faz |
|---|---|
| Escola de ML | 14 lições com analogia, demonstração interactiva e mini-teste (inclui limpeza de dados, escalas e fuga de informação) |
| O Meu Primeiro Modelo | Treino em 1 clique, com 8 problemas (alguns angolanos e sem internet) e resultado em linguagem simples |
| Comparador de Modelos | Palpite, porquê ganhou, onde erram, fronteiras de decisão, "mexer e ver", curva de aprendizagem, custo real, relatório PDF |
| Diagnóstico de CSV | Ao carregar dados, avisa de valores em falta, IDs, outliers, classes desequilibradas... |
| Revisão do dia | Repete as perguntas que falhaste ao fim de 1, 3, 7, 14 e 30 dias |
| Projecto Final | Projecto guiado em 5 passos com relatório em PDF |
| Conquistas | Medalhas, sequência de dias, desafio da semana e ranking opcional da turma |
| Certificados | PDF com QR code e página pública de verificação (`?verificar=CODIGO`) |
| Painel do Professor | Turmas com código, progresso dos alunos, lições mais difíceis, CSV, repor palavra-passe |
| Contas | Registo, turma, mudar palavra-passe, papéis (aluno/professor/admin), Google opcional |
| Pesquisa global | Caixa no menu que encontra lições, termos e páginas |

---

## Instalação local
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt                     # tudo: núcleo, PyTorch, gymnasium e statsmodels
streamlit run app.py
```
Sem credenciais Supabase a app usa uma base local (`data/db.json`) e liga as contas demo.

## Pôr online com Supabase (progresso que não se perde)
1. Cria um projecto em supabase.com e abre o **SQL Editor**. Cola e executa o ficheiro `schema.sql`.
2. Em **Project Settings > API** copia o URL e a chave **service_role**.
3. No Streamlit Cloud: **App > Settings > Secrets**. Cola o conteúdo de `.streamlit/secrets.toml.example` e preenche
   `supabase_url`, `supabase_key`, `cookie_key`, `admin_code`, `admin_email`, `app_url` e os dados de email (`smtp_*`).
4. Faz **Reboot app**. Cria a tua conta no separador **Criar conta**, escolhe «Sou professor» e usa o `admin_code` como código de convite (só funciona para o primeiro administrador). Depois cria convites de professor em *Administração > Convites*.
5. Cria turmas no **Painel do Professor** e partilha o código com os alunos.

Segurança: a chave `service_role` fica só no servidor (nos secrets). As tabelas têm RLS activo sem políticas,
por isso a chave pública não consegue ler nada. Nunca ponhas o `secrets.toml` no GitHub.

> Nota: os projectos gratuitos do Supabase podem ser pausados após um período sem actividade. Confirma as regras actuais no site deles.

## Testes
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Estrutura
```
app.py                     # Entrada, menu e roteador (com tratamento de erros)
schema.sql                 # Tabelas do Supabase
modules/
├── settings.py  db.py     # Configuração e camada de dados (Supabase ou TinyDB local)
├── auth.py                # Login, registo, perfil, Google opcional
├── escola.py  revisao.py  gamificacao.py  projeto.py  certificados.py  professor.py
├── primeiro_modelo.py  comparador.py  explicacoes.py  diagnostico.py  ajuda.py  pesquisa.py
├── utils.py  dashboard.py
├── supervisionado/ nao_supervisionado/ redes_neurais/ deep_learning/ reforco/
tests/test_smoke.py
```

## O que ainda não existe
- Versão em inglês (a interface está toda escrita em português).

## Painéis e organizações (Fase 1)

- Cada pessoa entra no seu painel: **Administração**, **Professor**, **Aluno** ou **Business** (menu próprio; páginas de outros painéis são recusadas).
- **Escolas e empresas** são organizações (tabelas `organizacoes` e `membros`). As organizações não são criadas pelo admin; os membros entram com o código de convite (ao criar conta ou em *Perfil*).
- Quem é membro de uma empresa ganha o painel Business; o **administrador do sistema só tem o painel de Administração**: não acede aos painéis de professor, aluno ou business, não cria turmas nem organizações e não entra nelas (em *Organizações* só vê totais e pode suspender/reactivar).
- **Quem cria o quê:**
  - **Professor:** recebe um **convite de uso único** do admin (por email, válido 7 dias, ligado ao seu email). Cria a sua **escola** em *Perfil* e as suas **turmas**.
  - **Aluno:** regista-se com o código da turma, ou é **criado pelo professor** (em lote, com palavra-passe temporária que tem de trocar no primeiro acesso). Por defeito o professor tem de **aceitar** quem entra pelo código (pode desligar).
  - **Professor sem convite:** no ecrã *Criar conta* > «Sou professor» há «Não tens convite? Pede acesso». Preenche nome, email e escola; o admin recebe o pedido (email e aviso na Visão geral) e aprova ou recusa em *Administração > Convites*. Ao aprovar, o convite de uso único sai para o email do pedido. Volta a executar `schema.sql` para criar a tabela `pedidos_convite`.
  - **Empresa (Business):** o dono regista-se e a empresa fica **pendente até o admin aprovar**; depois recebe o código para os colegas.
  - **Admin:** cria convites, aprova ou recusa empresas, suspende organizações. Não entra nos painéis dos utilizadores.
- **Email (Brevo):** preenche `smtp_*`, `email_from` e `admin_email` nos segredos (ver `.streamlit/secrets.toml.example`). Em *Administração > Email* vês a quota diária, a fila e podes enviar um email de teste. Sem email configurado, tudo funciona: os códigos e links aparecem no ecrã para os enviares à mão. O código `admin_code` só serve para criar o primeiro administrador.
- **Base de dados já existente:** volta a executar `schema.sql` no Supabase (é seguro repetir) para criar as tabelas novas e a coluna `turmas.org_id`.
- **Painel Business (completo):** uma conta de empresa (papel `empresa`) só vê este painel: nada de lições, pontos, turmas ou ranking. Páginas:
  - **Início** (estado da empresa, 4 passos, problemas), **Dados** (CSV/Excel, diagnóstico automático, biblioteca partilhada, dados de exemplo),
    **Análises** (prever vendas, procura/stock, clientes que saem, fraude/anomalias, segmentos), **Previsões** (ficheiro novo, dados guardados ou um caso à mão),
    **Relatórios** (histórico + PDF executivo), **Equipa** (código de convite, papéis, remover) e **Perfil**.
  - Papéis: **gestor** (tudo), **analista** (carrega dados, treina, gera relatórios) e **membro** (consulta e usa os modelos).
  - A lógica está em `modules/business_core.py` (sem interface, testável) e os ecrãs em `modules/business.py`. Os dados e as análises guardam-se nas tabelas `bn_datasets` e `bn_analises` (**volta a correr o `schema.sql` no Supabase**; em modo local é automático).
  - Contas de empresa antigas (papel `aluno` + empresa, sem turma nem escola) passam a ver só o painel Business.
- **IA, opcional (gratuita com o Gemini ou paga com o Grok):** em **Análises** há um assistente que sugere o problema e as colunas a partir de uma frase, e em **Relatórios** um comentário para a gestão que também entra no PDF. Só seguem para a IA os nomes das colunas, as categorias e os números já calculados, **nunca linhas de dados**; tudo o que a IA propõe é validado contra os dados reais. Liga-se com `ia_api_key` e `ia_provedor` (`gemini` ou `xai`) nos segredos (opcionais: `ia_model`, `ia_base_url`, `ia_limite_diario`); serve qualquer serviço compatível com a API da OpenAI. Sem a chave a IA fica desligada. Limite diário por empresa (por omissão 30 pedidos). No nível gratuito da Google o conteúdo enviado pode ser usado para melhorar os produtos deles. Código em `modules/ia.py`.
- **Tutor de IA para o aluno:** «Perguntar à IA sobre isto» aparece nas **lições** (dúvidas sobre o que leste), nos **resultados dos mini-testes** (a IA não dá a resposta de perguntas ainda por responder) e **depois de cada treino** (o que significam os números, se é bom ou mau, como melhorar). Usa a mesma chave da IA (`ia_api_key`); cada aluno tem o seu limite diário (`ia_limite_aluno`, por omissão 20). Só segue texto das lições e números dos resultados, nunca nomes nem dados pessoais. Código em `modules/tutor_ia.py` e `perguntar_tutor` em `modules/ia.py`.
- **Poupar o limite da IA:** o limite do Gemini gratuito é da chave e não de cada aluno, por isso há (1) **cache** das respostas do tutor nas lições e nos testes, partilhada entre alunos (guarda só a resposta, nunca a pergunta nem quem a fez; não gasta limite); (2) **tecto global** por dia e por minuto (`ia_limite_global_dia`, `ia_limite_global_minuto`; os valores por omissão são prudentes: **confirma os limites do teu plano na consola e ajusta**) e **intervalo mínimo** entre pedidos do mesmo aluno (`ia_intervalo`, 5 s); (3) **modelo mais leve** para o tutor (`ia_model_tutor`); (4) **prioridade** a empresas e professores: uma parte do tecto diário (`ia_reserva_pct`, 30%) é só deles, e o tutor dos alunos «descansa» quando o resto acaba; (5) **explicações padrão sem IA** depois de cada treino (a IA só entra quando o aluno pergunta mais); (6) **chave de reserva** opcional (`ia_api_key_2`). O administrador vê o uso da chave em **Visão geral**.
- **IA no painel do professor:** resumo da turma, sugestão de aula de reforço (sobre a lição mais errada), exercícios de escolha múltipla, mensagens para um aluno ou para a turma e rascunho de feedback ao Projecto Final. Tudo aparece como **rascunho editável** para o professor rever. Os nomes dos alunos **nunca** seguem para a IA (a mensagem leva um marcador `{nome}` que o servidor preenche, ou retira); no feedback o professor cola o texto, emails e telefones são retirados automaticamente e tem de confirmar que tirou o nome. Cada professor tem o seu limite (`ia_limite_professor`, por omissão 30). Código em `modules/professor.py` e `modules/ia.py`.
- **Sair:** apaga o cookie no navegador **e** revoga no servidor as sessões abertas até esse momento (chave `sessao:<utilizador>` na tabela `progress`), por isso actualizar a página já não reabre a conta.
- **Inglês:** o motor de tradução aceita agora modelos com valores (`"Passo {} de {}"` → `"Step {} of {}"`) para as frases montadas com f-strings.

## Tema claro (branco)

- Interruptor **Tema claro** no menu lateral (por cima de *Sair*) e escolha **Escuro / Claro** em *Perfil*. A escolha fica guardada na conta e aplica-se ao entrar. **O tema por omissão é o claro (branco)**, incluindo no ecrã de login.
- Como funciona: as cores da paleta (`C_BG`, `C_TEXT`...) continuam a ser hex em Python (gráficos) e passam a variáveis CSS `--df-*` dentro de HTML, por isso o mesmo código serve os dois temas. As duas paletas estão em `TEMAS` (`modules/utils.py`); para ajustar uma cor, muda-a aí.
- Gráficos: `st.pyplot` e `st.plotly_chart` são envolvidos por `instalar_tema_graficos()` (chamado em `app.py`) e adaptam fundo, eixos, legenda e texto ao tema; as cores dos dados não mudam. Tabelas (`st.dataframe`) e gráficos nativos (`st.bar_chart`) convertem-se por filtro CSS no tema claro.
- Texto novo em HTML: usa `{C_TEXT}`, `{C_TEXT_SEC}`... ou `color:var(--df-text)`; evita `#FFFFFF` e `#161B27` fixos, porque ficam errados num dos temas.

## Idioma / Language (Português · English)

A interface está em português e pode mudar para inglês a qualquer momento: no ecrã de login e no menu lateral há um seletor **Idioma / Language**. A escolha fica guardada na conta (como o tema) e também aceita `?lang=en` no endereço.

Como funciona (para quem mexe no código):
- O código-fonte continua em português. `modules/i18n.py` traduz, no momento de mostrar, tudo o que passa por `st.*` (texto, botões, opções, tabelas, gráficos Plotly/Matplotlib e PDFs), usando `modules/locales/en.json`.
- O valor devolvido pelos widgets não muda (as opções só são traduzidas na apresentação), por isso a lógica da app não depende do idioma.
- Para acrescentar ou corrigir uma tradução, edita `modules/locales/en.json` (`"texto em português": "English text"`). Frases sem tradução aparecem em português.
- Ficam em português de propósito: os e-mails enviados, e os textos de exemplo dos datasets de NLP (são dados, não interface).
- Testes: `pytest -q tests/test_idioma.py`.

## Laboratório de ML (treino, comparação, AutoML, séries, modelos guardados)

Página **Laboratório de ML** (menu do aluno › Ferramentas). Carrega um CSV/Excel e:

- **Biblioteca**: 16 algoritmos de regressão, 13 de classificação e 11 de séries temporais, com descrição e disponibilidade (XGBoost, LightGBM, statsmodels e PyTorch são opcionais: se faltarem, o algoritmo aparece como «indisponível» e nada parte).
- **Treinar e comparar**: pré-processamento em `Pipeline` (ajustado só com o treino), validação cruzada, tabela comparativa ordenável, gráficos (real × previsto, matriz de confusão, ROC, Precision-Recall) e recomendação justificada. A recomendação usa só a validação cruzada do treino; o teste nunca decide.
- **AutoML**: escolhe os algoritmos compatíveis, com limite de modelos e de tempo. Não afina hiperparâmetros. «Melhor» = melhor entre os avaliados.
- **Séries temporais**: ordem cronológica, divisão treino/validação/teste temporal, validação progressiva, referências ingénuas, MAE/RMSE/MAPE/MASE; intervalos só nos modelos que os dão (ARIMA/SARIMAX).
- **Previsões** e **Modelos guardados**: modelos guardados por utilizador em `artifacts/modelos/` (joblib + metadados JSON + versões), com assinatura HMAC verificada antes de abrir. Define `modelos_key` em `secrets.toml` (ver `secrets.toml.example`). Em alojamentos com disco temporário os modelos podem desaparecer ao reiniciar.

O código está em `modules/ml_engine/` (sem Streamlit, testável) e `modules/laboratorio.py` (interface).
Testes do motor: `pytest -q tests/test_data_loader.py tests/test_preprocessing.py tests/test_regression.py tests/test_classification.py tests/test_model_comparison.py tests/test_auto_ml.py tests/test_model_persistence.py tests/test_prediction_service.py tests/test_forecasting.py`

## Países (mundo lusófono)

O DataForge EDU adapta-se a **Angola, Moçambique, Brasil e Portugal** (escolha em «País / Country», no menu lateral e no ecrã de entrada; fica guardada na conta).
O que muda: moeda, cidade e bairros dos datasets de exemplo e das lições, escala das notas (0–20, ou 0–10 no Brasil), ortografia
(Acordo de 1990 fora de Angola), a variante do português (no Brasil: «você», «tela», «arquivo»...) e os exemplos e a variante que o tutor de IA usa.

- Toda a configuração está em `modules/regiao.py` (dicionário `REGIOES`). Para acrescentar um país, copia uma entrada e preenche-a.
- Os nomes internos dos datasets não mudam; só o texto mostrado (`regiao.localizar`, chamada por `i18n.tr`). Angola é o ponto de partida e não é alterada.
- A adaptação do português do Brasil é automática (regras e dicionário). **Deve ser revista por um falante brasileiro** antes de se divulgar.
- O painel Business e o conteúdo do e-mail continuam com exemplos de Angola.
- Testes: `tests/test_regiao.py`.
