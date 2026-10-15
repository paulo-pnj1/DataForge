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
pip install -r requirements.txt                     # núcleo
pip install -r requirements-dl.txt                  # opcional: Deep Learning e Reforço (PyTorch)
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
- O painel Business ainda é um marcador; o MVP vem na Fase 3.

## Tema claro (branco)

- Interruptor **Tema claro** no menu lateral (por cima de *Sair*) e escolha **Escuro / Claro** em *Perfil*. A escolha fica guardada na conta e aplica-se ao entrar; o ecrã de login é sempre escuro.
- Como funciona: as cores da paleta (`C_BG`, `C_TEXT`...) continuam a ser hex em Python (gráficos) e passam a variáveis CSS `--df-*` dentro de HTML, por isso o mesmo código serve os dois temas. As duas paletas estão em `TEMAS` (`modules/utils.py`); para ajustar uma cor, muda-a aí.
- Gráficos: `st.pyplot` e `st.plotly_chart` são envolvidos por `instalar_tema_graficos()` (chamado em `app.py`) e adaptam fundo, eixos, legenda e texto ao tema; as cores dos dados não mudam. Tabelas (`st.dataframe`) e gráficos nativos (`st.bar_chart`) convertem-se por filtro CSS no tema claro.
- Texto novo em HTML: usa `{C_TEXT}`, `{C_TEXT_SEC}`... ou `color:var(--df-text)`; evita `#FFFFFF` e `#161B27` fixos, porque ficam errados num dos temas.
