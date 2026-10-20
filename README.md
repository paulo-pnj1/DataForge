# muanaPay

Cobranças e lembretes por WhatsApp, com SMS opcional (Ombala). Parte do ecossistema **muanaOS**.

**Várias instituições, um só número.** Cada cliente (EBD, igreja, escola…) é uma *instituição* com o seu admin, participantes e campanhas, totalmente separados. Todas enviam pelo mesmo número de WhatsApp ("MuanaPay") e o nome da instituição vai dentro de cada mensagem.

- O **participante** só usa o WhatsApp: recebe lembretes e envia o comprovativo (foto ou PDF).
- O **admin** só usa o painel web: vê os comprovativos, confirma o valor e aprova ou rejeita.
- Cada comprovativo aprovado soma ao total. Quando o total chega ao valor devido, os lembretes param e o participante recebe os parabéns. Na página de cada campanha podes editar o valor devido de cada participante; se o novo valor ficar igual ou abaixo do que já foi pago, o participante passa a "Pago" e recebe os parabéns.

## 1. Instalar

```bash
npm install
cp .env.example .env     # preenche as variáveis
npm run dev              # http://localhost:3000
```

Gera os segredos com `openssl rand -hex 32` (`SESSION_SECRET` e `CRON_SECRET`).

### Firebase (Firestore)

1. Na consola do Firebase cria um projeto e ativa o **Firestore Database** (modo produção).
2. Em **Definições do projeto > Contas de serviço** clica em **Gerar nova chave privada**.
3. Do JSON descarregado copia para o `.env`: `project_id` → `FIREBASE_PROJECT_ID`, `client_email` → `FIREBASE_CLIENT_EMAIL`, `private_key` → `FIREBASE_PRIVATE_KEY` (entre aspas, com os `\n`). Na Vercel cola a chave tal como está.
4. As coleções (`instituicoes`, `campanhas`, `professores`, `inscricoes`, `comprovativos`, `mensagens`) são criadas sozinhas. Só é preciso **um índice composto** (mensagens por instituição e data), que está em `firestore.indexes.json`: publica-o com `firebase deploy --only firestore:indexes`, ou abre o link que o Firestore mostra no erro da primeira vez que o Resumo carregar.
5. O acesso é só pelo servidor (Admin SDK). Mantém as regras do Firestore fechadas (`allow read, write: if false;`).

### Cloudinary

No dashboard do Cloudinary copia `Cloud name`, `API Key` e `API Secret` para o `.env`. Os comprovativos são guardados como ficheiros **authenticated** (privados): só abrem com um URL assinado, e o painel gera-o depois de confirmar a sessão do admin.

## 2. Configurar a WhatsApp Cloud API (Meta)

1. Em developers.facebook.com cria uma app do tipo **Business** e adiciona o produto **WhatsApp**.
2. Em **WhatsApp > API Setup** copia o `Phone number ID` para `WHATSAPP_PHONE_NUMBER_ID`.
3. Cria um **System User** no Business Manager, dá-lhe acesso à app e gera um **token permanente** com as permissões `whatsapp_business_messaging` e `whatsapp_business_management`. Põe em `WHATSAPP_ACCESS_TOKEN`.
4. Em **App settings > Basic** copia o **App Secret** para `WHATSAPP_APP_SECRET`.
5. Em **WhatsApp > Configuration > Webhook**:
   - Callback URL: `https://O-TEU-DOMINIO/api/whatsapp/webhook`
   - Verify token: o mesmo texto que puseste em `WHATSAPP_VERIFY_TOKEN`
   - Subscreve o campo **messages**.
6. Para testar localmente usa um túnel (ex.: `ngrok http 3000`).

## 3. Criar os 5 modelos de mensagem

Em **WhatsApp Manager > Message templates** cria estes modelos (o lembrete tem de ser o `muanapay_lembrete_v2`). Categoria **Utility**, idioma **Portuguese (PT)** (`pt_PT`; se usares outro, muda `WHATSAPP_TEMPLATE_LANG`). Os **nomes têm de ser exatamente estes**. A aprovação demora de minutos a 24 horas.

O nome da instituição é a variável `{{2}}` (ou `{{4}}`): **os mesmos 5 modelos servem todas as instituições**, sem nova aprovação da Meta a cada cliente.

**`muanapay_lembrete_v2`** (8 variáveis)

```
Olá {{1}}! Lembrete de pagamento, {{2}}. Campanha: {{3}}. Faltam {{4}}. Prazo: {{5}}. Hoje podes enviar {{6}}. Paga por: {{7}}. {{8}} Quando pagares, responde a esta mensagem com o comprovativo (foto ou PDF).
```
Exemplos: Maria / EBD Igreja X / Material 2026 / 2.500 Kz / 30/11/2026 (restam 9 dias) / 500 Kz / Multicaixa Express 923 456 789 (titular: EBD Igreja X); Transferência IBAN AO06 0040 0000 1234 5678 9012 3 (titular: EBD Igreja X) / Passo a passo chegamos lá: cada contribuição conta.

A variável `{{7}}` vem dos **Dados de pagamento** que cada instituição preenche no painel (menu *Dados de pagamento*). Se a instituição ainda não os preencheu, a variável diz "pede os dados de pagamento à tua instituição".

**Botão de resposta rápida (obrigatório neste modelo):** ao criar o modelo, em *Buttons* adiciona um botão **Quick reply** com o texto `Ver dados de pagamento`. Quando o participante lhe toca, o sistema responde com os dados e envia o número do Multicaixa e o IBAN **cada um numa mensagem só dele**, para copiar com um toque prolongado. O mesmo acontece se o participante escrever "iban", "multicaixa", "dados"...

O modelo antigo `muanapay_lembrete` (7 variáveis) deixa de ser usado: depois de o `_v2` ser aprovado podes apagá-lo.

**`muanapay_aprovado`** (6 variáveis)

```
Olá {{1}}! {{2}}: o teu comprovativo foi aprovado e registámos {{3}} em {{4}}. Ainda faltam {{5}} (prazo: {{6}}). Podes enviar o restante por aqui quando quiseres.
```
Exemplos: Maria / EBD Igreja X / 2.000 Kz / Material 2026 / 3.000 Kz / 30/11/2026 (restam 9 dias)

**`muanapay_recebido`** (6 variáveis): usado quando o admin regista um pagamento em mãos e ainda falta algum valor

```
Olá {{1}}! {{2}}: registámos o teu pagamento de {{3}} em {{4}}. Ainda faltam {{5}} (prazo: {{6}}). Obrigado!
```
Exemplos: Maria / EBD Igreja X / 2.000 Kz / Material 2026 / 3.000 Kz / 30/11/2026 (restam 9 dias)

**`muanapay_parabens`** (5 variáveis)

```
Parabéns, {{1}}! 🎉 O teu pagamento de {{2}} ({{3}}) em {{4}} está completo e confirmado. {{5}} Obrigado!
```
Exemplos: Maria / Material 2026 / 5.000 Kz / EBD Igreja X / Que Deus abençoe a tua dedicação.

**`muanapay_rejeitado`** (3 variáveis)

```
Olá {{1}}! {{2}}: não foi possível aprovar o comprovativo que enviaste. Motivo: {{3}}. Por favor, envia novamente a foto ou o PDF correto respondendo a esta mensagem.
```
Exemplos: Maria / EBD Igreja X / imagem cortada

Se a Meta reclassificar o `muanapay_lembrete` como **Marketing**, aceita: continua a funcionar, só custa mais por mensagem.

## 4. Lembretes automáticos

`GET /api/cron/lembretes` com o cabeçalho `Authorization: Bearer $CRON_SECRET` envia os lembretes em atraso.

- Só envia entre `LEMBRETES_HORA_INICIO` e `LEMBRETES_HORA_FIM` (hora de Luanda, por defeito 8h–20h).
- Respeita a frequência de cada campanha (campo "Lembrar a cada (dias)").
- Salta quem já enviou um comprovativo à espera de aprovação.
- Continua a lembrar mesmo depois do prazo, até o pagamento estar completo.

Na **Vercel**, o `vercel.json` já agenda uma chamada diária às 08:00 UTC (09:00 em Luanda), e a Vercel envia o `CRON_SECRET` sozinha. No plano Hobby o mínimo é 1 vez por dia. Se quiseres mais vezes, usa o plano Pro ou um serviço externo (cron-job.org, GitHub Actions) a chamar o endpoint a cada hora.

O botão **Enviar lembretes agora**, na página de cada campanha, envia já e ignora horário e frequência.

## 5. Instituições e fluxo completo

**Dono da plataforma (tu):** entras com `ADMIN_EMAIL`/`ADMIN_PASSWORD`, vais a **Instituições**, crias cada cliente (nome que aparece nas mensagens, email e senha do admin) e podes **Gerir** qualquer uma. Podes desativar uma instituição (pára lembretes e bloqueia o login) ou redefinir a senha do admin.

**Admin da instituição:** entra com o email e a senha que lhe deste e só vê os dados da sua instituição.

1. **Dados de pagamento:** preenche o Multicaixa Express e/ou o IBAN, cada um com o nome do titular. Vão em cada lembrete.
2. **Participantes:** adiciona ou importa nome e número de WhatsApp.
3. **Campanhas:** cria com valor total e prazo (inclui todos os participantes).
4. O cron envia o lembrete. A sugestão de valor ("hoje podes enviar 500 Kz") é calculada a partir do que falta e dos dias até ao prazo, e varia a cada envio, junto com uma frase de motivação.
5. O participante responde com a foto ou o PDF do comprovativo (ou paga em mãos, ver abaixo). É guardado e ele recebe a confirmação de receção.
6. **Comprovativos:** o admin vê a imagem, digita o valor e aprova ou rejeita. O participante é avisado no WhatsApp.
7. Pagamento completo: estado "Pago", lembretes param, mensagem de parabéns.

Se o participante escrever qualquer texto, o sistema responde com o estado do pagamento.

**Mesmo número em duas instituições:** o webhook decide a instituição (1) pela única que tem pagamento em aberto, (2) pela mensagem a que o participante respondeu com "Responder", (3) pela que lhe escreveu por último. Se continuar ambíguo, pede-lhe que responda diretamente ao lembrete. Por isso os modelos dizem "responde a esta mensagem".

### Pagamento em mãos

Na página de cada campanha, a coluna **Pagou em mãos** permite ao admin registar o valor que o participante entregou (com uma nota opcional, ex.: "Entregue ao tesoureiro"). O valor soma ao total junto com os comprovativos aprovados; se completar o pagamento, os lembretes param e o participante recebe os parabéns. A caixa **Avisar no WhatsApp** (ligada por defeito) envia-lhe a confirmação. Se registares por engano, **Anular** remove o valor e recalcula tudo (não avisa o participante). Os registos ficam na coleção `pagamentosManuais`.

### Campanhas por ano: pagamento único, mensal ou semanal

Cada campanha tem um **ano** (ex.: "Campanha 2026"). Em *Resumo* e *Campanhas* há um **seletor de ano** (o ano atual por defeito): escolhes 2025 e vês a campanha desse ano, com os totais, o ranking e a lista para imprimir. Os anos passados ficam para consulta e histórico. As campanhas que já tinhas ficam no ano do seu prazo.

Ao criar a campanha escolhes o **Pagamento**:

| Tipo | Como funciona |
|---|---|
| **Único** | Um valor e um prazo final, como sempre. |
| **Mensal** | Uma cota por mês, no **dia do mês** que escolheres, de janeiro a dezembro (ou desde a data de "Começa em"). Se o mês não tiver esse dia, o prazo é o último dia dele. |
| **Semanal** | Uma cota por semana, no **dia da semana** que escolheres (ex.: todas as quartas-feiras do ano: 52 ou 53). |

- **A cota de um mês entra em curso no dia 1 desse mês; a de uma semana, na segunda-feira dessa semana**, e vence no prazo.
- **O que ficou por pagar soma-se à cota seguinte.** Quem não pagou uma quarta-feira deve, na seguinte, as duas (`devido = cota × períodos em curso`). **Não há juros.**
- **Cada pagamento abate primeiro o que é mais antigo.** Quem pagou adiantado fica em dia até a cota acabar.
- Quem estava "Pago" e passa a dever o período novo volta a "Pendente" ou "Parcial", e os lembretes retomam (o cron faz isto sozinho, e o painel também ao abrir).
- **Lembretes e respostas:** o "Faltam" já inclui o que está em atraso, e o prazo passa a dizer, por exemplo, "2 semanas em atraso (desde 14/10/2026)". Sem modelos novos na Meta: cabe nas mesmas variáveis.
- **Cota própria:** na tabela da campanha o valor de cada participante passa a ser a **cota** dele (a coluna "Devido até agora" mostra o total).
- **Quem entra a meio** (adicionado depois do início) só paga do período em curso para a frente. Ao criar a campanha, todos começam no 1.º período (se a data de início for no passado, devem desde aí).
- **Editar** a cota, o 1.º prazo ou o nº de períodos recalcula o devido de todos (quem tem cota própria mantém-na).

**Passar de um ano para o seguinte.** Ao criar a campanha de 2027, em **"Copiar os participantes de"** escolhe a de 2026:
- os participantes ativos entram na nova (e as cotas próprias que tinham);
- com **"Passar as dívidas para a nova campanha"** ligado, o que cada um ficou a dever em 2026 **passa para 2027**: soma-se ao devido e **é abatido primeiro** pelos pagamentos novos. Os lembretes dizem "inclui X Kz em dívida do ano anterior";
- na campanha de 2026 essa dívida fica registada como **"Transitou X Kz"**, o valor devido passa a ser o que já foi pago, e a campanha **fecha** (pára de lembrar);
- só funciona depois de a campanha copiada ter terminado (último prazo passado). Antes disso o sistema avisa: cria a nova depois, ou desliga a opção para só copiar os participantes.

### Editar e apagar

Tudo o que se cria pode ser corrigido ou apagado no painel:

| O quê | Editar | Apagar |
|---|---|---|
| Instituição (só o dono da plataforma) | Nome, email do admin, tom cristão, nova senha | Pede para escrever o nome exato; apaga **tudo** da instituição |
| Participante | Nome e WhatsApp (página do participante) | Apaga também inscrições, comprovativos, pagamentos em mãos e mensagens dele |
| Campanha | Nome, valor, prazo e frequência (opção de aplicar o novo valor a quem ainda não pagou tudo) | Apaga também inscrições, comprovativos e pagamentos em mãos |
| Participante numa campanha | Valor devido; pagamento em mãos | "Remover da campanha" |
| Comprovativo aprovado | Corrigir o valor | Apagar (o total pago é recalculado) |
| Comprovativo rejeitado | | Apagar |

Os botões de apagar pedem confirmação. Apagar comprovativos apaga também o ficheiro no Cloudinary. Para parar lembretes sem perder dados, usa "Pausar" (participante), "Encerrar" (campanha) ou "Desativar" (instituição).

### Pesquisa e paginação

Todas as tabelas (instituições, participantes, campanhas, participantes de cada campanha, mensagens do Resumo e comprovativos) têm pesquisa (ignora acentos e maiúsculas), filtro por estado e paginação de 10 em 10 (6 nos comprovativos). O estado fica no endereço (`?q=maria&estado=ativos&pagina=2`), por isso funciona sem JavaScript e podes partilhar ou recarregar a página.

### Aspecto

Tema escuro com as cores do logotipo (ciano `#00d7fc`, azul `#0068fb`, violeta `#af5efb`, magenta `#ff3bff` sobre preto). O painel adapta-se a qualquer ecrã: barra lateral fixa em computadores, menu em gaveta em tablets e telemóveis, e as tabelas passam a cartões em ecrãs pequenos. Os logotipos estão em `public/` (fundo transparente) e os ícones do separador em `src/app/icon.png` e `src/app/apple-icon.png`.

### Enviar a campanha só a participantes escolhidos

- **Ao criar a campanha:** em "Quem recebe esta campanha?" escolhe *Todos os participantes ativos*, *Só os participantes que eu escolher* (lista com pesquisa e "Marcar todos") ou *Ninguém por agora*. Só quem entra na campanha recebe lembretes, automáticos ou manuais.
- **Depois de criada:** na página da campanha, "Adicionar participantes" mostra só os que ainda estão fora.
- **Lembrete manual a alguns:** marca os participantes na tabela e usa "Enviar lembrete" (ignora o horário; quem já pagou tudo ou está pausado é saltado). Também podes "Remover da campanha" os marcados. Máximo de 100 de cada vez; a seleção vale para a página que estás a ver.

### Rápido e leve

- **Sem recarregar a página:** guardar, apagar, aprovar, registar pagamentos e enviar lembretes correm em segundo plano. O botão fica a rodar, aparece uma mensagem no ecrã e só os dados que mudaram são atualizados. A pesquisa das tabelas atualiza ao escreveres (espera 300 ms).
- **Carregamento:** ao clicar numa ligação aparece logo um esqueleto da página. Não há fontes externas (usa as do sistema), o CSS é plano (sem degradés pesados nem desfoque) e os logotipos são WebP de poucos KB, guardados uma semana no navegador.
- **Menos leituras na base de dados:** a instituição é lida uma só vez por pedido, o Resumo carrega as 60 mensagens mais recentes e os comprovativos aprovados/rejeitados mostram os 100 mais recentes (os por analisar, até 300).
- **Feedback do clique:** botões e ligações têm um toque curto de "premido" (sem animações pesadas) e não esperam 300 ms em telemóvel.

## 6. SMS (Ombala)

Os SMS saem pela [Ombala](https://app.useombala.ao), um gateway angolano. Serve para quem não tem WhatsApp ou não tem internet, e para avisos escritos por ti no painel.

**Configurar**

1. Na Ombala cria a conta, gera o **token** da API e regista um **remetente** (nome, ex.: `MUANAPAY`, ou número). O remetente tem de ficar **aprovado** antes de o SMS sair.
2. No `.env` (e na Vercel) define `OMBALA_TOKEN`, `OMBALA_FROM` e, para os SMS dizerem para onde mandar o comprovativo, `WHATSAPP_NUMERO_PUBLICO` (ex.: `244923456789`). `OMBALA_BASE_URL` só muda se a Ombala mudar o endereço da API.
3. Em **Instituições > Gerir**, escolhe o **Canal das mensagens automáticas** de cada instituição. O SMS gasta créditos da tua conta Ombala, por isso só o dono da plataforma o liga.

**Canais**

| Canal | O que acontece |
|---|---|
| Só WhatsApp | Como sempre. SMS desligado e sem página SMS. |
| WhatsApp, e SMS se o WhatsApp falhar | Envia por WhatsApp. Se a Meta recusar logo, ou aceitar mas depois não entregar (ex.: número sem WhatsApp), envia o SMS, uma só vez. |
| Só SMS | Lembretes, aprovado, recebido, rejeitado e parabéns saem só por SMS. |
| WhatsApp e SMS | Os dois canais ao mesmo tempo. |

**Modelos dos SMS**

São os mesmos 5 avisos do WhatsApp, em versão curta. **Não precisam de aprovação**: ao contrário da Meta, a Ombala aceita qualquer texto. Para os mudar, edita as frases em `src/lib/mensagens.ts` (dentro de cada função `enviar...`, na variável `sms`). O texto sai sempre sem acentos (ver "Bom saber").

**Lembrete**
```
{instituição}: Olá {nome}, faltam {falta} da campanha {campanha}. Prazo: {prazo}. Paga por {pagamento}. Envia o comprovativo por WhatsApp para {número}.
```
Exemplo: `EBD Igreja X: Ola Maria, faltam 3.000 Kz da campanha Material 2026. Prazo: 28/10/2026 (restam 9 dias). Paga por Multicaixa Express 923 456 789. Envia o comprovativo por WhatsApp para +244 999 888 777.`

`{pagamento}` vem dos **Dados de pagamento** (Multicaixa Express e/ou IBAN, sem o nome do titular, para poupar caracteres). Se a instituição ainda não os preencheu, a frase passa a "Pede os dados de pagamento à instituição." Este SMS não leva a sugestão de valor do dia nem a frase de motivação do WhatsApp.

**Aprovado**
```
{instituição}: o teu comprovativo foi aprovado e registámos {valor} em {campanha}. Ainda faltam {falta} (prazo: {prazo}).
```

**Recebido** (pagamento em mãos)
```
{instituição}: registámos o teu pagamento de {valor} em {campanha}. Ainda faltam {falta} (prazo: {prazo}). Obrigado!
```

**Parabéns**
```
Parabéns, {nome}! {instituição}: o teu pagamento de {valor} ({campanha}) está completo e confirmado. Obrigado!
```

**Rejeitado**
```
{instituição}: não foi possível aprovar o comprovativo que enviaste (motivo: {motivo}). Envia de novo a foto ou o PDF correto por WhatsApp para {número}.
```

`{número}` é o `WHATSAPP_NUMERO_PUBLICO`; se não estiver definido, o SMS diz "por WhatsApp para o número da muanaPay". O lembrete ocupa normalmente 2 SMS; os outros ocupam 1 ou 2, conforme o tamanho do nome da instituição e da campanha.

**Página SMS** (menu, só com SMS ligado): o admin escreve um aviso e envia a todos os participantes ativos ou só aos escolhidos. Mostra quantos SMS a mensagem ocupa e o texto tal como chega.

**Bom saber**

- Um SMS **não aceita fotos nem respostas**: os SMS automáticos pedem o comprovativo por WhatsApp (`WHATSAPP_NUMERO_PUBLICO`).
- Os acentos são retirados (`ação` vira `acao`): com acentos o telemóvel usa um alfabeto que só deixa 70 caracteres por SMS em vez de 160. Cada SMS tem 160 caracteres (153 se a mensagem tiver mais de um SMS); o máximo aqui são 3 SMS.
- Só telemóveis angolanos (9 dígitos, começam por 9). Outros números falham com essa mensagem.
- Os SMS aparecem em **Resumo > Mensagens** com a etiqueta SMS. O estado é "enviada" quando a Ombala aceita; o relatório de entrega do SMS não é lido.
- O SMS de reserva só dispara se o webhook do WhatsApp estiver ligado (secção 2), porque é o webhook que avisa que a mensagem não foi entregue.

## Notas importantes

- **Consentimento:** a Meta exige que os participantes tenham aceitado receber mensagens da instituição pelo WhatsApp.
- **Limite diário:** todas as instituições partilham o limite de mensagens do número. Com a verificação do negócio na Meta o limite sobe.
- **Um número por instituição** (nome próprio, limites separados) é possível mais tarde, com Embedded Signup e estatuto de Tech Provider; os dados já estão separados por instituição para isso.
- **Janela de 24h:** respostas livres (receção do comprovativo, estado) só funcionam dentro de 24h após a mensagem do participante. Tudo o que o sistema inicia (lembrete, aprovado, rejeitado, parabéns) usa os modelos.
- **Comprovativos** ficam no Cloudinary (até 15 MB cada); no Firestore guarda-se só a referência e o estado.
- **Segurança:** o webhook valida a assinatura da Meta; o painel usa cookie assinado. O login não tem limite de tentativas, por isso usa uma senha forte (ou põe o painel atrás de uma proteção extra).

## Estrutura

```
src/lib/tipos.ts               modelo de dados (coleções do Firestore)
src/lib/instituicoes.ts         leitura de uma instituição
src/app/(admin)/instituicoes    gestão de instituições (dono da plataforma)
src/lib/db.ts                   ligação ao Firestore
src/lib/cloudinary.ts           upload e URL assinado dos comprovativos
src/lib/consultas.ts            leituras do painel
src/lib/whatsapp.ts             envio, download de mídia, assinatura
src/lib/sms.ts                  envio de SMS pela Ombala (um ou vários números)
src/lib/smsTexto.ts             texto de SMS: sem acentos, contagem de SMS
src/lib/canais.ts               canal de envio de cada instituição (WhatsApp, SMS, ambos)
src/lib/recorrencia.ts          campanhas mensais e semanais: períodos, acumulação, dívida transitada
src/lib/ano.ts                  ano das campanhas e seletor de ano
src/lib/mensagens.ts            os modelos de WhatsApp + os SMS equivalentes, e o envio pelo canal da instituição
src/lib/motivacao.ts            sugestão de valor e frases
src/lib/lembretes.ts            lógica do cron
src/lib/pagamentos.ts           aprovar/rejeitar e somar
src/components/Shell.tsx        barra lateral / menu em gaveta
src/components/FormAcao.tsx     formulário que corre ações sem recarregar a página
src/components/Toaster.tsx      mensagens rápidas
src/components/Seletor.tsx      escolher participantes (ao criar/adicionar à campanha)
src/components/BarraLote.tsx    ações sobre os participantes marcados na tabela
src/components/Filtro.tsx       pesquisa e filtros das tabelas (ao vivo)
src/components/Paginacao.tsx    paginação das tabelas
src/components/BotaoConfirmar.tsx  botão que pede confirmação (apagar)
src/components/CopiarBotao.tsx  botão "Copiar" (Multicaixa Express / IBAN)
src/components/FormularioSms.tsx  campos do SMS do admin (contador, pré-visualização, destinatários)
src/components/SeletorAno.tsx   escolher o ano das campanhas no painel
src/lib/lista.ts                pesquisa e paginação em memória
src/lib/apagar.ts               apagar em cascata (participante, campanha, instituição)
src/lib/dadosPagamento.ts       validação e formatação do Multicaixa Express / IBAN
src/app/api/whatsapp/webhook    recebe mensagens e comprovativos
src/app/api/cron/lembretes      endpoint do agendador
src/app/(admin)/sms              página para enviar SMS aos participantes
src/app/(admin)/participantes       gestão dos participantes
src/app/(admin)/…               painel web
```
