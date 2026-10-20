export type StatusInscricao = "PENDENTE" | "PARCIAL" | "QUITADO";
export type StatusComprovativo = "PENDENTE" | "APROVADO" | "REJEITADO";

/** Como a instituição envia as mensagens automáticas. Em falta = só WhatsApp. */
export type CanalEnvio = "WHATSAPP" | "WHATSAPP_COM_SMS_RESERVA" | "SMS" | "WHATSAPP_E_SMS";

/** Como os professores pagam à instituição. Todos os campos são opcionais: pode haver só Multicaixa Express ou só IBAN. */
export type DadosPagamento = {
  multicaixaNumero: string; // número associado ao Multicaixa Express, só dígitos (9)
  multicaixaTitular: string;
  iban: string; // sem espaços, maiúsculas: AO06...
  ibanTitular: string;
};

/** Cliente da muanaPay (uma EBD, uma igreja, uma escola...). Tudo o resto pertence a uma instituição. */
export type Instituicao = {
  id: string;
  nome: string; // aparece nas mensagens do WhatsApp, ex.: "EBD Igreja X"
  adminEmail: string; // minúsculas
  adminSenha: string; // scrypt: "sal:hash" em hexadecimal
  tomCristao: boolean; // frases bíblicas nas mensagens
  canal?: CanalEnvio; // WhatsApp, SMS ou ambos (só o dono da plataforma altera: o SMS gasta créditos)
  pagamento?: DadosPagamento | null; // dados bancários que aparecem nos lembretes
  ativa: boolean;
  criadaEm: Date;
};

export type Campanha = {
  id: string;
  instituicaoId: string;
  nome: string;
  valorTotal: number; // Kz
  prazo: Date; // pagamento único: o prazo final. Mensal: o prazo do último mês
  frequenciaHoras: number;
  ativa: boolean;
  criadaEm: Date;
  // Campanha mensal (cotas): valorTotal é a cota de cada mês e o que fica por pagar soma-se ao mês seguinte
  ano?: number; // ano da campanha (em falta: o ano do prazo)
  recorrente?: boolean; // mensal ou semanal
  periodicidade?: "MENSAL" | "SEMANAL"; // em falta = mensal
  diaMes?: number; // mensal: dia do mês do prazo
  periodos?: number; // quantos meses ou semanas dura
  meses?: number; // campanhas mensais antigas guardavam aqui o nº de meses (ver periodos)
  primeiroPrazo?: Date; // prazo do 1.º período; os outros caem no mesmo dia do mês ou da semana
  periodosAplicados?: number; // quantos períodos já estão somados nos valorDevido das inscrições
};

export type Professor = {
  id: string;
  instituicaoId: string;
  nome: string;
  telefone: string; // só dígitos, com indicativo: 244923456789
  ativo: boolean;
  criadoEm: Date;
  ultimoContactoEm?: Date | null; // última mensagem que lhe enviámos (desempata quando o número está em várias instituições)
};

export type Inscricao = {
  id: string; // `${campanhaId}_${professorId}`
  instituicaoId: string;
  campanhaId: string;
  professorId: string;
  valorDevido: number; // mensal: cota × meses em curso (já inclui o que ficou por pagar)
  valorPeriodo?: number; // mensal: cota própria deste participante (em falta = a cota da campanha)
  periodoInicial?: number; // 1.º período que lhe é cobrado (quem entra a meio não paga os anteriores)
  saldoInicial?: number; // dívida transitada do ano anterior (já está dentro de valorDevido)
  valorTransitado?: number; // dívida desta inscrição que passou para a campanha do ano seguinte
  valorDevidoOriginal?: number; // o valorDevido antes de a dívida transitar
  valorPago: number; // soma dos comprovativos aprovados
  status: StatusInscricao;
  ultimoLembreteEm: Date | null;
  lembretesEnviados: number;
  quitadoEm: Date | null;
};

export type Comprovativo = {
  id: string; // id da mensagem do WhatsApp (evita duplicados)
  instituicaoId: string;
  inscricaoId: string;
  mediaId: string;
  mimeType: string;
  nomeArquivo: string | null;
  legenda: string | null;
  cloudinaryId: string;
  cloudinaryTipo: string; // image | raw | video
  cloudinaryFormato: string | null;
  status: StatusComprovativo;
  valorAprovado: number | null;
  motivoRejeicao: string | null;
  recebidoEm: Date;
  decididoEm: Date | null;
};

/** Pagamento entregue em mãos ao responsável da instituição e registado à mão no painel. */
export type PagamentoManual = {
  id: string;
  instituicaoId: string;
  inscricaoId: string;
  valor: number; // Kz
  nota: string | null; // ex.: "Entregue ao tesoureiro"
  estado: "ATIVO" | "ANULADO"; // anular em vez de apagar: fica o histórico
  registadoEm: Date;
  anuladoEm: Date | null;
};

export type Mensagem = {
  id: string;
  instituicaoId: string;
  professorId: string | null;
  direcao: "ENVIADA" | "RECEBIDA";
  canal?: "WHATSAPP" | "SMS"; // em falta = WhatsApp
  tipo: string;
  conteudo: string;
  waMessageId: string | null;
  smsId?: string | null; // id da mensagem na Ombala
  status: string | null; // sent, delivered, read, failed
  erro: string | null;
  criadaEm: Date;
};
