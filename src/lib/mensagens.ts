import { db, ler } from "./db";
import type { Campanha, Instituicao, Professor } from "./tipos";
import { enviarModelo, enviarTexto, type Resultado } from "./whatsapp";
import { enviarSms, enviarSmsLote } from "./sms";
import { paraSms } from "./smsTexto";
import { canalDe } from "./canais";
import type { Unidade } from "./recorrencia";
import { diasRestantes, fmtData, kz, pluralDias, primeiroNome } from "./format";
import { fmtNumero, pagamentoCurto, pagamentoEmLinha } from "./dadosPagamento";
import { fraseMotivacional, fraseParabens, sugestaoDoDia } from "./motivacao";

/** Nomes dos modelos na Meta. Os mesmos 4 servem todas as instituições: o nome vai como variável. */
/** Payload do botão "Ver dados de pagamento" do modelo do lembrete. */
export const BOTAO_DADOS = "DADOS_PAGAMENTO";

export const MODELOS = {
  lembrete: "muanapay_lembrete_v2", // v2: 8 variáveis, inclui os dados de pagamento
  parabens: "muanapay_parabens",
  aprovado: "muanapay_aprovado",
  recebido: "muanapay_recebido", // pagamento em mãos registado pelo admin
  rejeitado: "muanapay_rejeitado",
} as const;

export type Ctx = {
  instituicao: Instituicao;
  professor: Professor;
  campanha: Campanha;
  valorDevido: number;
  valorPago: number;
  emAtraso?: number; // campanha mensal ou semanal: períodos cujo prazo já passou e ainda não estão pagos (o prazo da campanha é o do mais antigo)
  unidade?: Unidade; // mês ou semana
  anterior?: number; // dívida do ano anterior ainda por pagar
};

async function registar(
  instituicaoId: string,
  professorId: string | null,
  tipo: string,
  conteudo: string,
  r: Resultado,
  marcarContacto = true,
  extra: Record<string, unknown> = {},
) {
  const agora = new Date();
  if (!r.ok) console.error("[whatsapp] envio falhou:", tipo, r.erro);
  await db().collection("mensagens").add({
    instituicaoId,
    professorId,
    direcao: "ENVIADA",
    canal: "WHATSAPP",
    tipo,
    conteudo,
    waMessageId: r.id ?? null,
    status: r.ok ? "sent" : "failed",
    erro: r.erro ?? null,
    criadaEm: agora,
    ...extra,
  });
  // Se o mesmo número estiver em várias instituições, a última a escrever-lhe é a mais provável de receber a resposta.
  if (professorId && r.ok && marcarContacto) {
    await db().collection("professores").doc(professorId).update({ ultimoContactoEm: agora });
  }
}

/** Regista um SMS. Não mexe em ultimoContactoEm: o professor não pode responder a um SMS. */
async function registarSms(instituicaoId: string, professorId: string | null, tipo: string, conteudo: string, r: Resultado) {
  if (!r.ok) console.error("[sms] envio falhou:", tipo, r.erro);
  await db().collection("mensagens").add({
    instituicaoId,
    professorId,
    direcao: "ENVIADA",
    canal: "SMS",
    tipo,
    conteudo,
    waMessageId: null,
    smsId: r.id ?? null,
    status: r.ok ? "sent" : "failed",
    erro: r.erro ?? null,
    criadaEm: new Date(),
  });
}

function descreverPrazo(
  prazo: Date,
  emAtraso = 0,
  unidade: Unidade = { singular: "mês", plural: "meses" },
  anterior = 0,
): string {
  const partes: string[] = [];
  if (anterior > 0) partes.push(`inclui ${kz(anterior)} em dívida do ano anterior`);
  if (emAtraso > 0) {
    partes.push(`${emAtraso} ${emAtraso === 1 ? unidade.singular : unidade.plural} em atraso (desde ${fmtData(prazo)})`);
  } else {
    const d = diasRestantes(prazo);
    if (d > 1) partes.push(`${fmtData(prazo)} (restam ${pluralDias(d)})`);
    else if (d === 1) partes.push(`${fmtData(prazo)} (resta 1 dia)`);
    else if (d === 0) partes.push(`${fmtData(prazo)} (termina hoje)`);
    else partes.push(`${fmtData(prazo)} (já terminou)`);
  }
  return partes.join("; ");
}

/** Onde o professor deve mandar o comprovativo: um SMS não aceita respostas com foto. */
function viaWhatsapp(): string {
  const n = (process.env.WHATSAPP_NUMERO_PUBLICO ?? "").replace(/\D/g, "");
  if (!n) return "por WhatsApp para o número da muanaPay";
  return `por WhatsApp para ${n.length === 12 && n.startsWith("244") ? `+244 ${fmtNumero(n.slice(3))}` : fmtNumero(n)}`;
}

/**
 * Envia a mesma mensagem pelo canal da instituição:
 * - WHATSAPP: só o modelo do WhatsApp (como sempre foi);
 * - SMS: só o SMS;
 * - WHATSAPP_E_SMS: os dois;
 * - WHATSAPP_COM_SMS_RESERVA: WhatsApp; se a Meta recusar já, SMS logo; se a Meta aceitar mas depois não
 *   entregar (ex.: número sem WhatsApp), o webhook envia o SMS (guardado em `smsReserva`).
 * Conta como enviada se pelo menos um canal aceitou.
 */
async function despachar(
  inst: Instituicao,
  professor: Professor,
  tipo: string,
  logWhatsapp: string,
  viaWa: () => Promise<Resultado>,
  textoSms: string,
): Promise<Resultado> {
  const canal = canalDe(inst);
  const sms = paraSms(textoSms);
  const resultados: Resultado[] = [];

  if (canal !== "SMS") {
    const r = await viaWa();
    const extra = canal === "WHATSAPP_COM_SMS_RESERVA" && r.ok ? { smsReserva: sms } : {};
    await registar(inst.id, professor.id, tipo, logWhatsapp, r, true, extra);
    resultados.push(r);
  }

  const enviarAgora =
    canal === "SMS" || canal === "WHATSAPP_E_SMS" || (canal === "WHATSAPP_COM_SMS_RESERVA" && !resultados[0]?.ok);
  if (enviarAgora) {
    const r = await enviarSms(professor.telefone, sms);
    await registarSms(inst.id, professor.id, tipo, sms, r);
    resultados.push(r);
  }

  const bom = resultados.find((r) => r.ok);
  if (bom) return { ok: true, id: bom.id };
  return { ok: false, erro: resultados.map((r) => r.erro).filter(Boolean).join(" | ") || undefined };
}

export async function enviarLembrete(ctx: Ctx, n: number): Promise<Resultado> {
  const falta = Math.max(0, ctx.valorDevido - ctx.valorPago);
  const dias = diasRestantes(ctx.campanha.prazo);
  const params = [
    primeiroNome(ctx.professor.nome),
    ctx.instituicao.nome,
    ctx.campanha.nome,
    kz(falta),
    descreverPrazo(ctx.campanha.prazo, ctx.emAtraso, ctx.unidade, ctx.anterior),
    sugestaoDoDia(falta, dias, n),
    pagamentoEmLinha(ctx.instituicao),
    fraseMotivacional(n, ctx.instituicao.tomCristao),
  ];
  const pagar = pagamentoCurto(ctx.instituicao);
  const sms =
    `${ctx.instituicao.nome}: Olá ${primeiroNome(ctx.professor.nome)}, faltam ${kz(falta)} da campanha ${ctx.campanha.nome}. ` +
    `Prazo: ${descreverPrazo(ctx.campanha.prazo, ctx.emAtraso, ctx.unidade, ctx.anterior)}. ` +
    `${pagar ? `Paga por ${pagar}.` : "Pede os dados de pagamento à instituição."} Envia o comprovativo ${viaWhatsapp()}.`;
  return despachar(
    ctx.instituicao,
    ctx.professor,
    "lembrete",
    params.join(" | "),
    () => enviarModelo(ctx.professor.telefone, MODELOS.lembrete, params, BOTAO_DADOS),
    sms,
  );
}

export async function enviarParabens(ctx: Ctx): Promise<Resultado> {
  const params = [
    primeiroNome(ctx.professor.nome),
    ctx.campanha.nome,
    kz(ctx.valorPago),
    ctx.instituicao.nome,
    fraseParabens(ctx.instituicao.tomCristao),
  ];
  const sms =
    `Parabéns, ${primeiroNome(ctx.professor.nome)}! ${ctx.instituicao.nome}: o teu pagamento de ${kz(ctx.valorPago)} ` +
    `(${ctx.campanha.nome}) está completo e confirmado. Obrigado!`;
  return despachar(
    ctx.instituicao,
    ctx.professor,
    "parabens",
    params.join(" | "),
    () => enviarModelo(ctx.professor.telefone, MODELOS.parabens, params),
    sms,
  );
}

export async function enviarAprovado(ctx: Ctx, valorAprovado: number): Promise<Resultado> {
  const falta = Math.max(0, ctx.valorDevido - ctx.valorPago);
  const params = [
    primeiroNome(ctx.professor.nome),
    ctx.instituicao.nome,
    kz(valorAprovado),
    ctx.campanha.nome,
    kz(falta),
    descreverPrazo(ctx.campanha.prazo, ctx.emAtraso, ctx.unidade, ctx.anterior),
  ];
  const sms =
    `${ctx.instituicao.nome}: o teu comprovativo foi aprovado e registámos ${kz(valorAprovado)} em ${ctx.campanha.nome}. ` +
    `Ainda faltam ${kz(falta)} (prazo: ${descreverPrazo(ctx.campanha.prazo, ctx.emAtraso, ctx.unidade, ctx.anterior)}).`;
  return despachar(
    ctx.instituicao,
    ctx.professor,
    "aprovado",
    params.join(" | "),
    () => enviarModelo(ctx.professor.telefone, MODELOS.aprovado, params),
    sms,
  );
}

export async function enviarRecebido(ctx: Ctx, valorRecebido: number): Promise<Resultado> {
  const falta = Math.max(0, ctx.valorDevido - ctx.valorPago);
  const params = [
    primeiroNome(ctx.professor.nome),
    ctx.instituicao.nome,
    kz(valorRecebido),
    ctx.campanha.nome,
    kz(falta),
    descreverPrazo(ctx.campanha.prazo, ctx.emAtraso, ctx.unidade, ctx.anterior),
  ];
  const sms =
    `${ctx.instituicao.nome}: registámos o teu pagamento de ${kz(valorRecebido)} em ${ctx.campanha.nome}. ` +
    `Ainda faltam ${kz(falta)} (prazo: ${descreverPrazo(ctx.campanha.prazo, ctx.emAtraso, ctx.unidade, ctx.anterior)}). Obrigado!`;
  return despachar(
    ctx.instituicao,
    ctx.professor,
    "recebido",
    params.join(" | "),
    () => enviarModelo(ctx.professor.telefone, MODELOS.recebido, params),
    sms,
  );
}

export async function enviarRejeitado(instituicao: Instituicao, professor: Professor, motivo: string): Promise<Resultado> {
  const params = [primeiroNome(professor.nome), instituicao.nome, motivo];
  const sms =
    `${instituicao.nome}: não foi possível aprovar o comprovativo que enviaste (motivo: ${motivo}). ` +
    `Envia de novo a foto ou o PDF correto ${viaWhatsapp()}.`;
  return despachar(
    instituicao,
    professor,
    "rejeitado",
    params.join(" | "),
    () => enviarModelo(professor.telefone, MODELOS.rejeitado, params),
    sms,
  );
}

/** Resposta livre dentro da janela de 24h aberta pela mensagem do professor (só WhatsApp: o professor escreveu-nos lá). */
export async function responderTexto(professor: Professor, texto: string, marcarContacto = true): Promise<Resultado> {
  const r = await enviarTexto(professor.telefone, texto);
  await registar(professor.instituicaoId, professor.id, "texto", texto, r, marcarContacto);
  return r;
}

/**
 * Chamado pelo webhook quando o WhatsApp diz que NÃO entregou uma mensagem (ex.: número sem WhatsApp).
 * Se a mensagem guardou um SMS de reserva, envia-o, uma só vez.
 */
export async function enviarReservaSms(mensagemId: string): Promise<void> {
  const ref = db().collection("mensagens").doc(mensagemId);
  const reserva = await db().runTransaction(async (t) => {
    const d = (await t.get(ref)).data();
    if (!d?.smsReserva || d.smsReservaEnviadaEm || !d.professorId) return null;
    t.update(ref, { smsReservaEnviadaEm: new Date() });
    return { texto: String(d.smsReserva), instituicaoId: String(d.instituicaoId), professorId: String(d.professorId), tipo: String(d.tipo) };
  });
  if (!reserva) return;
  const professor = ler<Professor>(await db().collection("professores").doc(reserva.professorId).get());
  if (!professor) return;
  const r = await enviarSms(professor.telefone, reserva.texto);
  await registarSms(reserva.instituicaoId, professor.id, reserva.tipo, reserva.texto, r);
}

export type ResumoSms = { enviados: number; falhas: number; erro?: string };

/** SMS escrito pelo admin (página SMS), o mesmo texto para os professores escolhidos. */
export async function enviarSmsAosProfessores(
  instituicao: Instituicao,
  professores: Professor[],
  texto: string,
): Promise<ResumoSms> {
  const sms = paraSms(texto);
  const resultados = await enviarSmsLote(professores.map((p) => p.telefone), sms);

  let enviados = 0;
  let falhas = 0;
  let erro: string | undefined;
  for (let i = 0; i < professores.length; i += 400) {
    const lote = db().batch();
    for (const p of professores.slice(i, i + 400)) {
      const r = resultados.get(p.telefone) ?? { ok: false, erro: "Sem resposta da Ombala" };
      if (r.ok) enviados++;
      else {
        falhas++;
        erro ??= r.erro;
      }
      lote.set(db().collection("mensagens").doc(), {
        instituicaoId: instituicao.id,
        professorId: p.id,
        direcao: "ENVIADA",
        canal: "SMS",
        tipo: "sms",
        conteudo: sms,
        waMessageId: null,
        smsId: r.id ?? null,
        status: r.ok ? "sent" : "failed",
        erro: r.erro ?? null,
        criadaEm: new Date(),
      });
    }
    await lote.commit();
  }
  return { enviados, falhas, erro };
}
