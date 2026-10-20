import { db, listar, porIds } from "@/lib/db";
import { guardarComprovativo } from "@/lib/cloudinary";
import type { Campanha, Comprovativo, Inscricao, Instituicao, Mensagem, Professor } from "@/lib/tipos";
import { assinaturaValida, baixarMidia } from "@/lib/whatsapp";
import { BOTAO_DADOS, enviarReservaSms, responderTexto } from "@/lib/mensagens";
import { pagamentoEmBloco } from "@/lib/dadosPagamento";
import { diasRestantes, fmtData, kz } from "@/lib/format";
import { atualizarVariasCampanhas, comRecorrencia } from "@/lib/recorrencia";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type WaMsg = {
  id: string;
  from: string;
  type: string;
  context?: { id?: string }; // quando o professor "responde" a uma mensagem concreta
  text?: { body: string };
  button?: { payload?: string; text?: string }; // toque num botão de resposta rápida do modelo
  interactive?: { type?: string; button_reply?: { id?: string; title?: string } }; // outro formato de resposta a botão
  image?: { id: string; mime_type?: string; caption?: string };
  document?: { id: string; mime_type?: string; filename?: string; caption?: string };
};
type WaStatus = { id: string; status: string; errors?: { title?: string; message?: string }[] };
type WaValue = {
  metadata?: { phone_number_id?: string; display_phone_number?: string };
  messages?: WaMsg[];
  statuses?: WaStatus[];
};
type Payload = {
  entry?: { changes?: { value?: WaValue }[] }[];
};

const LIMITE_BYTES = 15 * 1024 * 1024;

// Passo de verificação do webhook (a Meta chama uma vez ao configurar)
export async function GET(req: Request) {
  const p = new URL(req.url).searchParams;
  if (p.get("hub.mode") === "subscribe" && p.get("hub.verify_token") === process.env.WHATSAPP_VERIFY_TOKEN) {
    return new Response(p.get("hub.challenge") ?? "", { status: 200 });
  }
  return new Response("Forbidden", { status: 403 });
}

export async function POST(req: Request) {
  const corpo = await req.text();
  if (!assinaturaValida(corpo, req.headers.get("x-hub-signature-256"))) {
    return new Response("Assinatura inválida", { status: 401 });
  }

  let payload: Payload;
  try {
    payload = JSON.parse(corpo) as Payload;
  } catch {
    return new Response("ok", { status: 200 });
  }

  // ID do número de WhatsApp do muanaPay. Eventos de qualquer outro número (ex.: muanaVenda) são ignorados.
  const MEU_PHONE_ID = process.env.WHATSAPP_PHONE_NUMBER_ID;
  if (!MEU_PHONE_ID) {
    console.warn("[webhook] WHATSAPP_PHONE_NUMBER_ID não definido: não consigo filtrar eventos de outros números!");
  }

  for (const entry of payload.entry ?? []) {
    for (const change of entry.changes ?? []) {
      const value = change.value;
      const phoneIdEvento = value?.metadata?.phone_number_id;
      if (MEU_PHONE_ID && phoneIdEvento !== MEU_PHONE_ID) {
        console.log("[webhook] evento ignorado: é de outro número", phoneIdEvento, value?.metadata?.display_phone_number);
        continue;
      }
      for (const s of value?.statuses ?? []) await atualizarEstado(s);
      for (const m of value?.messages ?? []) {
        try {
          await tratarMensagem(m);
        } catch (e) {
          console.error("Erro ao tratar mensagem", m.id, e);
        }
      }
    }
  }
  return new Response("ok", { status: 200 });
}

async function atualizarEstado(s: WaStatus) {
  const r = await db().collection("mensagens").where("waMessageId", "==", s.id).get();
  const dados = {
    status: s.status,
    erro: s.status === "failed" ? (s.errors?.[0]?.message ?? s.errors?.[0]?.title ?? "Falhou") : null,
  };
  await Promise.all(r.docs.map((d) => d.ref.update(dados)));

  // O WhatsApp não entregou (ex.: número sem WhatsApp): se a mensagem tem SMS de reserva, envia-o agora.
  if (s.status === "failed") {
    await Promise.all(
      r.docs.map((d) =>
        enviarReservaSms(d.id).catch((e) => console.error("[sms] reserva falhou:", d.id, e)),
      ),
    );
  }
}

const idSeguro = (waId: string) => waId.replace(/[\/]/g, "_");

type Candidata = { professor: Professor; instituicao: Instituicao; inscricao: Inscricao; campanha: Campanha };

/**
 * O mesmo número pode ser professor em várias instituições. Descobrimos de qual é esta mensagem:
 * 1. só uma instituição tem pagamento em aberto para ele -> essa;
 * 2. ele respondeu (função "Responder") a uma mensagem nossa -> a instituição dessa mensagem;
 * 3. a instituição que lhe escreveu por último;
 * 4. senão, pedimos-lhe que responda diretamente ao lembrete.
 */
async function escolherProfessor(
  m: WaMsg,
  validos: Professor[],
  candidatas: Candidata[],
): Promise<Professor | "ambiguo"> {
  if (validos.length === 1) return validos[0];

  const ids = [...new Set(candidatas.map((c) => c.professor.id))];
  if (ids.length === 0) return validos[0];
  if (ids.length === 1) return candidatas[0].professor;

  if (m.context?.id) {
    const r = await listar<Mensagem>(db().collection("mensagens").where("waMessageId", "==", m.context.id).limit(1));
    const daConversa = candidatas.find((c) => c.professor.instituicaoId === r[0]?.instituicaoId);
    if (daConversa) return daConversa.professor;
  }

  const ordenados = ids
    .map((id) => candidatas.find((c) => c.professor.id === id)!.professor)
    .sort((a, b) => (b.ultimoContactoEm?.getTime() ?? 0) - (a.ultimoContactoEm?.getTime() ?? 0));
  const [primeiro, segundo] = ordenados;
  if ((primeiro.ultimoContactoEm?.getTime() ?? 0) > (segundo.ultimoContactoEm?.getTime() ?? 0)) return primeiro;
  return "ambiguo";
}

/**
 * A Meta pode entregar a mesma mensagem mais de uma vez (reenvio, ou webhook configurado em vários níveis).
 * Reservamos o id da mensagem de forma atómica: só o primeiro pedido a conseguir criar o documento a trata.
 */
async function primeiraVez(waId: string): Promise<boolean> {
  try {
    await db().collection("eventosWhatsapp").doc(idSeguro(waId)).create({ criadoEm: new Date() });
    return true;
  } catch (e) {
    const err = e as { code?: number | string; message?: string };
    if (err.code === 6 || err.code === "already-exists" || /ALREADY_EXISTS/i.test(err.message ?? "")) return false;
    throw e; // outro erro: deixa tratar a mensagem (melhor duplicar do que perder)
  }
}


/**
 * Resposta automática de "estado do pagamento": no máximo 1 por professor por minuto.
 * Reserva atómica (create) para funcionar mesmo com vários pedidos em paralelo.
 */
async function podeResponderEstado(professorId: string): Promise<boolean> {
  const janela = Math.floor(Date.now() / 60_000);
  try {
    await db().collection("eventosWhatsapp").doc(`estado_${idSeguro(professorId)}_${janela}`).create({ criadoEm: new Date() });
    return true;
  } catch (e) {
    const err = e as { code?: number | string; message?: string };
    if (err.code === 6 || err.code === "already-exists" || /ALREADY_EXISTS/i.test(err.message ?? "")) return false;
    return true;
  }
}

async function tratarMensagem(m: WaMsg) {
  if (!(await primeiraVez(m.id).catch(() => true))) {
    console.log("[webhook] mensagem repetida ignorada", m.id);
    return;
  }
  console.log("[webhook] mensagem", m.id, m.type, "de", m.from, JSON.stringify(m.button ?? m.interactive ?? m.text ?? {}));
  const professores = await listar<Professor>(db().collection("professores").where("telefone", "==", m.from));
  if (professores.length === 0) {
    console.warn("[webhook] número sem participante registado:", m.from);
    return; // número desconhecido: ignora
  }

  const instituicoes = await porIds<Instituicao>("instituicoes", professores.map((p) => p.instituicaoId));
  const validos = professores.filter((p) => instituicoes.get(p.instituicaoId)?.ativa);
  if (validos.length === 0) {
    console.warn("[webhook] participante sem instituição ativa:", m.from);
    return;
  }

  // Inscrições em aberto (campanha ativa), por ordem de prazo
  const candidatas: Candidata[] = [];
  for (const p of validos) {
    const instituicao = instituicoes.get(p.instituicaoId) as Instituicao;
    let minhas = await listar<Inscricao>(db().collection("inscricoes").where("professorId", "==", p.id));
    const campanhas = await porIds<Campanha>("campanhas", minhas.map((i) => i.campanhaId));
    // Campanhas mensais: soma o mês que entrou em curso antes de ver o que está em aberto.
    if (await atualizarVariasCampanhas([...campanhas.values()])) {
      minhas = await listar<Inscricao>(db().collection("inscricoes").where("professorId", "==", p.id));
    }
    const abertas = minhas.filter((i) => i.status !== "QUITADO");
    for (const inscricao of abertas) {
      const campanha = campanhas.get(inscricao.campanhaId);
      if (campanha?.ativa) candidatas.push({ professor: p, instituicao, inscricao, campanha });
    }
  }
  candidatas.sort((a, b) => a.campanha.prazo.getTime() - b.campanha.prazo.getTime());

  const escolha = await escolherProfessor(m, validos, candidatas);
  const professor = escolha === "ambiguo" ? validos[0] : escolha;
  const emVarias = validos.length > 1;

  const midia = m.image ?? m.document;
  await db().collection("mensagens").add({
    instituicaoId: professor.instituicaoId,
    professorId: professor.id,
    direcao: "RECEBIDA",
    tipo: m.type,
    conteudo: m.text?.body ?? m.button?.text ?? midia?.caption ?? `[${m.type}]`,
    waMessageId: m.id,
    status: null,
    erro: null,
    criadaEm: new Date(),
  });

  if (escolha === "ambiguo") {
    const nomes = [...new Set(candidatas.map((c) => c.instituicao.nome))].join(" e ");
    await responderTexto(
      professor,
      `Tens pagamentos em aberto em mais de uma instituição (${nomes}). Para sabermos a qual te referes, responde diretamente à mensagem de lembrete: toca nela e escolhe "Responder", e depois envia o comprovativo.`,
      false,
    );
    return;
  }

  const alvo = candidatas.find((c) => c.professor.id === professor.id);
  if (!alvo) {
    await responderTexto(professor, "Não tens nenhum pagamento em aberto neste momento. Obrigado! 🙏", !emVarias);
    return;
  }
  const { inscricao, campanha, instituicao } = alvo;
  const falta = Math.max(0, inscricao.valorDevido - inscricao.valorPago);

  // Comprovativo: imagem ou PDF
  if (midia) {
    const ref = db().collection("comprovativos").doc(idSeguro(m.id));
    if ((await ref.get()).exists) return; // a Meta reenviou o mesmo evento

    try {
      const { buf, mime } = await baixarMidia(midia.id);
      if (buf.length > LIMITE_BYTES) throw new Error("Ficheiro demasiado grande");
      const guardado = await guardarComprovativo(buf, idSeguro(m.id), instituicao.id);
      const dados: Omit<Comprovativo, "id"> = {
        instituicaoId: instituicao.id,
        inscricaoId: inscricao.id,
        mediaId: midia.id,
        mimeType: mime || midia.mime_type || "application/octet-stream",
        nomeArquivo: m.document?.filename ?? null,
        legenda: midia.caption ?? null,
        cloudinaryId: guardado.publicId,
        cloudinaryTipo: guardado.tipo,
        cloudinaryFormato: guardado.formato,
        status: "PENDENTE",
        valorAprovado: null,
        motivoRejeicao: null,
        recebidoEm: new Date(),
        decididoEm: null,
      };
      await ref.create(dados);
      await responderTexto(
        professor,
        `Recebemos o teu comprovativo para "${campanha.nome}" (${instituicao.nome}). ✅ Assim que for aprovado, avisamos-te por aqui.`,
      );
    } catch (e) {
      console.error("Falha ao guardar comprovativo", e);
      await responderTexto(
        professor,
        "Não conseguimos receber o ficheiro. Por favor, envia novamente a foto ou o PDF do comprovativo.",
      );
    }
    return;
  }

  // Botão "Ver dados de pagamento" (ou o professor pergunta por IBAN, Multicaixa, conta...)
  // Aceita o payload que enviámos, o texto do botão (o payload pode vir vazio ou diferente) e o formato "interactive".
  const botaoPayload = m.button?.payload ?? m.interactive?.button_reply?.id;
  const botaoTexto = m.button?.text ?? m.interactive?.button_reply?.title ?? "";
  const pedeDados =
    botaoPayload === BOTAO_DADOS ||
    /ver dados/i.test(botaoTexto) ||
    /iban|nib|multicaixa|express|dados|transfer|como pag/i.test(m.text?.body ?? "");
  if (pedeDados) {
    await podeResponderEstado(professor.id); // ocupa a janela de 1 min: evita a resposta de estado logo a seguir
    const p = instituicao.pagamento;
    if (!p || (!p.multicaixaNumero && !p.iban)) {
      await responderTexto(
        professor,
        `${instituicao.nome} ainda não indicou os dados de pagamento por aqui. Fala com o responsável da instituição, por favor.`,
      );
      return;
    }
    await responderTexto(
      professor,
      `Para pagar "${campanha.nome}" (faltam ${kz(falta)}), usa uma destas opções. Cada dado vem numa mensagem à parte: toca e mantém premido para copiar.\n\n${pagamentoEmBloco(instituicao)}`,
    );
    if (p.multicaixaNumero) await responderTexto(professor, p.multicaixaNumero); // só o número, fácil de copiar
    if (p.iban) await responderTexto(professor, p.iban); // só o IBAN, sem espaços: os apps dos bancos aceitam
    await responderTexto(professor, "Depois de pagares, envia aqui a foto ou o PDF do comprovativo. 🙏");
    return;
  }

  // Reações, stickers, etc. não precisam de resposta automática
  if (m.type !== "text") return;
  // Várias mensagens seguidas do mesmo professor recebem só uma resposta de estado
  if (!(await podeResponderEstado(professor.id))) {
    console.log("[webhook] resposta de estado omitida (já enviada há menos de 1 min)", m.id);
    return;
  }

  // Qualquer outro texto: estado do pagamento + instrução
  const rec = comRecorrencia(campanha, inscricao);
  const dias = diasRestantes(rec.campanha.prazo);
  const anterior = rec.anterior > 0 ? `Inclui ${kz(rec.anterior)} em dívida do ano anterior. ` : "";
  const prazo =
    anterior +
    (rec.emAtraso > 0
      ? `Tens ${rec.emAtraso} ${rec.emAtraso === 1 ? rec.unidade.singular : rec.unidade.plural} em atraso (desde ${fmtData(rec.campanha.prazo)}), já somados ao valor em falta.`
      : dias >= 0
        ? `Prazo: ${fmtData(rec.campanha.prazo)}.`
        : "O prazo já terminou, mas podes regularizar a qualquer momento.");
  const comoPagar = pagamentoEmBloco(instituicao);
  await responderTexto(
    professor,
    `Olá, ${professor.nome.split(" ")[0]}! ${instituicao.nome} · "${campanha.nome}": já pagaste ${kz(inscricao.valorPago)} de ${kz(inscricao.valorDevido)}. Faltam ${kz(falta)}. ${prazo}` +
      (comoPagar ? `\n\nComo pagar:\n${comoPagar}` : "") +
      `\n\nDepois de pagares, envia aqui a foto ou o PDF do comprovativo.`,
  );
}
