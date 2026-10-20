import { apagarFicheiro } from "./cloudinary";
import { db, ler, listar } from "./db";
import { obterInstituicao } from "./instituicoes";
import { enviarAprovado, enviarParabens, enviarRecebido, enviarRejeitado } from "./mensagens";
import { atualizarPeriodos, atualizarPorInscricao, comRecorrencia, situacao } from "./recorrencia";
import type { Campanha, Comprovativo, Inscricao, PagamentoManual, Professor } from "./tipos";

type Res = { ok: true } | { ok: false; erro: string };

const JA_TRATADO = "JA_TRATADO";
const NAO_ENCONTRADO = "Comprovativo não encontrado ou já tratado.";

/** Marca o comprovativo como decidido, uma só vez (transação evita aprovar duas vezes). Só da própria instituição. */
async function decidir(id: string, instituicaoId: string, dados: Partial<Comprovativo>): Promise<Comprovativo | null> {
  const ref = db().collection("comprovativos").doc(id);
  try {
    return await db().runTransaction(async (tx) => {
      const c = ler<Comprovativo>(await tx.get(ref));
      if (!c || c.instituicaoId !== instituicaoId || c.status !== "PENDENTE") throw new Error(JA_TRATADO);
      tx.update(ref, { ...dados, decididoEm: new Date() });
      return c;
    });
  } catch (e) {
    if (e instanceof Error && e.message === JA_TRATADO) return null;
    throw e;
  }
}

/** Total pago = comprovativos aprovados + pagamentos em mãos ativos. É a única fonte de verdade do valorPago. */
async function somaPaga(inscricaoId: string): Promise<number> {
  const [aprovados, manuais] = await Promise.all([
    listar<Comprovativo>(
      db().collection("comprovativos").where("inscricaoId", "==", inscricaoId).where("status", "==", "APROVADO"),
    ),
    listar<PagamentoManual>(
      db().collection("pagamentosManuais").where("inscricaoId", "==", inscricaoId).where("estado", "==", "ATIVO"),
    ),
  ]);
  return aprovados.reduce((s, x) => s + (x.valorAprovado ?? 0), 0) + manuais.reduce((s, x) => s + x.valor, 0);
}

export async function aprovarComprovativo(instituicaoId: string, id: string, valor: number): Promise<Res> {
  if (!Number.isInteger(valor) || valor <= 0) return { ok: false, erro: "Indica um valor válido em Kz." };

  const c = await decidir(id, instituicaoId, { status: "APROVADO", valorAprovado: valor });
  if (!c) return { ok: false, erro: NAO_ENCONTRADO };

  await atualizarPorInscricao(c.inscricaoId); // campanha mensal: soma o mês que entrou em curso
  const pago = await somaPaga(c.inscricaoId);

  const iRef = db().collection("inscricoes").doc(c.inscricaoId);
  const i = ler<Inscricao>(await iRef.get());
  if (!i) return { ok: false, erro: "Inscrição não encontrada." };

  const quitado = pago >= i.valorDevido;
  const jaEstavaQuitado = i.status === "QUITADO";
  await iRef.update({
    valorPago: pago,
    status: quitado ? "QUITADO" : pago > 0 ? "PARCIAL" : "PENDENTE",
    quitadoEm: quitado ? (i.quitadoEm ?? new Date()) : null,
  });

  const instituicao = await obterInstituicao(instituicaoId);
  const professor = ler<Professor>(await db().collection("professores").doc(i.professorId).get());
  const campanha = ler<Campanha>(await db().collection("campanhas").doc(i.campanhaId).get());
  if (instituicao && professor && campanha) {
    const ctx = { instituicao, professor, ...comRecorrencia(campanha, { ...i, valorPago: pago }), valorDevido: i.valorDevido, valorPago: pago };
    if (quitado && !jaEstavaQuitado) await enviarParabens(ctx); // lembretes param: status QUITADO
    else if (!quitado) await enviarAprovado(ctx, valor);
  }
  return { ok: true };
}

export async function rejeitarComprovativo(instituicaoId: string, id: string, motivo: string): Promise<Res> {
  const texto = motivo.trim() || "o comprovativo não está legível ou não corresponde ao pagamento";
  const c = await decidir(id, instituicaoId, { status: "REJEITADO", motivoRejeicao: texto });
  if (!c) return { ok: false, erro: NAO_ENCONTRADO };

  const instituicao = await obterInstituicao(instituicaoId);
  const i = ler<Inscricao>(await db().collection("inscricoes").doc(c.inscricaoId).get());
  const professor = i && ler<Professor>(await db().collection("professores").doc(i.professorId).get());
  if (instituicao && professor) await enviarRejeitado(instituicao, professor, texto);
  return { ok: true };
}

/**
 * Muda o valor devido de um professor numa campanha e recalcula o estado.
 * Campanha mensal: o valor é a cota mensal deste professor (o total devido sai da cota × meses em curso).
 */
export async function alterarValorDevido(instituicaoId: string, inscricaoId: string, novoValor: number): Promise<Res> {
  if (!Number.isInteger(novoValor) || novoValor <= 0) return { ok: false, erro: "Indica um valor válido em Kz." };

  const ref = db().collection("inscricoes").doc(inscricaoId);
  const atual = ler<Inscricao>(await ref.get());
  if (!atual || atual.instituicaoId !== instituicaoId) return { ok: false, erro: "Inscrição não encontrada." };
  let campanha = ler<Campanha>(await db().collection("campanhas").doc(atual.campanhaId).get());
  if (campanha?.recorrente) campanha = (await atualizarPeriodos(campanha)).campanha;

  const resultado = await db().runTransaction(async (tx) => {
    const i = ler<Inscricao>(await tx.get(ref));
    if (!i || i.instituicaoId !== instituicaoId) return null;
    const mensal = campanha?.recorrente ? situacao(campanha, { ...i, valorPeriodo: novoValor }) : null;
    const devido = mensal ? mensal.devido : novoValor;
    const quitado = i.valorPago >= devido;
    tx.update(ref, {
      valorDevido: devido,
      ...(mensal ? { valorPeriodo: novoValor } : {}),
      status: quitado ? "QUITADO" : i.valorPago > 0 ? "PARCIAL" : "PENDENTE",
      quitadoEm: quitado ? (i.quitadoEm ?? new Date()) : null,
    });
    return { antes: i, devido };
  });
  if (!resultado) return { ok: false, erro: "Inscrição não encontrada." };
  const { antes, devido } = resultado;

  // Baixar o valor até ao que já foi pago quita o professor: lembretes param e recebe parabéns.
  const passouAQuitado = antes.status !== "QUITADO" && antes.valorPago >= devido;
  if (passouAQuitado) {
    const instituicao = await obterInstituicao(instituicaoId);
    const professor = ler<Professor>(await db().collection("professores").doc(antes.professorId).get());
    if (instituicao && professor && campanha) {
      await enviarParabens({ instituicao, professor, campanha, valorDevido: devido, valorPago: antes.valorPago });
    }
  }
  return { ok: true };
}

/**
 * O professor pagou em mãos: o admin regista o valor. Soma ao total, atualiza o estado e (opcional) avisa no WhatsApp.
 * Se o pagamento ficar completo, os lembretes param e o professor recebe os parabéns.
 */
export async function registarPagamentoManual(
  instituicaoId: string,
  inscricaoId: string,
  valor: number,
  nota: string,
  avisar: boolean,
): Promise<Res> {
  if (!Number.isInteger(valor) || valor <= 0) return { ok: false, erro: "Indica um valor válido em Kz." };

  await atualizarPorInscricao(inscricaoId); // campanha mensal: soma o mês que entrou em curso
  const iRef = db().collection("inscricoes").doc(inscricaoId);
  const antes = ler<Inscricao>(await iRef.get());
  if (!antes || antes.instituicaoId !== instituicaoId) return { ok: false, erro: "Inscrição não encontrada." };

  await db().collection("pagamentosManuais").add({
    instituicaoId,
    inscricaoId,
    valor,
    nota: nota.trim() || null,
    estado: "ATIVO",
    registadoEm: new Date(),
    anuladoEm: null,
  } satisfies Omit<PagamentoManual, "id">);

  const pago = await somaPaga(inscricaoId);
  const quitado = pago >= antes.valorDevido;
  await iRef.update({
    valorPago: pago,
    status: quitado ? "QUITADO" : pago > 0 ? "PARCIAL" : "PENDENTE",
    quitadoEm: quitado ? (antes.quitadoEm ?? new Date()) : null,
  });

  if (avisar) {
    const instituicao = await obterInstituicao(instituicaoId);
    const professor = ler<Professor>(await db().collection("professores").doc(antes.professorId).get());
    const campanha = ler<Campanha>(await db().collection("campanhas").doc(antes.campanhaId).get());
    if (instituicao && professor && campanha) {
      const ctx = { instituicao, professor, ...comRecorrencia(campanha, { ...antes, valorPago: pago }), valorDevido: antes.valorDevido, valorPago: pago };
      if (quitado && antes.status !== "QUITADO") await enviarParabens(ctx);
      else if (!quitado) await enviarRecebido(ctx, valor);
    }
  }
  return { ok: true };
}

/** Engano ao registar? Anula: deixa de contar e o estado é recalculado. Não avisa o professor. */
export async function anularPagamentoManual(instituicaoId: string, id: string): Promise<Res> {
  const ref = db().collection("pagamentosManuais").doc(id);
  const p = ler<PagamentoManual>(await ref.get());
  if (!p || p.instituicaoId !== instituicaoId || p.estado !== "ATIVO") {
    return { ok: false, erro: "Pagamento não encontrado ou já anulado." };
  }
  await ref.update({ estado: "ANULADO", anuladoEm: new Date() });

  await atualizarPorInscricao(p.inscricaoId);
  const iRef = db().collection("inscricoes").doc(p.inscricaoId);
  const i = ler<Inscricao>(await iRef.get());
  if (!i) return { ok: false, erro: "Inscrição não encontrada." };
  const pago = await somaPaga(p.inscricaoId);
  const quitado = pago >= i.valorDevido;
  await iRef.update({
    valorPago: pago,
    status: quitado ? "QUITADO" : pago > 0 ? "PARCIAL" : "PENDENTE",
    quitadoEm: quitado ? (i.quitadoEm ?? new Date()) : null,
  });
  return { ok: true };
}

/** Recalcula valor pago e estado de uma inscrição a partir dos comprovativos aprovados e dos pagamentos em mãos. */
async function recalcularInscricao(inscricaoId: string): Promise<void> {
  await atualizarPorInscricao(inscricaoId);
  const iRef = db().collection("inscricoes").doc(inscricaoId);
  const i = ler<Inscricao>(await iRef.get());
  if (!i) return;
  const pago = await somaPaga(inscricaoId);
  const quitado = pago >= i.valorDevido;
  await iRef.update({
    valorPago: pago,
    status: quitado ? "QUITADO" : pago > 0 ? "PARCIAL" : "PENDENTE",
    quitadoEm: quitado ? (i.quitadoEm ?? new Date()) : null,
  });
}

/** Corrige o valor de um comprovativo já aprovado (engano ao aprovar). Não avisa o professor. */
export async function corrigirValorAprovado(instituicaoId: string, id: string, valor: number): Promise<Res> {
  if (!Number.isInteger(valor) || valor <= 0) return { ok: false, erro: "Indica um valor válido em Kz." };
  const ref = db().collection("comprovativos").doc(id);
  const c = ler<Comprovativo>(await ref.get());
  if (!c || c.instituicaoId !== instituicaoId || c.status !== "APROVADO") {
    return { ok: false, erro: "Só se pode corrigir o valor de um comprovativo aprovado." };
  }
  await ref.update({ valorAprovado: valor });
  await recalcularInscricao(c.inscricaoId);
  return { ok: true };
}

/** Apaga um comprovativo (e o ficheiro). Se estava aprovado, o total pago é recalculado. */
export async function apagarComprovativo(instituicaoId: string, id: string): Promise<Res> {
  const ref = db().collection("comprovativos").doc(id);
  const c = ler<Comprovativo>(await ref.get());
  if (!c || c.instituicaoId !== instituicaoId) return { ok: false, erro: "Comprovativo não encontrado." };
  await apagarFicheiro(c.cloudinaryId, c.cloudinaryTipo);
  await ref.delete();
  if (c.status === "APROVADO") await recalcularInscricao(c.inscricaoId);
  return { ok: true };
}
