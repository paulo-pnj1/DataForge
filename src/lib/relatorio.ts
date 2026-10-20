import type { StatusInscricao } from "./tipos";

type Base = { valorDevido: number; valorPago: number; status: StatusInscricao };
type ParaRanking = Base & { quitadoEm: Date | null; professor: { nome: string } };

/** Quanto ainda falta a um professor (nunca negativo). */
export const faltaDe = (i: { valorDevido: number; valorPago: number }) => Math.max(0, i.valorDevido - i.valorPago);

/** Percentagem paga, de 0 a 100. */
export function percentagem(i: { valorDevido: number; valorPago: number }): number {
  return i.valorDevido > 0 ? Math.min(100, Math.round((i.valorPago / i.valorDevido) * 100)) : 0;
}

export type Totais = {
  devido: number; // total a pagar
  pago: number; // total pago
  falta: number; // soma do que falta a cada participante
  excedente: number; // pago acima do valor devido (não abate o que falta aos outros)
  quitados: number;
  total: number;
};

export function totais(linhas: Base[]): Totais {
  const t: Totais = { devido: 0, pago: 0, falta: 0, excedente: 0, quitados: 0, total: linhas.length };
  for (const i of linhas) {
    t.devido += i.valorDevido;
    t.pago += i.valorPago;
    t.falta += faltaDe(i);
    t.excedente += Math.max(0, i.valorPago - i.valorDevido);
    if (i.status === "QUITADO") t.quitados += 1;
  }
  return t;
}

export type Posicao<T> = { pos: number; item: T; pct: number };

/** Compara dois professores: quem quitou primeiro, depois quem pagou a maior parte. */
function comparar(a: ParaRanking, b: ParaRanking): number {
  const aq = a.status === "QUITADO";
  const bq = b.status === "QUITADO";
  if (aq !== bq) return aq ? -1 : 1;
  if (aq && bq) return (a.quitadoEm?.getTime() ?? Infinity) - (b.quitadoEm?.getTime() ?? Infinity);
  const ra = a.valorDevido > 0 ? a.valorPago / a.valorDevido : 0;
  const rb = b.valorDevido > 0 ? b.valorPago / b.valorDevido : 0;
  if (ra !== rb) return rb - ra;
  return b.valorPago - a.valorPago;
}

/**
 * Ranking de uma campanha. Quem já pagou tudo vem primeiro (por ordem de chegada),
 * depois quem pagou mais em percentagem. Empates partilham a posição.
 * Quem ainda não pagou nada fica à parte, sem posição.
 */
export function ranking<T extends ParaRanking>(linhas: T[]): { posicoes: Posicao<T>[]; semPagamento: T[] } {
  const nome = (a: T, b: T) => a.professor.nome.localeCompare(b.professor.nome, "pt");
  const com = linhas.filter((l) => l.valorPago > 0 || l.status === "QUITADO");
  const sem = linhas.filter((l) => !(l.valorPago > 0 || l.status === "QUITADO")).sort(nome);

  com.sort((a, b) => comparar(a, b) || nome(a, b));
  const posicoes: Posicao<T>[] = [];
  com.forEach((item, idx) => {
    const anterior = posicoes[idx - 1];
    const pos = anterior && comparar(anterior.item, item) === 0 ? anterior.pos : idx + 1;
    posicoes.push({ pos, item, pct: percentagem(item) });
  });
  return { posicoes, semPagamento: sem };
}

const MEDALHAS = ["🥇", "🥈", "🥉"];

/** Texto pronto a colar no grupo de WhatsApp: só posição, nome e percentagem (sem valores em Kz). */
export function textoRanking(opcoes: {
  instituicao: string;
  campanha: string;
  prazo: string;
  posicoes: Posicao<ParaRanking>[];
  totais: Totais;
  limite?: number;
}): string {
  const { instituicao, campanha, prazo, posicoes, totais: t, limite = 10 } = opcoes;
  const linhas = posicoes.slice(0, limite).map((p) => {
    const marca = p.pos <= 3 ? MEDALHAS[p.pos - 1] : `${p.pos}º`;
    const estado = p.item.status === "QUITADO" ? "pago ✅" : `${p.pct}%`;
    return `${marca} ${p.item.professor.nome}: ${estado}`;
  });
  const pct = t.devido > 0 ? Math.min(100, Math.round((t.pago / t.devido) * 100)) : 0;
  return [
    `🏆 *Ranking: ${campanha}*`,
    instituicao,
    "",
    ...linhas,
    "",
    `Juntos já chegámos a ${pct}% do valor. ${t.quitados} de ${t.total} já estão com tudo pago.`,
    `Prazo: ${prazo}. Quem ainda não pagou vai a tempo, cada passo conta! 🙏`,
  ].join("\n");
}
