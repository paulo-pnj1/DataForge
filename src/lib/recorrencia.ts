import { db, ler, listar } from "./db";
import { dataISO, fimDoDia } from "./format";
import type { Campanha, Inscricao } from "./tipos";

/**
 * Campanhas mensais e semanais (cotas).
 *
 * Cada mês (ou semana) é uma prestação com o seu prazo:
 * - mensal: o mesmo dia de cada mês (se o mês não tiver esse dia, o último dia dele);
 * - semanal: o mesmo dia da semana (ex.: todas as quartas-feiras).
 * A cota de um mês entra em curso no dia 1 desse mês; a de uma semana, na segunda-feira dessa semana.
 * O que ficou por pagar continua em dívida e soma-se à cota seguinte:
 *   valorDevido = dívida transitada do ano anterior + cota × períodos em curso
 * Cada pagamento abate primeiro o que é mais antigo (a dívida transitada, depois o período mais antigo).
 * Não há juros: só acumula o que ficou por pagar.
 */

export type Periodicidade = "MENSAL" | "SEMANAL";
export type Unidade = { singular: string; plural: string };

export const MAX_PERIODOS = 120;
const MES: Unidade = { singular: "mês", plural: "meses" };
const SEMANA: Unidade = { singular: "semana", plural: "semanas" };

export const DIAS_SEMANA = ["Domingo", "Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado"];

type Calendario = Pick<Campanha, "recorrente" | "periodicidade" | "diaMes" | "periodos" | "meses" | "primeiroPrazo">;

export const periodicidadeDe = (c: Pick<Campanha, "periodicidade">): Periodicidade => c.periodicidade ?? "MENSAL";
export const unidadeDe = (c: Pick<Campanha, "periodicidade">): Unidade => (periodicidadeDe(c) === "SEMANAL" ? SEMANA : MES);
/** Quantos períodos tem a campanha (as antigas guardavam em `meses`). */
export const totalPeriodos = (c: Pick<Campanha, "periodos" | "meses">): number => c.periodos ?? c.meses ?? 0;

type Recorrente = Campanha & { recorrente: true; primeiroPrazo: Date };
const eRecorrente = (c: Calendario & { valorTotal?: number }): boolean =>
  Boolean(c.recorrente && c.primeiroPrazo && totalPeriodos(c) > 0);

const dois = (n: number) => String(n).padStart(2, "0");
const iso = (d: Date) => `${d.getUTCFullYear()}-${dois(d.getUTCMonth() + 1)}-${dois(d.getUTCDate())}`;
const utc = (isoData: string) => {
  const [a, m, d] = isoData.split("-").map(Number);
  return new Date(Date.UTC(a, m - 1, d));
};
const somarDias = (d: Date, n: number) => new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + n));
const ultimoDiaDoMes = (ano: number, mesIdx: number) => new Date(Date.UTC(ano, mesIdx + 1, 0)).getUTCDate();

/** Data (AAAA-MM-DD, em Luanda) do prazo do período k (0 = primeiro). */
function dataDoPeriodo(c: Calendario, k: number): string {
  const base = utc(dataISO(c.primeiroPrazo!));
  if (periodicidadeDe(c) === "SEMANAL") return iso(somarDias(base, 7 * k));
  const dia = c.diaMes ?? base.getUTCDate();
  const alvo = new Date(Date.UTC(base.getUTCFullYear(), base.getUTCMonth() + k, 1));
  const a = alvo.getUTCFullYear();
  const m = alvo.getUTCMonth();
  return `${a}-${dois(m + 1)}-${dois(Math.min(dia, ultimoDiaDoMes(a, m)))}`;
}

/** Prazo (fim do dia em Luanda) do período k. */
export const prazoDoPeriodo = (c: Calendario, k: number): Date => fimDoDia(dataDoPeriodo(c, k));

/** Prazo do último período: o fim da campanha. */
export const prazoFinal = (c: Calendario): Date => prazoDoPeriodo(c, totalPeriodos(c) - 1);

/** Início (00:00 em Luanda) do período k: o dia 1 do mês do prazo, ou a segunda-feira da semana do prazo. */
function inicioDoPeriodo(c: Calendario, k: number): Date {
  const data = dataDoPeriodo(c, k);
  if (periodicidadeDe(c) === "MENSAL") return new Date(`${data.slice(0, 7)}-01T00:00:00+01:00`);
  const d = utc(data);
  const segunda = somarDias(d, -((d.getUTCDay() + 6) % 7));
  return new Date(`${iso(segunda)}T00:00:00+01:00`);
}

/** Quantos períodos já estão em curso: o 1.º logo, e os seguintes a partir do início de cada um. */
export function periodosEmCurso(c: Calendario, agora = new Date()): number {
  const total = totalPeriodos(c);
  let n = 1;
  while (n < total && agora >= inicioDoPeriodo(c, n)) n++;
  return n;
}

/**
 * Calendário de uma campanha de um ano: o 1.º prazo e quantos períodos cabem até 31 de dezembro.
 * Semanal: todas as semanas no dia da semana escolhido. Mensal: o dia escolhido de cada mês.
 * `inicio` (AAAA-MM-DD) permite começar a meio do ano; por defeito é 1 de janeiro.
 */
export function calendarioDoAno(opts: {
  periodicidade: Periodicidade;
  ano: number;
  inicio?: string;
  diaSemana?: number; // 0 = domingo ... 6 = sábado
  diaMes?: number; // 1 a 31
}): { primeiroPrazo: Date; periodos: number } | null {
  const inicio = opts.inicio && /^\d{4}-\d{2}-\d{2}$/.test(opts.inicio) ? opts.inicio : `${opts.ano}-01-01`;
  const fim = utc(`${opts.ano}-12-31`);
  let primeiro: string;
  let periodos = 0;

  if (opts.periodicidade === "SEMANAL") {
    const alvo = opts.diaSemana ?? 3;
    let d = utc(inicio);
    while (d.getUTCDay() !== alvo) d = somarDias(d, 1);
    primeiro = iso(d);
    for (let x = d; x <= fim; x = somarDias(x, 7)) periodos++;
  } else {
    const dia = opts.diaMes ?? 10;
    const ini = utc(inicio);
    let m = 0;
    let cand: Date;
    do {
      const a = ini.getUTCFullYear();
      const mes = ini.getUTCMonth() + m;
      const alvo = new Date(Date.UTC(a, mes, 1));
      cand = new Date(Date.UTC(alvo.getUTCFullYear(), alvo.getUTCMonth(), Math.min(dia, ultimoDiaDoMes(alvo.getUTCFullYear(), alvo.getUTCMonth()))));
      m++;
    } while (cand < ini);
    primeiro = iso(cand);
    for (let j = 0; ; j++) {
      const alvo = new Date(Date.UTC(cand.getUTCFullYear(), cand.getUTCMonth() + j, 1));
      const d = new Date(Date.UTC(alvo.getUTCFullYear(), alvo.getUTCMonth(), Math.min(dia, ultimoDiaDoMes(alvo.getUTCFullYear(), alvo.getUTCMonth()))));
      if (d > fim) break;
      periodos++;
    }
  }
  return periodos >= 1 ? { primeiroPrazo: fimDoDia(primeiro), periodos } : null;
}

export type Situacao = {
  periodosCobrados: number; // períodos em curso que este participante deve pagar
  devido: number; // total acumulado (dívida transitada + cotas)
  emAtraso: number; // períodos cujo prazo já passou e ainda não estão pagos
  prazo: Date; // prazo do período mais antigo por pagar (o que ainda importa)
  anterior: number; // dívida transitada do ano anterior ainda por pagar
  unidade: Unidade;
};

type InscricaoMin = Pick<Inscricao, "valorPago" | "valorPeriodo" | "periodoInicial" | "saldoInicial">;

/** Situação de uma inscrição numa campanha recorrente (null se a campanha for de pagamento único). */
export function situacao(c: Campanha, i: InscricaoMin, agora = new Date()): Situacao | null {
  if (!eRecorrente(c)) return null;
  const total = totalPeriodos(c);
  const cota = i.valorPeriodo ?? c.valorTotal;
  const inicial = i.periodoInicial ?? 0;
  const saldo = i.saldoInicial ?? 0;
  const emCurso = periodosEmCurso(c, agora);
  const cobrados = Math.max(1, emCurso - inicial);
  const devido = saldo + cota * cobrados;
  const anterior = Math.max(0, saldo - i.valorPago);
  const cobertos = cota > 0 ? Math.floor(Math.max(0, i.valorPago - saldo) / cota) : 0; // períodos totalmente pagos
  const maisAntigo = inicial + cobertos; // índice do período mais antigo por pagar
  const unidade = unidadeDe(c);

  if (maisAntigo >= total) {
    return { periodosCobrados: cobrados, devido, emAtraso: 0, prazo: prazoDoPeriodo(c, total - 1), anterior, unidade };
  }
  let emAtraso = 0;
  for (let k = maisAntigo; k < emCurso; k++) if (agora > prazoDoPeriodo(c, k)) emAtraso++;
  return { periodosCobrados: cobrados, devido, emAtraso, prazo: prazoDoPeriodo(c, maisAntigo), anterior, unidade };
}

/**
 * Para as mensagens: a campanha com o prazo do período que importa, quantos estão em atraso e a dívida do ano anterior.
 * Nas campanhas de pagamento único devolve a campanha tal como está.
 */
export function comRecorrencia(
  campanha: Campanha,
  inscricao: InscricaoMin,
  agora = new Date(),
): { campanha: Campanha; emAtraso: number; unidade: Unidade; anterior: number } {
  const s = situacao(campanha, inscricao, agora);
  if (!s) return { campanha, emAtraso: 0, unidade: MES, anterior: Math.max(0, (inscricao.saldoInicial ?? 0) - inscricao.valorPago) };
  return { campanha: { ...campanha, prazo: s.prazo }, emAtraso: s.emAtraso, unidade: s.unidade, anterior: s.anterior };
}

const estadoDe = (pago: number, devido: number) => (pago >= devido ? "QUITADO" : pago > 0 ? "PARCIAL" : "PENDENTE");

/**
 * Põe os valorDevido das inscrições de uma campanha recorrente em dia (soma o período que entrou em curso).
 * Só escreve quando o número de períodos mudou, ou com `forcar` (depois de editar a campanha).
 * Quem estava quitado e passa a dever o período novo volta a "Pendente/Parcial" e os lembretes retomam.
 * Inscrições cuja dívida já transitou para o ano seguinte não mudam.
 */
export async function atualizarPeriodos(
  c: Campanha,
  opts: { forcar?: boolean; agora?: Date } = {},
): Promise<{ campanha: Campanha; mudou: boolean }> {
  if (!eRecorrente(c)) return { campanha: c, mudou: false };
  const agora = opts.agora ?? new Date();
  const emCurso = periodosEmCurso(c, agora);
  if (!opts.forcar && c.periodosAplicados === emCurso) return { campanha: c, mudou: false };

  const inscs = await listar<Inscricao>(db().collection("inscricoes").where("campanhaId", "==", c.id));
  for (let k = 0; k < inscs.length; k += 20) {
    await Promise.all(
      inscs.slice(k, k + 20).map((x) =>
        db().runTransaction(async (t) => {
          const ref = db().collection("inscricoes").doc(x.id);
          const i = ler<Inscricao>(await t.get(ref)); // valorPago mais recente
          if (!i || i.valorTransitado) return;
          const devido = situacao(c, i, agora)?.devido ?? i.valorDevido;
          const estado = estadoDe(i.valorPago, devido);
          t.update(ref, {
            valorDevido: devido,
            status: estado,
            quitadoEm: estado === "QUITADO" ? (i.quitadoEm ?? new Date()) : null,
          });
        }),
      ),
    );
  }
  await db().collection("campanhas").doc(c.id).update({ periodosAplicados: emCurso });
  return { campanha: { ...c, periodosAplicados: emCurso }, mudou: true };
}

/** Atualiza várias campanhas. Devolve true se alguma mudou (para reler as inscrições). */
export async function atualizarVariasCampanhas(campanhas: Campanha[]): Promise<boolean> {
  const r = await Promise.all(campanhas.filter((c) => c.recorrente).map((c) => atualizarPeriodos(c)));
  return r.some((x) => x.mudou);
}

/** Atalho: atualiza a campanha de uma inscrição (usado antes de aprovar ou registar pagamentos). */
export async function atualizarPorInscricao(inscricaoId: string): Promise<void> {
  const i = ler<Inscricao>(await db().collection("inscricoes").doc(inscricaoId).get());
  if (!i) return;
  const c = ler<Campanha>(await db().collection("campanhas").doc(i.campanhaId).get());
  if (c?.recorrente) await atualizarPeriodos(c);
}
