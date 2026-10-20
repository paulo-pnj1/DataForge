import { db, porIds } from "./db";
import { periodosEmCurso } from "./recorrencia";
import type { Campanha, Inscricao } from "./tipos";

type CampanhaMin = Pick<Campanha, "id" | "instituicaoId" | "valorTotal"> &
  Partial<Pick<Campanha, "recorrente" | "periodicidade" | "diaMes" | "periodos" | "meses" | "primeiroPrazo">>;

type Opcoes = {
  /** Campanha recorrente: quem entra depois do início só paga do período em curso para a frente. */
  desdeAgora?: boolean;
  /** Dívida transitada do ano anterior, por participante (soma-se ao devido e abate-se primeiro). */
  saldos?: Map<string, number>;
  /** Cota própria de cada participante (campanha recorrente). */
  cotas?: Map<string, number>;
};

/**
 * Inscreve participantes numa campanha; ignora quem já está. Devolve quantos foram novos.
 * Campanha recorrente: ao criar a campanha todos começam no 1.º período; quem entra depois (`desdeAgora`)
 * só paga do período em curso para a frente, não os anteriores.
 */
export async function inscreverProfessores(
  campanha: CampanhaMin,
  professorIds: string[],
  opcoes: Opcoes = {},
): Promise<number> {
  const alvo = professorIds.map((p) => ({ professorId: p, id: `${campanha.id}_${p}` }));
  const existentes = await porIds<Inscricao>("inscricoes", alvo.map((a) => a.id));
  const novos = alvo.filter((a) => !existentes.has(a.id));

  const recorrente = Boolean(campanha.recorrente && campanha.primeiroPrazo && (campanha.periodos ?? campanha.meses));
  const emCurso = recorrente ? periodosEmCurso(campanha) : 0;
  const periodoInicial = recorrente ? (opcoes.desdeAgora ? emCurso - 1 : 0) : undefined;

  for (let i = 0; i < novos.length; i += 400) {
    const lote = db().batch();
    for (const n of novos.slice(i, i + 400)) {
      const saldo = opcoes.saldos?.get(n.professorId) ?? 0;
      const cotaPropria = recorrente ? opcoes.cotas?.get(n.professorId) : undefined;
      const cota = cotaPropria ?? campanha.valorTotal;
      const valorDevido = saldo + (recorrente ? cota * (emCurso - (periodoInicial ?? 0)) : campanha.valorTotal);
      const dados: Omit<Inscricao, "id"> = {
        instituicaoId: campanha.instituicaoId,
        campanhaId: campanha.id,
        professorId: n.professorId,
        valorDevido,
        ...(periodoInicial !== undefined ? { periodoInicial } : {}),
        ...(cotaPropria !== undefined && cotaPropria !== campanha.valorTotal ? { valorPeriodo: cotaPropria } : {}),
        ...(saldo > 0 ? { saldoInicial: saldo } : {}),
        valorPago: 0,
        status: "PENDENTE",
        ultimoLembreteEm: null,
        lembretesEnviados: 0,
        quitadoEm: null,
      };
      lote.set(db().collection("inscricoes").doc(n.id), dados);
    }
    await lote.commit();
  }
  return novos.length;
}
