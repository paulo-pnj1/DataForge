import { db, listar, ler, porIds } from "./db";
import { anoDaCampanha } from "./ano";
import { atualizarPeriodos, atualizarVariasCampanhas, situacao } from "./recorrencia";
import type { Campanha, Comprovativo, Inscricao, Mensagem, PagamentoManual, Professor, StatusComprovativo } from "./tipos";

/** Coleção filtrada pela instituição (só filtros de igualdade: não precisam de índices compostos). */
const da = (colecao: string, instituicaoId: string) => db().collection(colecao).where("instituicaoId", "==", instituicaoId);

export async function contarPendentes(instituicaoId: string): Promise<number> {
  const r = await da("comprovativos", instituicaoId).where("status", "==", "PENDENTE").count().get();
  return r.data().count;
}

export async function campanhasComInscricoes(instituicaoId: string, soAtivas = false, ano?: number) {
  const base = da("campanhas", instituicaoId);
  const todas = await listar<Campanha>(soAtivas ? base.where("ativa", "==", true) : base);
  const campanhas = ano === undefined ? todas : todas.filter((c) => anoDaCampanha(c) === ano);
  await atualizarVariasCampanhas(campanhas); // recorrentes: soma o período que entrou em curso
  const inscricoes = await listar<Inscricao>(da("inscricoes", instituicaoId));
  campanhas.sort((a, b) =>
    soAtivas ? a.prazo.getTime() - b.prazo.getTime() : b.criadaEm.getTime() - a.criadaEm.getTime(),
  );
  return campanhas.map((c) => ({ ...c, inscricoes: inscricoes.filter((i) => i.campanhaId === c.id) }));
}

/** Todas as campanhas da instituição, sem inscrições (para o seletor de ano e a escolha da campanha a copiar). */
export async function todasAsCampanhas(instituicaoId: string): Promise<Campanha[]> {
  return listar<Campanha>(da("campanhas", instituicaoId));
}

/** Precisa do índice composto de firestore.indexes.json (instituicaoId + criadaEm). */
export async function ultimasMensagens(instituicaoId: string, n: number) {
  const msgs = await listar<Mensagem>(da("mensagens", instituicaoId).orderBy("criadaEm", "desc").limit(n));
  const profs = await porIds<Professor>("professores", msgs.flatMap((m) => (m.professorId ? [m.professorId] : [])));
  return msgs.map((m) => ({ ...m, professor: m.professorId ? (profs.get(m.professorId) ?? null) : null }));
}

export async function campanhaDetalhe(instituicaoId: string, id: string) {
  const c0 = ler<Campanha>(await db().collection("campanhas").doc(id).get());
  if (!c0 || c0.instituicaoId !== instituicaoId) return null;
  const c = (await atualizarPeriodos(c0)).campanha;
  const inscs = await listar<Inscricao>(db().collection("inscricoes").where("campanhaId", "==", id));
  const profs = await porIds<Professor>("professores", inscs.map((i) => i.professorId));
  const pendentes = await listar<Comprovativo>(da("comprovativos", instituicaoId).where("status", "==", "PENDENTE"));
  const manuais = await listar<PagamentoManual>(da("pagamentosManuais", instituicaoId).where("estado", "==", "ATIVO"));
  const inscricoes = inscs
    .flatMap((i) => {
      const professor = profs.get(i.professorId);
      if (!professor) return [];
      return [{
        ...i,
        professor,
        atraso: situacao(c, i)?.emAtraso ?? 0, // meses em atraso (campanha mensal)
        comprovativos: pendentes.filter((p) => p.inscricaoId === i.id).map((p) => ({ id: p.id })),
        manuais: manuais.filter((m) => m.inscricaoId === i.id).sort((a, b) => b.registadoEm.getTime() - a.registadoEm.getTime()),
      }];
    })
    .sort((a, b) => a.professor.nome.localeCompare(b.professor.nome, "pt"));
  return { ...c, inscricoes };
}

export async function comprovativosPorEstado(instituicaoId: string, estado: StatusComprovativo, limite = 500) {
  const lista = await listar<Comprovativo>(da("comprovativos", instituicaoId).where("status", "==", estado));
  lista.sort((a, b) =>
    estado === "PENDENTE" ? a.recebidoEm.getTime() - b.recebidoEm.getTime() : b.recebidoEm.getTime() - a.recebidoEm.getTime(),
  );
  const pagina = lista.slice(0, limite);
  let inscs = await porIds<Inscricao>("inscricoes", pagina.map((c) => c.inscricaoId));
  const camps = await porIds<Campanha>("campanhas", [...inscs.values()].map((i) => i.campanhaId));
  if (await atualizarVariasCampanhas([...camps.values()])) {
    inscs = await porIds<Inscricao>("inscricoes", pagina.map((c) => c.inscricaoId)); // valores já com o mês novo
  }
  const profs = await porIds<Professor>("professores", [...inscs.values()].map((i) => i.professorId));

  return pagina.flatMap((c) => {
    const i = inscs.get(c.inscricaoId);
    const campanha = i && camps.get(i.campanhaId);
    const professor = i && profs.get(i.professorId);
    if (!i || !campanha || !professor) return [];
    return [{
      ...c,
      inscricao: {
        valorDevido: i.valorDevido,
        valorPago: i.valorPago,
        campanha: { nome: campanha.nome },
        professor: { nome: professor.nome, telefone: professor.telefone },
      },
    }];
  });
}
