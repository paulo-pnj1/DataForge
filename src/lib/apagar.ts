import type { Query } from "firebase-admin/firestore";
import { apagarFicheiro } from "./cloudinary";
import { db, ler, listar, porIds } from "./db";
import type { Campanha, Comprovativo, Inscricao, Professor } from "./tipos";

type Res = { ok: true } | { ok: false; erro: string };

const LOTE = 400;

function partes<T>(lista: T[], n: number): T[][] {
  const r: T[][] = [];
  for (let i = 0; i < lista.length; i += n) r.push(lista.slice(i, i + n));
  return r;
}

/** Apaga todos os documentos de uma consulta, em lotes. */
async function apagarConsulta(q: Query): Promise<void> {
  for (;;) {
    const snap = await q.limit(LOTE).get();
    if (snap.empty) return;
    const lote = db().batch();
    snap.docs.forEach((d) => lote.delete(d.ref));
    await lote.commit();
    if (snap.size < LOTE) return;
  }
}

/** Apaga comprovativos e também os ficheiros guardados no Cloudinary. */
async function apagarComprovativos(q: Query): Promise<void> {
  for (;;) {
    const snap = await q.limit(LOTE).get();
    if (snap.empty) return;
    await Promise.all(
      snap.docs.map((d) => {
        const c = d.data() as Partial<Comprovativo>;
        return c.cloudinaryId ? apagarFicheiro(c.cloudinaryId, c.cloudinaryTipo ?? "image") : Promise.resolve();
      }),
    );
    const lote = db().batch();
    snap.docs.forEach((d) => lote.delete(d.ref));
    await lote.commit();
    if (snap.size < LOTE) return;
  }
}

/** Inscrições e tudo o que depende delas: comprovativos (com ficheiros) e pagamentos em mãos. */
async function apagarInscricoes(ids: string[]): Promise<void> {
  for (const parte of partes(ids, 30)) {
    await apagarComprovativos(db().collection("comprovativos").where("inscricaoId", "in", parte));
    await apagarConsulta(db().collection("pagamentosManuais").where("inscricaoId", "in", parte));
  }
  for (const parte of partes(ids, LOTE)) {
    const lote = db().batch();
    parte.forEach((id) => lote.delete(db().collection("inscricoes").doc(id)));
    await lote.commit();
  }
}

export async function apagarProfessorCompleto(instituicaoId: string, id: string): Promise<Res> {
  const ref = db().collection("professores").doc(id);
  const p = ler<Professor>(await ref.get());
  if (!p || p.instituicaoId !== instituicaoId) return { ok: false, erro: "Participante não encontrado." };

  const inscs = await listar<Inscricao>(db().collection("inscricoes").where("professorId", "==", id));
  await apagarInscricoes(inscs.map((i) => i.id));
  await apagarConsulta(db().collection("mensagens").where("professorId", "==", id));
  await ref.delete();
  return { ok: true };
}

export async function apagarCampanhaCompleta(instituicaoId: string, id: string): Promise<Res> {
  const ref = db().collection("campanhas").doc(id);
  const c = ler<Campanha>(await ref.get());
  if (!c || c.instituicaoId !== instituicaoId) return { ok: false, erro: "Campanha não encontrada." };

  const inscs = await listar<Inscricao>(db().collection("inscricoes").where("campanhaId", "==", id));
  await apagarInscricoes(inscs.map((i) => i.id));
  await ref.delete();
  return { ok: true };
}

/** Tira um professor de uma campanha (apaga a inscrição e os pagamentos dela). */
export async function removerInscricaoCompleta(instituicaoId: string, inscricaoId: string): Promise<Res> {
  const i = ler<Inscricao>(await db().collection("inscricoes").doc(inscricaoId).get());
  if (!i || i.instituicaoId !== instituicaoId) return { ok: false, erro: "Inscrição não encontrada." };
  await apagarInscricoes([inscricaoId]);
  return { ok: true };
}

/** Tira vários professores de uma campanha de uma vez. Só conta as inscrições que são desta instituição e campanha. */
export async function removerInscricoesCompleto(
  instituicaoId: string,
  campanhaId: string,
  inscricaoIds: string[],
): Promise<number> {
  const achadas = await porIds<Inscricao>("inscricoes", inscricaoIds);
  const validas = [...achadas.values()].filter((i) => i.instituicaoId === instituicaoId && i.campanhaId === campanhaId);
  await apagarInscricoes(validas.map((i) => i.id));
  return validas.length;
}

/** Só o dono da plataforma: apaga a instituição e TUDO o que lhe pertence. */
export async function apagarInstituicaoCompleta(id: string): Promise<void> {
  const da = (colecao: string) => db().collection(colecao).where("instituicaoId", "==", id);
  await apagarComprovativos(da("comprovativos"));
  for (const colecao of ["pagamentosManuais", "mensagens", "inscricoes", "campanhas", "professores"]) {
    await apagarConsulta(da(colecao));
  }
  await db().collection("instituicoes").doc(id).delete();
}
