import { db, ler, listar, porIds } from "./db";
import type { Campanha, Comprovativo, Inscricao, Instituicao, Professor } from "./tipos";
import { enviarLembrete } from "./mensagens";
import { horaEmLuanda } from "./format";
import { atualizarPeriodos, atualizarVariasCampanhas, comRecorrencia } from "./recorrencia";

export type ResumoLembretes = {
  enviados: number;
  falhas: number;
  ignorados: number;
  motivo?: string;
};


/**
 * Reserva atómica da inscrição antes de enviar: se o cron correr 2-3 vezes ao mesmo tempo
 * (reenvio da Vercel, vários projetos/domínios, chamada externa), só uma execução consegue reservar.
 */
async function reservar(id: string, intervaloMs: number): Promise<boolean> {
  const ref = db().collection("inscricoes").doc(id);
  return db().runTransaction(async (t) => {
    const d = ler<Inscricao & { lembreteReservadoEm?: Date | null }>(await t.get(ref));
    if (!d) return false;
    const agora = Date.now();
    if (d.ultimoLembreteEm && agora - d.ultimoLembreteEm.getTime() < intervaloMs) return false;
    if (d.lembreteReservadoEm && agora - d.lembreteReservadoEm.getTime() < 10 * 60_000) return false;
    t.update(ref, { lembreteReservadoEm: new Date(agora) });
    return true;
  });
}

const pausa = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Envia lembretes a quem ainda não completou o pagamento, em todas as instituições ativas.
 * - automático (cron): respeita o horário e a frequência de cada campanha,
 *   e salta quem tem comprovativo à espera de aprovação.
 * - forçado (botão "Enviar agora"): ignora horário e frequência; usa-se sempre com campanhaId e instituicaoId.
 */
export async function executarLembretes(
  opts: { forcar?: boolean; campanhaId?: string; instituicaoId?: string } = {},
): Promise<ResumoLembretes> {
  const agora = new Date();

  if (!opts.forcar) {
    const h = horaEmLuanda(agora);
    const ini = Number(process.env.LEMBRETES_HORA_INICIO ?? 8);
    const fim = Number(process.env.LEMBRETES_HORA_FIM ?? 20);
    if (h < ini || h >= fim) {
      return { enviados: 0, falhas: 0, ignorados: 0, motivo: "Fora do horário de envio" };
    }
  }

  // Campanhas mensais: soma o mês que entrou em curso (quem estava quitado volta a ter algo por pagar).
  const mensais = (await listar<Campanha>(db().collection("campanhas").where("recorrente", "==", true))).filter(
    (c) =>
      c.ativa &&
      (!opts.instituicaoId || c.instituicaoId === opts.instituicaoId) &&
      (!opts.campanhaId || c.id === opts.campanhaId),
  );
  await atualizarVariasCampanhas(mensais);

  const abertas = opts.instituicaoId
    ? (await listar<Inscricao>(db().collection("inscricoes").where("instituicaoId", "==", opts.instituicaoId))).filter(
        (i) => i.status !== "QUITADO",
      )
    : await listar<Inscricao>(db().collection("inscricoes").where("status", "in", ["PENDENTE", "PARCIAL"]));

  const instituicoes = await porIds<Instituicao>("instituicoes", abertas.map((i) => i.instituicaoId));
  const campanhas = await porIds<Campanha>("campanhas", abertas.map((i) => i.campanhaId));
  const professores = await porIds<Professor>("professores", abertas.map((i) => i.professorId));
  const pendentes = await listar<Comprovativo>(db().collection("comprovativos").where("status", "==", "PENDENTE"));
  const comPendente = new Set(pendentes.map((c) => c.inscricaoId));

  const inscricoes = abertas.flatMap((i) => {
    const instituicao = instituicoes.get(i.instituicaoId);
    const campanha = campanhas.get(i.campanhaId);
    const professor = professores.get(i.professorId);
    if (!instituicao?.ativa || !campanha?.ativa || !professor?.ativo) return [];
    if (opts.campanhaId && campanha.id !== opts.campanhaId) return [];
    return [{ ...i, instituicao, campanha, professor, pendentes: comPendente.has(i.id) ? 1 : 0 }];
  });

  let enviados = 0;
  let falhas = 0;
  let ignorados = 0;

  const jaNestaExecucao = new Set<string>(); // mesmo telefone + campanha só uma vez por execução
  for (const i of inscricoes) {
    if (!opts.forcar) {
      const intervalo = i.campanha.frequenciaHoras * 3_600_000;
      const venceu = !i.ultimoLembreteEm || agora.getTime() - i.ultimoLembreteEm.getTime() >= intervalo;
      const chave = `${i.professor.telefone}|${i.campanhaId}`;
      if (!venceu || i.pendentes > 0 || jaNestaExecucao.has(chave) || !(await reservar(i.id, intervalo))) {
        ignorados++;
        continue;
      }
      jaNestaExecucao.add(chave);
    }

    const r = await enviarLembrete(
      {
        instituicao: i.instituicao,
        professor: i.professor,
        ...comRecorrencia(i.campanha, i),
        valorDevido: i.valorDevido,
        valorPago: i.valorPago,
      },
      i.lembretesEnviados,
    );

    if (r.ok) {
      enviados++;
      await db().collection("inscricoes").doc(i.id).update({
        ultimoLembreteEm: new Date(),
        lembretesEnviados: i.lembretesEnviados + 1,
        lembreteReservadoEm: null,
      });
    } else {
      falhas++; // não atualiza ultimoLembreteEm: tenta de novo no próximo ciclo
      if (!opts.forcar) await db().collection("inscricoes").doc(i.id).update({ lembreteReservadoEm: null });
    }
    await pausa(250);
  }

  return { enviados, falhas, ignorados };
}

/**
 * Envia o lembrete já, só às inscrições escolhidas de uma campanha (ignora horário e frequência).
 * Salta quem já pagou tudo ou está pausado. Envia 5 de cada vez para ser rápido sem sobrecarregar o WhatsApp.
 */
export async function enviarLembretesSelecionados(
  instituicaoId: string,
  campanhaId: string,
  inscricaoIds: string[],
): Promise<ResumoLembretes> {
  const [instituicao, campanha] = await Promise.all([
    db().collection("instituicoes").doc(instituicaoId).get().then((d) => ler<Instituicao>(d)),
    db().collection("campanhas").doc(campanhaId).get().then((d) => ler<Campanha>(d)),
  ]);
  if (!instituicao || !campanha || campanha.instituicaoId !== instituicaoId) {
    return { enviados: 0, falhas: 0, ignorados: inscricaoIds.length, motivo: "Campanha não encontrada" };
  }
  if (campanha.recorrente) await atualizarPeriodos(campanha);

  const inscricoes = [...(await porIds<Inscricao>("inscricoes", inscricaoIds)).values()].filter(
    (i) => i.instituicaoId === instituicaoId && i.campanhaId === campanhaId,
  );
  const professores = await porIds<Professor>("professores", inscricoes.map((i) => i.professorId));

  let enviados = 0;
  let falhas = 0;
  let ignorados = inscricaoIds.length - inscricoes.length;

  const alvos = inscricoes.flatMap((i) => {
    const professor = professores.get(i.professorId);
    if (i.status === "QUITADO" || !professor?.ativo) {
      ignorados++;
      return [];
    }
    return [{ i, professor }];
  });

  for (let k = 0; k < alvos.length; k += 5) {
    await Promise.all(
      alvos.slice(k, k + 5).map(async ({ i, professor }) => {
        const r = await enviarLembrete(
          { instituicao, professor, ...comRecorrencia(campanha, i), valorDevido: i.valorDevido, valorPago: i.valorPago },
          i.lembretesEnviados,
        );
        if (r.ok) {
          enviados++;
          await db().collection("inscricoes").doc(i.id).update({
            ultimoLembreteEm: new Date(),
            lembretesEnviados: i.lembretesEnviados + 1,
          });
        } else {
          falhas++;
        }
      }),
    );
    if (k + 5 < alvos.length) await pausa(200);
  }
  return { enviados, falhas, ignorados };
}
