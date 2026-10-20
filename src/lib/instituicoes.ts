import { cache } from "react";
import { db, ler } from "./db";
import type { Instituicao } from "./tipos";

/** Sem memória: usar nas ações, que acabam de alterar dados. */
export async function obterInstituicaoAtual(id: string): Promise<Instituicao | null> {
  return ler<Instituicao>(await db().collection("instituicoes").doc(id).get());
}

/** Lida uma só vez por pedido: o menu lateral e a página pedem a mesma instituição. */
export const obterInstituicao = cache(obterInstituicaoAtual);
