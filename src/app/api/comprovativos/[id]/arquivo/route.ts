import { db, ler } from "@/lib/db";
import { lerSessao } from "@/lib/auth";
import { urlAssinada } from "@/lib/cloudinary";
import type { Comprovativo } from "@/lib/tipos";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Só quem tem sessão chega aqui, e só vê comprovativos da sua instituição (o dono da plataforma vê todos).
export async function GET(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const sessao = await lerSessao();
  if (!sessao) return new Response("Não autorizado", { status: 401 });
  const { id } = await ctx.params;
  const c = ler<Comprovativo>(await db().collection("comprovativos").doc(id).get());
  if (!c) return new Response("Não encontrado", { status: 404 });
  if (sessao.papel === "inst" && sessao.instituicaoId !== c.instituicaoId) {
    return new Response("Não encontrado", { status: 404 });
  }
  return Response.redirect(urlAssinada(c.cloudinaryId, c.cloudinaryTipo, c.cloudinaryFormato), 302);
}
