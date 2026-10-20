import crypto from "node:crypto";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { db, listar } from "./db";
import { obterInstituicao, obterInstituicaoAtual } from "./instituicoes";
import type { Instituicao } from "./tipos";

const COOKIE = "muanapay_sessao";
const SETE_DIAS = 7 * 24 * 60 * 60;

/**
 * - super: dono da plataforma (ADMIN_EMAIL/ADMIN_PASSWORD). Cria instituições e pode "entrar" em qualquer uma.
 * - inst: admin de uma instituição. Só vê os dados dessa instituição.
 */
export type Sessao = { papel: "super" | "inst"; instituicaoId: string | null };

function segredo(): string {
  const s = process.env.SESSION_SECRET;
  if (!s) throw new Error("SESSION_SECRET não definido");
  return s;
}

function assinar(valor: string): string {
  return crypto.createHmac("sha256", segredo()).update(valor).digest("hex");
}

function hash(v: string): Buffer {
  return crypto.createHash("sha256").update(v).digest();
}

export function hashSenha(senha: string): string {
  const sal = crypto.randomBytes(16);
  return `${sal.toString("hex")}:${crypto.scryptSync(senha, sal, 64).toString("hex")}`;
}

function senhaConfere(senha: string, guardada: string): boolean {
  const [salHex, hHex] = guardada.split(":");
  if (!salHex || !hHex) return false;
  const esperado = Buffer.from(hHex, "hex");
  const h = crypto.scryptSync(senha, Buffer.from(salHex, "hex"), esperado.length);
  return crypto.timingSafeEqual(h, esperado);
}

export async function autenticar(email: string, senha: string): Promise<Sessao | null> {
  const e = email.trim().toLowerCase();

  const eSuper = (process.env.ADMIN_EMAIL ?? "").trim().toLowerCase();
  const sSuper = process.env.ADMIN_PASSWORD ?? "";
  if (eSuper && sSuper) {
    const okEmail = crypto.timingSafeEqual(hash(e), hash(eSuper));
    const okSenha = crypto.timingSafeEqual(hash(senha), hash(sSuper));
    if (okEmail && okSenha) return { papel: "super", instituicaoId: null };
  }

  const achadas = await listar<Instituicao>(db().collection("instituicoes").where("adminEmail", "==", e).limit(1));
  const i = achadas[0];
  if (i && i.ativa && senhaConfere(senha, i.adminSenha)) return { papel: "inst", instituicaoId: i.id };
  return null;
}

export async function criarSessao(s: Sessao): Promise<void> {
  const exp = String(Date.now() + SETE_DIAS * 1000);
  const corpo = `${s.papel}.${s.instituicaoId ?? "-"}.${exp}`;
  (await cookies()).set(COOKIE, `${corpo}.${assinar(corpo)}`, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: SETE_DIAS,
  });
}

export async function encerrarSessao(): Promise<void> {
  (await cookies()).delete(COOKIE);
}

export async function lerSessao(): Promise<Sessao | null> {
  const token = (await cookies()).get(COOKIE)?.value;
  if (!token) return null;
  const partes = token.split(".");
  if (partes.length !== 4) return null;
  const [papel, inst, exp, assinatura] = partes;
  const esperada = assinar(`${papel}.${inst}.${exp}`);
  if (assinatura.length !== esperada.length) return null;
  if (!crypto.timingSafeEqual(Buffer.from(assinatura), Buffer.from(esperada))) return null;
  if (!(Number(exp) > Date.now())) return null;
  if (papel !== "super" && papel !== "inst") return null;
  if (papel === "inst" && inst === "-") return null;
  return { papel, instituicaoId: inst === "-" ? null : inst };
}

export async function requireSessao(): Promise<Sessao> {
  const s = await lerSessao();
  if (!s) redirect("/login");
  return s;
}

export async function requireSuper(): Promise<Sessao> {
  const s = await requireSessao();
  if (s.papel !== "super") redirect("/");
  return s;
}

/** A instituição com que a sessão está a trabalhar. Todas as páginas e ações do painel passam por aqui. */
export async function requireInstituicao(): Promise<Instituicao> {
  const s = await requireSessao();
  if (!s.instituicaoId) redirect("/instituicoes");
  const i = await obterInstituicao(s.instituicaoId);
  if (!i || (s.papel === "inst" && !i.ativa)) redirect("/login");
  return i;
}

/** Igual a requireInstituicao, mas sem memória: para ações que alteram a instituição. */
export async function requireInstituicaoAcao(): Promise<Instituicao> {
  const s = await requireSessao();
  if (!s.instituicaoId) redirect("/instituicoes");
  const i = await obterInstituicaoAtual(s.instituicaoId);
  if (!i || (s.papel === "inst" && !i.ativa)) redirect("/login");
  return i;
}
