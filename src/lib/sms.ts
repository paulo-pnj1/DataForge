import { paraSms } from "./smsTexto";

/**
 * Envio de SMS pela Ombala (gateway angolano), só no servidor.
 *
 * API: POST {OMBALA_BASE_URL}/messages
 *   Authorization: Token <OMBALA_TOKEN>
 *   corpo: { message, from, to, schedule? }
 * `to` é um número de 9 dígitos (923456789) ou vários separados por vírgula.
 * `from` é o remetente (nome ou número) já aprovado no painel da Ombala.
 */

export type ResultadoSms = { ok: boolean; id?: string; erro?: string };

const baseUrl = () => (process.env.OMBALA_BASE_URL?.trim() || "https://api.useombala.ao/v1").replace(/\/+$/, "");

/** O token pode vir com ou sem o prefixo "Token ". */
const token = () => (process.env.OMBALA_TOKEN ?? "").trim().replace(/^Token\s+/i, "");
const remetente = () => (process.env.OMBALA_FROM ?? "").trim();

export function smsConfigurado(): boolean {
  return Boolean(token() && remetente());
}

/** "244923456789", "+244 923 456 789" ou "923456789" -> "923456789". null se não for telemóvel angolano. */
export function numeroSms(telefone: string): string | null {
  let d = telefone.replace(/\D/g, "");
  if (d.startsWith("00244")) d = d.slice(5);
  else if (d.startsWith("244") && d.length === 12) d = d.slice(3);
  return /^9\d{8}$/.test(d) ? d : null;
}

type Destinatario = { phone_number?: string; message_status?: string; message_id?: string | number };
type RespostaOmbala = {
  id?: string | number;
  cost?: string | number;
  recipients?: Destinatario[];
  message?: unknown;
  error?: unknown;
  detail?: unknown;
  errors?: unknown;
};

function erroDe(json: RespostaOmbala | null, status: number): string {
  const bruto = json?.message ?? json?.error ?? json?.detail ?? json?.errors;
  const texto = (typeof bruto === "string" ? bruto : bruto ? JSON.stringify(bruto) : "").slice(0, 200);
  if (status === 401 || status === 403) return `A Ombala recusou o token${texto ? `: ${texto}` : ""}`;
  return texto || `A Ombala respondeu HTTP ${status}`;
}

const FALHOU = /fail|reject|error|invalid|undeliver/i;
const POR_PEDIDO = 50; // números por chamada à Ombala

/** Uma chamada à Ombala com até POR_PEDIDO números (já validados, 9 dígitos) e o mesmo texto. */
async function enviarGrupo(numeros: string[], texto: string): Promise<Map<string, ResultadoSms>> {
  const mapa = new Map<string, ResultadoSms>();
  const falharTodos = (erro: string) => numeros.forEach((n) => mapa.set(n, { ok: false, erro }));

  if (!smsConfigurado()) {
    falharTodos("SMS não configurado (faltam OMBALA_TOKEN e OMBALA_FROM)");
    return mapa;
  }
  try {
    const res = await fetch(`${baseUrl()}/messages`, {
      method: "POST",
      headers: { Authorization: `Token ${token()}`, "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ message: texto, from: remetente(), to: numeros.join(",") }),
      signal: AbortSignal.timeout(15_000),
    });
    const json = (await res.json().catch(() => null)) as RespostaOmbala | null;
    if (!res.ok) {
      falharTodos(erroDe(json, res.status));
      return mapa;
    }
    const porNumero = new Map<string, Destinatario>();
    for (const d of json?.recipients ?? []) {
      const n = d.phone_number ? numeroSms(d.phone_number) : null;
      if (n) porNumero.set(n, d);
    }
    for (const n of numeros) {
      const d = porNumero.get(n);
      if (d?.message_status && FALHOU.test(d.message_status)) {
        mapa.set(n, { ok: false, erro: `Ombala: ${d.message_status}` });
      } else {
        const id = d?.message_id ?? json?.id;
        mapa.set(n, { ok: true, id: id != null ? String(id) : undefined });
      }
    }
  } catch (e) {
    falharTodos(
      e instanceof Error ? (e.name === "TimeoutError" ? "A Ombala demorou demasiado a responder" : e.message) : "Erro de rede",
    );
  }
  return mapa;
}

/** Um SMS. O texto é convertido para ASCII sem acentos (ver smsTexto.ts). */
export async function enviarSms(telefone: string, texto: string): Promise<ResultadoSms> {
  const n = numeroSms(telefone);
  if (!n) return { ok: false, erro: "SMS só funciona com telemóveis angolanos (9 dígitos)" };
  const corpo = paraSms(texto);
  if (!corpo) return { ok: false, erro: "Mensagem vazia" };
  return (await enviarGrupo([n], corpo)).get(n) ?? { ok: false, erro: "Sem resposta da Ombala" };
}

/**
 * O mesmo SMS para vários números (a Ombala aceita vários por chamada).
 * Devolve o resultado de cada telefone, indexado pelo telefone tal como foi recebido.
 */
export async function enviarSmsLote(telefones: string[], texto: string): Promise<Map<string, ResultadoSms>> {
  const resultado = new Map<string, ResultadoSms>();
  const corpo = paraSms(texto);
  const validos: string[] = [];
  const doTelefone = new Map<string, string>(); // telefone original -> 9 dígitos

  for (const t of telefones) {
    const n = numeroSms(t);
    if (!n) resultado.set(t, { ok: false, erro: "SMS só funciona com telemóveis angolanos (9 dígitos)" });
    else {
      doTelefone.set(t, n);
      if (!validos.includes(n)) validos.push(n);
    }
  }
  if (!corpo) {
    for (const [t] of doTelefone) resultado.set(t, { ok: false, erro: "Mensagem vazia" });
    return resultado;
  }

  const porNumero = new Map<string, ResultadoSms>();
  for (let i = 0; i < validos.length; i += POR_PEDIDO) {
    const parcial = await enviarGrupo(validos.slice(i, i + POR_PEDIDO), corpo);
    for (const [n, r] of parcial) porNumero.set(n, r);
  }
  for (const [t, n] of doTelefone) resultado.set(t, porNumero.get(n) ?? { ok: false, erro: "Sem resposta da Ombala" });
  return resultado;
}
