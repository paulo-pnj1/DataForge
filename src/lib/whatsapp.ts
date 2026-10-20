import crypto from "node:crypto";

const GRAPH = `https://graph.facebook.com/${process.env.WHATSAPP_API_VERSION ?? "v21.0"}`;

export type Resultado = { ok: boolean; id?: string; erro?: string };

/** Parâmetros de modelo não aceitam quebras de linha, tabs ou mais de 4 espaços seguidos. */
function limpar(t: string): string {
  return t.replace(/[\r\n\t]+/g, " ").replace(/ {2,}/g, " ").trim() || "-"; // a Meta recusa parâmetros vazios
}

async function chamar(body: Record<string, unknown>): Promise<Resultado> {
  const phoneId = process.env.WHATSAPP_PHONE_NUMBER_ID;
  const token = process.env.WHATSAPP_ACCESS_TOKEN;
  if (!phoneId || !token) {
    return { ok: false, erro: "WhatsApp não configurado (faltam variáveis de ambiente)" };
  }
  try {
    const res = await fetch(`${GRAPH}/${phoneId}/messages`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ messaging_product: "whatsapp", ...body }),
    });
    const json = (await res.json().catch(() => ({}))) as {
      messages?: { id: string }[];
      error?: { message?: string };
    };
    if (!res.ok) return { ok: false, erro: json.error?.message ?? `HTTP ${res.status}` };
    return { ok: true, id: json.messages?.[0]?.id };
  } catch (e) {
    return { ok: false, erro: e instanceof Error ? e.message : "Erro de rede" };
  }
}

/** Mensagem iniciada por nós: tem de ser um modelo aprovado pela Meta. */
export function enviarModelo(
  para: string,
  modelo: string,
  parametros: string[],
  /** Payload do botão de resposta rápida, se o modelo tiver um (índice 0). */
  botaoPayload?: string,
): Promise<Resultado> {
  const components: Record<string, unknown>[] = [
    {
      type: "body",
      parameters: parametros.map((p) => ({ type: "text", text: limpar(p) })),
    },
  ];
  if (botaoPayload) {
    components.push({
      type: "button",
      sub_type: "quick_reply",
      index: "0",
      parameters: [{ type: "payload", payload: botaoPayload }],
    });
  }
  return chamar({
    to: para,
    type: "template",
    template: {
      name: modelo,
      language: { code: process.env.WHATSAPP_TEMPLATE_LANG ?? "pt_PT" },
      components,
    },
  });
}

/** Texto livre: só funciona dentro de 24h depois da última mensagem do professor. */
export function enviarTexto(para: string, texto: string): Promise<Resultado> {
  return chamar({ to: para, type: "text", text: { body: texto } });
}

export async function baixarMidia(mediaId: string): Promise<{ buf: Buffer; mime: string }> {
  const token = process.env.WHATSAPP_ACCESS_TOKEN;
  if (!token) throw new Error("WHATSAPP_ACCESS_TOKEN não definido");
  const headers = { Authorization: `Bearer ${token}` };

  const meta = await fetch(`${GRAPH}/${mediaId}`, { headers });
  if (!meta.ok) throw new Error(`Falha ao obter mídia (${meta.status})`);
  const { url, mime_type } = (await meta.json()) as { url: string; mime_type: string };

  const ficheiro = await fetch(url, { headers });
  if (!ficheiro.ok) throw new Error(`Falha ao descarregar mídia (${ficheiro.status})`);
  return { buf: Buffer.from(await ficheiro.arrayBuffer()), mime: mime_type };
}

export function assinaturaValida(corpo: string, cabecalho: string | null): boolean {
  const segredo = process.env.WHATSAPP_APP_SECRET;
  if (!segredo || !cabecalho) return false;
  const esperada = "sha256=" + crypto.createHmac("sha256", segredo).update(corpo).digest("hex");
  if (esperada.length !== cabecalho.length) return false;
  return crypto.timingSafeEqual(Buffer.from(esperada), Buffer.from(cabecalho));
}
