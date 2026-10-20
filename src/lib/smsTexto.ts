/**
 * Texto de SMS. Sem dependências do servidor: usado também no navegador (contador da página SMS).
 *
 * Um SMS simples (alfabeto GSM) leva 160 caracteres; acima disso cada parte leva 153.
 * Acentos como ã, á, ê, õ obrigam o telemóvel a usar UCS-2 (70 caracteres por SMS),
 * por isso retiramo-los: "ação" -> "acao". Assim o texto cabe em menos SMS e chega sempre igual.
 */

const TROCAS: Record<string, string> = {
  "“": '"',
  "”": '"',
  "„": '"',
  "‘": "'",
  "’": "'",
  "–": "-",
  "—": "-",
  "…": "...",
  "•": "-",
  "\u00a0": " ",
  "€": "EUR",
};

/** Texto como será enviado: ASCII, sem acentos, sem emojis, sem espaços a mais. */
export function paraSms(texto: string): string {
  return texto
    .replace(/[“”„‘’–—…•\u00a0€]/g, (c) => TROCAS[c] ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\r/g, "")
    .replace(/[^\x20-\x7E\n]/g, "") // emojis e símbolos que não cabem num SMS simples
    .replace(/[ \t]{2,}/g, " ")
    .replace(/ *\n */g, "\n")
    .trim();
}

const EXTENSAO = new Set(["[", "]", "{", "}", "\\", "^", "|", "~"]); // contam 2 no alfabeto GSM

/** Tamanho em "septetos" do alfabeto GSM (≈ caracteres). */
export function septetos(texto: string): number {
  let n = 0;
  for (const c of texto) n += EXTENSAO.has(c) ? 2 : 1;
  return n;
}

/** Quantos SMS o texto ocupa (e é cobrado). */
export function segmentos(texto: string): number {
  const n = septetos(texto);
  if (n === 0) return 0;
  return n <= 160 ? 1 : Math.ceil(n / 153);
}

export const MAX_SEGMENTOS = 3;
export const MAX_SEPTETOS = 153 * MAX_SEGMENTOS; // 459
