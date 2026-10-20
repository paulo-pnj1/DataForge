import type { DadosPagamento, Instituicao } from "./tipos";

/** "923456789" -> "923 456 789" */
export function fmtNumero(n: string): string {
  const d = n.replace(/\D/g, "");
  return d.length === 9 ? d.replace(/(\d{3})(?=\d)/g, "$1 ") : d;
}

/** "AO06004000001234567890123" -> "AO06 0040 0000 1234 5678 9012 3" */
export function fmtIban(iban: string): string {
  return iban.replace(/\s+/g, "").toUpperCase().replace(/(.{4})(?=.)/g, "$1 ");
}

/** Só números de 9 dígitos que começam por 9 (telemóvel angolano). Aceita +244 / 244 à frente. */
export function normalizarMulticaixa(raw: string): string {
  let d = raw.replace(/\D/g, "");
  if (d.startsWith("00244")) d = d.slice(5);
  else if (d.startsWith("244") && d.length === 12) d = d.slice(3);
  return d;
}

export function multicaixaValido(n: string): boolean {
  return /^9\d{8}$/.test(n);
}

/** IBAN angolano: "AO" + 23 dígitos (25 caracteres), com o dígito de controlo mod 97 certo. */
export function ibanValido(raw: string): boolean {
  const iban = raw.replace(/\s+/g, "").toUpperCase();
  if (!/^AO\d{23}$/.test(iban)) return false;
  const rodado = iban.slice(4) + iban.slice(0, 4);
  const numerico = rodado.replace(/[A-Z]/g, (c) => String(c.charCodeAt(0) - 55));
  let resto = 0;
  for (const ch of numerico) resto = (resto * 10 + Number(ch)) % 97;
  return resto === 1;
}

export function temDadosPagamento(i: Pick<Instituicao, "pagamento">): boolean {
  const p = i.pagamento;
  return Boolean(p && (p.multicaixaNumero || p.iban));
}

function partes(p: DadosPagamento): { metodo: string; dados: string; titular: string }[] {
  const r: { metodo: string; dados: string; titular: string }[] = [];
  if (p.multicaixaNumero) {
    r.push({ metodo: "Multicaixa Express", dados: fmtNumero(p.multicaixaNumero), titular: p.multicaixaTitular });
  }
  if (p.iban) {
    r.push({ metodo: "Transferência IBAN", dados: fmtIban(p.iban), titular: p.ibanTitular });
  }
  return r;
}

/**
 * Uma só linha, para a variável {{7}} do modelo (a Meta não aceita quebras de linha em variáveis).
 * Ex.: "Multicaixa Express 923 456 789 (titular: Igreja X); Transferência IBAN AO06 ... (titular: Igreja X)"
 */
export function pagamentoEmLinha(i: Pick<Instituicao, "pagamento">): string {
  const p = i.pagamento;
  const lista = p ? partes(p) : [];
  if (lista.length === 0) return "pede os dados de pagamento à tua instituição";
  return lista.map((x) => `${x.metodo} ${x.dados}${x.titular ? ` (titular: ${x.titular})` : ""}`).join("; ");
}

/** Várias linhas, para respostas de texto livre (dentro da janela de 24h). Vazio se não houver dados. */
export function pagamentoEmBloco(i: Pick<Instituicao, "pagamento">): string {
  const p = i.pagamento;
  const lista = p ? partes(p) : [];
  return lista
    .map((x) => `• ${x.metodo}: ${x.dados}${x.titular ? `\n  Titular: ${x.titular}` : ""}`)
    .join("\n");
}

/** Versão curta para SMS: sem o nome do titular. Vazio se não houver dados. */
export function pagamentoCurto(i: Pick<Instituicao, "pagamento">): string {
  const p = i.pagamento;
  const lista = p ? partes(p) : [];
  return lista.map((x) => `${x.metodo} ${x.dados}`).join(" ou ");
}
