import { dataISO } from "./format";
import type { Campanha } from "./tipos";

/** Ano civil de hoje, em Luanda. */
export const anoAtual = (): number => Number(dataISO(new Date()).slice(0, 4));

/** Ano da campanha. As antigas não o guardam: usa o ano do primeiro prazo. */
export const anoDaCampanha = (c: Pick<Campanha, "ano" | "primeiroPrazo" | "prazo">): number =>
  c.ano ?? Number(dataISO(c.primeiroPrazo ?? c.prazo).slice(0, 4));

/** Anos com campanhas, mais o atual, do mais recente para o mais antigo (para o seletor de ano). */
export function anosDisponiveis(campanhas: Pick<Campanha, "ano" | "primeiroPrazo" | "prazo">[]): number[] {
  const anos = new Set<number>([anoAtual(), ...campanhas.map(anoDaCampanha)]);
  return [...anos].sort((a, b) => b - a);
}

/** Ano escolhido no endereço (?ano=2025); sem escolha, ou inválido, é o ano atual. */
export function escolherAno(param: string | undefined, anos: number[]): number {
  const n = Number(param);
  return Number.isInteger(n) && anos.includes(n) ? n : anoAtual();
}
