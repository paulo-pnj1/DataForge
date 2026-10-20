/** Linhas por página em todas as tabelas. */
export const POR_PAGINA = 10;

/** "João" -> "joao": a pesquisa ignora acentos e maiúsculas. */
export function semAcentos(s: string): string {
  return s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
}

/** Todas as palavras pesquisadas têm de aparecer em algum dos campos. */
export function corresponde(campos: (string | null | undefined)[], q: string): boolean {
  const termos = semAcentos(q).split(/\s+/).filter(Boolean);
  if (termos.length === 0) return true;
  const palheiro = semAcentos(campos.filter(Boolean).join(" "));
  return termos.every((t) => palheiro.includes(t));
}

export type Pagina<T> = { itens: T[]; pagina: number; paginas: number; total: number; inicio: number };

/** Corta a lista na página pedida (valores inválidos caem na página 1 ou na última). */
export function paginar<T>(itens: T[], paginaPedida: string | undefined, tamanho = POR_PAGINA): Pagina<T> {
  const total = itens.length;
  const paginas = Math.max(1, Math.ceil(total / tamanho));
  const n = parseInt(paginaPedida ?? "1", 10);
  const pagina = Number.isFinite(n) ? Math.min(Math.max(1, n), paginas) : 1;
  const inicio = (pagina - 1) * tamanho;
  return { itens: itens.slice(inicio, inicio + tamanho), pagina, paginas, total, inicio };
}

/** Só aceita um dos valores permitidos (evita estados inventados no URL). */
export function umDe<T extends string>(valor: string | undefined, permitidos: readonly T[]): T | "" {
  return permitidos.find((p) => p === valor) ?? "";
}
