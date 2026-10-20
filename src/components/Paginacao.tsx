import Link from "next/link";

/** Controlos de página + "1–10 de 37". Os filtros em `params` são mantidos ao mudar de página. */
export function Paginacao({
  base,
  params,
  pagina,
  paginas,
  total,
  inicio,
  mostrados,
}: {
  base: string;
  params: Record<string, string | undefined>;
  pagina: number;
  paginas: number;
  total: number;
  inicio: number;
  mostrados: number;
}) {
  if (total === 0) return null;

  const href = (p: number) => {
    const u = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v) u.set(k, v);
    if (p > 1) u.set("pagina", String(p));
    const qs = u.toString();
    return qs ? `${base}?${qs}` : base;
  };

  const numeros = [...new Set([1, paginas, pagina - 1, pagina, pagina + 1].filter((n) => n >= 1 && n <= paginas))].sort(
    (a, b) => a - b,
  );
  const itens: (number | null)[] = [];
  numeros.forEach((n, i) => {
    if (i > 0 && n - numeros[i - 1] > 1) itens.push(null);
    itens.push(n);
  });

  return (
    <div className="paginacao">
      <span className="suave">
        {inicio + 1}–{inicio + mostrados} de {total}
      </span>
      {paginas > 1 && (
        <nav aria-label="Páginas">
          {pagina > 1 ? (
            <Link className="pag" href={href(pagina - 1)} rel="prev">‹ Anterior</Link>
          ) : (
            <span className="pag desativada" aria-disabled="true">‹ Anterior</span>
          )}
          {itens.map((n, i) =>
            n === null ? (
              <span key={`e${i}`} className="pag-num reticencias" aria-hidden="true">…</span>
            ) : (
              <Link
                key={n}
                className="pag pag-num"
                href={href(n)}
                aria-current={n === pagina ? "page" : undefined}
                aria-label={`Página ${n}`}
              >
                {n}
              </Link>
            ),
          )}
          {pagina < paginas ? (
            <Link className="pag" href={href(pagina + 1)} rel="next">Seguinte ›</Link>
          ) : (
            <span className="pag desativada" aria-disabled="true">Seguinte ›</span>
          )}
        </nav>
      )}
    </div>
  );
}
