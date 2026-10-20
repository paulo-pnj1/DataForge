"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, useTransition, type FormEvent } from "react";

type Opcao = { valor: string; rotulo: string };
type Select = { nome: string; rotulo: string; valor: string; opcoes: Opcao[] };

/**
 * Pesquisa e filtros "ao vivo": atualiza o endereço sem recarregar a página (espera 300 ms depois de escreveres).
 * `manter` guarda outros parâmetros do endereço (ex.: o separador dos comprovativos).
 */
export function Filtro({
  base,
  q,
  placeholder = "Pesquisar…",
  selects = [],
  manter = {},
}: {
  base: string;
  q: string;
  placeholder?: string;
  selects?: Select[];
  manter?: Record<string, string>;
}) {
  const router = useRouter();
  const [pendente, iniciar] = useTransition();
  const [texto, setTexto] = useState(q);
  const [valores, setValores] = useState<Record<string, string>>(() => Object.fromEntries(selects.map((s) => [s.nome, s.valor])));
  const espera = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ultimoUrl = useRef<string>("");

  // Se o endereço mudar por fora (botão "voltar"), acompanha-o.
  const chaveServidor = JSON.stringify([q, selects.map((s) => s.valor)]);
  useEffect(() => {
    if (chaveServidor === ultimoUrl.current) return;
    setTexto(q);
    setValores(Object.fromEntries(selects.map((s) => [s.nome, s.valor])));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chaveServidor]);

  useEffect(() => () => { if (espera.current) clearTimeout(espera.current); }, []);

  function aplicar(novoTexto: string, novosValores: Record<string, string>) {
    const qs = new URLSearchParams(manter);
    if (novoTexto.trim()) qs.set("q", novoTexto.trim());
    for (const [k, v] of Object.entries(novosValores)) if (v) qs.set(k, v);
    const url = qs.toString() ? `${base}?${qs}` : base;
    ultimoUrl.current = JSON.stringify([novoTexto.trim(), selects.map((s) => novosValores[s.nome] ?? "")]);
    iniciar(() => router.replace(url, { scroll: false }));
  }

  function aoEscrever(v: string) {
    setTexto(v);
    if (espera.current) clearTimeout(espera.current);
    espera.current = setTimeout(() => aplicar(v, valores), 300);
  }

  function aoEscolher(nome: string, v: string) {
    const novos = { ...valores, [nome]: v };
    setValores(novos);
    aplicar(texto, novos);
  }

  function limpar() {
    const vazios = Object.fromEntries(selects.map((s) => [s.nome, ""]));
    setTexto("");
    setValores(vazios);
    aplicar("", vazios);
  }

  function aoSubmeter(e: FormEvent) {
    e.preventDefault();
    if (espera.current) clearTimeout(espera.current);
    aplicar(texto, valores);
  }

  const ativo = Boolean(texto) || Object.values(valores).some(Boolean);

  return (
    <form className="filtro" role="search" onSubmit={aoSubmeter} aria-busy={pendente}>
      <div className="filtro-busca">
        <label htmlFor="filtro-q">Pesquisar</label>
        <input
          id="filtro-q"
          type="search"
          value={texto}
          onChange={(e) => aoEscrever(e.target.value)}
          placeholder={placeholder}
          autoComplete="off"
        />
      </div>
      {selects.map((s) => (
        <div key={s.nome}>
          <label htmlFor={`filtro-${s.nome}`}>{s.rotulo}</label>
          <select id={`filtro-${s.nome}`} value={valores[s.nome] ?? ""} onChange={(e) => aoEscolher(s.nome, e.target.value)}>
            {s.opcoes.map((o) => (
              <option key={o.valor} value={o.valor}>{o.rotulo}</option>
            ))}
          </select>
        </div>
      ))}
      <div className="filtro-acoes">
        {ativo && <button className="btn ghost" type="button" onClick={limpar}>Limpar</button>}
      </div>
    </form>
  );
}
