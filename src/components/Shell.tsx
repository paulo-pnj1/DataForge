"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { Marca } from "./ui";
import { Toaster } from "./Toaster";

type Item = { href: string; rotulo: string; contagem?: number };

/**
 * Estrutura do painel. Em ecrãs largos a barra lateral fica fixa;
 * em telemóvel e tablet passa a gaveta que abre pelo botão de menu.
 */
export function Shell({
  itens,
  subtitulo,
  rodape,
  children,
}: {
  itens: Item[];
  subtitulo: string;
  rodape: ReactNode;
  children: ReactNode;
}) {
  const [aberto, setAberto] = useState(false);
  const caminho = usePathname();

  useEffect(() => {
    setAberto(false); // fecha a gaveta ao navegar
  }, [caminho]);

  useEffect(() => {
    if (!aberto) return;
    const aoTecla = (e: KeyboardEvent) => {
      if (e.key === "Escape") setAberto(false);
    };
    window.addEventListener("keydown", aoTecla);
    return () => window.removeEventListener("keydown", aoTecla);
  }, [aberto]);

  const ativo = (href: string) => (href === "/" ? caminho === "/" : caminho === href || caminho.startsWith(`${href}/`));

  return (
    <div className="app">
      <header className="topbar">
        <button
          type="button"
          className="menu-btn"
          aria-label={aberto ? "Fechar menu" : "Abrir menu"}
          aria-expanded={aberto}
          aria-controls="lateral"
          onClick={() => setAberto((v) => !v)}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            {aberto ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
          </svg>
        </button>
        <Marca compacta />
      </header>

      {aberto && <div className="cortina" onClick={() => setAberto(false)} aria-hidden="true" />}

      <aside id="lateral" className={`lateral${aberto ? " aberta" : ""}`}>
        <div className="lateral-topo">
          <Marca />
          <small>{subtitulo}</small>
        </div>
        <nav aria-label="Principal">
          {itens.map((i) => (
            <Link key={i.href} href={i.href} aria-current={ativo(i.href) ? "page" : undefined}>
              {i.rotulo}
              {i.contagem ? <span className="contagem">{i.contagem}</span> : null}
            </Link>
          ))}
        </nav>
        <div className="lateral-rodape">{rodape}</div>
      </aside>

      <main className="conteudo">{children}</main>
      <Toaster />
    </div>
  );
}
