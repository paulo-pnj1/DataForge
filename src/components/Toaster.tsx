"use client";

import { useEffect, useState } from "react";

type Tipo = "ok" | "erro";
type Aviso = { id: number; mensagem: string; tipo: Tipo };

const EVENTO = "muanapay:aviso";

/** Mostra uma mensagem rápida no canto do ecrã (sem recarregar nada). */
export function avisar(mensagem: string, tipo: Tipo = "ok"): void {
  window.dispatchEvent(new CustomEvent(EVENTO, { detail: { mensagem, tipo } }));
}

export function Toaster() {
  const [avisos, setAvisos] = useState<Aviso[]>([]);

  useEffect(() => {
    let n = 0;
    const aoAvisar = (e: Event) => {
      const { mensagem, tipo } = (e as CustomEvent<{ mensagem: string; tipo: Tipo }>).detail;
      const id = ++n;
      setAvisos((l) => [...l.slice(-2), { id, mensagem, tipo }]);
      window.setTimeout(() => setAvisos((l) => l.filter((a) => a.id !== id)), tipo === "erro" ? 7000 : 4000);
    };
    window.addEventListener(EVENTO, aoAvisar);
    return () => window.removeEventListener(EVENTO, aoAvisar);
  }, []);

  return (
    <div className="toasts" aria-live="polite">
      {avisos.map((a) => (
        <div
          key={a.id}
          className={`toast ${a.tipo}`}
          role={a.tipo === "erro" ? "alert" : "status"}
          onClick={() => setAvisos((l) => l.filter((x) => x.id !== a.id))}
        >
          {a.mensagem}
        </div>
      ))}
    </div>
  );
}
