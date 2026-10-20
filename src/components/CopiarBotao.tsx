"use client";

import { useState } from "react";

/** Copia `valor` para a área de transferência. Funciona também sem HTTPS/permissões (plano B com textarea). */
export function CopiarBotao({ valor, rotulo = "Copiar" }: { valor: string; rotulo?: string }) {
  const [estado, setEstado] = useState<"" | "ok" | "erro">("");

  async function copiar() {
    let ok = false;
    try {
      await navigator.clipboard.writeText(valor);
      ok = true;
    } catch {
      const t = document.createElement("textarea");
      t.value = valor;
      t.style.position = "fixed";
      t.style.opacity = "0";
      document.body.appendChild(t);
      t.select();
      try {
        ok = document.execCommand("copy");
      } catch {
        ok = false;
      }
      document.body.removeChild(t);
    }
    setEstado(ok ? "ok" : "erro");
    setTimeout(() => setEstado(""), 2000);
  }

  return (
    <button className="btn sec mini" type="button" onClick={copiar} aria-live="polite">
      {estado === "ok" ? "Copiado ✓" : estado === "erro" ? "Não foi possível" : rotulo}
    </button>
  );
}
