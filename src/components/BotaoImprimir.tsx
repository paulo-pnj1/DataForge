"use client";

export function BotaoImprimir({ rotulo = "Imprimir" }: { rotulo?: string }) {
  return (
    <button className="btn" type="button" onClick={() => window.print()}>
      {rotulo}
    </button>
  );
}
