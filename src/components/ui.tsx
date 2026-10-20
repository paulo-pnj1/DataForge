import type { ReactNode } from "react";

export function Aviso({ erro, ok }: { erro?: string; ok?: string }) {
  if (erro) return <div className="aviso erro" role="alert">{erro}</div>;
  if (ok) return <div className="aviso ok" role="status">{ok}</div>;
  return null;
}

export function Tag({ tom = "neutro", children }: { tom?: "neutro" | "ok" | "aviso" | "erro"; children: ReactNode }) {
  return <span className={`tag ${tom === "neutro" ? "" : tom}`}>{children}</span>;
}

export function Progresso({ pago, total, grande }: { pago: number; total: number; grande?: boolean }) {
  const pct = total > 0 ? Math.min(100, Math.round((pago / total) * 100)) : 0;
  return (
    <div
      className={`progresso ${grande ? "grande" : ""}`}
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="Valor aprovado"
    >
      <i style={{ width: `${pct}%` }} />
    </div>
  );
}

export const rotuloInscricao = { PENDENTE: "Por pagar", PARCIAL: "Parcial", QUITADO: "Pago" } as const;
export const tomInscricao = { PENDENTE: "erro", PARCIAL: "aviso", QUITADO: "ok" } as const;

/** Logotipo: símbolo + nome. Imagens pequenas em WebP (≈ 17 KB no total). */
export function Marca({ compacta }: { compacta?: boolean }) {
  return (
    <span className={`marca${compacta ? " compacta" : ""}`}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className="marca-simbolo" src="/logo-simbolo.webp" alt="" width={150} height={82} decoding="async" />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className="marca-texto" src="/logo-texto.webp" alt="MuanaPay" width={300} height={54} decoding="async" />
    </span>
  );
}

/** Logotipo grande (página de entrada). */
export function LogoGrande() {
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img className="logo-grande" src="/logo-completo.webp" alt="MuanaPay, uma solução da muanaOS" width={400} height={267} decoding="async" />
  );
}
