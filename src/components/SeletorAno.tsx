"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";

/** Escolhe o ano das campanhas que o painel mostra (?ano=2025). */
export function SeletorAno({ anos, atual }: { anos: number[]; atual: number }) {
  const router = useRouter();
  const caminho = usePathname();
  const params = useSearchParams();

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <label htmlFor="ano" style={{ margin: 0 }}>Ano</label>
      <select
        id="ano"
        value={atual}
        style={{ width: "auto" }}
        onChange={(e) => {
          const p = new URLSearchParams(params.toString());
          p.set("ano", e.target.value);
          p.delete("pagina");
          router.push(`${caminho}?${p.toString()}`);
        }}
      >
        {anos.map((a) => (
          <option key={a} value={a}>{a}</option>
        ))}
      </select>
    </div>
  );
}
