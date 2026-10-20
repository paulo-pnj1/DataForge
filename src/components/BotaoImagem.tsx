"use client";

import { useEffect, useState } from "react";

type Props = {
  /** `id` do elemento que vira imagem. */
  alvo: string;
  /** Nome do ficheiro, sem extensão. */
  ficheiro: string;
  rotulo?: string;
  /** Esconde na imagem tudo o que tiver a classe `valor-kz` (valores em Kz). */
  ocultarValores?: boolean;
};

/** Gera um PNG do elemento `alvo` e descarrega-o. Em telemóveis que permitem, oferece também "Partilhar" (WhatsApp, etc.). */
export function BotaoImagem({ alvo, ficheiro, rotulo = "Baixar imagem", ocultarValores = false }: Props) {
  const [estado, setEstado] = useState<"" | "a-gerar" | "erro">("");
  const [podePartilhar, setPodePartilhar] = useState(false);

  useEffect(() => {
    try {
      const teste = new File([""], "teste.png", { type: "image/png" });
      setPodePartilhar(typeof navigator.canShare === "function" && navigator.canShare({ files: [teste] }));
    } catch {
      setPodePartilhar(false);
    }
  }, []);

  async function gerarPng(): Promise<Blob> {
    const el = document.getElementById(alvo);
    if (!el) throw new Error("Elemento não encontrado");
    const { toBlob } = await import("html-to-image");

    el.classList.add("a-capturar");
    if (ocultarValores) el.classList.add("sem-valores");
    try {
      await new Promise((r) => requestAnimationFrame(() => r(null))); // deixa o ecrã assentar com as classes novas
      const caixa = el.getBoundingClientRect();
      let escala = 2;
      // Os navegadores limitam o tamanho do canvas; em listas longas baixa-se a escala.
      while (escala > 0.5 && (caixa.width * escala) * (caixa.height * escala) > 16_000_000) escala -= 0.25;

      let fundo = getComputedStyle(el).backgroundColor;
      if (!fundo || fundo === "transparent" || fundo === "rgba(0, 0, 0, 0)") fundo = getComputedStyle(document.body).backgroundColor;

      const blob = await toBlob(el, { pixelRatio: escala, cacheBust: true, backgroundColor: fundo });
      if (!blob) throw new Error("Imagem vazia");
      return blob;
    } finally {
      el.classList.remove("a-capturar", "sem-valores");
    }
  }

  async function baixar() {
    setEstado("a-gerar");
    try {
      const blob = await gerarPng();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${ficheiro}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      setEstado("");
    } catch {
      setEstado("erro");
      setTimeout(() => setEstado(""), 3000);
    }
  }

  async function partilhar() {
    setEstado("a-gerar");
    try {
      const blob = await gerarPng();
      const file = new File([blob], `${ficheiro}.png`, { type: "image/png" });
      await navigator.share({ files: [file], title: ficheiro });
      setEstado("");
    } catch (e) {
      // Fechar a janela de partilha não é erro.
      if (e instanceof DOMException && e.name === "AbortError") setEstado("");
      else {
        setEstado("erro");
        setTimeout(() => setEstado(""), 3000);
      }
    }
  }

  const ocupado = estado === "a-gerar";
  return (
    <>
      <button className="btn sec" type="button" onClick={baixar} disabled={ocupado} aria-live="polite">
        {ocupado ? "A gerar…" : estado === "erro" ? "Não foi possível gerar" : rotulo}
      </button>
      {podePartilhar && (
        <button className="btn sec" type="button" onClick={partilhar} disabled={ocupado}>
          Partilhar
        </button>
      )}
    </>
  );
}
