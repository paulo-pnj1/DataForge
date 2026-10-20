"use client";

import { useEffect, useState } from "react";
import { BotaoConfirmar } from "./BotaoConfirmar";

/**
 * Barra de ações para os professores marcados na tabela. As caixas de seleção ficam nas linhas
 * (ligadas a este formulário pelo atributo `form`), por isso aqui só se conta e se marca a página toda.
 */
export function BarraLote({ formId }: { formId: string }) {
  const [n, setN] = useState(0);
  const [todos, setTodos] = useState(false);

  useEffect(() => {
    const caixas = () => Array.from(document.querySelectorAll<HTMLInputElement>(`input[name="inscricaoIds"][form="${formId}"]`));
    const contar = () => {
      const c = caixas();
      const marcadas = c.filter((x) => x.checked).length;
      setN(marcadas);
      setTodos(c.length > 0 && marcadas === c.length);
    };
    const aoMudar = (e: Event) => {
      const alvo = e.target as HTMLElement | null;
      if (alvo instanceof HTMLInputElement && alvo.name === "inscricaoIds" && alvo.getAttribute("form") === formId) contar();
    };
    const form = document.getElementById(formId);
    const aoLimpar = () => setTimeout(contar, 0);
    document.addEventListener("change", aoMudar);
    form?.addEventListener("reset", aoLimpar);
    contar();
    return () => {
      document.removeEventListener("change", aoMudar);
      form?.removeEventListener("reset", aoLimpar);
    };
  }, [formId]);

  function marcarTodos(valor: boolean) {
    document
      .querySelectorAll<HTMLInputElement>(`input[name="inscricaoIds"][form="${formId}"]`)
      .forEach((c) => { c.checked = valor; });
    setN(valor ? document.querySelectorAll(`input[name="inscricaoIds"][form="${formId}"]`).length : 0);
    setTodos(valor);
  }

  return (
    <div className="barra-lote">
      <label className="linha-check">
        <input type="checkbox" checked={todos} onChange={(e) => marcarTodos(e.target.checked)} />
        <span>Marcar a página</span>
      </label>
      <span className="suave" aria-live="polite">{n} {n === 1 ? "marcado" : "marcados"}</span>
      <div className="acoes">
        <BotaoConfirmar
          className="btn mini"
          name="acao"
          value="lembrar"
          disabled={n === 0}
          mensagem={`Enviar um lembrete no WhatsApp a ${n} ${n === 1 ? "participante" : "participantes"}? Quem já pagou tudo é ignorado.`}
        >
          Enviar lembrete
        </BotaoConfirmar>
        <BotaoConfirmar
          className="btn perigo mini"
          name="acao"
          value="remover"
          disabled={n === 0}
          mensagem={`Remover ${n} ${n === 1 ? "participante" : "participantes"} desta campanha? Os comprovativos e pagamentos deles nesta campanha também são apagados.`}
        >
          Remover da campanha
        </BotaoConfirmar>
      </div>
    </div>
  );
}
