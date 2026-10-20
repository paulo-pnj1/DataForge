"use client";

import { useEffect, useRef, useState } from "react";
import { MAX_SEGMENTOS, MAX_SEPTETOS, paraSms, segmentos, septetos } from "@/lib/smsTexto";
import { SeletorProfessores } from "./Seletor";
import { BotaoConfirmar } from "./BotaoConfirmar";

type Prof = { id: string; nome: string; telefone: string };

/**
 * Campos do SMS escrito pelo admin: texto com contador (o que se paga é o número de SMS),
 * pré-visualização do que chega ao telemóvel e escolha de quem recebe.
 * Vai dentro de um <FormAcao>.
 */
export function FormularioSms({
  nomeInstituicao,
  professores,
  desativado,
}: {
  nomeInstituicao: string;
  professores: Prof[];
  desativado?: boolean;
}) {
  const raiz = useRef<HTMLDivElement>(null);
  const [texto, setTexto] = useState("");
  const [assinar, setAssinar] = useState(true);
  const [modo, setModo] = useState<"todos" | "escolher">("todos");

  // Depois de enviar, o FormAcao limpa o formulário: o estado acompanha.
  useEffect(() => {
    const form = raiz.current?.closest("form");
    if (!form) return;
    const aoLimpar = () => {
      setTexto("");
      setAssinar(true);
      setModo("todos");
    };
    form.addEventListener("reset", aoLimpar);
    return () => form.removeEventListener("reset", aoLimpar);
  }, []);

  const final = paraSms(assinar && texto.trim() ? `${nomeInstituicao}: ${texto.trim()}` : texto);
  const tamanho = septetos(final);
  const partes = segmentos(final);
  const demais = tamanho > MAX_SEPTETOS;

  return (
    <div ref={raiz} style={{ display: "grid", gap: 16 }}>
      <div>
        <label htmlFor="mensagem">Mensagem</label>
        <textarea
          id="mensagem"
          name="mensagem"
          rows={4}
          required
          disabled={desativado}
          placeholder="Ex.: A reunião dos participantes é sábado às 9h. Contamos contigo!"
          onChange={(e) => setTexto(e.target.value)}
        />
        <div className="suave" style={{ fontSize: 13, marginTop: 6, color: demais ? "var(--perigo)" : undefined }}>
          {tamanho} {tamanho === 1 ? "caractere" : "caracteres"} · {partes} {partes === 1 ? "SMS" : "SMS"} por participante
          {demais ? ` (o máximo são ${MAX_SEPTETOS} caracteres, ${MAX_SEGMENTOS} SMS)` : ""}
        </div>
      </div>

      <div className="linha-check">
        <input id="assinar" name="assinar" type="checkbox" defaultChecked disabled={desativado} onChange={(e) => setAssinar(e.target.checked)} />
        <label htmlFor="assinar">Começar com o nome da instituição ({nomeInstituicao}:)</label>
      </div>

      {final && (
        <p className="suave" style={{ fontSize: 13, whiteSpace: "pre-wrap" }}>
          Assim chega (sem acentos, para caber em menos SMS): “{final}”
        </p>
      )}

      <fieldset className="quem-recebe">
        <legend>Quem recebe?</legend>
        <div className="linha-check">
          <input id="sms-todos" type="radio" name="modo" value="todos" defaultChecked disabled={desativado} onChange={() => setModo("todos")} />
          <label htmlFor="sms-todos">Todos os participantes ativos ({professores.length})</label>
        </div>
        <div className="linha-check">
          <input id="sms-escolher" type="radio" name="modo" value="escolher" disabled={desativado} onChange={() => setModo("escolher")} />
          <label htmlFor="sms-escolher">Só os participantes que eu escolher</label>
        </div>
        {modo === "escolher" && (
          <SeletorProfessores professores={professores} vazio="Ainda não há participantes ativos. Adiciona primeiro na página Participantes." />
        )}
      </fieldset>

      <div>
        <BotaoConfirmar
          className="btn"
          disabled={desativado || demais}
          mensagem="Enviar este SMS? Cada SMS gasta créditos da Ombala e não dá para cancelar."
        >
          Enviar SMS
        </BotaoConfirmar>
      </div>
    </div>
  );
}
