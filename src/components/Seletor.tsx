"use client";

import { useMemo, useState } from "react";

type Prof = { id: string; nome: string; telefone: string };

const semAcentos = (s: string) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

/**
 * Lista de professores com pesquisa e caixas de seleção. Os escolhidos vão no formulário como `professorIds`,
 * mesmo os que ficarem escondidos pela pesquisa.
 */
export function SeletorProfessores({ professores, vazio = "Não há participantes para escolher." }: { professores: Prof[]; vazio?: string }) {
  const [escolhidos, setEscolhidos] = useState<Set<string>>(new Set());
  const [busca, setBusca] = useState("");

  const visiveis = useMemo(() => {
    const termos = semAcentos(busca).split(/\s+/).filter(Boolean);
    if (termos.length === 0) return professores;
    return professores.filter((p) => {
      const palheiro = semAcentos(`${p.nome} ${p.telefone}`);
      return termos.every((t) => palheiro.includes(t));
    });
  }, [professores, busca]);

  function alternar(id: string) {
    setEscolhidos((atual) => {
      const novo = new Set(atual);
      if (novo.has(id)) novo.delete(id);
      else novo.add(id);
      return novo;
    });
  }

  function marcarVisiveis() {
    setEscolhidos((atual) => new Set([...atual, ...visiveis.map((p) => p.id)]));
  }

  if (professores.length === 0) return <p className="vazio">{vazio}</p>;

  return (
    <div className="seletor">
      {[...escolhidos].map((id) => (
        <input key={id} type="hidden" name="professorIds" value={id} />
      ))}
      <div className="seletor-topo">
        <input
          type="search"
          value={busca}
          onChange={(e) => setBusca(e.target.value)}
          placeholder="Pesquisar participante"
          aria-label="Pesquisar participante"
          autoComplete="off"
        />
        <button type="button" className="btn sec mini" onClick={marcarVisiveis}>Marcar {busca ? "os que aparecem" : "todos"}</button>
        <button type="button" className="btn ghost mini" onClick={() => setEscolhidos(new Set())}>Limpar</button>
      </div>
      <div className="seletor-lista" role="group" aria-label="Participantes">
        {visiveis.length === 0 ? (
          <p className="vazio" style={{ padding: "12px" }}>Nenhum participante corresponde.</p>
        ) : (
          visiveis.map((p) => (
            <label key={p.id} className="seletor-item">
              <input type="checkbox" checked={escolhidos.has(p.id)} onChange={() => alternar(p.id)} />
              <span>
                {p.nome} <span className="suave">+{p.telefone}</span>
              </span>
            </label>
          ))
        )}
      </div>
      <div className="seletor-rodape suave">{escolhidos.size} {escolhidos.size === 1 ? "escolhido" : "escolhidos"} de {professores.length}</div>
    </div>
  );
}

/** "Quem recebe esta campanha?": todos, só alguns, ou ninguém por agora. */
export function QuemRecebe({ professores }: { professores: Prof[] }) {
  const [modo, setModo] = useState<"todos" | "escolher" | "nenhum">("todos");
  return (
    <fieldset className="quem-recebe">
      <legend>Quem recebe esta campanha?</legend>
      <div className="linha-check">
        <input id="modo-todos" type="radio" name="modo" value="todos" checked={modo === "todos"} onChange={() => setModo("todos")} />
        <label htmlFor="modo-todos">Todos os participantes ativos ({professores.length})</label>
      </div>
      <div className="linha-check">
        <input id="modo-escolher" type="radio" name="modo" value="escolher" checked={modo === "escolher"} onChange={() => setModo("escolher")} />
        <label htmlFor="modo-escolher">Só os participantes que eu escolher</label>
      </div>
      <div className="linha-check">
        <input id="modo-nenhum" type="radio" name="modo" value="nenhum" checked={modo === "nenhum"} onChange={() => setModo("nenhum")} />
        <label htmlFor="modo-nenhum">Ninguém por agora (adiciono depois)</label>
      </div>
      {modo === "escolher" && <SeletorProfessores professores={professores} vazio="Ainda não há participantes ativos. Adiciona primeiro na página Participantes." />}
    </fieldset>
  );
}
