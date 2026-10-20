import { FormAcao } from "@/components/FormAcao";
import Link from "next/link";
import { comprovativosPorEstado } from "@/lib/consultas";
import { fmtDataHora, kz } from "@/lib/format";
import { corresponde, paginar } from "@/lib/lista";
import { Aviso, Tag } from "@/components/ui";
import { Filtro } from "@/components/Filtro";
import { Paginacao } from "@/components/Paginacao";
import { BotaoConfirmar } from "@/components/BotaoConfirmar";
import { requireInstituicao } from "@/lib/auth";
import { aprovarAction, apagarComprovativoAction, corrigirComprovativoAction, rejeitarAction } from "../../actions";

const ESTADOS = [
  { chave: "PENDENTE", rotulo: "Em análise" },
  { chave: "APROVADO", rotulo: "Aprovados" },
  { chave: "REJEITADO", rotulo: "Rejeitados" },
] as const;

const POR_PAGINA_COMPROVATIVOS = 6; // cada comprovativo é um cartão grande

export default async function Comprovativos({
  searchParams,
}: {
  searchParams: Promise<{ estado?: string; erro?: string; ok?: string; q?: string; pagina?: string }>;
}) {
  const inst = await requireInstituicao();
  const sp = await searchParams;
  const q = (sp.q ?? "").trim();
  const atual = ESTADOS.find((e) => e.chave === sp.estado)?.chave ?? "PENDENTE";

  const todos = await comprovativosPorEstado(inst.id, atual, atual === "PENDENTE" ? 300 : 100);
  const filtrados = todos.filter((c) =>
    corresponde([c.inscricao.professor.nome, c.inscricao.professor.telefone, c.inscricao.campanha.nome, c.legenda], q),
  );
  const pg = paginar(filtrados, sp.pagina, POR_PAGINA_COMPROVATIVOS);

  return (
    <>
      <div className="topo">
        <div>
          <h1>Comprovativos</h1>
          <p>Enviados pelos participantes no WhatsApp. Confirma o valor ao aprovar.</p>
        </div>
      </div>
      <Aviso erro={sp.erro} ok={sp.ok} />

      <nav className="abas" aria-label="Estado">
        {ESTADOS.map((e) => (
          <Link key={e.chave} href={`/comprovativos?estado=${e.chave}`} aria-current={e.chave === atual ? "page" : undefined}>
            {e.rotulo}
          </Link>
        ))}
      </nav>

      {todos.length > 0 && (
        <Filtro base="/comprovativos" q={q} placeholder="Participante, número ou campanha" manter={{ estado: atual }} />
      )}

      {todos.length === 0 && <div className="bloco"><p className="vazio">Nenhum comprovativo aqui.</p></div>}
      {todos.length > 0 && pg.total === 0 && <div className="bloco"><p className="vazio">Nenhum comprovativo corresponde à pesquisa.</p></div>}

      {pg.itens.map((c) => {
        const falta = Math.max(0, c.inscricao.valorDevido - c.inscricao.valorPago);
        const ehImagem = c.mimeType.startsWith("image/");
        const url = `/api/comprovativos/${c.id}/arquivo`;
        return (
          <article key={c.id} className="bloco comp">
            <div className="previa">
              {ehImagem ? (
                <a href={url} target="_blank" rel="noreferrer">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={url} alt={`Comprovativo de ${c.inscricao.professor.nome}`} />
                </a>
              ) : (
                <a className="btn sec" href={url} target="_blank" rel="noreferrer">Abrir {c.nomeArquivo ?? "PDF"}</a>
              )}
            </div>
            <div>
              <h2>{c.inscricao.professor.nome}</h2>
              <div className="suave">
                {c.inscricao.campanha.nome} · recebido {fmtDataHora(c.recebidoEm)} · +{c.inscricao.professor.telefone}
              </div>
              <p style={{ margin: "10px 0 0" }}>
                Já aprovado: <b>{kz(c.inscricao.valorPago)}</b> de {kz(c.inscricao.valorDevido)} · falta <b>{kz(falta)}</b>
              </p>
              {c.legenda && <p className="suave">“{c.legenda}”</p>}

              {c.status === "PENDENTE" ? (
                <div className="decisao">
                  <FormAcao action={aprovarAction}>
                    <input type="hidden" name="id" value={c.id} />
                    <div>
                      <label htmlFor={`v-${c.id}`}>Valor deste comprovativo (Kz)</label>
                      <input id={`v-${c.id}`} name="valor" inputMode="numeric" placeholder={String(falta)} required />
                    </div>
                    <button className="btn" type="submit">Aprovar</button>
                  </FormAcao>
                  <FormAcao action={rejeitarAction}>
                    <input type="hidden" name="id" value={c.id} />
                    <div>
                      <label htmlFor={`m-${c.id}`}>Motivo da rejeição</label>
                      <input id={`m-${c.id}`} name="motivo" placeholder="Ex.: imagem cortada, valor ilegível" />
                    </div>
                    <button className="btn perigo" type="submit">Rejeitar</button>
                  </FormAcao>
                </div>
              ) : c.status === "APROVADO" ? (
                <div className="decisao">
                  <p style={{ margin: 0 }}><Tag tom="ok">Aprovado</Tag> {kz(c.valorAprovado ?? 0)}</p>
                  <FormAcao action={corrigirComprovativoAction}>
                    <input type="hidden" name="id" value={c.id} />
                    <input type="hidden" name="estado" value={atual} />
                    <div>
                      <label htmlFor={`c-${c.id}`}>Corrigir valor (Kz)</label>
                      <input id={`c-${c.id}`} name="valor" inputMode="numeric" defaultValue={c.valorAprovado ?? ""} required />
                    </div>
                    <button className="btn sec" type="submit">Atualizar</button>
                  </FormAcao>
                  <FormAcao action={apagarComprovativoAction}>
                    <input type="hidden" name="id" value={c.id} />
                    <input type="hidden" name="estado" value={atual} />
                    <BotaoConfirmar className="btn perigo mini" mensagem="Apagar este comprovativo aprovado? O valor deixa de contar para o participante.">
                      Apagar comprovativo
                    </BotaoConfirmar>
                  </FormAcao>
                </div>
              ) : (
                <div className="decisao">
                  <p style={{ margin: 0 }}><Tag tom="erro">Rejeitado</Tag> <span className="suave">{c.motivoRejeicao}</span></p>
                  <FormAcao action={apagarComprovativoAction}>
                    <input type="hidden" name="id" value={c.id} />
                    <input type="hidden" name="estado" value={atual} />
                    <BotaoConfirmar className="btn perigo mini" mensagem="Apagar este comprovativo rejeitado?">
                      Apagar comprovativo
                    </BotaoConfirmar>
                  </FormAcao>
                </div>
              )}
            </div>
          </article>
        );
      })}

      <Paginacao
        base="/comprovativos"
        params={{ estado: atual, q: q || undefined }}
        pagina={pg.pagina}
        paginas={pg.paginas}
        total={pg.total}
        inicio={pg.inicio}
        mostrados={pg.itens.length}
      />
    </>
  );
}
