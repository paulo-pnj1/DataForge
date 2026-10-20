import Link from "next/link";
import { campanhasComInscricoes, contarPendentes, todasAsCampanhas, ultimasMensagens } from "@/lib/consultas";
import { anoAtual, anosDisponiveis, escolherAno } from "@/lib/ano";
import { SeletorAno } from "@/components/SeletorAno";
import { periodicidadeDe, unidadeDe } from "@/lib/recorrencia";
import { diasRestantes, fmtData, fmtDataHora, kz, pluralDias } from "@/lib/format";
import { corresponde, paginar, umDe } from "@/lib/lista";
import { Progresso, Tag } from "@/components/ui";
import { Filtro } from "@/components/Filtro";
import { Paginacao } from "@/components/Paginacao";
import { requireInstituicao } from "@/lib/auth";

const ESTADOS = ["recebida", "falhou", "lida", "entregue", "enviada"] as const;

export default async function Resumo({
  searchParams,
}: {
  searchParams: Promise<{ q?: string; estado?: string; pagina?: string; ano?: string }>;
}) {
  const inst = await requireInstituicao();
  const sp = await searchParams;
  const q = (sp.q ?? "").trim();
  const estado = umDe(sp.estado, ESTADOS);

  const anos = anosDisponiveis(await todasAsCampanhas(inst.id));
  const ano = escolherAno(sp.ano, anos);
  const [todasDoAno, pendentes, mensagens] = await Promise.all([
    campanhasComInscricoes(inst.id, false, ano),
    contarPendentes(inst.id),
    ultimasMensagens(inst.id, 60),
  ]);
  // No ano atual só as campanhas ativas; nos anos passados, todas.
  const campanhas = (ano === anoAtual() ? todasDoAno.filter((c) => c.ativa) : todasDoAno).sort(
    (a, b) => a.prazo.getTime() - b.prazo.getTime(),
  );

  const pagoTotal = campanhas.reduce((s, c) => s + c.inscricoes.reduce((x, i) => x + i.valorPago, 0), 0);
  const devidoTotal = campanhas.reduce((s, c) => s + c.inscricoes.reduce((x, i) => x + i.valorDevido, 0), 0);

  const estadoMsg = (m: (typeof mensagens)[number]) =>
    m.direcao === "RECEBIDA" ? "recebida" : m.status === "failed" ? "falhou" : m.status === "read" ? "lida" : m.status === "delivered" ? "entregue" : "enviada";
  const filtradas = mensagens.filter(
    (m) =>
      corresponde([m.professor?.nome, m.professor?.telefone, m.tipo], q) && (estado === "" || estadoMsg(m) === estado),
  );
  const pg = paginar(filtradas, sp.pagina);

  return (
    <>
      <div className="topo">
        <div>
          <h1>Resumo</h1>
          <p>Como está o pagamento dos participantes.</p>
        </div>
        <SeletorAno anos={anos} atual={ano} />
        {pendentes > 0 && (
          <Link className="btn" href="/comprovativos">
            {pendentes} {pendentes === 1 ? "comprovativo à espera" : "comprovativos à espera"}
          </Link>
        )}
      </div>

      {campanhas.length === 0 ? (
        <div className="bloco">
          <p className="vazio">Ainda não há campanhas ativas.</p>
          <Link className="btn" href="/campanhas">Criar campanha</Link>
        </div>
      ) : (
        <>
          <div className="estatisticas">
            <div className="bloco"><div className="rotulo">Recebido</div><div className="valor">{kz(pagoTotal)}</div></div>
            <div className="bloco"><div className="rotulo">Em falta</div><div className="valor">{kz(Math.max(0, devidoTotal - pagoTotal))}</div></div>
            <div className="bloco"><div className="rotulo">Campanhas ativas</div><div className="valor">{campanhas.length}</div></div>
          </div>
          <div className="grelha" style={{ marginBottom: 20 }}>
            {campanhas.map((c) => {
              const pago = c.inscricoes.reduce((s, i) => s + i.valorPago, 0);
              const devido = c.inscricoes.reduce((s, i) => s + i.valorDevido, 0);
              const quitados = c.inscricoes.filter((i) => i.status === "QUITADO").length;
              const dias = diasRestantes(c.prazo);
              return (
                <Link key={c.id} href={`/campanhas/${c.id}`} className="bloco campanha-cartao" style={{ marginBottom: 0 }}>
                  <h3>{c.nome}</h3>
                  <div className="suave" style={{ marginBottom: 14 }}>
                    {c.recorrente
                      ? `${periodicidadeDe(c) === "SEMANAL" ? "Semanal" : "Mensal"} · ${kz(c.valorTotal)}/${unidadeDe(c).singular} · até ${fmtData(c.prazo)}`
                      : <>Prazo {fmtData(c.prazo)} · {dias >= 0 ? `restam ${pluralDias(dias)}` : "prazo terminado"}</>}
                  </div>
                  <div className="valor-grande">
                    {kz(pago)} <small>de {kz(devido)}</small>
                  </div>
                  <div style={{ marginTop: 12 }}>
                    <Progresso pago={pago} total={devido} grande />
                  </div>
                  <div className="meta">
                    <span>{quitados} de {c.inscricoes.length} participantes com tudo pago</span>
                    <span>Faltam {kz(Math.max(0, devido - pago))}</span>
                  </div>
                </Link>
              );
            })}
          </div>
        </>
      )}

      <section className="bloco">
        <h2>Mensagens</h2>
        {mensagens.length === 0 ? (
          <p className="vazio">Nada enviado ou recebido ainda.</p>
        ) : (
          <>
            <Filtro
              base="/"
              q={q}
              placeholder="Participante, número ou tipo"
              selects={[
                {
                  nome: "estado",
                  rotulo: "Estado",
                  valor: estado,
                  opcoes: [
                    { valor: "", rotulo: "Todos" },
                    { valor: "recebida", rotulo: "Recebidas" },
                    { valor: "enviada", rotulo: "Enviadas" },
                    { valor: "entregue", rotulo: "Entregues" },
                    { valor: "lida", rotulo: "Lidas" },
                    { valor: "falhou", rotulo: "Falharam" },
                  ],
                },
              ]}
            />
            {pg.total === 0 ? (
              <p className="vazio">Nenhuma mensagem corresponde à pesquisa.</p>
            ) : (
              <div className="tabela-rolagem">
                <table className="tabela">
                  <thead>
                    <tr><th>Quando</th><th>Participante</th><th>Tipo</th><th>Estado</th></tr>
                  </thead>
                  <tbody>
                    {pg.itens.map((m) => (
                      <tr key={m.id}>
                        <td data-label="Quando" className="suave">{fmtDataHora(m.criadaEm)}</td>
                        <td data-label="Participante">{m.professor?.nome ?? "—"}</td>
                        <td data-label="Tipo">
                          {m.direcao === "RECEBIDA" ? `Recebida (${m.tipo})` : m.tipo}
                          {m.canal === "SMS" && <> <Tag>SMS</Tag></>}
                        </td>
                        <td data-label="Estado">
                          {m.direcao === "RECEBIDA" ? (
                            <Tag>recebida</Tag>
                          ) : m.status === "failed" ? (
                            <span title={m.erro ?? ""}><Tag tom="erro">falhou</Tag> <span className="suave">{m.erro}</span></span>
                          ) : m.status === "read" ? (
                            <Tag tom="ok">lida</Tag>
                          ) : m.status === "delivered" ? (
                            <Tag tom="ok">entregue</Tag>
                          ) : (
                            <Tag tom="aviso">enviada</Tag>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <Paginacao
              base="/"
              params={{ q: q || undefined, estado: estado || undefined }}
              pagina={pg.pagina}
              paginas={pg.paginas}
              total={pg.total}
              inicio={pg.inicio}
              mostrados={pg.itens.length}
            />
          </>
        )}
      </section>
    </>
  );
}
