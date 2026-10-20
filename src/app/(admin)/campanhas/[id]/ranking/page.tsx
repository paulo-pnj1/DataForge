import Link from "next/link";
import { notFound } from "next/navigation";
import { requireInstituicao } from "@/lib/auth";
import { campanhaDetalhe } from "@/lib/consultas";
import { dataISO, fmtData, fmtDataHora, kz } from "@/lib/format";
import { ranking, textoRanking, totais } from "@/lib/relatorio";
import { Progresso, Tag } from "@/components/ui";
import { CopiarBotao } from "@/components/CopiarBotao";
import { BotaoImagem } from "@/components/BotaoImagem";

const slug = (t: string) =>
  t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "campanha";

const MEDALHAS = ["🥇", "🥈", "🥉"];

export default async function RankingCampanha({ params }: { params: Promise<{ id: string }> }) {
  const inst = await requireInstituicao();
  const { id } = await params;
  const c = await campanhaDetalhe(inst.id, id);
  if (!c) notFound();

  const t = totais(c.inscricoes);
  const { posicoes, semPagamento } = ranking(c.inscricoes);
  const podio = posicoes.slice(0, 3);
  const resto = posicoes.slice(3);
  const pctGeral = t.devido > 0 ? Math.min(100, Math.round((t.pago / t.devido) * 100)) : 0;

  const texto = textoRanking({
    instituicao: inst.nome,
    campanha: c.nome,
    prazo: fmtData(c.prazo),
    posicoes,
    totais: t,
  });

  return (
    <>
      <div className="topo">
        <div>
          <Link className="voltar" href={`/campanhas/${c.id}`}>← {c.nome}</Link>
          <h1>Ranking</h1>
          <p>
            Quem pagou tudo primeiro lidera; depois, quem já pagou a maior parte. Atualizado em {fmtDataHora(new Date())}.
          </p>
        </div>
        {posicoes.length > 0 && (
          <div className="acoes">
            <BotaoImagem alvo="captura-ranking" ficheiro={`ranking-${slug(c.nome)}-${dataISO(new Date())}`} ocultarValores />
            <CopiarBotao valor={texto} rotulo="Copiar para o WhatsApp" />
          </div>
        )}
      </div>

      <div id="captura-ranking" className="captura">
      <div className="so-captura captura-topo">
        <h2>{inst.nome}</h2>
        <p>Ranking: {c.nome} · Prazo: {fmtData(c.prazo)} · {fmtDataHora(new Date())}</p>
      </div>
      <div className="estatisticas">
        <div className="bloco"><div className="rotulo">Já pagaram tudo</div><div className="valor">{t.quitados} de {t.total}</div></div>
        <div className="bloco valor-kz"><div className="rotulo">Total pago</div><div className="valor">{kz(t.pago)}</div></div>
        <div className="bloco"><div className="rotulo">Progresso geral</div><div className="valor">{pctGeral}%</div></div>
      </div>

      {posicoes.length === 0 ? (
        <section className="bloco">
          <p className="vazio">Ainda ninguém pagou. Assim que houver pagamentos aprovados ou registados em mãos, o ranking aparece aqui.</p>
        </section>
      ) : (
        <>
          <section className="podio" aria-label="Pódio">
            {podio.map((p) => (
              <div key={p.item.id} className={`podio-item lugar-${Math.min(p.pos, 3)}`}>
                <div className="podio-medalha" aria-hidden="true">{MEDALHAS[Math.min(p.pos, 3) - 1]}</div>
                <div className="podio-lugar">{p.pos}º lugar</div>
                <div className="podio-nome">{p.item.professor.nome}</div>
                <div className="podio-valor valor-kz">{kz(p.item.valorPago)}</div>
                <div className="suave" style={{ fontSize: 13 }}>
                  {p.item.status === "QUITADO"
                    ? `pagou tudo${p.item.quitadoEm ? ` em ${fmtData(p.item.quitadoEm)}` : ""}`
                    : `${p.pct}% pago`}
                </div>
              </div>
            ))}
          </section>

          {resto.length > 0 && (
            <section className="bloco">
              <h2>Restantes classificados</h2>
              <div className="tabela-rolagem">
                <table className="tabela">
                  <thead>
                    <tr><th>Lugar</th><th>Participante</th><th>Progresso</th><th className="num valor-kz">Pago</th><th>Estado</th></tr>
                  </thead>
                  <tbody>
                    {resto.map((p) => (
                      <tr key={p.item.id}>
                        <td data-label="Lugar"><strong>{p.pos}º</strong></td>
                        <td data-label="Participante"><Link href={`/participantes/${p.item.professor.id}`}>{p.item.professor.nome}</Link></td>
                        <td data-label="Progresso" style={{ minWidth: 140 }}>
                          <Progresso pago={p.item.valorPago} total={p.item.valorDevido} />
                        </td>
                        <td data-label="Pago" className="num valor-kz">{kz(p.item.valorPago)}</td>
                        <td data-label="Estado">
                          {p.item.status === "QUITADO" ? <Tag tom="ok">Pago</Tag> : <Tag tom="aviso">{p.pct}%</Tag>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}

      <div className="so-captura captura-rodape">Gerado pela MuanaPay</div>
      </div>

      {semPagamento.length > 0 && (
        <details className="bloco">
          <summary>Ainda sem pagamentos ({semPagamento.length})</summary>
          <p className="suave" style={{ margin: 0 }}>
            {semPagamento.map((i) => i.professor.nome).join(" · ")}
          </p>
        </details>
      )}
    </>
  );
}
