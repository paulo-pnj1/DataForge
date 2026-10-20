import Link from "next/link";
import { notFound } from "next/navigation";
import { requireInstituicao } from "@/lib/auth";
import { campanhaDetalhe } from "@/lib/consultas";
import { dataISO, fmtData, fmtDataHora, kz } from "@/lib/format";
import { umDe } from "@/lib/lista";
import { faltaDe, totais } from "@/lib/relatorio";
import { rotuloInscricao } from "@/components/ui";
import { BotaoImprimir } from "@/components/BotaoImprimir";
import { BotaoImagem } from "@/components/BotaoImagem";

const slug = (t: string) =>
  t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "campanha";

const FILTROS = ["faltam"] as const;
const classeEstado = { PENDENTE: "st-falta", PARCIAL: "st-parcial", QUITADO: "st-ok" } as const;

export default async function ImprimirLista({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ so?: string }>;
}) {
  const inst = await requireInstituicao();
  const { id } = await params;
  const sp = await searchParams;
  const so = umDe(sp.so, FILTROS);

  const c = await campanhaDetalhe(inst.id, id);
  if (!c) notFound();

  const base = `/campanhas/${c.id}/imprimir`;
  const linhas = c.inscricoes.filter((i) => so === "" || faltaDe(i) > 0); // já vêm por ordem alfabética
  const t = totais(linhas);
  const nomeFicheiro = `lista-${slug(c.nome)}${so === "faltam" ? "-em-falta" : ""}-${dataISO(new Date())}`;

  return (
    <>
      <div className="nao-imprimir">
        <div className="topo">
          <div>
            <Link className="voltar" href={`/campanhas/${c.id}`}>← {c.nome}</Link>
            <h1>Imprimir lista</h1>
            <p>Pré-visualização da folha. O menu e os botões não saem no papel.</p>
          </div>
          <div className="acoes">
            <BotaoImagem alvo="folha-lista" ficheiro={nomeFicheiro} />
            <BotaoImprimir />
          </div>
        </div>
        <nav className="abas" aria-label="Quem imprimir">
          <Link href={base} aria-current={so === "" ? "page" : undefined}>Todos os participantes</Link>
          <Link href={`${base}?so=faltam`} aria-current={so === "faltam" ? "page" : undefined}>Só quem ainda falta pagar</Link>
        </nav>
      </div>

      <article className="folha" id="folha-lista">
        <header className="folha-topo">
          <h2>{inst.nome}</h2>
          <p className="folha-titulo">Lista de pagamentos: {c.nome}</p>
          <p className="folha-meta">
            Prazo: {fmtData(c.prazo)} · Emitida em {fmtDataHora(new Date())} · {linhas.length}{" "}
            {linhas.length === 1 ? "participante" : "participantes"}
            {so === "faltam" ? " (só quem ainda falta pagar)" : ""}
          </p>
        </header>

        <div className="folha-resumo">
          <div><span>Total a pagar</span><strong>{kz(t.devido)}</strong></div>
          <div><span>Total pago</span><strong>{kz(t.pago)}</strong></div>
          <div><span>Total em falta</span><strong>{kz(t.falta)}</strong></div>
        </div>

        {linhas.length === 0 ? (
          <p className="folha-vazio">Ninguém para mostrar: todos os participantes já pagaram.</p>
        ) : (
          <table className="tab-imprimir">
            <thead>
              <tr>
                <th className="n">Nº</th>
                <th>Participante</th>
                <th className="num">A pagar</th>
                <th className="num">Pago</th>
                <th className="num">Falta</th>
                <th>Estado</th>
              </tr>
            </thead>
            <tbody>
              {linhas.map((i, idx) => (
                <tr key={i.id}>
                  <td className="n">{idx + 1}</td>
                  <td>{i.professor.nome}</td>
                  <td className="num">{kz(i.valorDevido)}</td>
                  <td className="num">{kz(i.valorPago)}</td>
                  <td className="num">{kz(faltaDe(i))}</td>
                  <td className={classeEstado[i.status]}>{rotuloInscricao[i.status]}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <td></td>
                <td>TOTAL</td>
                <td className="num">{kz(t.devido)}</td>
                <td className="num">{kz(t.pago)}</td>
                <td className="num">{kz(t.falta)}</td>
                <td>{t.quitados} de {t.total} pagos</td>
              </tr>
            </tfoot>
          </table>
        )}

        {t.excedente > 0 && (
          <p className="folha-nota">
            O total pago inclui {kz(t.excedente)} entregues acima do valor devido por alguns participantes; esse valor não abate o que falta aos outros.
          </p>
        )}
        <footer className="folha-rodape">Gerado pela MuanaPay</footer>
      </article>
    </>
  );
}
