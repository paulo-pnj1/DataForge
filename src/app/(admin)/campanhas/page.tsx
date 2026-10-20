import { FormAcao } from "@/components/FormAcao";
import Link from "next/link";
import { campanhasComInscricoes, todasAsCampanhas } from "@/lib/consultas";
import { anoAtual, anoDaCampanha, anosDisponiveis, escolherAno } from "@/lib/ano";
import { DIAS_SEMANA, totalPeriodos, unidadeDe } from "@/lib/recorrencia";
import { SeletorAno } from "@/components/SeletorAno";
import { fmtData, kz } from "@/lib/format";
import { corresponde, paginar, umDe } from "@/lib/lista";
import { Aviso, Progresso, Tag } from "@/components/ui";
import { Filtro } from "@/components/Filtro";
import { Paginacao } from "@/components/Paginacao";
import { BotaoConfirmar } from "@/components/BotaoConfirmar";
import { requireInstituicao } from "@/lib/auth";
import { QuemRecebe } from "@/components/Seletor";
import { db, listar } from "@/lib/db";
import type { Professor } from "@/lib/tipos";
import { apagarCampanha, criarCampanha } from "../../actions";

const ESTADOS = ["ativas", "encerradas"] as const;

export default async function Campanhas({
  searchParams,
}: {
  searchParams: Promise<{ erro?: string; ok?: string; q?: string; estado?: string; pagina?: string; ano?: string }>;
}) {
  const inst = await requireInstituicao();
  const sp = await searchParams;
  const q = (sp.q ?? "").trim();
  const estado = umDe(sp.estado, ESTADOS);

  const semInscricoes = await todasAsCampanhas(inst.id);
  const anos = anosDisponiveis(semInscricoes);
  const ano = escolherAno(sp.ano, anos);
  const [todas, professoresAtivos] = await Promise.all([
    campanhasComInscricoes(inst.id, false, ano),
    listar<Professor>(db().collection("professores").where("instituicaoId", "==", inst.id).where("ativo", "==", true)),
  ]);
  const paraEscolher = professoresAtivos
    .sort((a, b) => a.nome.localeCompare(b.nome, "pt"))
    .map((p) => ({ id: p.id, nome: p.nome, telefone: p.telefone }));
  const filtradas = todas.filter((c) => corresponde([c.nome], q) && (estado === "" || (estado === "ativas") === c.ativa));
  const pg = paginar(filtradas, sp.pagina);

  return (
    <>
      <div className="topo">
        <div>
          <h1>Campanhas</h1>
          <p>Uma campanha por ano (ex.: Campanha 2026): pagamento único, ou cotas mensais ou semanais em que o que ficar por pagar soma-se à seguinte.</p>
        </div>
        <SeletorAno anos={anos} atual={ano} />
      </div>
      <Aviso erro={sp.erro} ok={sp.ok} />

      <details className="bloco" open={todas.length === 0 || Boolean(sp.erro)}>
        <summary>Nova campanha</summary>
        <FormAcao action={criarCampanha} style={{ display: "grid", gap: 16 }}>
          <div className="campos">
            <div>
              <label htmlFor="ano">Ano</label>
              <input id="ano" name="ano" type="number" min={2000} max={2100} defaultValue={anoAtual()} required />
            </div>
            <div>
              <label htmlFor="nome">Nome (se ficar vazio: “Campanha {anoAtual()}”)</label>
              <input id="nome" name="nome" placeholder={`Campanha ${anoAtual()}`} />
            </div>
            <div>
              <label htmlFor="repete">Pagamento</label>
              <select id="repete" name="repete" defaultValue="UNICA">
                <option value="UNICA">Único (um valor, um prazo)</option>
                <option value="MENSAL">Todos os meses (cotas mensais)</option>
                <option value="SEMANAL">Todas as semanas (cotas semanais)</option>
              </select>
            </div>
            <div>
              <label htmlFor="valorTotal">Valor por participante (Kz): total, ou cota de cada mês/semana</label>
              <input id="valorTotal" name="valorTotal" inputMode="numeric" placeholder="5000" required />
            </div>
            <div>
              <label htmlFor="prazo">Prazo final (só no pagamento único)</label>
              <input id="prazo" name="prazo" type="date" />
            </div>
            <div>
              <label htmlFor="diaMes">Dia do mês do prazo (só mensal)</label>
              <input id="diaMes" name="diaMes" type="number" min={1} max={31} defaultValue={10} />
            </div>
            <div>
              <label htmlFor="diaSemana">Dia da semana do prazo (só semanal)</label>
              <select id="diaSemana" name="diaSemana" defaultValue="3">
                {DIAS_SEMANA.map((d, k) => <option key={d} value={k}>{d}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor="inicio">Começa em (mensal e semanal; vazio = 1 de janeiro)</label>
              <input id="inicio" name="inicio" type="date" />
            </div>
            <div>
              <label htmlFor="frequenciaDias">Lembrar a cada (dias)</label>
              <input id="frequenciaDias" name="frequenciaDias" type="number" min={1} defaultValue={2} required />
            </div>
          </div>
          <p className="suave" style={{ fontSize: 13 }}>
            Nas campanhas mensais e semanais há um prazo por mês (no dia escolhido) ou por semana (no dia da semana escolhido),
            até 31 de dezembro. A cota de um mês entra em curso no dia 1 desse mês; a de uma semana, na segunda-feira dessa semana.
            O que ficar por pagar continua em dívida e soma-se à cota seguinte: quem não pagou uma quarta-feira deve, na seguinte,
            as duas. Cada pagamento abate primeiro o que é mais antigo. Não há juros.
          </p>
          <div className="campos">
            <div>
              <label htmlFor="copiarDe">Copiar os participantes de</label>
              <select id="copiarDe" name="copiarDe" defaultValue="">
                <option value="">Ninguém: escolho abaixo</option>
                {semInscricoes
                  .sort((a, b) => b.criadaEm.getTime() - a.criadaEm.getTime())
                  .map((c) => <option key={c.id} value={c.id}>{c.nome} ({anoDaCampanha(c)})</option>)}
              </select>
            </div>
          </div>
          <div className="linha-check">
            <input id="transitar" name="transitar" type="checkbox" defaultChecked />
            <label htmlFor="transitar">Passar as dívidas para a nova campanha (só se a campanha copiada já terminou; ela fecha)</label>
          </div>
          <QuemRecebe professores={paraEscolher} />
          <div><button className="btn" type="submit">Criar campanha</button></div>
        </FormAcao>
      </details>

      <section className="bloco">
        <h2>Todas as campanhas</h2>
        {todas.length === 0 ? (
          <p className="vazio">Ainda não criaste nenhuma campanha.</p>
        ) : (
          <>
            <Filtro
              base="/campanhas"
              q={q}
              placeholder="Nome da campanha"
              selects={[
                {
                  nome: "estado",
                  rotulo: "Estado",
                  valor: estado,
                  opcoes: [
                    { valor: "", rotulo: "Todas" },
                    { valor: "ativas", rotulo: "Ativas" },
                    { valor: "encerradas", rotulo: "Encerradas" },
                  ],
                },
              ]}
            />
            {pg.total === 0 ? (
              <p className="vazio">Nenhuma campanha corresponde à pesquisa.</p>
            ) : (
              <div className="tabela-rolagem">
                <table className="tabela">
                  <thead>
                    <tr><th>Campanha</th><th>Prazo</th><th className="num">Total</th><th style={{ width: 180 }}>Progresso</th><th>Estado</th><th></th></tr>
                  </thead>
                  <tbody>
                    {pg.itens.map((c) => {
                      const pago = c.inscricoes.reduce((s, i) => s + i.valorPago, 0);
                      const devido = c.inscricoes.reduce((s, i) => s + i.valorDevido, 0);
                      return (
                        <tr key={c.id}>
                          <td data-label="Campanha"><Link href={`/campanhas/${c.id}`}>{c.nome}</Link></td>
                          <td data-label="Prazo" className="suave">{c.recorrente ? `${totalPeriodos(c)} ${unidadeDe(c).plural} · até ${fmtData(c.prazo)}` : fmtData(c.prazo)}</td>
                          <td data-label="Total" className="num">{c.recorrente ? `${kz(c.valorTotal)}/${unidadeDe(c).singular}` : kz(c.valorTotal)}</td>
                          <td data-label="Progresso"><Progresso pago={pago} total={devido} /></td>
                          <td data-label="Estado">{c.ativa ? <Tag tom="ok">ativa</Tag> : <Tag>encerrada</Tag>}</td>
                          <td className="celula-acoes">
                            <div className="linha-acoes">
                              <Link className="btn sec mini" href={`/campanhas/${c.id}`}>Abrir / editar</Link>
                              <FormAcao action={apagarCampanha}>
                                <input type="hidden" name="id" value={c.id} />
                                <BotaoConfirmar mensagem={`Apagar a campanha "${c.nome}"? Isto apaga as inscrições, comprovativos e pagamentos dela. Não dá para desfazer.`}>
                                  Apagar
                                </BotaoConfirmar>
                              </FormAcao>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            <Paginacao
              base="/campanhas"
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
