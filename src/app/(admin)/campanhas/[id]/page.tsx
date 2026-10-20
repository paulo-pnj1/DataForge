import { FormAcao } from "@/components/FormAcao";
import Link from "next/link";
import { notFound } from "next/navigation";
import { campanhaDetalhe } from "@/lib/consultas";
import { dataISO, diasRestantes, fmtData, fmtDataHora, kz, pluralDias } from "@/lib/format";
import { corresponde, paginar, umDe } from "@/lib/lista";
import { periodicidadeDe, periodosEmCurso, totalPeriodos, unidadeDe } from "@/lib/recorrencia";
import { anoDaCampanha } from "@/lib/ano";
import { Aviso, Progresso, Tag, rotuloInscricao, tomInscricao } from "@/components/ui";
import { Filtro } from "@/components/Filtro";
import { Paginacao } from "@/components/Paginacao";
import { BotaoConfirmar } from "@/components/BotaoConfirmar";
import { BarraLote } from "@/components/BarraLote";
import { SeletorProfessores } from "@/components/Seletor";
import { db, listar } from "@/lib/db";
import type { Professor } from "@/lib/tipos";
import { requireInstituicao } from "@/lib/auth";
import {
  acaoEmLote,
  adicionarProfessoresACampanha,
  alternarCampanha,
  anularPagamentoMao,
  apagarCampanha,
  editarCampanha,
  editarValorDevido,
  enviarLembretesAgora,
  inscreverTodos,
  registarPagamentoMao,
  removerInscricao,
} from "../../../actions";

const ESTADOS = ["PENDENTE", "PARCIAL", "QUITADO"] as const;

// Enviar lembretes a muita gente demora: damos até 60 s à ação.
export const maxDuration = 60;

export default async function CampanhaDetalhe({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ erro?: string; ok?: string; q?: string; estado?: string; pagina?: string }>;
}) {
  const inst = await requireInstituicao();
  const { id } = await params;
  const sp = await searchParams;
  const q = (sp.q ?? "").trim();
  const estado = umDe(sp.estado, ESTADOS);

  const c = await campanhaDetalhe(inst.id, id);
  if (!c) notFound();

  const todosProfessores = await listar<Professor>(
    db().collection("professores").where("instituicaoId", "==", inst.id).where("ativo", "==", true),
  );
  const jaNaCampanha = new Set(c.inscricoes.map((i) => i.professor.id));
  const porAdicionar = todosProfessores
    .filter((p) => !jaNaCampanha.has(p.id))
    .sort((a, b) => a.nome.localeCompare(b.nome, "pt"))
    .map((p) => ({ id: p.id, nome: p.nome, telefone: p.telefone }));

  const pago = c.inscricoes.reduce((s, i) => s + i.valorPago, 0);
  const devido = c.inscricoes.reduce((s, i) => s + i.valorDevido, 0);
  const dias = diasRestantes(c.prazo);
  const emCurso = c.recorrente && c.primeiroPrazo && totalPeriodos(c) > 0 ? periodosEmCurso(c) : 0;
  const un = unidadeDe(c);
  const semanal = periodicidadeDe(c) === "SEMANAL";

  const filtradas = c.inscricoes.filter(
    (i) => corresponde([i.professor.nome, i.professor.telefone], q) && (estado === "" || i.status === estado),
  );
  const pg = paginar(filtradas, sp.pagina);
  const base = `/campanhas/${c.id}`;

  return (
    <>
      <div className="topo">
        <div>
          <Link className="voltar" href="/campanhas">← Campanhas</Link>
          <h1>{c.nome}</h1>
          <p>
            {c.recorrente
              ? `${semanal ? "Semanal" : "Mensal"}: cota de ${kz(c.valorTotal)} · ${un.singular} ${emCurso} de ${totalPeriodos(c)} em curso · termina em ${fmtData(c.prazo)}`
              : `Prazo ${fmtData(c.prazo)} · ${dias >= 0 ? `restam ${pluralDias(dias)}` : "prazo terminado"}`}
            {" "}· lembretes a cada {Math.round(c.frequenciaHoras / 24)} dia(s)
          </p>
        </div>
        <div className="acoes">
          <Link className="btn sec" href={`${base}/ranking`}>Ranking</Link>
          <Link className="btn sec" href={`${base}/imprimir`}>Imprimir lista</Link>
          <FormAcao action={inscreverTodos}>
            <input type="hidden" name="id" value={c.id} />
            <button className="btn sec" type="submit">Incluir todos os participantes</button>
          </FormAcao>
          {c.ativa && (
            <FormAcao action={enviarLembretesAgora}>
              <input type="hidden" name="id" value={c.id} />
              <button className="btn" type="submit">Enviar lembretes agora</button>
            </FormAcao>
          )}
          <FormAcao action={alternarCampanha}>
            <input type="hidden" name="id" value={c.id} />
            <button className="btn ghost" type="submit">{c.ativa ? "Encerrar campanha" : "Reativar campanha"}</button>
          </FormAcao>
        </div>
      </div>
      <Aviso erro={sp.erro} ok={sp.ok} />

      <section className="bloco">
        <div className="valor-grande">
          {kz(pago)} <small>de {kz(devido)} pagos (comprovativos aprovados + em mãos)</small>
        </div>
        <div style={{ marginTop: 12 }}>
          <Progresso pago={pago} total={devido} grande />
        </div>
        <div className="meta">
          <span>{c.inscricoes.filter((i) => i.status === "QUITADO").length} de {c.inscricoes.length} participantes com tudo pago</span>
          <span>Faltam {kz(Math.max(0, devido - pago))}</span>
        </div>
      </section>

      <details className="bloco" open={c.inscricoes.length === 0 && porAdicionar.length > 0}>
        <summary>Adicionar participantes ({porAdicionar.length} fora da campanha)</summary>
        <FormAcao action={adicionarProfessoresACampanha} style={{ display: "grid", gap: 14 }}>
          <input type="hidden" name="id" value={c.id} />
          <SeletorProfessores
            key={porAdicionar.length}
            professores={porAdicionar}
            vazio="Todos os participantes ativos já estão nesta campanha."
          />
          <div><button className="btn" type="submit">Adicionar escolhidos</button></div>
        </FormAcao>
      </details>

      <details className="bloco" open={Boolean(sp.erro)}>
        <summary>Editar campanha</summary>
        <FormAcao action={editarCampanha} style={{ display: "grid", gap: 16 }}>
          <input type="hidden" name="id" value={c.id} />
          <div className="campos">
            <div>
              <label htmlFor="nome">Nome</label>
              <input id="nome" name="nome" defaultValue={c.nome} required />
            </div>
            <div>
              <label htmlFor="valorTotal">{c.recorrente ? `Cota ${semanal ? "semanal" : "mensal"} por participante (Kz)` : "Valor total por participante (Kz)"}</label>
              <input id="valorTotal" name="valorTotal" inputMode="numeric" defaultValue={c.valorTotal} required />
            </div>
            <div>
              <label htmlFor="prazo">{c.recorrente ? `Prazo ${semanal ? "da 1.ª semana" : "do 1.º mês"}` : "Prazo final"}</label>
              <input id="prazo" name="prazo" type="date" defaultValue={dataISO(c.recorrente && c.primeiroPrazo ? c.primeiroPrazo : c.prazo)} required />
            </div>
            <div>
              <label htmlFor="ano">Ano</label>
              <input id="ano" name="ano" type="number" min={2000} max={2100} defaultValue={anoDaCampanha(c)} required />
            </div>
            {c.recorrente && (
              <div>
                <label htmlFor="periodos">Nº de {un.plural}</label>
                <input id="periodos" name="periodos" type="number" min={1} max={120} defaultValue={totalPeriodos(c)} required />
              </div>
            )}
            <div>
              <label htmlFor="frequenciaDias">Lembrar a cada (dias)</label>
              <input id="frequenciaDias" name="frequenciaDias" type="number" min={1} defaultValue={Math.max(1, Math.round(c.frequenciaHoras / 24))} required />
            </div>
          </div>
          {c.recorrente ? (
            <p className="suave" style={{ fontSize: 13 }}>
              Mudar a cota, o 1.º prazo ou o nº de períodos recalcula o valor devido de todos (quem tem cota própria mantém-na).
            </p>
          ) : (
            <div className="linha-check">
              <input id="aplicarATodos" name="aplicarATodos" type="checkbox" />
              <label htmlFor="aplicarATodos">Aplicar o novo valor também a quem ainda não pagou tudo (quem já pagou o suficiente é avisado)</label>
            </div>
          )}
          <div><button className="btn" type="submit">Guardar alterações</button></div>
        </FormAcao>
      </details>

      <section className="bloco">
        <h2>Participantes</h2>
        {c.inscricoes.length === 0 ? (
          <p className="vazio">Ninguém está nesta campanha. Usa “Incluir todos os participantes”.</p>
        ) : (
          <>
            <Filtro
              base={base}
              q={q}
              placeholder="Nome ou número"
              selects={[
                {
                  nome: "estado",
                  rotulo: "Estado",
                  valor: estado,
                  opcoes: [
                    { valor: "", rotulo: "Todos" },
                    { valor: "PENDENTE", rotulo: rotuloInscricao.PENDENTE },
                    { valor: "PARCIAL", rotulo: rotuloInscricao.PARCIAL },
                    { valor: "QUITADO", rotulo: rotuloInscricao.QUITADO },
                  ],
                },
              ]}
            />
            {pg.total === 0 ? (
              <p className="vazio">Nenhum participante corresponde à pesquisa.</p>
            ) : (
              <FormAcao id="lote" action={acaoEmLote} reiniciar>
                <input type="hidden" name="campanhaId" value={c.id} />
                <BarraLote formId="lote" />
              </FormAcao>
            )}
            {pg.total > 0 && (
              <div className="tabela-rolagem">
                <table className="tabela">
                  <thead>
                    <tr>
                      <th className="selecionar"><span className="sr-only">Marcar</span></th><th>Participante</th><th>{c.recorrente ? `Cota ${semanal ? "semanal" : "mensal"}` : "Valor devido"}</th><th className="num">Pago</th><th className="num">Falta</th><th>Estado</th><th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {pg.itens.map((i) => (
                      <tr key={i.id}>
                        <td data-label="Marcar" className="selecionar">
                          <input type="checkbox" name="inscricaoIds" value={i.id} form="lote" aria-label={`Marcar ${i.professor.nome}`} />
                        </td>
                        <td data-label="Participante">
                          <Link href={`/participantes/${i.professor.id}`}>{i.professor.nome}</Link>
                          <div className="suave" style={{ fontSize: 13 }}>+{i.professor.telefone}</div>
                        </td>
                        <td data-label="Valor devido" className="largo">
                          <FormAcao action={editarValorDevido} className="form-linha">
                            <input type="hidden" name="campanhaId" value={c.id} />
                            <input type="hidden" name="inscricaoId" value={i.id} />
                            <input
                              name="valor"
                              type="number"
                              min={1}
                              step={1}
                              defaultValue={c.recorrente ? (i.valorPeriodo ?? c.valorTotal) : i.valorDevido}
                              aria-label={`${c.recorrente ? "Cota" : "Valor devido"} de ${i.professor.nome} em Kz`}
                              style={{ maxWidth: 120 }}
                              required
                            />
                            <button className="btn sec mini" type="submit">Guardar</button>
                          </FormAcao>
                          {c.recorrente && <div className="suave" style={{ fontSize: 13, marginTop: 4 }}>Devido até agora: {kz(i.valorDevido)}</div>}
                          {i.saldoInicial ? <div className="suave" style={{ fontSize: 13, marginTop: 4 }}>Inclui {kz(i.saldoInicial)} do ano anterior</div> : null}
                        </td>
                        <td data-label="Pago" className="num">{kz(i.valorPago)}</td>
                        <td data-label="Falta" className="num">{kz(Math.max(0, i.valorDevido - i.valorPago))}</td>
                        <td data-label="Estado">
                          <Tag tom={tomInscricao[i.status]}>{rotuloInscricao[i.status]}</Tag>{" "}
                          {i.atraso > 0 && <><Tag tom="erro">{i.atraso} {i.atraso === 1 ? un.singular : un.plural} em atraso</Tag>{" "}</>}
                          {i.valorTransitado ? <><Tag>Transitou {kz(i.valorTransitado)}</Tag>{" "}</> : null}
                          {i.comprovativos.length > 0 && <Tag tom="aviso">{i.comprovativos.length} em análise</Tag>}
                          <div className="suave" style={{ fontSize: 13, marginTop: 2 }}>
                            {i.status === "QUITADO"
                              ? "lembretes parados"
                              : `${i.lembretesEnviados} lembretes${i.ultimoLembreteEm ? ` · último ${fmtDataHora(i.ultimoLembreteEm)}` : ""}`}
                          </div>
                        </td>
                        <td className="celula-acoes">
                          <details className="gerir">
                            <summary className="btn sec mini">Gerir</summary>
                            <div className="gerir-painel">
                              <FormAcao reiniciar action={registarPagamentoMao} style={{ display: "grid", gap: 8 }}>
                                <input type="hidden" name="campanhaId" value={c.id} />
                                <input type="hidden" name="inscricaoId" value={i.id} />
                                <label htmlFor={`mao-${i.id}`}>Pagou em mãos</label>
                                <div className="form-linha">
                                  <input
                                    id={`mao-${i.id}`}
                                    name="valor"
                                    type="number"
                                    min={1}
                                    step={1}
                                    placeholder="Valor (Kz)"
                                    required
                                  />
                                  <input name="nota" placeholder="Nota (opcional)" aria-label="Nota" maxLength={120} />
                                </div>
                                <div className="form-linha" style={{ justifyContent: "space-between" }}>
                                  <span className="linha-check">
                                    <input id={`avisar-${i.id}`} name="avisar" type="checkbox" defaultChecked />
                                    <label htmlFor={`avisar-${i.id}`}>Avisar no WhatsApp</label>
                                  </span>
                                  <button className="btn sec mini" type="submit">Registar</button>
                                </div>
                              </FormAcao>
                              {i.manuais.map((m) => (
                                <FormAcao key={m.id} action={anularPagamentoMao} className="pagamento-mao">
                                  <input type="hidden" name="campanhaId" value={c.id} />
                                  <input type="hidden" name="id" value={m.id} />
                                  <span className="suave">
                                    {kz(m.valor)} em mãos · {fmtData(m.registadoEm)}
                                    {m.nota ? ` · ${m.nota}` : ""}
                                  </span>
                                  <BotaoConfirmar mensagem={`Anular o pagamento em mãos de ${kz(m.valor)}?`}>Anular</BotaoConfirmar>
                                </FormAcao>
                              ))}
                              <hr />
                              <FormAcao action={removerInscricao}>
                                <input type="hidden" name="campanhaId" value={c.id} />
                                <input type="hidden" name="inscricaoId" value={i.id} />
                                <BotaoConfirmar mensagem={`Remover ${i.professor.nome} desta campanha? Os comprovativos e pagamentos dele nesta campanha também são apagados.`}>
                                  Remover da campanha
                                </BotaoConfirmar>
                              </FormAcao>
                            </div>
                          </details>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <Paginacao
              base={base}
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

      <section className="bloco zona-perigo">
        <h2>Apagar campanha</h2>
        <p className="suave">
          Apaga a campanha e as inscrições, comprovativos e pagamentos em mãos dela. Não dá para desfazer.
          Se só queres parar os lembretes, usa “Encerrar campanha”.
        </p>
        <FormAcao action={apagarCampanha}>
          <input type="hidden" name="ir" value="/campanhas" />
          <input type="hidden" name="id" value={c.id} />
          <BotaoConfirmar className="btn perigo" mensagem={`Apagar a campanha "${c.nome}" com todos os pagamentos? Não dá para desfazer.`}>
            Apagar campanha
          </BotaoConfirmar>
        </FormAcao>
      </section>
    </>
  );
}
