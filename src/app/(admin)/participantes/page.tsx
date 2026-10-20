import { FormAcao } from "@/components/FormAcao";
import Link from "next/link";
import { db, listar } from "@/lib/db";
import type { Professor } from "@/lib/tipos";
import { Aviso, Tag } from "@/components/ui";
import { Filtro } from "@/components/Filtro";
import { Paginacao } from "@/components/Paginacao";
import { BotaoConfirmar } from "@/components/BotaoConfirmar";
import { requireInstituicao } from "@/lib/auth";
import { corresponde, paginar, umDe } from "@/lib/lista";
import { alternarProfessor, apagarProfessor, criarProfessor, importarProfessores } from "../../actions";

const ESTADOS = ["ativos", "pausados"] as const;

export default async function Professores({
  searchParams,
}: {
  searchParams: Promise<{ erro?: string; ok?: string; q?: string; estado?: string; pagina?: string }>;
}) {
  const inst = await requireInstituicao();
  const sp = await searchParams;
  const q = (sp.q ?? "").trim();
  const estado = umDe(sp.estado, ESTADOS);

  const todos = (await listar<Professor>(db().collection("professores").where("instituicaoId", "==", inst.id))).sort((a, b) =>
    a.nome.localeCompare(b.nome, "pt"),
  );
  const filtrados = todos.filter(
    (p) => corresponde([p.nome, p.telefone], q) && (estado === "" || (estado === "ativos") === p.ativo),
  );
  const pg = paginar(filtrados, sp.pagina);

  return (
    <>
      <div className="topo">
        <div>
          <h1>Participantes</h1>
          <p>Quem recebe os lembretes no WhatsApp.</p>
        </div>
      </div>
      <Aviso erro={sp.erro} ok={sp.ok} />

      <details className="bloco" open={todos.length === 0 || Boolean(sp.erro)}>
        <summary>Adicionar participante</summary>
        <FormAcao reiniciar action={criarProfessor} className="campos">
          <div>
            <label htmlFor="nome">Nome</label>
            <input id="nome" name="nome" required />
          </div>
          <div>
            <label htmlFor="telefone">WhatsApp</label>
            <input id="telefone" name="telefone" inputMode="tel" placeholder="923 456 789" required />
          </div>
          <div><button className="btn" type="submit">Adicionar participante</button></div>
        </FormAcao>
      </details>

      <details className="bloco">
        <summary>Importar vários de uma vez</summary>
        <FormAcao reiniciar action={importarProfessores} style={{ display: "grid", gap: 12 }}>
          <div>
            <label htmlFor="lista">Um participante por linha: nome; número</label>
            <textarea id="lista" name="lista" placeholder={"Maria Fernandes; 923 456 789\nJoão Pedro; 912 345 678"} required />
          </div>
          <div><button className="btn sec" type="submit">Importar lista</button></div>
        </FormAcao>
      </details>

      <section className="bloco">
        <h2>{todos.length} {todos.length === 1 ? "participante" : "participantes"}</h2>
        {todos.length === 0 ? (
          <p className="vazio">Adiciona o primeiro participante acima.</p>
        ) : (
          <>
            <Filtro
              base="/participantes"
              q={q}
              placeholder="Nome ou número"
              selects={[
                {
                  nome: "estado",
                  rotulo: "Estado",
                  valor: estado,
                  opcoes: [
                    { valor: "", rotulo: "Todos" },
                    { valor: "ativos", rotulo: "Recebem lembretes" },
                    { valor: "pausados", rotulo: "Pausados" },
                  ],
                },
              ]}
            />
            {pg.total === 0 ? (
              <p className="vazio">Nenhum participante corresponde à pesquisa.</p>
            ) : (
              <div className="tabela-rolagem">
                <table className="tabela">
                  <thead><tr><th>Nome</th><th>WhatsApp</th><th>Estado</th><th></th></tr></thead>
                  <tbody>
                    {pg.itens.map((p) => (
                      <tr key={p.id}>
                        <td data-label="Nome"><Link href={`/participantes/${p.id}`}>{p.nome}</Link></td>
                        <td data-label="WhatsApp" className="suave">+{p.telefone}</td>
                        <td data-label="Estado">{p.ativo ? <Tag tom="ok">recebe lembretes</Tag> : <Tag>pausado</Tag>}</td>
                        <td className="celula-acoes">
                          <div className="linha-acoes">
                            <Link className="btn sec mini" href={`/participantes/${p.id}`}>Editar</Link>
                            <FormAcao action={alternarProfessor}>
                              <input type="hidden" name="id" value={p.id} />
                              <button className="btn ghost mini" type="submit">{p.ativo ? "Pausar" : "Reativar"}</button>
                            </FormAcao>
                            <FormAcao action={apagarProfessor}>
                              <input type="hidden" name="id" value={p.id} />
                              <BotaoConfirmar mensagem={`Apagar ${p.nome}? Isto apaga também os pagamentos, comprovativos e mensagens dele. Não dá para desfazer.`}>
                                Apagar
                              </BotaoConfirmar>
                            </FormAcao>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <Paginacao
              base="/participantes"
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
