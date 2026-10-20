import { FormAcao } from "@/components/FormAcao";
import Link from "next/link";
import { db, listar } from "@/lib/db";
import type { Instituicao } from "@/lib/tipos";
import { CANAIS } from "@/lib/canais";
import { requireSuper } from "@/lib/auth";
import { fmtData } from "@/lib/format";
import { corresponde, paginar, umDe } from "@/lib/lista";
import { Aviso, Tag } from "@/components/ui";
import { Filtro } from "@/components/Filtro";
import { Paginacao } from "@/components/Paginacao";
import { alternarInstituicao, criarInstituicao, entrarInstituicao } from "../../actions";

const ESTADOS = ["ativas", "desativadas"] as const;

export default async function Instituicoes({
  searchParams,
}: {
  searchParams: Promise<{ erro?: string; ok?: string; q?: string; estado?: string; pagina?: string }>;
}) {
  await requireSuper();
  const sp = await searchParams;
  const q = (sp.q ?? "").trim();
  const estado = umDe(sp.estado, ESTADOS);

  const todas = (await listar<Instituicao>(db().collection("instituicoes"))).sort((a, b) => a.nome.localeCompare(b.nome, "pt"));
  const filtradas = todas.filter((i) => corresponde([i.nome, i.adminEmail], q) && (estado === "" || (estado === "ativas") === i.ativa));
  const pg = paginar(filtradas, sp.pagina);

  return (
    <>
      <div className="topo">
        <div>
          <h1>Instituições</h1>
          <p>Os clientes da MuanaPay. Cada um tem o seu admin, participantes e campanhas.</p>
        </div>
      </div>
      <Aviso erro={sp.erro} ok={sp.ok} />

      <details className="bloco" open={todas.length === 0 || Boolean(sp.erro)}>
        <summary>Nova instituição</summary>
        <FormAcao reiniciar action={criarInstituicao} style={{ display: "grid", gap: 16 }}>
          <div className="campos">
            <div>
              <label htmlFor="nome">Nome (aparece nas mensagens)</label>
              <input id="nome" name="nome" placeholder="EBD Igreja X" required />
            </div>
            <div>
              <label htmlFor="adminEmail">Email do admin</label>
              <input id="adminEmail" name="adminEmail" type="email" required />
            </div>
            <div>
              <label htmlFor="senha">Senha inicial (mín. 8)</label>
              <input id="senha" name="senha" type="password" minLength={8} autoComplete="new-password" required />
            </div>
          </div>
          <div className="campos">
            <div>
              <label htmlFor="canal">Canal das mensagens automáticas</label>
              <select id="canal" name="canal" defaultValue="WHATSAPP">
                {CANAIS.map((c) => <option key={c.valor} value={c.valor}>{c.rotulo}</option>)}
              </select>
            </div>
          </div>
          <div className="linha-check">
            <input id="tomCristao" name="tomCristao" type="checkbox" defaultChecked />
            <label htmlFor="tomCristao">Frases cristãs nas mensagens</label>
          </div>
          <div><button className="btn" type="submit">Criar instituição</button></div>
        </FormAcao>
      </details>

      <section className="bloco">
        <h2>{todas.length} {todas.length === 1 ? "instituição" : "instituições"}</h2>
        {todas.length === 0 ? (
          <p className="vazio">Cria a primeira instituição acima.</p>
        ) : (
          <>
            <Filtro
              base="/instituicoes"
              q={q}
              placeholder="Nome ou email do admin"
              selects={[
                {
                  nome: "estado",
                  rotulo: "Estado",
                  valor: estado,
                  opcoes: [
                    { valor: "", rotulo: "Todas" },
                    { valor: "ativas", rotulo: "Ativas" },
                    { valor: "desativadas", rotulo: "Desativadas" },
                  ],
                },
              ]}
            />
            {pg.total === 0 ? (
              <p className="vazio">Nenhuma instituição corresponde à pesquisa.</p>
            ) : (
              <div className="tabela-rolagem">
                <table className="tabela">
                  <thead>
                    <tr><th>Nome</th><th>Admin</th><th>Estado</th><th>Criada</th><th></th></tr>
                  </thead>
                  <tbody>
                    {pg.itens.map((i) => (
                      <tr key={i.id}>
                        <td data-label="Nome"><Link href={`/instituicoes/${i.id}`}>{i.nome}</Link></td>
                        <td data-label="Admin" className="suave">{i.adminEmail}</td>
                        <td data-label="Estado">{i.ativa ? <Tag tom="ok">ativa</Tag> : <Tag>desativada</Tag>}</td>
                        <td data-label="Criada" className="suave">{fmtData(i.criadaEm)}</td>
                        <td className="celula-acoes">
                          <div className="linha-acoes">
                            <form action={entrarInstituicao}>
                              <input type="hidden" name="id" value={i.id} />
                              <button className="btn mini" type="submit">Gerir</button>
                            </form>
                            <Link className="btn sec mini" href={`/instituicoes/${i.id}`}>Editar</Link>
                            <FormAcao action={alternarInstituicao}>
                              <input type="hidden" name="id" value={i.id} />
                              <button className="btn ghost mini" type="submit">{i.ativa ? "Desativar" : "Ativar"}</button>
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
              base="/instituicoes"
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
